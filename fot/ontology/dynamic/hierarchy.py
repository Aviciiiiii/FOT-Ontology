"""
Dynamic FOT Hierarchy Construction

This module implements the complete hierarchy building pipeline from the original
/src/hierarchy.py script, including:
- SciBERT embedding generation with caching
- UMAP dimensionality reduction for clustering
- Recursive HDBSCAN clustering with GPU acceleration (optional)
- Hierarchical clustering within each HDBSCAN cluster
- Layer assignment based on tree depth (L4/L5/L6 for dynamic entities)
- Parent node assignment using FAISS similarity search
- Static-dynamic layer integration with ID offset
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import cdist
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ...utils.logging import get_logger

logger = get_logger(__name__)

# Optional GPU-accelerated dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available - parent node assignment will be slower")

try:
    import cupy as cp
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("cuML/CuPy not available - falling back to CPU implementations")
    # CPU fallback imports
    from sklearn.cluster import DBSCAN
    from sklearn.manifold import TSNE


# ============ Entity Dataset for Batch Processing ============
class EntityDataset:
    """Dataset wrapper for batch embedding computation."""

    def __init__(self, entities: List[Dict[str, Any]]):
        self.entities = entities

    def __len__(self) -> int:
        return len(self.entities)

    def __getitem__(self, idx: int) -> Tuple[int, str]:
        return self.entities[idx]['id'], self.entities[idx]['name']


# ============ BERT Embedding Generation ============
def get_bert_embedding(texts: List[str], model, tokenizer, device: str) -> np.ndarray:
    """
    Compute SciBERT embeddings for a batch of texts.

    Args:
        texts: List of entity names
        model: Pretrained transformer model
        tokenizer: Corresponding tokenizer
        device: 'cuda' or 'cpu'

    Returns:
        NumPy array of shape (len(texts), hidden_dim)
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Use [CLS] token representation
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def get_or_create_embeddings(
    entities: List[Dict[str, Any]],
    embedding_file: str,
    model_path: str,
    batch_size: int = 64,
    device: str = "cpu"
) -> Dict[int, np.ndarray]:
    """
    Load cached embeddings or compute them using SciBERT.

    Args:
        entities: List of entity dicts with 'id' and 'name' keys
        embedding_file: Path to cache file (.npz format)
        model_path: Path to SciBERT model directory
        batch_size: Batch size for embedding computation
        device: 'cuda' or 'cpu'

    Returns:
        Dictionary mapping entity_id -> embedding vector
    """
    embedding_path = Path(embedding_file)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing embeddings
    entity_to_embedding = {}
    if embedding_path.exists():
        logger.info(f"Loading cached embeddings from {embedding_file}")
        loaded_data = np.load(embedding_file, allow_pickle=True)
        entity_to_embedding = dict(zip(loaded_data['ids'], loaded_data['embeddings']))
        existing_ids = set(entity_to_embedding.keys())

        # Find new entities not in cache
        new_entities = [entity for entity in entities if entity['id'] not in existing_ids]

        if not new_entities:
            logger.info(f"All {len(entities)} entities found in cache")
            return entity_to_embedding

        logger.info(f"Computing embeddings for {len(new_entities)} new entities")
    else:
        logger.info(f"Computing embeddings for all {len(entities)} entities")
        new_entities = entities

    # Load model for new embeddings
    if new_entities:
        logger.info(f"Loading SciBERT model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model = model.to(device)
        model.eval()

        # Compute embeddings in batches
        dataset = EntityDataset(new_entities)
        new_embeddings = {}

        for i in tqdm(range(0, len(dataset), batch_size), desc="Computing embeddings"):
            batch_entities = new_entities[i:i + batch_size]
            batch_ids = [e['id'] for e in batch_entities]
            batch_texts = [e['name'] for e in batch_entities]

            batch_embeddings = get_bert_embedding(batch_texts, model, tokenizer, device)

            for entity_id, embedding in zip(batch_ids, batch_embeddings):
                new_embeddings[entity_id] = embedding

        # Merge with existing embeddings
        entity_to_embedding.update(new_embeddings)

        # Save updated cache
        ids = list(entity_to_embedding.keys())
        embeddings = list(entity_to_embedding.values())
        np.savez_compressed(embedding_file, ids=ids, embeddings=embeddings)
        logger.info(f"Saved {len(ids)} embeddings to {embedding_file}")

    return entity_to_embedding


# ============ Entity Loading ============
def load_entities(file_path: str) -> List[Dict[str, Any]]:
    """
    Load entities from tab-separated file.

    Expected format:
        id\tname\tipc_code\t[layer]\t[parent]
    or:
        name\tid\tipc_code\t[layer]\t[parent]

    Args:
        file_path: Path to entity file

    Returns:
        List of entity dictionaries
    """
    entities = []
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"Entity file not found: {file_path}")
        return entities

    with open(file_path, 'r', encoding='utf-8') as f:
        header = next(f, None)  # Skip header

        for line_num, line in enumerate(f, start=2):
            parts = line.strip().split('\t')

            try:
                if len(parts) >= 2:
                    # Detect format: id first or name first
                    if parts[1].isdigit():
                        # Format: name\tid\tipc_code
                        entities.append({
                            'id': int(parts[1]),
                            'name': parts[0],
                            'ipc_code': parts[2] if len(parts) > 2 else ''
                        })
                    else:
                        # Format: id\tname\tipc_code
                        entities.append({
                            'id': int(parts[0]),
                            'name': parts[1],
                            'ipc_code': parts[2] if len(parts) > 2 else ''
                        })
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping invalid line {line_num}: {line.strip()[:50]}... Error: {e}")

    logger.info(f"Loaded {len(entities)} entities from {file_path}")
    return entities


# ============ UMAP Dimensionality Reduction ============
def apply_umap(
    embeddings: np.ndarray,
    n_components: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction.

    Args:
        embeddings: Input embeddings (N, D)
        n_components: Target dimensionality
        n_neighbors: Number of neighbors for graph construction
        min_dist: Minimum distance between points
        use_gpu: Use GPU-accelerated cuML if available

    Returns:
        Reduced embeddings (N, n_components)
    """
    if use_gpu and GPU_AVAILABLE:
        logger.info(f"Applying GPU-accelerated UMAP (n_components={n_components})")
        umap_model = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine'
        )
        return umap_model.fit_transform(embeddings)
    else:
        logger.info(f"Applying CPU TSNE (n_components={n_components})")
        # Fallback to TSNE on CPU (UMAP requires umap-learn)
        from sklearn.manifold import TSNE
        tsne_model = TSNE(
            n_components=min(n_components, 50),  # TSNE limited to <=3 typically, use 50 for compatibility
            metric='cosine',
            n_jobs=-1
        )
        return tsne_model.fit_transform(embeddings)


# ============ Recursive HDBSCAN Clustering ============
def recursive_hdbscan_clustering(
    embeddings: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 7,
    cluster_size_threshold: int = 5000,
    use_gpu: bool = True
) -> List[List[int]]:
    """
    Recursively cluster embeddings using HDBSCAN.

    Large clusters exceeding the threshold are recursively split.

    Args:
        embeddings: Input embeddings (N, D) - NumPy or CuPy array
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for core points
        cluster_size_threshold: Split clusters larger than this
        use_gpu: Use GPU-accelerated cuML if available

    Returns:
        List of cluster index lists
    """
    logger.info(f"Starting recursive HDBSCAN clustering on {len(embeddings)} embeddings")

    if use_gpu and GPU_AVAILABLE:
        # Ensure CuPy array
        if not isinstance(embeddings, cp.ndarray):
            embeddings = cp.array(embeddings)
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    else:
        # CPU fallback: use DBSCAN instead of HDBSCAN
        logger.info("Using CPU DBSCAN (HDBSCAN requires cuML)")
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(eps=0.5, min_samples=min_samples, metric='cosine', n_jobs=-1)

    cluster_labels = clusterer.fit_predict(embeddings)

    # Convert to numpy if on GPU
    if use_gpu and GPU_AVAILABLE:
        cluster_labels = cp.asnumpy(cluster_labels)

    # Group indices by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        clusters[int(label)].append(idx)

    # Recursively split large clusters
    final_clusters = []
    for label, indices in clusters.items():
        if len(indices) > cluster_size_threshold:
            logger.info(f"Cluster {label} size {len(indices)} exceeds threshold {cluster_size_threshold}, recursively splitting")
            sub_embeddings = embeddings[indices]
            sub_clusters = recursive_hdbscan_clustering(
                sub_embeddings,
                min_cluster_size,
                min_samples,
                cluster_size_threshold,
                use_gpu
            )
            for sub_cluster in sub_clusters:
                final_clusters.append([indices[i] for i in sub_cluster])
        else:
            final_clusters.append(indices)

    logger.info(f"Created {len(final_clusters)} final clusters")
    return final_clusters


# ============ Hierarchical Clustering Utilities ============
def get_leaf_depths(linkage_matrix: np.ndarray) -> Tuple[Dict[int, int], int]:
    """
    Compute depth of each leaf node in hierarchical clustering tree.

    Args:
        linkage_matrix: Scipy linkage matrix

    Returns:
        (leaf_depths, max_depth) where leaf_depths maps leaf_id -> depth
    """
    tree, _ = to_tree(linkage_matrix, rd=True)
    leaf_depths = {}
    max_depth = [0]

    def assign_depths(node, current_depth):
        if node.is_leaf():
            leaf_depths[node.id] = current_depth
            if current_depth > max_depth[0]:
                max_depth[0] = current_depth
        else:
            assign_depths(node.left, current_depth + 1)
            assign_depths(node.right, current_depth + 1)

    assign_depths(tree, 0)
    return leaf_depths, max_depth[0]


def hierarchical_clustering_on_cluster(cluster_embeddings: np.ndarray) -> np.ndarray:
    """
    Perform Ward hierarchical clustering on a single cluster.

    Args:
        cluster_embeddings: Embeddings for cluster members (N, D)

    Returns:
        Scipy linkage matrix
    """
    # Convert CuPy to NumPy if needed
    if GPU_AVAILABLE and isinstance(cluster_embeddings, cp.ndarray):
        cluster_embeddings = cp.asnumpy(cluster_embeddings)

    linkage_matrix = linkage(cluster_embeddings, method='ward')
    return linkage_matrix


def assign_layers(
    depths: List[int],
    n_layers: int = 3,
    ratio: Tuple[int, ...] = (1, 2, 4)
) -> List[int]:
    """
    Assign layer numbers based on tree depth distribution.

    The ratio (1, 2, 4) means:
    - Top 1/7 of depths → Layer 4
    - Next 2/7 → Layer 5
    - Bottom 4/7 → Layer 6

    Args:
        depths: List of depth values for each entity
        n_layers: Number of layers (should match len(ratio))
        ratio: Distribution ratio for layer assignment

    Returns:
        List of layer assignments (4, 5, or 6)
    """
    sorted_depths = sorted(depths)
    total = len(sorted_depths)

    # Compute cumulative ratio thresholds
    cumulative_ratio = [sum(ratio[:i+1]) for i in range(len(ratio))]
    total_ratio = sum(ratio)

    # Find depth thresholds for each layer boundary
    thresholds = [
        sorted_depths[int(r * total / total_ratio) - 1]
        for r in cumulative_ratio[:-1]
    ]

    # Assign layers based on depth
    layers = []
    for depth in depths:
        layer = 4  # Default to layer 4 (shallowest)
        for i, threshold in enumerate(thresholds):
            if depth > threshold:
                layer = 5 + i  # 5, 6, ...
        layers.append(layer)

    return layers


# ============ Main Hierarchy Building Function ============
def build_hierarchy(
    entities: List[Dict[str, Any]],
    embeddings: np.ndarray,
    cluster_size_threshold: int = 10000,
    n_layers: int = 3,
    umap_config: Dict[str, Any] = None,
    hdbscan_config: Dict[str, Any] = None,
    use_gpu: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Build hierarchical structure for dynamic layer entities.

    This function:
    1. Applies UMAP dimensionality reduction
    2. Performs recursive HDBSCAN clustering
    3. Applies Ward hierarchical clustering within each cluster
    4. Assigns layers based on tree depth distribution

    Args:
        entities: List of entity dictionaries
        embeddings: Precomputed embeddings (N, D)
        cluster_size_threshold: Max cluster size before recursive split
        n_layers: Number of layers (3 for L4/L5/L6)
        umap_config: UMAP parameters
        hdbscan_config: HDBSCAN parameters
        use_gpu: Use GPU acceleration if available

    Returns:
        Dictionary mapping entity_id -> entity_dict (with 'layer' field added)
    """
    logger.info("Building hierarchy structure")

    # Default configurations
    if umap_config is None:
        umap_config = {'n_components': 100, 'n_neighbors': 10, 'min_dist': 0.1}
    if hdbscan_config is None:
        hdbscan_config = {'min_cluster_size': 10, 'min_samples': 7}

    # Step 1: UMAP dimensionality reduction
    umap_emb = apply_umap(embeddings, use_gpu=use_gpu, **umap_config)

    # Convert to GPU array if available
    if use_gpu and GPU_AVAILABLE:
        embeddings_gpu = cp.array(umap_emb)
    else:
        embeddings_gpu = umap_emb

    # Step 2: Recursive HDBSCAN clustering
    initial_clusters = recursive_hdbscan_clustering(
        embeddings_gpu,
        cluster_size_threshold=cluster_size_threshold,
        use_gpu=use_gpu,
        **hdbscan_config
    )

    # Step 3: Hierarchical clustering within each cluster
    hierarchy = {}

    for cluster_indices in tqdm(initial_clusters, desc="Processing clusters"):
        cluster_embeddings = embeddings_gpu[cluster_indices]
        cluster_entities = [entities[i] for i in cluster_indices]

        # Single entity clusters get assigned to top layer
        if len(cluster_entities) < 2:
            for entity in cluster_entities:
                entity['layer'] = 6  # Bottom layer for isolated entities
                hierarchy[entity['id']] = entity
            continue

        # Perform Ward hierarchical clustering
        linkage_matrix = hierarchical_clustering_on_cluster(cluster_embeddings)

        # Compute tree depths
        leaf_depths, max_depth = get_leaf_depths(linkage_matrix)
        depths = list(leaf_depths.values())

        # Assign layers based on depth distribution
        layers = assign_layers(depths, n_layers=n_layers, ratio=(1, 2, 4))

        # Update entity layer information
        for i, entity in enumerate(cluster_entities):
            entity['layer'] = layers[i]
            hierarchy[entity['id']] = entity

    # Log layer distribution
    layer_counts = Counter(entity['layer'] for entity in hierarchy.values())
    logger.info(f"Layer distribution: {dict(sorted(layer_counts.items()))}")

    return hierarchy


# ============ IPC Code Dictionary Construction ============
def build_ipc_code_dict(entities: List[Dict[str, Any]]) -> Dict[str, Dict[int, List[Dict]]]:
    """
    Build IPC code dictionary for parent node search.

    Structure: ipc_code_dict[main_class][layer] = [entities...]

    Args:
        entities: List of all entities (static + dynamic)

    Returns:
        Nested dictionary for IPC-based entity lookup
    """
    ipc_code_dict = defaultdict(lambda: defaultdict(list))

    for entity in entities:
        layer = entity.get('layer', 3)
        ipc_codes = entity.get('ipc_code', '').split(',')

        for code in ipc_codes:
            code = code.strip()
            if code:
                main_class = code[0]  # First character (A, B, C, ...)
                ipc_code_dict[main_class][layer].append(entity)

    return ipc_code_dict


def find_elbow_point(
    sim_scores: np.ndarray,
    min_parents: int = 1,
    max_parents: int = 5
) -> int:
    """
    Find elbow point in similarity score curve.

    Uses perpendicular distance from line connecting first and last points.

    Args:
        sim_scores: Sorted similarity scores (descending)
        min_parents: Minimum number of parents to select
        max_parents: Maximum number of parents to select

    Returns:
        Number of parents to select
    """
    if len(sim_scores) <= min_parents:
        return len(sim_scores)

    # Compute perpendicular distances from line
    points = np.array(list(enumerate(sim_scores)))
    line_vec = points[-1] - points[0]
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    vec_from_first = points - points[0]
    distances = np.abs(np.cross(line_vec_norm, vec_from_first))

    elbow_index = np.argmax(distances) + 1
    return max(min(elbow_index, max_parents), min_parents)


# ============ Parent Node Assignment ============
def assign_parent_nodes(
    hierarchy: Dict[int, Dict[str, Any]],
    ipc_code_dict: Dict[str, Dict[int, List[Dict]]],
    entity_to_embedding: Dict[int, np.ndarray],
    static_layer_entities: List[Dict[str, Any]],
    static_entity_to_embedding: Dict[int, np.ndarray],
    min_parents: int = 1,
    max_parents: int = 5,
    similarity_threshold: float = 0.86,
    use_faiss: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Assign parent nodes to dynamic layer entities using FAISS similarity search.

    This function uses batched FAISS queries and multi-threading for efficiency.

    Args:
        hierarchy: Dictionary of dynamic layer entities
        ipc_code_dict: IPC-based entity lookup dictionary
        entity_to_embedding: Embedding dictionary for all entities
        static_layer_entities: List of static layer entities
        static_entity_to_embedding: Embedding dictionary for static entities
        min_parents: Minimum number of parents per entity
        max_parents: Maximum number of parents per entity
        similarity_threshold: Minimum cosine similarity for valid parent
        use_faiss: Use FAISS for fast similarity search

    Returns:
        Updated hierarchy with 'parents' field added to each entity
    """
    logger.info("Assigning parent nodes with FAISS optimization")

    if not FAISS_AVAILABLE or not use_faiss:
        logger.warning("FAISS not available - using slow cosine similarity fallback")
        return _assign_parent_nodes_fallback(
            hierarchy, ipc_code_dict, entity_to_embedding,
            min_parents, max_parents, similarity_threshold
        )

    # Build FAISS index for each (IPC main class, layer) combination
    index_dict = {}
    id_map = {}

    for main_class in ipc_code_dict.keys():
        for layer in ipc_code_dict[main_class].keys():
            parent_entities = ipc_code_dict[main_class][layer]
            if not parent_entities:
                continue

            parent_embeddings = np.array([
                entity_to_embedding[e['id']] for e in parent_entities
            ]).astype('float32')
            parent_ids = np.array([e['id'] for e in parent_entities])

            # Build FAISS index with inner product (after L2 normalization = cosine similarity)
            index = faiss.IndexFlatIP(parent_embeddings.shape[1])
            faiss.normalize_L2(parent_embeddings)
            index.add(parent_embeddings)

            index_dict[(main_class, layer)] = index
            id_map[(main_class, layer)] = parent_ids

    # Prepare entity list and normalized embeddings
    entity_list = list(hierarchy.values())
    entity_embeddings = np.array([
        entity_to_embedding[e['id']] for e in entity_list
    ]).astype('float32')
    faiss.normalize_L2(entity_embeddings)

    # Build key mapping for each entity
    entity_to_keys = {}
    for entity in entity_list:
        current_layer = entity['layer']
        parent_layer = current_layer - 1
        ipc_codes = [
            code.strip()[0] for code in entity['ipc_code'].split(',') if code.strip()
        ]

        keys = []
        for main_class in ipc_codes:
            key = (main_class, parent_layer)
            if key in index_dict:
                keys.append(key)
            else:
                # Fallback to static layer 3
                static_key = (main_class, 3)
                if static_key in index_dict:
                    keys.append(static_key)

        entity_to_keys[entity['id']] = keys

    # Group entities by key for batched queries
    key_to_entity_indices = defaultdict(list)
    for i, entity in enumerate(entity_list):
        for key in entity_to_keys.get(entity['id'], []):
            key_to_entity_indices[key].append(i)

    # Process each key in parallel
    def process_key(key, indices):
        queries = entity_embeddings[indices]
        index = index_dict[key]
        K = max_parents
        D, I = index.search(queries, K)
        parent_ids = id_map[key]

        local_candidates = {}
        for idx, entity_idx in enumerate(indices):
            local_candidates[entity_idx] = []
            for sim, pid in zip(D[idx], parent_ids[I[idx]]):
                local_candidates[entity_idx].append((sim, pid))

        return local_candidates

    # Parallel processing
    candidate_results = {i: [] for i in range(len(entity_list))}
    max_workers = min(16, len(key_to_entity_indices))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_key, key, indices)
            for key, indices in key_to_entity_indices.items()
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="FAISS parent search"):
            local_candidates = future.result()
            for idx, cand_list in local_candidates.items():
                candidate_results[idx].extend(cand_list)

    # Merge candidates and select final parents
    for i, entity in enumerate(entity_list):
        candidates = candidate_results[i]

        if not candidates:
            entity['parents'] = []
            continue

        # Deduplicate and keep highest similarity
        parent_dict = {}
        for sim, pid in candidates:
            if pid not in parent_dict or sim > parent_dict[pid]:
                parent_dict[pid] = sim

        # Sort by similarity
        unique_parent_ids = np.array(list(parent_dict.keys()))
        unique_similarities = np.array(list(parent_dict.values()))
        sorted_indices = np.argsort(unique_similarities)[::-1]
        sorted_parent_ids = unique_parent_ids[sorted_indices]
        sorted_similarities = unique_similarities[sorted_indices]

        # Apply elbow point selection
        elbow_index = find_elbow_point(sorted_similarities, min_parents, max_parents)

        # Filter by similarity threshold
        valid_indices = np.where(sorted_similarities >= similarity_threshold)[0]
        final_indices = valid_indices[valid_indices < elbow_index]

        if len(final_indices) == 0:
            final_indices = [0]  # Keep at least the most similar parent

        entity['parents'] = sorted_parent_ids[final_indices].tolist()

    return hierarchy


def _assign_parent_nodes_fallback(
    hierarchy: Dict[int, Dict[str, Any]],
    ipc_code_dict: Dict[str, Dict[int, List[Dict]]],
    entity_to_embedding: Dict[int, np.ndarray],
    min_parents: int,
    max_parents: int,
    similarity_threshold: float
) -> Dict[int, Dict[str, Any]]:
    """Fallback implementation using cosine similarity (slower)."""
    logger.info("Using cosine similarity fallback for parent assignment")

    for entity in tqdm(hierarchy.values(), desc="Assigning parents"):
        current_layer = entity['layer']
        parent_layer = current_layer - 1
        entity_emb = entity_to_embedding[entity['id']]

        # Find candidate parents
        candidates = []
        ipc_codes = [code.strip()[0] for code in entity['ipc_code'].split(',') if code.strip()]

        for main_class in ipc_codes:
            parent_entities = ipc_code_dict.get(main_class, {}).get(parent_layer, [])
            if not parent_entities:
                parent_entities = ipc_code_dict.get(main_class, {}).get(3, [])

            for parent_entity in parent_entities:
                parent_emb = entity_to_embedding[parent_entity['id']]
                similarity = np.dot(entity_emb, parent_emb) / (
                    np.linalg.norm(entity_emb) * np.linalg.norm(parent_emb)
                )
                candidates.append((similarity, parent_entity['id']))

        # Sort and select parents
        if candidates:
            candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
            scores = np.array([c[0] for c in candidates])
            elbow = find_elbow_point(scores, min_parents, max_parents)
            valid = [(s, pid) for s, pid in candidates[:elbow] if s >= similarity_threshold]
            entity['parents'] = [pid for _, pid in (valid if valid else [candidates[0]])]
        else:
            entity['parents'] = []

    return hierarchy


# ============ Entity Merging and Saving ============
def merge_entities(entities1: List[Dict], entities2: List[Dict]) -> List[Dict]:
    """
    Merge two entity lists, removing duplicates by name.

    Args:
        entities1: First entity list
        entities2: Second entity list

    Returns:
        Merged entity list (deduplicated by name)
    """
    seen_names = set()
    merged_entities = []

    for entity in entities1 + entities2:
        if entity['name'] not in seen_names:
            seen_names.add(entity['name'])
            merged_entities.append(entity)

    return merged_entities


def save_entities(entities: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save entities to tab-separated file.

    Format: id\tname\tipc_code\tlayer\tparent

    Args:
        entities: List of entity dictionaries
        output_file: Path to output file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('id\tname\tipc_code\tlayer\tparent\n')

        for entity in entities:
            layer = entity.get('layer', '')
            parents = entity.get('parents', [])
            parent_str = ';'.join(map(str, parents)) if parents else ''
            f.write(f"{entity['id']}\t{entity['name']}\t{entity['ipc_code']}\t{layer}\t{parent_str}\n")

    logger.info(f"Saved {len(entities)} entities to {output_file}")


# ============ Main Pipeline Function ============
def run(
    cfg: Dict[str, Any],
    fot_entities_path: str,
    fot_layer_12_path: str,
    fot_layer_3_path: str,
    output_dynamic_path: str,
    output_full_path: str,
    model_path: str = "files/scibert_scivocab_uncased",
    embedding_cache: str = "artifacts/embeddings/hierarchy_embeddings.npz",
    id_offset: int = 11834,
    use_gpu: bool = False
) -> Dict[str, str]:
    """
    Complete hierarchy building pipeline.

    Args:
        cfg: Configuration dictionary
        fot_entities_path: Path to dynamic layer FOT entities (from chunk merge)
        fot_layer_12_path: Path to static layer L1/L2 entities
        fot_layer_3_path: Path to static layer L3 entities
        output_dynamic_path: Output path for dynamic layer with hierarchy
        output_full_path: Output path for complete FOT library
        model_path: Path to SciBERT model
        embedding_cache: Path to embedding cache file
        id_offset: ID offset for dynamic layer entities (default: 11834)
        use_gpu: Use GPU acceleration

    Returns:
        Dictionary with output file paths
    """
    logger.info("=" * 80)
    logger.info("Starting Dynamic FOT Hierarchy Construction")
    logger.info("=" * 80)

    # Load entities
    logger.info(f"Loading dynamic layer entities from {fot_entities_path}")
    fot_entities = load_entities(fot_entities_path)

    logger.info(f"Loading static layer L1/L2 from {fot_layer_12_path}")
    fot_layer_12 = load_entities(fot_layer_12_path)

    logger.info(f"Loading static layer L3 from {fot_layer_3_path}")
    fot_layer_3 = load_entities(fot_layer_3_path)

    static_layer_entities = merge_entities(fot_layer_12, fot_layer_3)
    logger.info(f"Total static layer entities: {len(static_layer_entities)}")

    # Apply ID offset to dynamic layer
    logger.info(f"Applying ID offset {id_offset} to dynamic layer entities")
    for entity in fot_entities:
        entity['id'] += id_offset

    # Merge for embedding computation
    all_entities = fot_entities + static_layer_entities
    logger.info(f"Total entities for embedding: {len(all_entities)}")

    # Compute or load embeddings
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Computing embeddings on device: {device}")

    entity_to_embedding = get_or_create_embeddings(
        all_entities,
        embedding_cache,
        model_path,
        batch_size=cfg.get('embeddings', {}).get('batch_size', 64),
        device=device
    )

    # Extract dynamic layer embeddings
    fot_embeddings = np.array([
        entity_to_embedding[entity['id']] for entity in fot_entities
    ])

    # Build hierarchy
    clustering_cfg = cfg.get('clustering', {})
    hierarchy = build_hierarchy(
        fot_entities,
        fot_embeddings,
        cluster_size_threshold=clustering_cfg.get('hdbscan', {}).get('cluster_size_threshold', 10000),
        n_layers=cfg.get('hierarchy', {}).get('n_layers', 3),
        umap_config=clustering_cfg.get('umap'),
        hdbscan_config=clustering_cfg.get('hdbscan'),
        use_gpu=use_gpu and GPU_AVAILABLE
    )

    # Update entity layer information
    for entity_id, entity_info in hierarchy.items():
        for fot_entity in fot_entities:
            if fot_entity['id'] == entity_id:
                fot_entity['layer'] = entity_info['layer']
                break

    # Build IPC code dictionary
    all_entities_with_layer = static_layer_entities + list(hierarchy.values())
    ipc_code_dict = build_ipc_code_dict(all_entities_with_layer)

    # Get static entity embeddings
    static_entity_to_embedding = {
        entity['id']: entity_to_embedding[entity['id']]
        for entity in static_layer_entities
    }

    # Assign parent nodes
    parent_cfg = cfg.get('parent_linking', {})
    hierarchy = assign_parent_nodes(
        hierarchy,
        ipc_code_dict,
        entity_to_embedding,
        static_layer_entities,
        static_entity_to_embedding,
        min_parents=parent_cfg.get('min_parents', 1),
        max_parents=parent_cfg.get('max_parents', 5),
        similarity_threshold=parent_cfg.get('similarity_threshold', 0.86),
        use_faiss=FAISS_AVAILABLE
    )

    # Log sample results
    entity_list = list(hierarchy.values())
    id_to_name = {entity['id']: entity['name'] for entity in all_entities}

    logger.info("First 10 dynamic entities:")
    for entity in entity_list[:10]:
        parent_names = [id_to_name.get(pid, str(pid)) for pid in entity.get('parents', [])]
        logger.info(f"  {entity['id']}\t{entity['name']}\tL{entity['layer']}\tParents: {parent_names}")

    # Save results
    logger.info(f"Saving dynamic layer hierarchy to {output_dynamic_path}")
    save_entities(entity_list, output_dynamic_path)

    logger.info(f"Saving full FOT library to {output_full_path}")
    full_entities = static_layer_entities + entity_list
    save_entities(full_entities, output_full_path)

    logger.info("=" * 80)
    logger.info("Hierarchy construction complete!")
    logger.info("=" * 80)

    return {
        'dynamic_hierarchy': output_dynamic_path,
        'full_library': output_full_path,
        'num_dynamic_entities': len(entity_list),
        'num_total_entities': len(full_entities)
    }
