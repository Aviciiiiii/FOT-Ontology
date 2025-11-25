from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from ...utils.logging import get_logger


def _ipc_prefix(ipc: str) -> str:
    return (ipc or "").strip()[:3]


def _encode_simple(text: str, dim: int = 128) -> List[float]:
    """Simple character frequency encoding (backward compatibility)."""
    vec = [0.0] * dim
    if not text:
        return vec
    total = 0.0
    for ch in text.lower():
        idx = (ord(ch) * 131) % dim
        vec[idx] += 1.0
        total += 1.0
    if total > 0:
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
    return vec


def _cosine_simple(a: List[float], b: List[float]) -> float:
    """Simple cosine similarity (backward compatibility)."""
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return float(num / (da * db))


class DistilBERTEncoder:
    """DistilBERT-based feature encoder for semantic similarity."""

    def __init__(self, model_name: str = "distilbert-base-uncased", device: Optional[str] = None):
        self.model_name = model_name
        self.device = None
        self.tokenizer = None
        self.model = None
        self.logger = get_logger("distilbert_encoder")

        # Initialize if dependencies available
        self._init_model(device)

    def _init_model(self, device: Optional[str] = None):
        """Initialize DistilBERT model and tokenizer."""
        try:
            import torch
            from transformers import DistilBertTokenizer, DistilBertModel

            # Set device
            if device:
                self.device = torch.device(device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load model and tokenizer
            self.logger.info("Loading DistilBERT model: %s", self.model_name)
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            self.model = DistilBertModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            self.logger.info("DistilBERT model loaded successfully on device: %s", self.device)

        except ImportError as e:
            self.logger.warning("DistilBERT dependencies not available: %s", e)
            self.logger.warning("Falling back to simple encoding")
        except Exception as e:
            self.logger.warning("Failed to load DistilBERT model: %s", e)
            self.logger.warning("Falling back to simple encoding")

    def is_available(self) -> bool:
        """Check if DistilBERT encoder is available."""
        return self.model is not None and self.tokenizer is not None

    def encode_text(self, text: str) -> List[float]:
        """Extract 768-dimensional features using DistilBERT."""
        if not self.is_available():
            # Fallback to simple encoding
            return _encode_simple(text, 768)

        try:
            import torch

            # Tokenize and encode
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                features = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                return features.squeeze().tolist()

        except Exception as e:
            self.logger.warning("Failed to encode text with DistilBERT: %s", e)
            # Fallback to simple encoding
            return _encode_simple(text, 768)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Encode multiple texts in batches for efficiency."""
        if not self.is_available():
            return [_encode_simple(text, 768) for text in texts]

        try:
            import torch

            all_features = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use [CLS] token embeddings
                    batch_features = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                    all_features.extend(batch_features.tolist())

            return all_features

        except Exception as e:
            self.logger.warning("Failed to encode batch with DistilBERT: %s", e)
            return [_encode_simple(text, 768) for text in texts]


class SKLearnKNN:
    """scikit-learn based KNN search for semantic similarity."""

    def __init__(self, n_neighbors: int = 1, metric: str = "cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.logger = get_logger("sklearn_knn")
        self._check_sklearn()

    def _check_sklearn(self):
        """Check if sklearn is available."""
        try:
            from sklearn.neighbors import NearestNeighbors
            import numpy as np
            self.available = True
        except ImportError:
            self.logger.warning("scikit-learn not available, falling back to simple cosine similarity")
            self.available = False

    def find_closest(self, query_features: List[float], candidate_features: List[List[float]]) -> Tuple[int, float]:
        """Find the closest candidate using KNN."""
        if not self.available or not candidate_features:
            # Fallback to simple cosine similarity
            return self._simple_closest(query_features, candidate_features)

        try:
            from sklearn.neighbors import NearestNeighbors
            import numpy as np

            # Convert to numpy arrays
            query_array = np.array([query_features])
            candidates_array = np.array(candidate_features)

            # Initialize KNN
            knn = NearestNeighbors(
                n_neighbors=min(self.n_neighbors, len(candidate_features)),
                algorithm='brute',
                metric=self.metric
            )
            knn.fit(candidates_array)

            # Find nearest neighbors
            distances, indices = knn.kneighbors(query_array)

            # Return best match (index and similarity)
            best_idx = indices[0][0]
            distance = distances[0][0]

            # Convert distance to similarity (cosine distance -> cosine similarity)
            if self.metric == "cosine":
                similarity = 1.0 - distance
            else:
                similarity = 1.0 / (1.0 + distance)  # Simple conversion for other metrics

            return best_idx, float(similarity)

        except Exception as e:
            self.logger.warning("Failed to use sklearn KNN: %s", e)
            return self._simple_closest(query_features, candidate_features)

    def _simple_closest(self, query_features: List[float], candidate_features: List[List[float]]) -> Tuple[int, float]:
        """Fallback: simple cosine similarity search."""
        if not candidate_features:
            return 0, 0.0

        best_idx = 0
        best_sim = -1.0

        for i, cand_features in enumerate(candidate_features):
            sim = _cosine_simple(query_features, cand_features)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return best_idx, best_sim


def prepare_level_dictionary(second_level_entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group Level 2 entities by IPC prefix (original script format)."""
    ipc_map = {}
    for entity in second_level_entities:
        key = _ipc_prefix(entity.get('ent_ipc', ''))
        if key not in ipc_map:
            ipc_map[key] = []
        ipc_map[key].append(entity)
    return ipc_map


def get_text(entity: Dict[str, Any]) -> str:
    """Extract text for encoding (original script format)."""
    name = entity.get('name', '')
    text = entity.get('text', '')
    return f"{name} {text}".strip()


def find_closest_parent(
    third_level_entity: Dict[str, Any],
    level_dict: Dict[str, List[Dict[str, Any]]],
    encoder: DistilBERTEncoder,
    knn_searcher: SKLearnKNN,
    k: int = 1
) -> Optional[Dict[str, Any]]:
    """Find closest parent using original script logic with enhanced features."""

    # Get IPC prefix for grouping
    ipc_prefix = _ipc_prefix(third_level_entity.get('ent_ipc', ''))
    candidate_entities = level_dict.get(ipc_prefix, [])

    if not candidate_entities:
        return None

    # Get features for the third level entity
    third_text = get_text(third_level_entity)
    third_features = encoder.encode_text(third_text)

    # Get features for all candidate entities
    candidate_texts = [get_text(ent) for ent in candidate_entities]
    candidate_features = encoder.encode_batch(candidate_texts)

    # Find closest parent using KNN
    best_idx, similarity = knn_searcher.find_closest(third_features, candidate_features)

    # Return the best match with similarity score
    best_match = candidate_entities[best_idx]
    best_match = dict(best_match)  # Copy to avoid modifying original
    best_match['similarity'] = similarity

    return best_match


def assign_parents(
    level2_path: str,
    l3_pred_path: str,
    out_path: str,
    *,
    encoder: str = "distilbert",
    knn_backend: str = "sklearn",
    bert_model: str = "distilbert-base-uncased",
    k_neighbors: int = 1,
    batch_size: int = 32,
    dry_run: bool = False,
    fast: bool = False,
) -> str:
    """Enhanced parent assignment with DistilBERT and sklearn KNN support."""
    logger = get_logger("parent_linker")

    # Load data
    with open(level2_path, "r", encoding="utf-8") as f:
        level2 = json.load(f)
    with open(l3_pred_path, "r", encoding="utf-8") as f:
        l3 = json.load(f)

    logger.info("Loaded %d Level 2 entities and %d Level 3 entities", len(level2), len(l3))

    # Choose encoding method
    if encoder == "distilbert":
        logger.info("Using DistilBERT encoder: %s", bert_model)
        bert_encoder = DistilBERTEncoder(bert_model)
        if not bert_encoder.is_available():
            logger.warning("DistilBERT not available, falling back to simple encoding")
            encoder = "simple"

    if encoder == "simple":
        logger.info("Using simple character frequency encoding")
        bert_encoder = None

    # Choose KNN backend
    if knn_backend == "sklearn":
        logger.info("Using sklearn KNN with %d neighbors", k_neighbors)
        knn_searcher = SKLearnKNN(n_neighbors=k_neighbors, metric="cosine")
    else:
        logger.info("Using simple cosine similarity")
        knn_searcher = SKLearnKNN(n_neighbors=1, metric="cosine")  # Will fallback to simple

    if encoder == "distilbert" and bert_encoder and bert_encoder.is_available():
        # Enhanced implementation with DistilBERT + KNN
        return _assign_parents_enhanced(
            level2, l3, out_path, bert_encoder, knn_searcher, logger, batch_size
        )
    else:
        # Fallback to simple implementation
        return _assign_parents_simple(level2, l3, out_path, logger)


def _assign_parents_enhanced(
    level2: List[Dict[str, Any]],
    l3: List[Dict[str, Any]],
    out_path: str,
    encoder: DistilBERTEncoder,
    knn_searcher: SKLearnKNN,
    logger,
    batch_size: int = 32
) -> str:
    """Enhanced parent assignment using DistilBERT + sklearn KNN."""

    # Prepare level dictionary (original script format)
    level_dict = prepare_level_dictionary(level2)

    # Log distribution
    dist = {k: len(v) for k, v in sorted(level_dict.items())}
    logger.info("L2 groups by IPC prefix: %s", dist)

    results = []
    linked_count = 0

    # Process each L3 entity
    for i, l3_entity in enumerate(l3):
        if i % 100 == 0 and i > 0:
            logger.info("Processed %d/%d entities", i, len(l3))

        # Find closest parent
        parent_entity = find_closest_parent(l3_entity, level_dict, encoder, knn_searcher)

        # Create output entity
        out_entity = dict(l3_entity)

        if parent_entity:
            # Original script format
            out_entity["parent_ent"] = parent_entity.get("name")

            # Enhanced format with additional info
            out_entity["parent"] = {
                "name": parent_entity.get("name"),
                "ent_ipc": parent_entity.get("ent_ipc", ""),
            }
            out_entity["sim"] = parent_entity.get("similarity", 0.0)
            linked_count += 1
        else:
            out_entity["parent_ent"] = None
            out_entity["parent"] = None
            out_entity["sim"] = 0.0

        results.append(out_entity)

    # Save results
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Enhanced parent assignment completed: %d/%d entities linked -> %s",
               linked_count, len(results), out_path)
    return out_path


def _assign_parents_simple(
    level2: List[Dict[str, Any]],
    l3: List[Dict[str, Any]],
    out_path: str,
    logger
) -> str:
    """Simple parent assignment (backward compatibility)."""

    # Group L2 by IPC prefix
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in level2:
        groups.setdefault(_ipc_prefix(e.get("ent_ipc", "")), []).append(e)

    # Log distribution summary
    dist = {k: len(v) for k, v in sorted(groups.items())}
    logger.info("L2 groups by IPC prefix: %s", dist)

    results: List[Dict[str, Any]] = []
    linked_count = 0

    for e in l3:
        pref = _ipc_prefix(e.get("ent_ipc", e.get("IPC_Classification", "")))
        candidates = groups.get(pref) or sum(groups.values(), [])  # flatten all L2 if group empty

        # Find best by cosine similarity
        q_text = f"{e.get('name','')} {e.get('text','')}".strip()
        q_vec = _encode_simple(q_text)
        best_sim = -1.0
        best_parent: Dict[str, Any] | None = None

        for cand in candidates:
            c_text = f"{cand.get('name','')} {cand.get('text','')}".strip()
            sim = _cosine_simple(q_vec, _encode_simple(c_text))
            if sim > best_sim:
                best_sim = sim
                best_parent = cand

        # Create output entity
        out_e = dict(e)
        if best_parent is not None:
            # Original script format
            out_e["parent_ent"] = best_parent.get("name")

            # Enhanced format
            out_e["parent"] = {
                "name": best_parent.get("name"),
                "ent_ipc": best_parent.get("ent_ipc", ""),
            }
            out_e["sim"] = float(best_sim if best_sim >= 0 else 0.0)
            linked_count += 1
        else:
            out_e["parent_ent"] = None
            out_e["parent"] = None
            out_e["sim"] = 0.0
        results.append(out_e)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Simple parent assignment completed: %d/%d entities linked -> %s",
               linked_count, len(results), out_path)
    return out_path


def run(pred_json: str, level2_json: str, out_json: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wrapper: assign parents and return stats used in Stage1 summary."""
    logger = get_logger("parent_linker")

    # Default configuration
    if config is None:
        config = {
            "encoder": "distilbert",
            "knn_backend": "sklearn",
            "bert_model": "distilbert-base-uncased",
            "k_neighbors": 1,
            "batch_size": 32
        }

    # Compute groups from L2 for stats
    try:
        with open(level2_json, "r", encoding="utf-8") as f:
            l2 = json.load(f)
    except Exception:
        l2 = []

    groups: Dict[str, int] = {}
    for e in l2:
        p = _ipc_prefix(e.get("ent_ipc", ""))
        groups[p] = groups.get(p, 0) + 1

    # Assign parents with configuration
    assign_parents(
        level2_json,
        pred_json,
        out_json,
        encoder=config.get("encoder", "distilbert"),
        knn_backend=config.get("knn_backend", "sklearn"),
        bert_model=config.get("bert_model", "distilbert-base-uncased"),
        k_neighbors=config.get("k_neighbors", 1),
        batch_size=config.get("batch_size", 32),
    )

    # Count linked entities
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            out_list = json.load(f)
    except Exception:
        out_list = []

    linked = sum(1 for e in out_list if e.get("parent"))
    linked_original = sum(1 for e in out_list if e.get("parent_ent"))

    logger.info("Parent linking summary: %d total, %d linked (enhanced), %d linked (original format)",
               len(out_list), linked, linked_original)

    return {
        "assigned": len(out_list),
        "linked_count": linked,
        "linked_count_original": linked_original,
        "groups": groups,
        "out_path": out_json,
        "config": config
    }