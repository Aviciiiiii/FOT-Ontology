from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from ...utils.logging import get_logger


def _encode_text_for_stub(text: str, dim: int = 128) -> List[float]:
    # Deterministic lightweight encoding without torch (for dry_run path safety)
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


def _load_entity_catalog(entity_catalog_path: str) -> Tuple[Dict[str, int], Dict[int, str], Dict[int, str], Dict[int, str]]:
    """Load entity catalog for Wikipedia linking (GAT mode)."""
    title2id = {}
    id2title = {}
    id2text = {}
    id2url = {}
    local_idx = 0

    with open(entity_catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            entity = json.loads(line)

            # Extract Wikipedia ID for URL construction
            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                    id2url[local_idx] = f"https://en.wikipedia.org/wiki?curid={wikipedia_id}"

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1

    return title2id, id2title, id2text, id2url


def _get_entity_features(text: str, tokenizer, bert_model, device):
    """Extract BERT features for entity text."""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = bert_model(**inputs)
    last_hidden_state = outputs[0]
    return last_hidden_state[:, 0, :].detach()  # Taking the embedding of [CLS] token


def _download_page_mock(url: str) -> str:
    """Mock Wikipedia page download for dry-run mode."""
    # Return mock HTML with some Wikipedia-style links
    mock_html = f'''
    <html><body>
    <a href="/wiki/Computer_science">Computer science</a>
    <a href="/wiki/Machine_learning">Machine learning</a>
    <a href="/wiki/Neural_network">Neural network</a>
    <a href="/wiki/Algorithm">Algorithm</a>
    <a href="/wiki/Data_structure">Data structure</a>
    </body></html>
    '''
    return mock_html


def _download_page_real(url: str, socks5_proxy: str = "127.0.0.1:7778") -> str:
    """Download Wikipedia page using curl with SOCKS5 proxy."""
    curl_command = f"curl --socks5-hostname {socks5_proxy} -L {url}"
    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
    return result.stdout


def _extract_wikipedia_links(html_content: str, entity_dict: Dict[str, int]) -> List[str]:
    """Extract Wikipedia entity links from HTML content."""
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        # Fallback to simple regex parsing
        import re
        links = re.findall(r'/wiki/([^"\\s]+)', html_content)
        entity_links = []
        for link in links:
            entity_name = link.replace('_', ' ')
            if entity_name in entity_dict:
                entity_links.append(entity_name)
        return entity_links

    soup = BeautifulSoup(html_content, 'html.parser')
    links = soup.find_all('a')
    entity_links = []

    for link in links:
        if 'href' in link.attrs and link['href'].startswith('/wiki/'):
            entity_name = link['href'][6:].replace('_', ' ')
            if entity_name in entity_dict:
                entity_links.append(entity_name)

    return entity_links


def _get_wikipedia_links(url: str, entity_dict: Dict[str, int], dry_run: bool = True, enable_fetch: bool = False, socks5_proxy: str = "127.0.0.1:7778") -> List[str]:
    """Get Wikipedia links from entity page."""
    if dry_run or not enable_fetch:
        html_content = _download_page_mock(url)
    else:
        html_content = _download_page_real(url, socks5_proxy)

    return _extract_wikipedia_links(html_content, entity_dict)


def _load_mag_entities(mag_file_path: str, title2id: Dict[str, int], id2text: Dict[int, str], id2url: Dict[int, str]) -> List[Dict[str, Any]]:
    """Load MAG entities from FieldsOfStudy.nt file."""
    mag_entities = []
    if not mag_file_path or not Path(mag_file_path).exists():
        return mag_entities

    levels = {}
    titles = {}

    try:
        with open(mag_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                components = line.strip().split(' ')
                if len(components) < 4:
                    continue

                subject = components[0]
                predicate = components[1]
                obj = ' '.join(components[2:])

                if '<https://makg.org/property/level>' in predicate:
                    level_value = obj.split('"')[1]
                    levels[subject] = level_value
                elif '<http://xmlns.com/foaf/0.1/name>' in predicate:
                    title_value = obj.split('"')[1]
                    titles[subject] = title_value

        # Filter for entities with level "0" and "1"
        level_0_titles = [title for entity, title in titles.items() if levels.get(entity) == '0']
        level_1_titles = [title for entity, title in titles.items() if levels.get(entity) == '1']

        for mag_title in level_0_titles + level_1_titles:
            if mag_title in title2id:
                mag_id = title2id[mag_title]
                mag_text = id2text.get(mag_id, "")
                mag_url = id2url.get(mag_id, "")
                mag_entities.append({
                    "id": mag_id,
                    "name": mag_title,
                    "text": mag_text,
                    "url": mag_url
                })
    except Exception as e:
        # Return empty list if MAG loading fails
        pass

    return mag_entities


def _load_level3_candidates(level3_path: str) -> List[Dict[str, Any]]:
    """Load Level 3 candidates for prediction (original script format)."""
    candidates = []
    if not level3_path or not Path(level3_path).exists():
        return candidates

    try:
        with open(level3_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for item in data:
            ipc_classification = item.get("IPC_Classification", "")
            level = item.get("level", "")
            text_entities = item.get("Texts_Entities", [])

            for text in text_entities:
                pending_entities = text.get("Crossencoder_Recommended_Entities", [])
                # Take top 21 entities as in original script
                selected_entities = pending_entities[:21]

                for entity in selected_entities:
                    candidates.append({
                        "id": entity.get("Entity_ID"),
                        "name": entity.get("Entity_Name"),
                        "text": entity.get("Entity_Text"),
                        "url": entity.get("Entity_URL"),
                        "ent_ipc": ipc_classification,
                        "level": level
                    })
    except Exception:
        pass

    return candidates


def _maybe_to_pth_path(path: str) -> str:
    p = Path(path)
    if p.suffix != ".pth":
        return str(p.with_suffix(".pth"))
    return str(p)


def run(
    level2_path: str,
    out_model_path: str,
    out_pred_path: str,
    *,
    mode: str = "mlp",
    entity_catalog_path: Optional[str] = None,
    mag_file_path: Optional[str] = None,
    level3_candidates_path: Optional[str] = None,
    encoder: str = "random",
    dry_run: bool = True,
    fast: bool = True,
    run_id: Optional[str] = None,
    gat_config: Optional[Dict[str, Any]] = None,
    network_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Train and predict using GAT or MLP classifier.

    Args:
        mode: "gat" for original GAT implementation, "mlp" for simplified MLP
        entity_catalog_path: Path to entity.jsonl for GAT mode
        dry_run: If True, use stub/mock implementations
        fast: If True, use reduced training epochs
        gat_config: Configuration for GAT mode (hidden_dim, epochs, etc.)
    """
    logger = get_logger("gat_classifier")
    with open(level2_path, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    # Paths
    metrics_path = Path("reports") / f"stage1_gat_metrics_{run_id or 'na'}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Default GAT config
    if gat_config is None:
        gat_config = {
            "hidden_dim": 128,
            "max_peripheral_nodes": 100,
            "epochs": 1 if fast else 100,
            "lr": 0.001,
            "batch_size": 32,
            "test_size": 0.20
        }

    if dry_run:
        # Stub: 写 JSON 元数据当作"模型"，并将所有样本作为预测正类
        model_type = f"{mode.upper()}-stub"
        meta = {"type": model_type, "mode": mode, "num_samples": len(seeds), "dry_run": True}
        Path(out_model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_model_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        preds_idx = list(range(len(seeds)))
        out_list = [seeds[i] for i in preds_idx]
        Path(out_pred_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_pred_path, "w", encoding="utf-8") as f:
            json.dump(out_list, f, ensure_ascii=False, indent=2)
        # Metrics stub
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"dry_run": True, "mode": mode, "train_samples": len(seeds), "metrics": {}}, f, ensure_ascii=False, indent=2)
        logger.info("[dry_run] %s stub model=%s preds=%s metrics=%s", model_type, out_model_path, out_pred_path, metrics_path)
        return out_model_path, out_pred_path

    # Real mode: GAT or MLP implementation
    # 延迟导入 torch，避免 dry-run 环境依赖问题
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.optim as optim  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for non-dry-run GAT classifier.") from e

    if mode == "gat":
        return _run_gat_mode(
            seeds, out_model_path, out_pred_path, metrics_path,
            entity_catalog_path, mag_file_path, level3_candidates_path,
            gat_config, network_config, logger
        )
    else:
        return _run_mlp_mode(
            seeds, out_model_path, out_pred_path, metrics_path,
            gat_config, logger
        )


def _run_mlp_mode(
    seeds: List[Dict[str, Any]],
    out_model_path: str,
    out_pred_path: str,
    metrics_path: Path,
    gat_config: Dict[str, Any],
    logger
) -> Tuple[str, str]:
    """Run simplified MLP mode (original behavior)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class TinyMLP(nn.Module):
        def __init__(self, in_dim: int = 128, hidden: int = 64, out_dim: int = 1):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # configs
    dim = 128
    hidden = gat_config.get("hidden_dim", 64)
    epochs = gat_config.get("epochs", 5)
    lr = gat_config.get("lr", 1e-3)

    # Build tensor inputs
    texts = [f"{e.get('name','')} {e.get('text','')}".strip() for e in seeds]
    feats = torch.tensor([_encode_text_for_stub(t, dim) for t in texts], dtype=torch.float32)

    model = TinyMLP(dim, hidden, 1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    target = torch.ones((feats.size(0), 1), dtype=torch.float32)

    model.train()
    losses: List[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(feats)
        preds = torch.sigmoid(logits)
        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(feats)).squeeze(1)
    selected_idx = (scores > 0.5).nonzero(as_tuple=True)[0].tolist()
    if not selected_idx:
        selected_idx = [int(torch.argmax(scores).item())]

    out_items = []
    for i in selected_idx:
        e = dict(seeds[i])
        try:
            sc = float(scores[i].item())
        except Exception:
            sc = 0.0
        e["score"] = sc
        out_items.append(e)
    Path(out_pred_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pred_path, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    real_model_path = _maybe_to_pth_path(out_model_path)
    Path(real_model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "dim": dim, "hidden": hidden, "mode": "mlp"}, real_model_path)

    metrics = {
        "mode": "mlp",
        "train_samples": int(feats.size(0)),
        "val_samples": 0,
        "loss": {"train_last": losses[-1] if losses else None, "train_curve": losses},
        "metrics": {"macro_f1": None, "accuracy": None, "topk": {"k1": None, "k3": None, "k5": None}},
        "selected_count": len(selected_idx),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("MLP trained epochs=%d -> model=%s preds=%s metrics=%s", epochs, real_model_path, out_pred_path, metrics_path)
    return real_model_path, out_pred_path


def _run_gat_mode(
    seeds: List[Dict[str, Any]],
    out_model_path: str,
    out_pred_path: str,
    metrics_path: Path,
    entity_catalog_path: Optional[str],
    mag_file_path: Optional[str],
    level3_candidates_path: Optional[str],
    gat_config: Dict[str, Any],
    network_config: Optional[Dict[str, Any]],
    logger
) -> Tuple[str, str]:
    """Run original GAT mode with graph neural networks."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    # Try to import PyTorch Geometric components
    try:
        from torch_geometric.nn import GATConv, TopKPooling, global_mean_pool  # type: ignore
        from torch_geometric.data import Data, DataLoader  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
    except ImportError as e:
        logger.warning("PyTorch Geometric not available, falling back to MLP mode: %s", e)
        return _run_mlp_mode(seeds, out_model_path, out_pred_path, metrics_path, gat_config, logger)

    # Try to import transformers for BERT
    try:
        from transformers import DistilBertTokenizer, DistilBertModel  # type: ignore
    except ImportError as e:
        logger.warning("Transformers not available, falling back to MLP mode: %s", e)
        return _run_mlp_mode(seeds, out_model_path, out_pred_path, metrics_path, gat_config, logger)

    if not entity_catalog_path:
        logger.warning("Entity catalog path required for GAT mode, falling back to MLP")
        return _run_mlp_mode(seeds, out_model_path, out_pred_path, metrics_path, gat_config, logger)

    # Network configuration
    if network_config is None:
        network_config = {"enable_fetch": False, "socks5": "127.0.0.1:7778"}

    enable_fetch = network_config.get("enable_fetch", False)
    socks5_proxy = network_config.get("socks5", "127.0.0.1:7778")

    # Load entity catalog
    try:
        title2id, id2title, id2text, id2url = _load_entity_catalog(entity_catalog_path)
        logger.info("Loaded entity catalog with %d entities", len(title2id))
    except Exception as e:
        logger.warning("Failed to load entity catalog: %s, falling back to MLP", e)
        return _run_mlp_mode(seeds, out_model_path, out_pred_path, metrics_path, gat_config, logger)

    # Load MAG entities
    mag_entities = _load_mag_entities(mag_file_path, title2id, id2text, id2url)
    logger.info("Loaded %d MAG entities", len(mag_entities))

    # Combine IPC seeds and MAG entities as positive samples
    fot_entities = seeds + mag_entities
    logger.info("Total positive entities: %d (IPC: %d, MAG: %d)", len(fot_entities), len(seeds), len(mag_entities))

    # Initialize BERT model (try to use lightweight version)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Try to load DistilBERT from files or download
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        bert_model.to(device)
        bert_model.eval()
        logger.info("Loaded DistilBERT model on device: %s", device)
    except Exception as e:
        logger.warning("Failed to load DistilBERT: %s, falling back to MLP", e)
        return _run_mlp_mode(seeds, out_model_path, out_pred_path, metrics_path, gat_config, logger)

    # GAT model definition
    class GATPoolClassifier(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super(GATPoolClassifier, self).__init__()
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.6)
            self.pool1 = TopKPooling(hidden_dim * 4, ratio=0.8)
            self.conv2 = GATConv(hidden_dim * 4, hidden_dim // 2, heads=2, concat=False, dropout=0.6)
            self.pool2 = TopKPooling(hidden_dim // 2, ratio=0.8)
            self.classifier = nn.Linear(hidden_dim // 2, output_dim)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.conv2(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            x = global_mean_pool(x, batch)
            x = self.classifier(x)
            return x

    # Graph dataset creation
    class CustomGraphDataset:
        def __init__(self, entities: List[Dict[str, Any]], positive_class: bool = True, negative_type: int = 0):
            """
            Args:
                entities: List of entities to process
                positive_class: True for positive samples, False for negative
                negative_type: 0 = positive samples, 1 = random center + original peripheral, 2 = fully random
            """
            self.data_list = []
            max_peripheral_nodes = gat_config.get("max_peripheral_nodes", 100)
            title_list = list(title2id.keys())

            for i, entity in enumerate(entities):
                if i % 50 == 0:
                    logger.info("Processing %s entity %d/%d",
                               "positive" if positive_class else f"negative-type{negative_type}", i, len(entities))

                try:
                    if positive_class or negative_type == 1:
                        # Positive samples or Type 1 negative: use entity's actual Wikipedia links
                        if positive_class:
                            center_text = f"{entity.get('name', '')} {entity.get('text', '')}".strip()
                        else:
                            # Type 1 negative: random center with original peripheral
                            random_entity = random.choice(title_list)
                            entity_id = title2id[random_entity]
                            center_text = f"{random_entity} {id2text.get(entity_id, '')}".strip()

                        center_features = _get_entity_features(center_text, tokenizer, bert_model, device)

                        # Get Wikipedia links
                        entity_url = entity.get("url", "")
                        if entity_url:
                            entity_links = _get_wikipedia_links(entity_url, title2id,
                                                               dry_run=(not enable_fetch),
                                                               enable_fetch=enable_fetch,
                                                               socks5_proxy=socks5_proxy)
                        else:
                            entity_links = []

                        # Limit peripheral nodes
                        if len(entity_links) > max_peripheral_nodes:
                            entity_links = entity_links[:max_peripheral_nodes]

                        # Handle empty link case
                        if not entity_links:
                            entity_links = ["Computer_science", "Algorithm"]  # fallback

                    else:
                        # Type 2 negative: completely random graph
                        random_center = random.choice(title_list)
                        center_id = title2id[random_center]
                        center_text = f"{random_center} {id2text.get(center_id, '')}".strip()
                        center_features = _get_entity_features(center_text, tokenizer, bert_model, device)

                        # Random number of peripheral nodes
                        num_peripheral = random.randint(10, max_peripheral_nodes)
                        entity_links = random.sample(title_list, min(num_peripheral, len(title_list)))

                    # Create feature matrix
                    num_peripheral = len(entity_links)
                    x = torch.zeros(num_peripheral + 1, 768)
                    x[0] = center_features.squeeze(0)

                    # Get peripheral entity features
                    for j, link_entity in enumerate(entity_links):
                        if link_entity in title2id:
                            entity_id = title2id[link_entity]
                            entity_text = f"{link_entity} {id2text.get(entity_id, '')}".strip()
                        else:
                            entity_text = link_entity

                        peripheral_features = _get_entity_features(entity_text, tokenizer, bert_model, device)
                        x[j + 1] = peripheral_features.squeeze(0)

                    # Create star graph (center connected to all peripheral)
                    edge_index = torch.tensor([[0] * num_peripheral, list(range(1, num_peripheral + 1))], dtype=torch.long)
                    y = torch.tensor([1 if positive_class else 0])

                    graph = Data(x=x, edge_index=edge_index, y=y)
                    self.data_list.append(graph)

                except Exception as e:
                    logger.warning("Failed to process entity %d: %s", i, e)
                    continue

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            return self.data_list[idx]

    # Create datasets
    logger.info("Creating positive dataset...")
    positive_dataset = CustomGraphDataset(fot_entities, positive_class=True, negative_type=0)

    # Create Type 1 negative samples: random center + original peripheral
    negative_seeds_type1 = fot_entities.copy()
    random.shuffle(negative_seeds_type1)
    logger.info("Creating Type 1 negative dataset (random center + original peripheral)...")
    negative_dataset_type1 = CustomGraphDataset(negative_seeds_type1[:len(fot_entities)], positive_class=False, negative_type=1)

    # Create Type 2 negative samples: completely random
    logger.info("Creating Type 2 negative dataset (fully random)...")
    negative_dataset_type2 = CustomGraphDataset(fot_entities, positive_class=False, negative_type=2)

    # Combine all datasets
    total_dataset = positive_dataset.data_list + negative_dataset_type1.data_list + negative_dataset_type2.data_list
    random.shuffle(total_dataset)

    logger.info("Dataset sizes - Positive: %d, Negative Type1: %d, Negative Type2: %d, Total: %d",
               len(positive_dataset.data_list), len(negative_dataset_type1.data_list),
               len(negative_dataset_type2.data_list), len(total_dataset))

    test_size = gat_config.get("test_size", 0.20)
    train_dataset, val_dataset = train_test_split(total_dataset, test_size=test_size, random_state=42)

    batch_size = gat_config.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    hidden_dim = gat_config.get("hidden_dim", 128)
    model = GATPoolClassifier(input_dim=768, hidden_dim=hidden_dim, output_dim=2)
    model.to(device)

    lr = gat_config.get("lr", 0.001)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Set class weights for imbalanced data
    class_weights = gat_config.get("class_weights", [1.0, 1.5])
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Training loop
    epochs = gat_config.get("epochs", 5)
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)
                total_val_loss += loss.item()

                pred = out.argmax(dim=1)
                correct += int((pred == batch.y).sum())
                total += batch.y.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        logger.info("Epoch %d/%d: Train Loss=%.4f, Val Loss=%.4f, Val Acc=%.4f",
                   epoch + 1, epochs, avg_train_loss, avg_val_loss, val_acc)

    # Prediction - choose prediction target
    prediction_entities = []
    if level3_candidates_path:
        # Predict on Level 3 candidates (original script behavior)
        prediction_entities = _load_level3_candidates(level3_candidates_path)
        logger.info("Loaded %d Level 3 candidates for prediction", len(prediction_entities))
    else:
        # Fallback to original seeds
        prediction_entities = seeds
        logger.info("Using %d seeds for prediction", len(prediction_entities))

    model.eval()
    predictions = []
    with torch.no_grad():
        for i, entity in enumerate(prediction_entities):
            if i % 100 == 0:
                logger.info("Predicting entity %d/%d", i, len(prediction_entities))

            try:
                # Create single entity graph for prediction
                center_text = f"{entity.get('name', '')} {entity.get('text', '')}".strip()
                center_features = _get_entity_features(center_text, tokenizer, bert_model, device)

                entity_url = entity.get("url", "")
                if entity_url:
                    entity_links = _get_wikipedia_links(entity_url, title2id,
                                                       dry_run=(not enable_fetch),
                                                       enable_fetch=enable_fetch,
                                                       socks5_proxy=socks5_proxy)
                else:
                    entity_links = ["Computer_science", "Algorithm"]

                max_peripheral_nodes = gat_config.get("max_peripheral_nodes", 100)
                if len(entity_links) > max_peripheral_nodes:
                    entity_links = entity_links[:max_peripheral_nodes]

                if not entity_links:
                    entity_links = ["Computer_science"]

                num_peripheral = len(entity_links)
                x = torch.zeros(num_peripheral + 1, 768, device=device)
                x[0] = center_features.squeeze(0)

                for j, link_entity in enumerate(entity_links):
                    if link_entity in title2id:
                        entity_id = title2id[link_entity]
                        entity_text = f"{link_entity} {id2text.get(entity_id, '')}".strip()
                    else:
                        entity_text = link_entity

                    peripheral_features = _get_entity_features(entity_text, tokenizer, bert_model, device)
                    x[j + 1] = peripheral_features.squeeze(0)

                edge_index = torch.tensor([[0] * num_peripheral, list(range(1, num_peripheral + 1))],
                                         dtype=torch.long, device=device)

                graph = Data(x=x, edge_index=edge_index, batch=torch.zeros(x.size(0), dtype=torch.long, device=device))
                output = model(graph)
                predicted_class = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, predicted_class].item()

                if predicted_class == 1:  # Positive prediction
                    entity_copy = dict(entity)
                    entity_copy["score"] = confidence
                    predictions.append(entity_copy)

            except Exception as e:
                logger.warning("Failed to predict entity %d: %s", i, e)
                continue

    # Save predictions
    Path(out_pred_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Save model
    real_model_path = _maybe_to_pth_path(out_model_path)
    Path(real_model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mode": "gat",
        "input_dim": 768,
        "hidden_dim": hidden_dim,
        "output_dim": 2,
        "config": gat_config
    }, real_model_path)

    # Save metrics
    metrics = {
        "mode": "gat",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "loss": {
            "train_last": train_losses[-1] if train_losses else None,
            "val_last": val_losses[-1] if val_losses else None,
            "train_curve": train_losses,
            "val_curve": val_losses
        },
        "metrics": {
            "accuracy": val_accuracies[-1] if val_accuracies else None,
            "val_acc_curve": val_accuracies,
            "best_val_loss": best_val_loss
        },
        "selected_count": len(predictions),
        "config": gat_config
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("GAT trained epochs=%d -> model=%s preds=%s metrics=%s",
               epochs, real_model_path, out_pred_path, metrics_path)
    logger.info("Selected %d entities out of %d prediction candidates", len(predictions), len(prediction_entities))
    if mag_entities:
        logger.info("Training included %d MAG entities + %d IPC seeds", len(mag_entities), len(seeds))

    return real_model_path, out_pred_path