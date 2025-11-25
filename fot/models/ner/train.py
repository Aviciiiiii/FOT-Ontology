"""
Complete NER training module with all original functionality restored.

This module implements the full NER training pipeline including:
- SciBERT/DistilBERT model support
- SpaCy POS tag integration
- Dynamic balanced sampling
- Multiple loss functions
- Training visualization
- Advanced evaluation metrics
- Complete checkpoint management
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from itertools import chain
import itertools

import numpy as np
from tqdm import tqdm

from ...utils.logging import get_logger
from ...utils.paths import load_yaml_once
from .entity_split import split_by_entity, verify_no_overlap


def _get_nested_config(cfg: Dict, *keys, default=None):
    """Helper function to get nested config values.

    Args:
        cfg: Configuration dictionary
        *keys: Sequence of keys to traverse (e.g., 'training', 'batch_size')
        default: Default value if key path doesn't exist

    Returns:
        Value at the key path or default

    Examples:
        _get_nested_config(cfg, 'training', 'batch_size', default=64)
        _get_nested_config(cfg, 'model', 'max_len', default=48)
    """
    current = cfg
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _load_sequences(p1: str, p2: str) -> List[Dict[str, Any]]:
    """Load sequences from two files.

    Supports both JSON array and JSONL formats:
    - JSON array: [{"tokens":[...],"tags":[...]}, ...]
    - JSONL: {"tokens":[...],"tags":[...]}\n{"tokens":[...],"tags":[...]}\n...
    """
    out: List[Dict[str, Any]] = []
    for p in (p1, p2):
        try:
            # Try JSON array format first
            data = json.loads(Path(p).read_text(encoding="utf-8"))
            if isinstance(data, list):
                out.extend(data)
        except json.JSONDecodeError:
            # Try JSONL format (one JSON object per line)
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                obj = json.loads(line)
                                out.append(obj)
                            except json.JSONDecodeError:
                                continue
            except Exception:
                pass
        except Exception:
            pass
    return out


def _maybe_pth(path: str) -> str:
    """Ensure path has .pth extension."""
    p = Path(path)
    return str(p if p.suffix == ".pth" else p.with_suffix(".pth"))


def get_model_path(cfg: Dict[str, Any]) -> str:
    """Get the appropriate model path based on configuration."""
    model_type = _get_nested_config(cfg, 'model', 'type', default='distilbert')

    if model_type == "scibert":
        # Check for SciBERT path in config or use default
        scibert_path = _get_nested_config(cfg, 'model', 'scibert_path', default='/root/autodl-tmp/try_GNN/Bert/scibert_scivocab_uncased/')
        if Path(scibert_path).exists():
            return scibert_path
        else:
            # Fallback to HuggingFace SciBERT
            return "allenai/scibert_scivocab_uncased"

    return _get_nested_config(cfg, 'model', 'distilbert_path', default='distilbert-base-uncased')


def setup_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Setup random seeds for reproducibility."""
    try:
        random.seed(seed)
        np.random.seed(seed)

        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def live_plot(data_dict: Dict[str, List[float]], figsize: Tuple[int, int] = (10, 6),
             title: str = 'Training Progress') -> None:
    """Real-time plotting of training curves."""
    try:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output

        clear_output(wait=True)
        plt.figure(figsize=figsize)

        for label, data in data_dict.items():
            if data:  # Only plot if we have data
                plt.plot(data, label=label, linewidth=2)

        plt.title(title, fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception:
        # If matplotlib/IPython not available, skip plotting
        pass


def evaluate_advanced(all_labels: List[List[str]], all_preds: List[List[str]],
                     logger, epoch: int = 0) -> Dict[str, float]:
    """Advanced evaluation with detailed metrics."""
    try:
        from sklearn.metrics import confusion_matrix, precision_score as sk_precision_score
        from sklearn.metrics import recall_score as sk_recall_score, f1_score as sk_f1_score
        from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
    except ImportError:
        logger.warning("sklearn or seqeval not available for advanced metrics")
        return {}

    # Overall sequence-level metrics
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    logger.info(f"Epoch {epoch} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Token-level FOT metrics
    fot_true = []
    fot_pred = []
    for true_seq, pred_seq in zip(all_labels, all_preds):
        fot_true.extend([1 if tag != 'O' else 0 for tag in true_seq])
        fot_pred.extend([1 if tag != 'O' else 0 for tag in pred_seq])

    if fot_true and fot_pred:
        fot_precision = sk_precision_score(fot_true, fot_pred, average='binary', zero_division=0)
        fot_recall = sk_recall_score(fot_true, fot_pred, average='binary', zero_division=0)
        fot_f1 = sk_f1_score(fot_true, fot_pred, average='binary', zero_division=0)

        logger.info(f"FOT Token-level - Precision: {fot_precision:.4f}, Recall: {fot_recall:.4f}, F1: {fot_f1:.4f}")

    # Confusion matrix
    try:
        all_true_tags = [tag for seq in all_labels for tag in seq]
        all_pred_tags = [tag for seq in all_preds for tag in seq]
        cm = confusion_matrix(all_true_tags, all_pred_tags, labels=['O', 'B-FOT', 'I-FOT'])
        logger.info("Confusion Matrix:")
        logger.info(f"{'':>8} {'O':>8} {'B-FOT':>8} {'I-FOT':>8}")
        labels = ['O', 'B-FOT', 'I-FOT']
        for i, label in enumerate(labels):
            row = f"{label:>8} " + " ".join(f"{cm[i][j]:>8}" for j in range(len(labels)))
            logger.info(row)
    except Exception as e:
        logger.warning(f"Could not compute confusion matrix: {e}")

    # FOT length distribution analysis
    try:
        true_fot_lengths = []
        pred_fot_lengths = []

        for seq in all_labels:
            current_fot_length = 0
            for tag in seq:
                if tag in ['B-FOT', 'I-FOT']:
                    current_fot_length += 1
                else:
                    if current_fot_length > 0:
                        true_fot_lengths.append(current_fot_length)
                        current_fot_length = 0
            if current_fot_length > 0:
                true_fot_lengths.append(current_fot_length)

        for seq in all_preds:
            current_fot_length = 0
            for tag in seq:
                if tag in ['B-FOT', 'I-FOT']:
                    current_fot_length += 1
                else:
                    if current_fot_length > 0:
                        pred_fot_lengths.append(current_fot_length)
                        current_fot_length = 0
            if current_fot_length > 0:
                pred_fot_lengths.append(current_fot_length)

        if true_fot_lengths:
            logger.info(f"True FOT average length: {np.mean(true_fot_lengths):.2f} (std: {np.std(true_fot_lengths):.2f})")
        if pred_fot_lengths:
            logger.info(f"Predicted FOT average length: {np.mean(pred_fot_lengths):.2f} (std: {np.std(pred_fot_lengths):.2f})")

    except Exception as e:
        logger.warning(f"Could not compute FOT length analysis: {e}")

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fot_precision": fot_precision if 'fot_precision' in locals() else 0.0,
        "fot_recall": fot_recall if 'fot_recall' in locals() else 0.0,
        "fot_f1": fot_f1 if 'fot_f1' in locals() else 0.0,
    }


def save_checkpoint(model, optimizer, scheduler, epoch: int, best_f1: float,
                   checkpoint_path: str, **kwargs) -> None:
    """Save complete training checkpoint."""
    try:
        import torch
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_f1': best_f1,
            **kwargs  # Additional data like pos2idx, idx2pos, etc.
        }
        torch.save(checkpoint_data, checkpoint_path)
    except Exception as e:
        logger = get_logger("train_checkpoint")
        logger.warning(f"Could not save checkpoint: {e}")


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load training checkpoint."""
    try:
        import torch
        # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return {}


def run_pretrain(
    mag_clean1: str,
    mag_clean2: str,
    mag_entities: str,
    out_model: str,
    *,
    dry_run: bool = True,
    fast: bool = True,
    run_id: str | None = None,
) -> str:
    logger = get_logger("ner_pretrain")

    # Load configuration
    cfg_dir_env = os.getenv("FOT_CONFIG_DIR", "configs")
    try:
        cfg = load_yaml_once(os.path.join(cfg_dir_env, "ner_train.yaml"))
    except Exception:
        cfg = {}

    # Setup seeds
    seed = int(_get_nested_config(cfg, 'advanced', 'seed', default=42))
    deterministic = bool(_get_nested_config(cfg, 'advanced', 'deterministic', default=True))
    setup_seed(seed, deterministic)

    # Load data
    seqs = _load_sequences(mag_clean1, mag_clean2)
    samples = len(seqs)

    metrics_path = Path("reports") / f"ner_pretrain_metrics_{run_id or 'na'}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        meta = {
            "stage": "pretrain",
            "samples": samples,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "params": {"dry_run": True, "fast": fast},
            "run_id": run_id,
        }
        p = Path(out_model)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps({"dry_run": True, "samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("NER pretrain stub written: %s (samples=%d)", p, samples)
        return str(p)

    # Real training mode
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        import os as _os
        _os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
        from transformers import AutoTokenizer
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        from seqeval.metrics import f1_score
        from .bert_crf import build_model, save_checkpoint as save_model_checkpoint, adjust_fot_weight
        from .datasets import NERDataset, DynamicBalancedBatchSampler, custom_collate_fn, preprocess_balanced_dataset
    except Exception as e:
        raise RuntimeError("PyTorch, transformers, torchcrf are required for NER pretrain.") from e

    # Model configuration
    pretrained_name = get_model_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    tag2idx = _get_nested_config(cfg, 'tags', 'tag2idx', default={"O": 0, "B-FOT": 1, "I-FOT": 2})

    # Load FOT entities for balanced sampling - CRITICAL FOR PERFORMANCE
    fot_entities = []
    logger.info(f"Loading FOT entities from: {mag_entities}")
    try:
        if Path(mag_entities).exists():
            with open(mag_entities, 'r', encoding='utf-8') as f:
                mag_data = json.load(f)

            # Handle different formats
            if isinstance(mag_data, list) and mag_data:
                if isinstance(mag_data[0], str):
                    fot_entities = mag_data
                    logger.info(f"Loaded {len(fot_entities)} FOT entities (string format)")
                elif isinstance(mag_data[0], dict):
                    # Try different key names
                    for key in ['name', 'entity', 'fot', 'term']:
                        if key in mag_data[0]:
                            fot_entities = [item[key] for item in mag_data if key in item]
                            logger.info(f"Loaded {len(fot_entities)} FOT entities from '{key}' field")
                            break

            if not fot_entities:
                logger.error("FOT entities loaded but empty or unrecognized format!")
                logger.error(f"Sample data: {mag_data[:2] if mag_data else 'empty'}")
        else:
            logger.error(f"FOT entities file not found: {mag_entities}")

    except Exception as e:
        logger.error(f"Failed to load FOT entities: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Enhanced POS weight dictionary from original script
    pos_weight_dict = {
        'NOUN': 1.3, 'PROPN': 1.3, 'ADJ': 1.1, 'VERB': 0.9, 'NUM': 0.8,
        'ADP': 0.6, 'DET': 0.5, 'CCONJ': 0.6, 'PART': 0.6, 'PRON': 0.5,
        'AUX': 0.5, 'ADV': 0.7, 'SCONJ': 0.6, 'INTJ': 0.4, 'SYM': 0.7,
        'X': 0.8, 'PAD': 1.0
    }
    idx2pos = {i: pos for i, pos in enumerate(['PAD'] + list(pos_weight_dict.keys()))}

    # Create dataset with POS processing
    max_len = int(_get_nested_config(cfg, 'model', 'max_len', default=48))

    # CRITICAL: Force balanced sampling if we have FOT entities
    if len(fot_entities) > 0 and not fast:
        logger.info(f"APPLYING BALANCED SAMPLING with {len(fot_entities)} FOT entities")
        balanced_seqs = preprocess_balanced_dataset(seqs, fot_entities, negative_ratio=2)
        logger.info(f"After balancing: {len(balanced_seqs)} samples (was {len(seqs)})")
        ds = NERDataset(balanced_seqs, tokenizer, max_len=max_len, tag2idx=tag2idx, add_pos_tags=True)
    else:
        if fast:
            logger.warning("Fast mode enabled - skipping balanced sampling (may have poor performance!)")
        else:
            logger.error("WARNING: No FOT entities - using UNBALANCED dataset (will have poor performance!)")
        ds = NERDataset(seqs, tokenizer, max_len=max_len, tag2idx=tag2idx, add_pos_tags=not fast)

    # Create train/validation/test split using ENTITY-LEVEL split to prevent data leakage
    # This ensures samples from the same entity are not split across train/val/test sets
    logger.info("Performing entity-level train/val/test split...")

    # Check if entity field exists in data
    has_entity_field = False
    if seqs and ('source_id' in seqs[0] or 'entity' in seqs[0]):
        has_entity_field = True
        entity_key = 'source_id' if 'source_id' in seqs[0] else 'entity'
        logger.info(f"Found entity field '{entity_key}' - using entity-level split")
    else:
        logger.warning("No entity field found - falling back to random sample-level split (may cause data leakage!)")

    if has_entity_field:
        # Entity-level split: 8:1:1 ratio (train:val:test)
        train_seqs, val_seqs, test_seqs = split_by_entity(
            seqs,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=42,
            entity_key=entity_key
        )

        # Verify no overlap
        verify_no_overlap(train_seqs, val_seqs, test_seqs, entity_key=entity_key)

        # Apply balanced sampling to training set only
        if len(fot_entities) > 0 and not fast:
            logger.info(f"APPLYING BALANCED SAMPLING to training set with {len(fot_entities)} FOT entities")
            train_seqs = preprocess_balanced_dataset(train_seqs, fot_entities, negative_ratio=2)
            logger.info(f"After balancing: {len(train_seqs)} train samples")

        # Create datasets
        train_ds = NERDataset(train_seqs, tokenizer, max_len=max_len, tag2idx=tag2idx, add_pos_tags=True)
        dev_ds = NERDataset(val_seqs, tokenizer, max_len=max_len, tag2idx=tag2idx, add_pos_tags=True)

        # Save test set for later evaluation
        test_file = Path(str(mag_clean1).replace('.json', '_test.json'))
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_seqs, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved test set to {test_file}")

    else:
        # Fallback: random sample-level split (not recommended)
        logger.warning("Using random sample-level split - this may cause data leakage!")
        split = int(len(ds) * 0.9)
        indices = list(range(len(ds)))
        train_ds, dev_ds = Subset(ds, indices[:split]), Subset(ds, indices[split:])

    # Training configuration
    batch_size = int(_get_nested_config(cfg, 'training', 'batch_size', default=64))
    epochs = int(_get_nested_config(cfg, 'training', 'epochs', default=10))
    lr = float(_get_nested_config(cfg, 'training', 'learning_rate', default=1e-5))

    # Dynamic balanced sampling
    use_dynamic = bool(_get_nested_config(cfg, 'dataset', 'balanced_sampling', 'enabled', default=False))

    if use_dynamic and hasattr(train_ds, '__len__'):
        train_sampler = DynamicBalancedBatchSampler(
            train_ds,
            batch_size,
            initial_positive_ratio=float(_get_nested_config(cfg, 'dataset', 'balanced_sampling', 'initial_positive_ratio', default=0.5)),
            final_positive_ratio=float(_get_nested_config(cfg, 'dataset', 'balanced_sampling', 'final_positive_ratio', default=0.3)),
            epochs=epochs,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=custom_collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    dev_loader = DataLoader(dev_ds, batch_size=batch_size, collate_fn=custom_collate_fn)

    # Build model with all original features
    model_kwargs = {
        "crf_weight": float(_get_nested_config(cfg, 'loss', 'weights', 'crf_loss', default=0.5)),
        "focal_weight": float(_get_nested_config(cfg, 'loss', 'weights', 'focal_loss', default=0.1)),
        "custom_weight": float(_get_nested_config(cfg, 'loss', 'weights', 'constraint_loss', default=0.5)),
        "use_focal": bool(_get_nested_config(cfg, 'loss', 'use_focal', default=True)),
        "tag2idx": tag2idx,
    }

    model = build_model(
        num_tags=len(tag2idx),
        pos_weight_dict=pos_weight_dict,
        idx2pos=ds.idx2pos if hasattr(ds, 'idx2pos') else idx2pos,
        lstm_hidden_dim=int(_get_nested_config(cfg, 'model', 'lstm_hidden_dim', default=256)),
        fot_weight=float(_get_nested_config(cfg, 'loss', 'fot_weight', default=1.8)),
        num_hidden_layers=int(_get_nested_config(cfg, 'model', 'num_hidden_layers', default=4)),
        pretrained_name_or_path=pretrained_name,
        last_k_layers=int(_get_nested_config(cfg, 'model', 'last_k_layers', default=4)),
        l2_lambda=float(_get_nested_config(cfg, 'loss', 'l2_lambda', default=1e-6)),
        **model_kwargs
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(_get_nested_config(cfg, 'training', 'scheduler', 'T_0', default=5)),
        T_mult=int(_get_nested_config(cfg, 'training', 'scheduler', 'T_mult', default=2)),
        eta_min=float(_get_nested_config(cfg, 'training', 'scheduler', 'eta_min', default=1e-6)),
    )

    # Training tracking
    train_losses: List[float] = []
    dev_losses: List[float] = []
    dev_f1s: List[float] = []
    fot_weights: List[float] = []

    best_f1 = -1.0
    best_dev_loss = float("inf")
    best_epoch = -1
    best_path = None

    patience = int(_get_nested_config(cfg, 'training', 'early_stopping', 'patience', default=3))
    use_amp = bool(_get_nested_config(cfg, 'training', 'use_amp', default=True))
    accumulate_steps = int(_get_nested_config(cfg, 'training', 'accumulate_steps', default=1))

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_crf_loss = 0.0
        total_focal_loss = 0.0
        total_custom_loss = 0.0
        total_l2_loss = 0.0

        optimizer.zero_grad()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                           desc=f"Epoch {epoch+1}/{epochs}")

        for step, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            tokens = batch.get("tokens", None)
            valid_mask = batch.get("valid_mask", None)
            if valid_mask is not None:
                valid_mask = valid_mask.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, emissions, crf_loss, focal_loss, custom_loss, l2_reg = model(
                    input_ids, attention_mask, labels=labels, pos_ids=pos_ids, tokens=tokens, valid_mask=valid_mask
                )
                loss = loss / accumulate_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += float(loss.item())
            total_crf_loss += float(crf_loss.item()) if hasattr(crf_loss, 'item') else float(crf_loss)
            total_focal_loss += float(focal_loss.item()) if hasattr(focal_loss, 'item') else float(focal_loss)
            total_custom_loss += float(custom_loss.item()) if hasattr(custom_loss, 'item') else float(custom_loss)
            total_l2_loss += float(l2_reg.item()) if hasattr(l2_reg, 'item') else float(l2_reg)

            # Update progress bar
            if (step + 1) % 50 == 0:
                avg_loss = epoch_loss / (step + 1)
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'CRF': f'{total_crf_loss/(step+1):.4f}',
                    'Focal': f'{total_focal_loss/(step+1):.4f}',
                    'Custom': f'{total_custom_loss/(step+1):.4f}'
                })

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_total_loss = 0.0
        all_preds: List[List[str]] = []
        all_labels: List[List[str]] = []
        id2tag = {v: k for k, v in tag2idx.items()}

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                pos_ids = batch["pos_ids"].to(device)
                tokens = batch.get("tokens", None)
                valid_mask = batch.get("valid_mask", None)
                if valid_mask is not None:
                    valid_mask = valid_mask.to(device)

                loss, emissions, *_ = model(
                    input_ids, attention_mask, labels=labels, pos_ids=pos_ids, tokens=tokens, valid_mask=valid_mask
                )
                val_total_loss += float(loss.item())

                # Decode predictions
                # CRITICAL FIX: Use valid_mask for decoding to exclude subword continuations
                decode_mask = (attention_mask.bool() & (labels != -100)) if labels is not None else attention_mask.bool()
                if valid_mask is not None:
                    decode_mask = (attention_mask.bool() & valid_mask)
                preds = model.crf.decode(emissions, mask=decode_mask, pos_ids=pos_ids)

                # Convert to tag strings
                for i in range(labels.size(0)):
                    gold = [id2tag[labels[i, j].item()] for j in range(labels.size(1)) if labels[i, j].item() != -100]
                    pred_tags = [id2tag.get(t, "O") for t in preds[i]]
                    m = min(len(gold), len(pred_tags))
                    all_labels.append(gold[:m])
                    all_preds.append(pred_tags[:m])

        avg_dev_loss = val_total_loss / len(dev_loader)
        dev_losses.append(avg_dev_loss)

        # Compute metrics
        dev_f1 = float(f1_score(all_labels, all_preds)) if all_labels else 0.0
        dev_f1s.append(dev_f1)

        # Advanced evaluation
        metrics = evaluate_advanced(all_labels, all_preds, logger, epoch + 1)

        # Dynamic FOT weight adjustment
        model.fot_weight = adjust_fot_weight(dev_f1, float(model.fot_weight))
        fot_weights.append(float(model.fot_weight))

        # Update scheduler
        scheduler.step()

        # Save best model
        if (dev_f1 > best_f1) or (dev_f1 == best_f1 and avg_dev_loss < best_dev_loss):
            best_f1 = dev_f1
            best_dev_loss = avg_dev_loss
            best_epoch = epoch

            out_p = Path(_maybe_pth(out_model))
            best_p = out_p.with_name((out_p.stem if out_p.suffix else out_p.name) + "_best.pth")

            checkpoint_data = {
                "state_dict": model.state_dict(),
                "tag2idx": tag2idx,
                "idx2tag": {v: k for k, v in tag2idx.items()},
                "pretrained_name": pretrained_name,
                "num_hidden_layers": int(_get_nested_config(cfg, 'model', 'num_hidden_layers', default=4)),
                "lstm_hidden_dim": int(_get_nested_config(cfg, 'model', 'lstm_hidden_dim', default=256)),
                "fot_weight": float(model.fot_weight),
                "last_k_layers": int(_get_nested_config(cfg, 'model', 'last_k_layers', default=4)),
                "l2_lambda": float(_get_nested_config(cfg, 'loss', 'l2_lambda', default=1e-6)),
                "use_focal": bool(_get_nested_config(cfg, 'loss', 'use_focal', default=True)),
                "pos_weight_dict": pos_weight_dict,
                "idx2pos": ds.idx2pos if hasattr(ds, 'idx2pos') else idx2pos,
                "pos2idx": ds.pos2idx if hasattr(ds, 'pos2idx') else {v: k for k, v in idx2pos.items()},
            }
            save_model_checkpoint(str(best_p), checkpoint_data)
            best_path = str(best_p)

        # Save checkpoint every epoch
        checkpoint_path = Path(out_model).parent / "checkpoint.pth"
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1, best_f1, str(checkpoint_path),
            pos2idx=ds.pos2idx if hasattr(ds, 'pos2idx') else {v: k for k, v in idx2pos.items()},
            idx2pos=ds.idx2pos if hasattr(ds, 'idx2pos') else idx2pos
        )

        # Training visualization
        if _get_nested_config(cfg, 'visualization', 'enabled', default=True):
            live_plot({
                'Train Loss': train_losses,
                'Dev Loss': dev_losses,
                'Dev F1': dev_f1s,
            }, title='NER Training Progress')

        logger.info(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
            f"Dev Loss: {avg_dev_loss:.4f}, Dev F1: {dev_f1:.4f}, "
            f"FOT Weight: {model.fot_weight:.3f}"
        )

        # Early stopping
        if patience > 0 and best_epoch >= 0 and (epoch - best_epoch) >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Save final model
    out_p = Path(_maybe_pth(out_model))
    out_p.parent.mkdir(parents=True, exist_ok=True)

    final_checkpoint_data = {
        "state_dict": model.state_dict(),
        "tag2idx": tag2idx,
        "idx2tag": {v: k for k, v in tag2idx.items()},
        "pretrained_name": pretrained_name,
        "num_hidden_layers": int(_get_nested_config(cfg, 'model', 'num_hidden_layers', default=4)),
        "lstm_hidden_dim": int(_get_nested_config(cfg, 'model', 'lstm_hidden_dim', default=256)),
        "fot_weight": float(model.fot_weight),
        "last_k_layers": int(_get_nested_config(cfg, 'model', 'last_k_layers', default=4)),
        "l2_lambda": float(_get_nested_config(cfg, 'loss', 'l2_lambda', default=1e-6)),
        "use_focal": bool(_get_nested_config(cfg, 'loss', 'use_focal', default=True)),
        "pos_weight_dict": pos_weight_dict,
        "idx2pos": ds.idx2pos if hasattr(ds, 'idx2pos') else idx2pos,
        "pos2idx": ds.pos2idx if hasattr(ds, 'pos2idx') else {v: k for k, v in idx2pos.items()},
    }
    save_model_checkpoint(str(out_p), final_checkpoint_data)

    # Save metrics
    metrics = {
        "stage": "pretrain",
        "samples": samples,
        "epochs": epochs,
        "train_loss": train_losses,
        "dev_loss": dev_losses,
        "dev_f1": dev_f1s,
        "best_f1": best_f1 if best_f1 >= 0 else None,
        "best_epoch": best_epoch if best_epoch >= 0 else None,
        "best_ckpt_path": best_path,
        "fot_weights": fot_weights,
        "final_metrics": evaluate_advanced(all_labels, all_preds, logger, epochs),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    final_path = best_path or str(out_p)
    logger.info("NER pretrain completed: final=%s best=%s (samples=%d)", out_p, best_path, samples)
    return final_path