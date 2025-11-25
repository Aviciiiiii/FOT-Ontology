"""
Complete NER fine-tuning module with all original functionality restored.

This module implements the full NER fine-tuning pipeline including:
- Loading and continuing from pre-trained models
- Advanced FOT entity-aware fine-tuning
- All features from train.py plus fine-tuning specific logic
- Enhanced evaluation and visualization
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from ...utils.logging import get_logger
from ...utils.paths import load_yaml_once

# Import all the training utilities
from .train import (
    _load_sequences, _maybe_pth, get_model_path, setup_seed, live_plot,
    evaluate_advanced, save_checkpoint, load_checkpoint
)


def run_finetune(
    fot_clean1: str,
    fot_clean2: str,
    onetwo_entities: str,
    third_entities: str,
    base_model: str,
    out_model: str,
    *,
    dry_run: bool = True,
    fast: bool = True,
    run_id: str | None = None,
) -> str:
    logger = get_logger("ner_finetune")

    # Load configuration
    cfg_dir = os.getenv("FOT_CONFIG_DIR", "configs")
    try:
        cfg = load_yaml_once(os.path.join(cfg_dir, "ner_finetune.yaml"))
    except Exception:
        cfg = {}

    # Setup seeds for reproducibility
    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    setup_seed(seed, deterministic)

    # Load data
    seqs = _load_sequences(fot_clean1, fot_clean2)
    samples = len(seqs)

    metrics_path = Path("reports") / f"ner_finetune_metrics_{run_id or 'na'}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        meta = {
            "stage": "finetune",
            "samples": samples,
            "base_model": base_model,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "params": {"dry_run": True, "fast": fast},
            "run_id": run_id,
        }
        p = Path(out_model)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps({"dry_run": True, "samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("NER finetune stub written: %s (samples=%d)", p, samples)
        return str(p)

    # Real fine-tuning mode
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        import os as _os
        _os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
        from transformers import AutoTokenizer
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        from seqeval.metrics import f1_score
        from .bert_crf import build_model, save_checkpoint as save_model_checkpoint, adjust_fot_weight, load_checkpoint as load_model_checkpoint
        from .datasets import NERDataset, DynamicBalancedBatchSampler, custom_collate_fn, preprocess_balanced_dataset
    except Exception as e:
        raise RuntimeError("PyTorch, transformers, torchcrf are required for NER finetune.") from e

    # Load base model metadata
    base_checkpoint = {}
    try:
        if Path(base_model).exists():
            base_checkpoint = load_model_checkpoint(base_model)
            logger.info(f"Loaded base model from {base_model}")
        else:
            logger.warning(f"Base model not found: {base_model}, starting from scratch")
    except Exception as e:
        logger.warning(f"Could not load base model: {e}")

    # Extract model configuration from base model or use defaults
    pretrained_name = base_checkpoint.get("pretrained_name", get_model_path(cfg))
    num_hidden_layers = int(base_checkpoint.get("num_hidden_layers", cfg.get("num_hidden_layers", 4)))
    lstm_hidden_dim = int(base_checkpoint.get("lstm_hidden_dim", cfg.get("lstm_hidden_dim", 256)))
    fot_weight = float(base_checkpoint.get("fot_weight", cfg.get("fot_weight", 1.8)))
    tag2idx = base_checkpoint.get("tag2idx", cfg.get("tag2idx", {"O": 0, "B-FOT": 1, "I-FOT": 2}))
    pos_weight_dict = base_checkpoint.get("pos_weight_dict", {
        'NOUN': 1.3, 'PROPN': 1.3, 'ADJ': 1.1, 'VERB': 0.9, 'NUM': 0.8,
        'ADP': 0.6, 'DET': 0.5, 'CCONJ': 0.6, 'PART': 0.6, 'PRON': 0.5,
        'AUX': 0.5, 'ADV': 0.7, 'SCONJ': 0.6, 'INTJ': 0.4, 'SYM': 0.7,
        'X': 0.8, 'PAD': 1.0
    })
    idx2pos = base_checkpoint.get("idx2pos", {i: pos for i, pos in enumerate(['PAD'] + list(pos_weight_dict.keys()))})

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    # Load FOT entities for fine-tuning
    fot_entities = []

    # Load Level 2 entities
    try:
        if Path(onetwo_entities).exists():
            onetwo_data = json.loads(Path(onetwo_entities).read_text(encoding="utf-8"))
            if isinstance(onetwo_data, list) and onetwo_data:
                if isinstance(onetwo_data[0], dict) and 'name' in onetwo_data[0]:
                    level2_entities = [item['name'] for item in onetwo_data]
                elif isinstance(onetwo_data[0], str):
                    level2_entities = onetwo_data
                else:
                    level2_entities = []
                fot_entities.extend(level2_entities)
                logger.info(f"Loaded {len(level2_entities)} Level 2 entities")
    except Exception as e:
        logger.warning(f"Could not load Level 2 entities: {e}")

    # Load third-party entities
    try:
        if Path(third_entities).exists():
            third_data = json.loads(Path(third_entities).read_text(encoding="utf-8"))
            if isinstance(third_data, list):
                fot_entities.extend(third_data)
                logger.info(f"Loaded {len(third_data)} third-party entities")
    except Exception as e:
        logger.warning(f"Could not load third-party entities: {e}")

    logger.info(f"Total FOT entities for fine-tuning: {len(fot_entities)}")

    # Create dataset with enhanced processing for fine-tuning
    max_len = int(cfg.get("max_len", 48 if fast else 128))

    # Apply balanced sampling specifically for fine-tuning
    if fot_entities and not fast:
        logger.info("Applying FOT entity-aware balanced dataset preprocessing for fine-tuning...")
        balanced_seqs = preprocess_balanced_dataset(seqs, fot_entities, negative_ratio=2)
        ds = NERDataset(balanced_seqs, tokenizer, max_len=max_len, tag2idx=tag2idx, add_pos_tags=True)
    else:
        ds = NERDataset(seqs, tokenizer, max_len=max_len, tag2idx=tag2idx, add_pos_tags=True)

    # Create train/validation split
    dev_files = [fot_clean1.replace('.json', '_dev.json'), fot_clean2.replace('.json', '_dev.json')]
    dev_seqs: List[Dict[str, Any]] = []
    has_dev = all(Path(p).exists() for p in dev_files)

    if has_dev:
        for p in dev_files:
            try:
                # Try JSON array format first
                data = json.loads(Path(p).read_text(encoding='utf-8'))
                if isinstance(data, list):
                    dev_seqs.extend(data)
            except json.JSONDecodeError:
                # Try JSONL format (one JSON object per line)
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    obj = json.loads(line)
                                    dev_seqs.append(obj)
                                except json.JSONDecodeError:
                                    continue
                except Exception:
                    pass
            except Exception:
                pass
        dev_ds = NERDataset(dev_seqs, tokenizer, max_len=max_len, tag2idx=tag2idx, add_pos_tags=True)
        train_ds = ds
    else:
        split = int(len(ds) * 0.9)
        indices = list(range(len(ds)))
        train_ds, dev_ds = Subset(ds, indices[:split]), Subset(ds, indices[split:])

    # Training configuration (typically more conservative for fine-tuning)
    batch_size = int(cfg.get("batch_size", 8 if fast else 32))  # Smaller batch size for fine-tuning
    epochs = int(cfg.get("epochs", 1 if fast else 5))  # Fewer epochs for fine-tuning
    lr = float(cfg.get("lr", 5e-6))  # Lower learning rate for fine-tuning

    # Dynamic balanced sampling with fine-tuning specific settings
    sampler_cfg = cfg.get("sampler", {})
    use_dynamic = bool(sampler_cfg.get("dynamic", True))

    if use_dynamic and hasattr(train_ds, '__len__'):
        train_sampler = DynamicBalancedBatchSampler(
            train_ds,
            batch_size,
            initial_positive_ratio=float(sampler_cfg.get("initial_positive_ratio", 0.6)),  # Higher for fine-tuning
            final_positive_ratio=float(sampler_cfg.get("final_positive_ratio", 0.4)),    # Higher for fine-tuning
            epochs=epochs,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=custom_collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    dev_loader = DataLoader(dev_ds, batch_size=batch_size, collate_fn=custom_collate_fn)

    # Build model with configuration from base model
    model_kwargs = {
        "crf_weight": float(cfg.get("crf_weight", 0.5)),
        "focal_weight": float(cfg.get("focal_weight", 0.1)),
        "custom_weight": float(cfg.get("custom_weight", 0.5)),
        "use_focal": bool(cfg.get("use_focal", base_checkpoint.get("use_focal", True))),
        "tag2idx": tag2idx,
    }

    model = build_model(
        num_tags=len(tag2idx),
        pos_weight_dict=pos_weight_dict,
        idx2pos=ds.idx2pos if hasattr(ds, 'idx2pos') else idx2pos,
        lstm_hidden_dim=lstm_hidden_dim,
        fot_weight=fot_weight,
        num_hidden_layers=num_hidden_layers,
        pretrained_name_or_path=pretrained_name,
        last_k_layers=int(base_checkpoint.get("last_k_layers", cfg.get("last_k_layers", 4))),
        l2_lambda=float(base_checkpoint.get("l2_lambda", cfg.get("l2_lambda", 1e-5))),
        **model_kwargs
    )

    # Load pre-trained weights
    if base_checkpoint and "state_dict" in base_checkpoint:
        try:
            # Load compatible weights (strict=False to handle architecture differences)
            model.load_state_dict(base_checkpoint["state_dict"], strict=False)
            logger.info("Successfully loaded pre-trained weights")
        except Exception as e:
            logger.warning(f"Could not load some pre-trained weights: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    # Optimizer and scheduler (more conservative for fine-tuning)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=float(cfg.get("weight_decay", 1e-4)))
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(cfg.get("T_0", 3 if not fast else 1)),  # Shorter cycles for fine-tuning
        T_mult=int(cfg.get("T_mult", 1)),               # No multiplication for fine-tuning
        eta_min=float(cfg.get("eta_min", 1e-7)),        # Lower minimum for fine-tuning
    )

    # Training tracking
    train_losses: List[float] = []
    dev_losses: List[float] = []
    dev_f1s: List[float] = []
    fot_weights: List[float] = []

    best_f1 = float(base_checkpoint.get("best_f1", -1.0))  # Start from base model's performance
    best_dev_loss = float("inf")
    best_epoch = -1
    best_path = None

    patience = int(cfg.get("early_stop_patience", 2))  # Lower patience for fine-tuning
    use_amp = bool(cfg.get("use_amp", True))
    accumulate_steps = int(cfg.get("accumulate_steps", 2))  # Higher accumulation for fine-tuning

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger.info(f"Starting fine-tuning from base F1: {best_f1:.4f}")

    # Fine-tuning loop
    from tqdm import tqdm

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_crf_loss = 0.0
        total_focal_loss = 0.0
        total_custom_loss = 0.0
        total_l2_loss = 0.0

        optimizer.zero_grad()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                           desc=f"Fine-tune Epoch {epoch+1}/{epochs}")

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Lower grad clip for fine-tuning
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += float(loss.item())
            total_crf_loss += float(crf_loss.item()) if hasattr(crf_loss, 'item') else float(crf_loss)
            total_focal_loss += float(focal_loss.item()) if hasattr(focal_loss, 'item') else float(focal_loss)
            total_custom_loss += float(custom_loss.item()) if hasattr(custom_loss, 'item') else float(custom_loss)
            total_l2_loss += float(l2_reg.item()) if hasattr(l2_reg, 'item') else float(l2_reg)

            # Update progress bar
            if (step + 1) % 20 == 0:  # More frequent updates for fine-tuning
                avg_loss = epoch_loss / (step + 1)
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'CRF': f'{total_crf_loss/(step+1):.4f}',
                    'FOT_W': f'{model.fot_weight:.3f}'
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
            for batch in tqdm(dev_loader, desc="Fine-tune Validation"):
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

        # Advanced evaluation with fine-tuning specific analysis
        metrics = evaluate_advanced(all_labels, all_preds, logger, epoch + 1)

        # FOT-specific analysis for fine-tuning
        fot_entity_hits = 0
        total_fot_predictions = 0
        for pred_seq, label_seq in zip(all_preds, all_labels):
            pred_text = " ".join([token for token, tag in zip(["dummy"] * len(pred_seq), pred_seq) if tag in ['B-FOT', 'I-FOT']])
            for fot_entity in fot_entities[:10]:  # Check top 10 entities
                if fot_entity.lower() in pred_text.lower():
                    fot_entity_hits += 1
            total_fot_predictions += sum(1 for tag in pred_seq if tag in ['B-FOT', 'I-FOT'])

        if total_fot_predictions > 0:
            logger.info(f"FOT entity recognition rate: {fot_entity_hits/min(len(fot_entities), 10):.3f}")

        # Dynamic FOT weight adjustment (more conservative for fine-tuning)
        old_weight = float(model.fot_weight)
        model.fot_weight = adjust_fot_weight(dev_f1, old_weight, min_weight=0.8, max_weight=1.5)  # Narrower range
        fot_weights.append(float(model.fot_weight))

        if abs(model.fot_weight - old_weight) > 0.05:
            logger.info(f"FOT weight adjusted: {old_weight:.3f} -> {model.fot_weight:.3f}")

        # Update scheduler
        scheduler.step()

        # Save best model (strict criteria for fine-tuning)
        improvement_threshold = 0.001  # Require meaningful improvement
        if (dev_f1 > best_f1 + improvement_threshold) or (abs(dev_f1 - best_f1) < improvement_threshold and avg_dev_loss < best_dev_loss):
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
                "num_hidden_layers": num_hidden_layers,
                "lstm_hidden_dim": lstm_hidden_dim,
                "fot_weight": float(model.fot_weight),
                "last_k_layers": int(cfg.get("last_k_layers", 4)),
                "l2_lambda": float(cfg.get("l2_lambda", 1e-5)),
                "use_focal": bool(cfg.get("use_focal", True)),
                "pos_weight_dict": pos_weight_dict,
                "idx2pos": ds.idx2pos if hasattr(ds, 'idx2pos') else idx2pos,
                "pos2idx": ds.pos2idx if hasattr(ds, 'pos2idx') else {v: k for k, v in idx2pos.items()},
                "base_model": base_model,
                "fot_entities": fot_entities[:100],  # Save sample of entities used
                "best_f1": best_f1,
            }
            save_model_checkpoint(str(best_p), checkpoint_data)
            best_path = str(best_p)
            logger.info(f"New best fine-tuned model saved: F1 {best_f1:.4f}")

        # Save checkpoint every epoch
        checkpoint_path = Path(out_model).parent / "finetune_checkpoint.pth"
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1, best_f1, str(checkpoint_path),
            pos2idx=ds.pos2idx if hasattr(ds, 'pos2idx') else {v: k for k, v in idx2pos.items()},
            idx2pos=ds.idx2pos if hasattr(ds, 'idx2pos') else idx2pos,
            base_model=base_model
        )

        # Training visualization
        if cfg.get("visualization", {}).get("enabled", True):
            live_plot({
                'Train Loss': train_losses,
                'Dev Loss': dev_losses,
                'Dev F1': dev_f1s,
            }, title='NER Fine-tuning Progress')

        logger.info(
            f"Fine-tune Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
            f"Dev Loss: {avg_dev_loss:.4f}, Dev F1: {dev_f1:.4f} (best: {best_f1:.4f}), "
            f"FOT Weight: {model.fot_weight:.3f}"
        )

        # Early stopping (strict for fine-tuning)
        if patience > 0 and best_epoch >= 0 and (epoch - best_epoch) >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs (no improvement since epoch {best_epoch+1})")
            break

    # Save final fine-tuned model
    out_p = Path(_maybe_pth(out_model))
    out_p.parent.mkdir(parents=True, exist_ok=True)

    final_checkpoint_data = {
        "state_dict": model.state_dict(),
        "tag2idx": tag2idx,
        "idx2tag": {v: k for k, v in tag2idx.items()},
        "pretrained_name": pretrained_name,
        "num_hidden_layers": num_hidden_layers,
        "lstm_hidden_dim": lstm_hidden_dim,
        "fot_weight": float(model.fot_weight),
        "last_k_layers": int(cfg.get("last_k_layers", 4)),
        "l2_lambda": float(cfg.get("l2_lambda", 1e-5)),
        "use_focal": bool(cfg.get("use_focal", True)),
        "pos_weight_dict": pos_weight_dict,
        "idx2pos": ds.idx2pos if hasattr(ds, 'idx2pos') else idx2pos,
        "pos2idx": ds.pos2idx if hasattr(ds, 'pos2idx') else {v: k for k, v in idx2pos.items()},
        "base_model": base_model,
        "fot_entities": fot_entities[:100],
        "final_f1": dev_f1s[-1] if dev_f1s else 0.0,
        "best_f1": best_f1,
    }
    save_model_checkpoint(str(out_p), final_checkpoint_data)

    # Save comprehensive fine-tuning metrics
    final_metrics = evaluate_advanced(all_labels, all_preds, logger, epochs) if all_labels else {}

    metrics = {
        "stage": "finetune",
        "samples": samples,
        "epochs": epochs,
        "train_loss": train_losses,
        "dev_loss": dev_losses,
        "dev_f1": dev_f1s,
        "best_f1": best_f1 if best_f1 >= 0 else None,
        "best_epoch": best_epoch if best_epoch >= 0 else None,
        "best_ckpt_path": best_path,
        "fot_weights": fot_weights,
        "base_model": base_model,
        "base_f1": float(base_checkpoint.get("best_f1", -1.0)),
        "improvement": best_f1 - float(base_checkpoint.get("best_f1", 0.0)),
        "fot_entities_count": len(fot_entities),
        "final_metrics": final_metrics,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    final_path = best_path or str(out_p)
    improvement = best_f1 - float(base_checkpoint.get("best_f1", 0.0))
    logger.info(
        f"NER fine-tuning completed: final=%s best=%s (samples=%d, improvement=+%.4f F1)",
        out_p, best_path, samples, improvement
    )

    return final_path