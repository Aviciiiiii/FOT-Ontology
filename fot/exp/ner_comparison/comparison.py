"""
NER Model Comparison Experiment Runner.

Comprehensive comparison of multiple NER models:
- PatentNER (full model with all features)
- PatentNER ablations (no POS, no customCRF, no focal loss, no L2)
- Baseline models (BiLSTM-CRF, BERT-CRF, SciBERT-CRF)
- Commercial NER (SpaCy, Stanford NER - optional)

Adapted from original /src/exp_scripts.py with modern architecture.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from itertools import chain, groupby
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, precision_score as sk_precision_score
from sklearn.metrics import recall_score as sk_recall_score, f1_score as sk_f1_score
from seqeval.metrics import f1_score, precision_score, recall_score

from ...utils.logging import get_logger
from ...utils.paths import load_yaml_once
from ...models.ner.bert_crf import BertBiLSTMCRF
from ...models.ner.datasets import NERDataset, custom_collate_fn
from ...models.ner.train import setup_seed, _get_nested_config, _load_sequences
from .models import BiLSTM_CRF, SciBERT_CRF, SpacyNERWrapper, StanfordNERWrapper
from .metrics import calculate_additional_metrics

logger = get_logger("ner_comparison")


def load_pretrained_model(checkpoint_path: str, num_tags: int, pos_weight_dict: Dict,
                          idx2pos: Dict, device: torch.device, pretrained_model: str = 'files/scibert_scivocab_uncased') -> BertBiLSTMCRF:
    """Load pretrained PatentNER model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load backbone
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(pretrained_model)

    # Build model with checkpoint parameters
    model = BertBiLSTMCRF(
        backbone=backbone,
        num_tags=num_tags,
        pos_weight_dict=pos_weight_dict,
        idx2pos=idx2pos,
        lstm_hidden_dim=checkpoint.get('lstm_hidden_dim', 256),
        fot_weight=checkpoint.get('fot_weight', 1.8),
        num_hidden_layers=checkpoint.get('num_hidden_layers', 4),
    )

    # Load state dict - handle both 'state_dict' and 'model_state_dict' keys
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)

    logger.info(f"Loaded pretrained model from {checkpoint_path}")
    return model


def train_model_single_epoch(model, train_loader, optimizer, scheduler, device, epoch, num_epochs):
    """Train model for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        pos_ids = batch.get('pos_ids', torch.zeros_like(input_ids)).to(device)
        tokens = batch.get('tokens', None)
        valid_mask = batch.get('valid_mask', None)  # NEW: get valid_mask
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)

        # Replace pos_ids with uniform NOUN if POS is disabled (for patentNERnopos ablation)
        # CRITICAL: Use NOUN instead of PAD to avoid triggering "non-NOUN/PROPN" penalties
        if getattr(model, 'disable_pos', False):
            neutral_idx = getattr(model, 'neutral_pos_idx', 0)
            pos_ids = torch.full_like(pos_ids, neutral_idx)

        optimizer.zero_grad()
        loss, *_ = model(input_ids, attention_mask, labels, pos_ids, tokens, valid_mask)

        if hasattr(loss, 'mean'):
            loss = loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, idx2tag, device, model_name: str) -> Dict[str, Any]:
    """Evaluate model and return comprehensive metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0

    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            pos_ids = batch.get('pos_ids', torch.zeros_like(input_ids)).to(device)
            tokens = batch.get('tokens', None)
            valid_mask = batch.get('valid_mask', None)  # NEW: get valid_mask
            if valid_mask is not None:
                valid_mask = valid_mask.to(device)

            # Replace pos_ids with uniform NOUN if POS is disabled (for patentNERnopos ablation)
            # CRITICAL: Use NOUN instead of PAD to avoid triggering "non-NOUN/PROPN" penalties
            if getattr(model, 'disable_pos', False):
                neutral_idx = getattr(model, 'neutral_pos_idx', 0)
                pos_ids = torch.full_like(pos_ids, neutral_idx)

            # Forward pass
            outputs = model(input_ids, attention_mask, labels, pos_ids, tokens, valid_mask)

            # Handle different model output formats
            if isinstance(outputs, tuple):
                loss = outputs[0]
                emissions = outputs[1] if len(outputs) > 1 else None
                total_loss += loss.item() if hasattr(loss, 'item') else (loss if isinstance(loss, (int, float)) else 0)
            else:
                # SpaCy/Stanford return predictions directly
                emissions = outputs
                total_loss += 0

            # Decode predictions
            if hasattr(model, 'crf') and emissions is not None:
                # CRF-based models - use valid_mask for decoding (only first subwords)
                decode_mask = (attention_mask.bool() & (labels != -100))
                if valid_mask is not None:
                    decode_mask = (attention_mask.bool() & valid_mask)
                preds_raw = model.crf.decode(emissions, mask=decode_mask, pos_ids=pos_ids)

                # CRITICAL FIX: Use decode_mask to extract only valid positions
                # This ensures predictions and labels are perfectly aligned
                for i in range(len(preds_raw)):
                    # preds_raw[i] already contains only valid positions (from mask-based decode)
                    # Convert predictions to tag strings (all positions in preds_raw are valid)
                    pred_tags = [idx2tag.get(p.item() if hasattr(p, 'item') else p, "O")
                                for p in preds_raw[i]]

                    # Extract true labels - only positions where labels != -100
                    true_tags = []
                    for j in range(labels.size(1)):
                        if labels[i][j].item() != -100:
                            true_tags.append(idx2tag[labels[i][j].item()])

                    # Both pred_tags and true_tags should have same length (num_valid positions)
                    # If they don't match, something is wrong with masking
                    if len(pred_tags) == len(true_tags) and len(pred_tags) > 0:
                        all_preds.append(pred_tags)
                        all_labels.append(true_tags)
                    elif len(pred_tags) > 0 and len(true_tags) > 0:
                        # Fallback: take minimum length to prevent crash
                        min_len = min(len(pred_tags), len(true_tags))
                        all_preds.append(pred_tags[:min_len])
                        all_labels.append(true_tags[:min_len])

            elif hasattr(model, '__class__') and model.__class__.__name__ in ['SpacyNERWrapper', 'StanfordNERWrapper']:
                # SpaCy/Stanford wrappers - outputs are already predictions
                if isinstance(outputs, tuple):
                    preds_tensor = outputs[0]
                else:
                    preds_tensor = outputs

                # Process each sequence
                for i in range(len(preds_tensor)):
                    seq_len = attention_mask[i].sum().item()

                    pred_tags = [idx2tag.get(preds_tensor[i][j].item() if hasattr(preds_tensor[i][j], 'item') else preds_tensor[i][j], "O") for j in range(min(len(preds_tensor[i]), seq_len))]

                    true_tags = []
                    for j in range(min(labels.size(1), seq_len)):
                        if j < labels.size(1) and labels[i][j].item() != -100:
                            true_tags.append(idx2tag[labels[i][j].item()])

                    min_len = min(len(pred_tags), len(true_tags))
                    if min_len > 0:
                        all_preds.append(pred_tags[:min_len])
                        all_labels.append(true_tags[:min_len])
            else:
                # Fallback for other model types
                logger.warning(f"Unknown model type, attempting to decode predictions...")
                if emissions is not None and hasattr(model, 'crf'):
                    preds_raw = model.crf.decode(emissions, mask=attention_mask.bool(), pos_ids=pos_ids)

                    for i in range(len(preds_raw)):
                        seq_len = attention_mask[i].sum().item()
                        pred_tags = [idx2tag.get(preds_raw[i][j].item() if hasattr(preds_raw[i][j], 'item') else preds_raw[i][j], "O") for j in range(min(len(preds_raw[i]), seq_len))]

                        true_tags = []
                        for j in range(min(labels.size(1), seq_len)):
                            if j < labels.size(1) and labels[i][j].item() != -100:
                                true_tags.append(idx2tag[labels[i][j].item()])

                        min_len = min(len(pred_tags), len(true_tags))
                        if min_len > 0:
                            all_preds.append(pred_tags[:min_len])
                            all_labels.append(true_tags[:min_len])
                else:
                    raise ValueError(f"Cannot decode predictions for model {model_name}")

    end_time = time.time()
    speed = len(test_loader.dataset) / (end_time - start_time)

    # Calculate sequence-level metrics
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    # FOT-specific token-level metrics
    fot_true = [1 if tag != 'O' else 0 for seq in all_labels for tag in seq]
    fot_pred = [1 if tag != 'O' else 0 for seq in all_preds for tag in seq]

    fot_precision = sk_precision_score(fot_true, fot_pred, average='binary', zero_division=0)
    fot_recall = sk_recall_score(fot_true, fot_pred, average='binary', zero_division=0)
    fot_f1 = sk_f1_score(fot_true, fot_pred, average='binary', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(
        [tag for seq in all_labels for tag in seq],
        [tag for seq in all_preds for tag in seq],
        labels=['O', 'B-FOT', 'I-FOT']
    )

    # Additional metrics
    b_prec, p_match, type_cons = calculate_additional_metrics(all_labels, all_preds)

    # FOT length distribution
    true_fot_lengths = [len(list(g)) for k, g in groupby(chain(*all_labels)) if k != 'O']
    pred_fot_lengths = [len(list(g)) for k, g in groupby(chain(*all_preds)) if k != 'O']

    logger.info(f"{model_name} Results:")
    logger.info(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    logger.info(f"  FOT F1: {fot_f1:.4f}, FOT Precision: {fot_precision:.4f}, FOT Recall: {fot_recall:.4f}")
    logger.info(f"  Speed: {speed:.2f} samples/sec")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': total_loss / len(test_loader),
        'confusion_matrix': cm.tolist(),
        'fot_precision': fot_precision,
        'fot_recall': fot_recall,
        'fot_f1': fot_f1,
        'b_prec': b_prec,
        'p_match': p_match,
        'type_cons': type_cons,
        'speed': speed,
        'true_fot_avg_length': float(np.mean(true_fot_lengths)) if true_fot_lengths else 0.0,
        'pred_fot_avg_length': float(np.mean(pred_fot_lengths)) if pred_fot_lengths else 0.0,
    }


def load_datasets_from_json(
    mag1_path: str,
    mag2_path: str,
    mag_entities_path: str,
    max_len: int,
    batch_size: int,
    sample_size: Optional[int] = None
):
    """
    Load and prepare datasets from JSON files (following fot/models/ner/train.py logic).

    Args:
        mag1_path: Path to MAG1 cleaned JSON file
        mag2_path: Path to MAG2 cleaned JSON file
        mag_entities_path: Path to MAG entities JSON file
        max_len: Maximum sequence length
        batch_size: Batch size for data loaders
        sample_size: Optional limit on dataset size

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, idx2pos)
    """
    from transformers import AutoTokenizer
    from ...models.ner.entity_split import split_by_entity

    logger.info("Loading data from JSON files...")

    # Load sequences from MAG files
    seqs = _load_sequences(mag1_path, mag2_path)
    logger.info(f"Loaded {len(seqs)} sequences from MAG data")

    # Optional sampling for testing
    if sample_size and sample_size < len(seqs):
        import random
        random.seed(42)
        seqs = random.sample(seqs, sample_size)
        logger.info(f"Sampled {len(seqs)} sequences for comparison")

    # Load FOT entities for balanced sampling
    fot_entities = []
    if Path(mag_entities_path).exists():
        try:
            with open(mag_entities_path, 'r', encoding='utf-8') as f:
                mag_data = json.load(f)

            if isinstance(mag_data, list) and mag_data:
                if isinstance(mag_data[0], str):
                    fot_entities = mag_data
                elif isinstance(mag_data[0], dict):
                    for key in ['name', 'entity', 'fot', 'term']:
                        if key in mag_data[0]:
                            fot_entities = [item[key] for item in mag_data if key in item]
                            break

            logger.info(f"Loaded {len(fot_entities)} FOT entities for balanced sampling")
        except Exception as e:
            logger.warning(f"Failed to load FOT entities: {e}")

    # CRITICAL: Use entity-level split to prevent data leakage
    # This ensures samples from the same entity are not split across train/val/test sets
    logger.info("Performing entity-level train/val/test split to prevent data leakage...")

    # Check if entity field exists in data
    has_entity_field = False
    if seqs and ('source_id' in seqs[0] or 'entity' in seqs[0]):
        has_entity_field = True
        entity_key = 'source_id' if 'source_id' in seqs[0] else 'entity'
        logger.info(f"✓ Found entity field '{entity_key}' - using entity-level split")
    else:
        logger.warning("⚠️  No entity field found - falling back to random split (may cause data leakage!)")

    if has_entity_field:
        # Entity-level split: 8:1:1 ratio (train:val:test)
        train_data, val_data, test_data = split_by_entity(
            seqs,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=42,
            entity_key=entity_key
        )
    else:
        # Fallback to random split (not recommended)
        from sklearn.model_selection import train_test_split
        logger.warning("⚠️  Using random sample-level split (may cause data leakage!)")
        train_data, test_data = train_test_split(seqs, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
        logger.info(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # CRITICAL FIX: Apply balanced sampling to ALL splits (train/val/test)
    # This ensures consistent positive/negative ratios across all datasets
    if len(fot_entities) > 0:
        from ...models.ner.datasets import preprocess_balanced_dataset
        logger.info("Applying balanced sampling to train/val/test splits...")
        train_data = preprocess_balanced_dataset(train_data, fot_entities, negative_ratio=2)
        val_data = preprocess_balanced_dataset(val_data, fot_entities, negative_ratio=2)
        test_data = preprocess_balanced_dataset(test_data, fot_entities, negative_ratio=2)
        logger.info(f"After balancing: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)} samples")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("files/scibert_scivocab_uncased")
    tag2idx = {"O": 0, "B-FOT": 1, "I-FOT": 2}

    # Create datasets
    logger.info("Creating NER datasets...")
    train_dataset = NERDataset(train_data, tokenizer, max_len=max_len, tag2idx=tag2idx, add_pos_tags=True)
    val_dataset = NERDataset(val_data, tokenizer, max_len=max_len, tag2idx=tag2idx, add_pos_tags=True)
    test_dataset = NERDataset(test_data, tokenizer, max_len=max_len, tag2idx=tag2idx, add_pos_tags=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    idx2pos = train_dataset.idx2pos

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, idx2pos


def run_ner_comparison_experiments(
    config_dir: str,
    data_dir: str,
    output_path: str,
    pretrain_model_path: Optional[str] = None,
    num_epochs: int = 20,
    batch_size: int = 64,
    sample_size: Optional[int] = None,
    skip_commercial: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run comprehensive NER model comparison experiments.

    Args:
        config_dir: Path to configuration directory
        data_dir: Path to training data directory (containing JSON files)
        output_path: Path to save results JSON
        pretrain_model_path: Path to pretrained PatentNER checkpoint
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        sample_size: Limit dataset size (for testing)
        skip_commercial: Skip SpaCy/Stanford NER (requires extra setup)
        dry_run: Run with minimal data for testing

    Returns:
        Dictionary with all experimental results
    """
    logger.info("="*80)
    logger.info("NER Model Comparison Experiments")
    logger.info("="*80)

    # Setup
    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load config
    cfg = load_yaml_once(os.path.join(config_dir, "ner_comparison.yaml"))

    # Tag mappings
    tag2idx = {"O": 0, "B-FOT": 1, "I-FOT": 2}
    idx2tag = {v: k for k, v in tag2idx.items()}
    num_tags = len(tag2idx)

    max_len = cfg.get("data", {}).get("max_len", 48)

    # Training configuration
    train_cfg = cfg.get("training", {})
    learning_rate = float(train_cfg.get("learning_rate", 5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    # POS weights (from config)
    pos_weights_cfg = cfg.get("pos_weights", {})
    pos_weight_dict = {
        'NOUN': pos_weights_cfg.get('NOUN', 1.3),
        'PROPN': pos_weights_cfg.get('PROPN', 1.3),
        'ADJ': pos_weights_cfg.get('ADJ', 1.1),
        'VERB': pos_weights_cfg.get('VERB', 0.9),
        'NUM': pos_weights_cfg.get('NUM', 0.8),
        'ADP': pos_weights_cfg.get('ADP', 0.6),
        'DET': pos_weights_cfg.get('DET', 0.5),
        'CCONJ': pos_weights_cfg.get('CCONJ', 0.6),
        'PART': pos_weights_cfg.get('PART', 0.6),
        'PRON': pos_weights_cfg.get('PRON', 0.5),
        'AUX': pos_weights_cfg.get('AUX', 0.5),
        'ADV': pos_weights_cfg.get('ADV', 0.7),
        'SCONJ': pos_weights_cfg.get('SCONJ', 0.6),
        'INTJ': pos_weights_cfg.get('INTJ', 0.4),
        'SYM': pos_weights_cfg.get('SYM', 0.7),
        'X': pos_weights_cfg.get('X', 0.8),
        'PAD': pos_weights_cfg.get('PAD', 1.0)
    }
    idx2pos = {i: pos for i, pos in enumerate(['PAD'] + list(pos_weight_dict.keys()))}

    # Load datasets from JSON files (like original script)
    logger.info(f"Loading data from {data_dir}")

    data_cfg = cfg.get("data", {})
    mag1_file = data_cfg.get("mag1_clean", "data/processed/cleaned_mag1_tagged_searched_sentences_with_entity.json")
    mag2_file = data_cfg.get("mag2_clean", "data/processed/cleaned_mag2_tagged_searched_sentences_with_entity.json")
    mag_entities_file = data_cfg.get("mag_entities", "data/interim/mag_entities.json")

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, idx2pos = load_datasets_from_json(
        mag1_path=mag1_file,
        mag2_path=mag2_file,
        mag_entities_path=mag_entities_file,
        max_len=max_len,
        batch_size=batch_size,
        sample_size=sample_size if not dry_run else 100
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Initialize tokenizer for vocab size
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("files/scibert_scivocab_uncased")
    vocab_size = len(tokenizer)

    # Model save directory
    model_save_dir = Path(output_path).parent / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Model configurations (from original exp_scripts.py)
    models_to_compare = {}

    # Get model configs from YAML
    models_cfg = cfg.get("models", {})

    # PatentNER variants
    if models_cfg.get("patentner_full", {}).get("enabled", True):
        models_to_compare['patentNER'] = {
            'type': 'patentner',
            'description': 'Full PatentNER with all features',
        }

    if models_cfg.get("patentner_no_pos", {}).get("enabled", True):
        models_to_compare['patentNERnopos'] = {
            'type': 'patentner',
            'description': 'Without POS-aware loss',
            'disable_pos': True,
        }

    if models_cfg.get("patentner_no_custom_crf", {}).get("enabled", True):
        models_to_compare['patentNER no customCRF'] = {
            'type': 'patentner',
            'description': 'Without custom CRF constraints',
            'disable_custom_crf': True,
        }

    if models_cfg.get("patentner_no_focal", {}).get("enabled", True):
        models_to_compare['patentNERnofocalloss'] = {
            'type': 'patentner',
            'description': 'Without focal loss',
            'disable_focal': True,
        }

    if models_cfg.get("patentner_no_l2", {}).get("enabled", True):
        models_to_compare['patentNERnol2'] = {
            'type': 'patentner',
            'description': 'Without L2 regularization',
            'disable_l2': True,
        }

    if pretrain_model_path and models_cfg.get("patentner_pretrained", {}).get("enabled", False):
        models_to_compare['patentNERpretrained'] = {
            'type': 'patentner',
            'description': 'With MAG pretraining',
            'pretrain_path': pretrain_model_path,
        }

    # Baseline models
    if models_cfg.get("bilstm_crf", {}).get("enabled", True):
        models_to_compare['BiLSTM-crf'] = {
            'type': 'bilstm',
            'description': 'Baseline BiLSTM-CRF',
        }

    if models_cfg.get("bert_crf", {}).get("enabled", True):
        models_to_compare['bert-crf'] = {
            'type': 'bert',
            'description': 'BERT-base + CRF',
            'pretrained_model': models_cfg.get("bert_crf", {}).get("pretrained_model", "bert-base-uncased"),
        }

    if models_cfg.get("scibert_crf", {}).get("enabled", True):
        models_to_compare['scibert-crf'] = {
            'type': 'scibert',
            'description': 'SciBERT + CRF',
            'pretrained_model': models_cfg.get("scibert_crf", {}).get("pretrained_model", "files/scibert_scivocab_uncased"),
        }

    # Commercial NER (optional)
    if not skip_commercial:
        if models_cfg.get("spacy_ner", {}).get("enabled", False):
            models_to_compare['spacyNER'] = {
                'type': 'spacy',
                'description': 'SpaCy pretrained NER',
            }

        if models_cfg.get("stanford_ner", {}).get("enabled", False):
            models_to_compare['stanfordNER'] = {
                'type': 'stanford',
                'description': 'Stanford CoreNLP NER',
            }

    all_results = {}

    # Run experiments for each model
    for model_name, model_config in models_to_compare.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Description: {model_config['description']}")
        logger.info(f"{'='*80}")

        try:
            model_type = model_config['type']

            # Check if this model needs FOT data (pretrained models should use FOT for fine-tuning)
            use_fot_data = 'pretrain_path' in model_config

            if use_fot_data:
                logger.info("Loading FOT data for fine-tuning pretrained model...")
                fot1_file = data_cfg.get("fot1_clean", "data/processed/cleaned_FOT1_tagged_searched_sentences_with_entity.json")
                fot2_file = data_cfg.get("fot2_clean", "data/processed/cleaned_FOT2_tagged_searched_sentences_with_entity.json")
                third_entities_file = data_cfg.get("third_entities", "data/interim/third_entities.json")

                # Load FOT datasets
                fot_train_dataset, fot_val_dataset, fot_test_dataset, fot_train_loader, fot_val_loader, fot_test_loader, fot_idx2pos = load_datasets_from_json(
                    mag1_path=fot1_file,
                    mag2_path=fot2_file,
                    mag_entities_path=third_entities_file,
                    max_len=max_len,
                    batch_size=batch_size,
                    sample_size=sample_size if not dry_run else 100
                )
                logger.info(f"FOT Data - Train: {len(fot_train_dataset)}, Val: {len(fot_val_dataset)}, Test: {len(fot_test_dataset)}")

                # Use FOT data for this model
                model_train_loader = fot_train_loader
                model_val_loader = fot_val_loader
                model_test_loader = fot_test_loader
            else:
                # Use default MAG data
                model_train_loader = train_loader
                model_val_loader = val_loader
                model_test_loader = test_loader

            # Check if model already trained
            best_model_path = model_save_dir / f"{model_name}_best_model.pth"
            checkpoint_path = model_save_dir / f"{model_name}_checkpoint.pth"

            # Instantiate model based on type
            if model_type == 'patentner':
                logger.info("Creating PatentNER model...")
                # Get pretrained model path from config
                pretrained_model = model_config.get('pretrained_model', models_cfg.get('patentner_full', {}).get('pretrained_model', 'files/scibert_scivocab_uncased'))

                # Load backbone
                from transformers import AutoModel
                backbone = AutoModel.from_pretrained(pretrained_model)

                model = BertBiLSTMCRF(
                    backbone=backbone,
                    num_tags=num_tags,
                    pos_weight_dict=pos_weight_dict,
                    idx2pos=idx2pos,
                    lstm_hidden_dim=256,
                    fot_weight=1.8,
                    num_hidden_layers=4
                )

                # Apply ablations
                if model_config.get('disable_pos'):
                    logger.info("Disabling POS-aware weights (keep FOT boost)")
                    # CRITICAL FIX: Complete POS removal while keeping FOT boost
                    # ① Set all POS weights to 1.0 (neutral)
                    model.crf.pos_weight_dict = {k: 1.0 for k in model.crf.pos_weight_dict}

                    # ② Mark model as POS-disabled (for training/eval loops)
                    model.disable_pos = True

                    # ③ Remove POS penalty in custom loss
                    model.rule_w_bad_pos = 0.0

                    # ④ Wrap _apply_constraints to pass uniform NOUN pos_ids
                    # CRITICAL: Use NOUN instead of PAD to avoid triggering "non-NOUN/PROPN" penalties
                    # This makes all POS-dependent rules see "no difference" rather than "always bad POS"
                    orig_apply_constraints = model.crf._apply_constraints
                    noun_idx = next(i for i, p in model.crf.idx2pos.items() if p == 'NOUN')
                    def _apply_constraints_no_pos(tags, pos_ids, tokens, mask):
                        if pos_ids is not None:
                            pos_ids = torch.full_like(pos_ids, noun_idx)
                        return orig_apply_constraints(tags, pos_ids, tokens, mask)
                    model.crf._apply_constraints = _apply_constraints_no_pos

                    # Store NOUN index for training/eval loops
                    model.neutral_pos_idx = noun_idx

                if model_config.get('disable_custom_crf'):
                    logger.info("Disabling CustomCRF entirely - replacing with StandardCRF")
                    # CRITICAL FIX: Replace CustomCRF with StandardCRF to remove ALL enhancements:
                    # - NO POS weights
                    # - NO constraint rules
                    # - NO FOT weight boosting
                    # This is the cleanest way to show CustomCRF's total contribution
                    from .models import StandardCRF
                    model.crf = StandardCRF(num_tags, batch_first=True).to(device)
                    # Also disable external custom loss
                    model._compute_custom_loss = lambda emissions, labels, pos_ids, mask, tokens: torch.tensor(0.0, device=emissions.device)

                if model_config.get('disable_focal'):
                    logger.info("Disabling focal loss")
                    model.focal_loss = None

                if model_config.get('disable_l2'):
                    logger.info("Disabling L2 regularization")
                    # CRITICAL FIX: Disable BOTH model L2 AND optimizer weight_decay
                    # to ensure only one factor is changed in ablation study
                    model.l2_lambda = 0
                    model.disable_weight_decay = True  # Flag for optimizer creation

                # Load pretrained if specified
                if 'pretrain_path' in model_config:
                    logger.info(f"Loading pretrained weights from {model_config['pretrain_path']}")
                    model = load_pretrained_model(model_config['pretrain_path'], num_tags, pos_weight_dict, idx2pos, device, pretrained_model)

                # Create optimizer with appropriate weight_decay
                model_weight_decay = 0.0 if getattr(model, 'disable_weight_decay', False) else weight_decay
                if model_weight_decay == 0.0:
                    logger.info(f"Creating optimizer with weight_decay=0.0 (L2 disabled)")
                optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=model_weight_decay)

            elif model_type == 'bilstm':
                logger.info("Creating BiLSTM-CRF model...")
                model = BiLSTM_CRF(
                    vocab_size=vocab_size,
                    embedding_dim=100,
                    hidden_dim=256,
                    num_labels=num_tags,
                    pos_weight_dict=pos_weight_dict,
                    idx2pos=idx2pos,
                    fot_weight=1.8
                )
                optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            elif model_type in ['bert', 'scibert']:
                logger.info(f"Creating {model_type.upper()}-CRF model...")
                pretrained_name = model_config.get('pretrained_model', 'files/scibert_scivocab_uncased')
                model = SciBERT_CRF(
                    num_labels=num_tags,
                    pos_weight_dict=pos_weight_dict,
                    idx2pos=idx2pos,
                    pretrained_model_name=pretrained_name,
                    fot_weight=1.8
                )
                optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            elif model_type == 'spacy':
                logger.info("Creating SpaCy NER wrapper...")
                model = SpacyNERWrapper()
                optimizer = None  # No training for SpaCy

            elif model_type == 'stanford':
                logger.info("Creating Stanford NER wrapper...")
                model = StanfordNERWrapper()
                optimizer = None  # No training for Stanford

            else:
                logger.error(f"Unknown model type: {model_type}")
                continue

            model = model.to(device)

            # Train or load model
            if optimizer is not None:  # Trainable models
                if best_model_path.exists():
                    logger.info(f"✓ Found existing trained model: {best_model_path}")
                    logger.info(f"  Loading weights and skipping training for {model_name}...")
                    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    best_val_f1 = checkpoint.get('val_f1', 0.0)
                    logger.info(f"  Loaded model with validation F1: {best_val_f1:.4f}")
                else:
                    logger.info(f"✗ No existing model found at {best_model_path}")

                    # Check for checkpoint to resume training
                    start_epoch = 0
                    best_val_f1 = 0.0

                    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
                    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

                    if checkpoint_path.exists():
                        logger.info(f"✓ Found checkpoint: {checkpoint_path}")
                        logger.info(f"  Resuming training from checkpoint...")
                        try:
                            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            if 'scheduler_state_dict' in checkpoint:
                                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                            start_epoch = checkpoint.get('epoch', 0)
                            best_val_f1 = checkpoint.get('best_val_f1', 0.0)
                            logger.info(f"  Resuming from epoch {start_epoch} with best F1: {best_val_f1:.4f}")
                        except Exception as e:
                            logger.warning(f"  Failed to load checkpoint: {e}")
                            logger.warning(f"  Starting training from scratch...")
                            start_epoch = 0
                            best_val_f1 = 0.0
                    else:
                        logger.info(f"  Starting training {model_name} from scratch for {num_epochs} epochs...")

                    for epoch in range(start_epoch, num_epochs):
                        # Training epoch
                        avg_loss = train_model_single_epoch(model, model_train_loader, optimizer, scheduler, device, epoch, num_epochs)

                        # Validation
                        model.eval()
                        val_preds = []
                        val_labels = []

                        with torch.no_grad():
                            for batch in model_val_loader:
                                input_ids = batch['input_ids'].to(device)
                                attention_mask = batch['attention_mask'].to(device)
                                labels = batch['labels'].to(device)
                                pos_ids = batch.get('pos_ids', torch.zeros_like(input_ids)).to(device)
                                tokens = batch.get('tokens', None)
                                valid_mask = batch.get('valid_mask', None)  # NEW: get valid_mask
                                if valid_mask is not None:
                                    valid_mask = valid_mask.to(device)

                                _, emissions, *_ = model(input_ids, attention_mask, labels, pos_ids, tokens, valid_mask)

                                if hasattr(model, 'crf'):
                                    # CRITICAL FIX: Use valid_mask for decoding (only first subwords)
                                    decode_mask = (attention_mask.bool() & (labels != -100)) if labels is not None else attention_mask.bool()
                                    if valid_mask is not None:
                                        decode_mask = (attention_mask.bool() & valid_mask)
                                    preds = model.crf.decode(emissions, mask=decode_mask, pos_ids=pos_ids)

                                    # CRITICAL FIX: Use mask-based alignment (same as evaluate_model)
                                    for i, pred in enumerate(preds):
                                        # preds[i] already contains only valid positions (from mask-based decode)
                                        pred_tags = [idx2tag.get(p.item() if hasattr(p, 'item') else p, "O")
                                                    for p in pred]

                                        # Extract true labels - only positions where labels != -100
                                        true_tags = []
                                        for j in range(labels.size(1)):
                                            if labels[i][j].item() != -100:
                                                true_tags.append(idx2tag[labels[i][j].item()])

                                        # Both should have same length (num_valid positions)
                                        if len(pred_tags) == len(true_tags) and len(pred_tags) > 0:
                                            val_preds.append(pred_tags)
                                            val_labels.append(true_tags)
                                        elif len(pred_tags) > 0 and len(true_tags) > 0:
                                            # Fallback: take minimum length to prevent crash
                                            min_len = min(len(pred_tags), len(true_tags))
                                            val_preds.append(pred_tags[:min_len])
                                            val_labels.append(true_tags[:min_len])

                        from seqeval.metrics import f1_score as seq_f1
                        val_f1 = seq_f1(val_labels, val_preds)

                        # CRITICAL: Dynamic FOT weight adjustment based on validation F1
                        # This prevents model from becoming too conservative in later epochs
                        # CRITICAL FIX: Adjust CRF's fot_weight (not model's), as _apply_weights uses model.crf.fot_weight
                        if hasattr(model, 'crf') and hasattr(model.crf, 'fot_weight'):
                            old_fot_weight = float(model.crf.fot_weight)
                            from ...models.ner.bert_crf import adjust_fot_weight
                            model.crf.fot_weight = adjust_fot_weight(val_f1, old_fot_weight)
                            if abs(model.crf.fot_weight - old_fot_weight) > 0.01:
                                logger.info(f"  FOT weight adjusted: {old_fot_weight:.3f} → {model.crf.fot_weight:.3f}")

                        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}")

                        # Save checkpoint every epoch
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'epoch': epoch + 1,
                            'val_f1': val_f1,
                            'best_val_f1': best_val_f1
                        }, checkpoint_path)

                        # Save best model (or first epoch)
                        if val_f1 > best_val_f1 or epoch == start_epoch:
                            best_val_f1 = val_f1
                            torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'epoch': epoch + 1,
                                'val_f1': val_f1
                            }, best_model_path)
                            logger.info(f"✓ Saved best model with F1: {val_f1:.4f} at epoch {epoch+1}")

                    # Load best model for evaluation
                    logger.info(f"Loading best model from {best_model_path}")
                    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded best model from epoch {checkpoint['epoch']} with F1: {checkpoint['val_f1']:.4f}")

            # Evaluate model
            logger.info(f"Evaluating {model_name}...")
            if model_type in ['spacy', 'stanford']:
                # Commercial NER systems - direct evaluation without training
                logger.info(f"  Running inference with {model_type.upper()} NER...")
                metrics = evaluate_model(model, model_test_loader, idx2tag, device, model_name)
            else:
                # Trained models - standard evaluation
                metrics = evaluate_model(model, model_test_loader, idx2tag, device, model_name)

            all_results[model_name] = metrics

            logger.info(f"✓ {model_name} complete: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

        except Exception as e:
            logger.error(f"✗ Error with {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_with_metadata = {
        'metadata': {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'device': str(device),
            'sample_size': sample_size,
            'data_files': {
                'mag1': mag1_file,
                'mag2': mag2_file,
                'entities': mag_entities_file
            }
        },
        'results': all_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info("="*80)
    logger.info("Comparison Experiments Complete")
    logger.info("="*80)

    # Print summary
    logger.info("\nFinal Results Summary:")
    logger.info(f"{'Model':<30} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Speed':>10}")
    logger.info("-" * 80)
    for model_name, metrics in all_results.items():
        logger.info(f"{model_name:<30} {metrics['f1']:>8.4f} {metrics['precision']:>10.4f} {metrics['recall']:>8.4f} {metrics['speed']:>10.2f}")

    return all_results
