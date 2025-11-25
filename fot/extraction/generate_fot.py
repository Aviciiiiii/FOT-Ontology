"""
FOT Generation from Patent Titles

This module implements dual-mode FOT extraction:
- Simple mode (dry-run): Quick pseudo-FOT extraction (first 2 words)
- NER mode (real): Full NER-based extraction with BertBiLSTMCRF model

Ported from /src/FOT_generator.py (593 lines).
"""

from __future__ import annotations

import csv
import gc
import glob
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logging import get_logger

logger = get_logger("generate_fot")


def run_generate(
    titles_tsv: str,
    ipc_tsv: str,
    chunks_dir: str,
    *,
    dry_run: bool = True,
    fast: bool = True,
    model_path: Optional[str] = None,
    config: Optional[Dict] = None
) -> str:
    """Generate FOT extraction results.

    Args:
        titles_tsv: Path to titles.txt (TSV with patent_id, title)
        ipc_tsv: Path to ipc_codes.txt (TSV with patent_id, ipc_code)
        chunks_dir: Output directory for chunk files
        dry_run: If True, use simple pseudo-FOT extraction
        fast: If True, use simplified processing
        model_path: Path to trained NER model checkpoint (required for real mode)
        config: Configuration dictionary with extraction parameters

    Returns:
        Path to output directory (chunk_000 or chunk_files/)

    Modes:
        - Dry-run/Simple: Takes first 2 words as pseudo-FOT (for testing)
        - Real/NER: Uses trained BertBiLSTMCRF model with SpaCy POS filtering
    """
    if dry_run or not model_path:
        logger.info("Running in SIMPLE mode (pseudo-FOT extraction)")
        return _generate_simple_fot(titles_tsv, ipc_tsv, chunks_dir)
    else:
        logger.info("Running in NER mode (full model-based extraction)")
        return _generate_ner_fot(titles_tsv, ipc_tsv, chunks_dir, model_path, config or {})


def _generate_simple_fot(titles_tsv: str, ipc_tsv: str, chunks_dir: str) -> str:
    """Simple pseudo-FOT generation (dry-run mode).

    Takes first 2 words from title as pseudo-FOT.
    Fast execution for testing and CI/CD.
    """
    # Load titles
    titles: List[Tuple[str, str]] = []
    with open(titles_tsv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            titles.append((row["patent_id"], row["title"]))

    # Load IPCs
    ipc_map: Dict[str, str] = {}
    with open(ipc_tsv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            ipc_map[row["patent_id"]] = row["ipc_code"]

    out_dir = Path(chunks_dir) / "chunk_000"
    out_dir.mkdir(parents=True, exist_ok=True)
    patent_out = out_dir / "patent_title_with_FOT.txt"
    fotmap_out = out_dir / "fot_id_mapping.txt"

    fot_id_counter = 1
    fot_map: Dict[str, int] = {}
    fot_rows: List[Tuple[int, str, str]] = []  # (id, name, ipc)
    patent_rows: List[Tuple[str, str, str]] = []  # (pid, title, FOT)

    for pid, title in titles:
        words = [w for w in title.split() if w.isalpha() or w.isalnum()]
        if not words:
            fot_str = ""
        else:
            # Take first two words as pseudo FOT
            end = min(2, len(words))
            fot_name = " ".join(words[:end])
            if fot_name not in fot_map:
                fot_map[fot_name] = fot_id_counter
                fot_rows.append((fot_id_counter, fot_name, ipc_map.get(pid, "")))
                fot_id_counter += 1
            fid = fot_map[fot_name]
            start_idx, end_idx = 0, end - 1
            fot_str = f"{fid}:{start_idx}:{end_idx}"
        patent_rows.append((pid, title, fot_str))

    with open(patent_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(["patentid", "title", "FOT"])
        for row in patent_rows:
            w.writerow(list(row))

    with open(fotmap_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(["fot_id", "fot_name", "ipc_code"])
        for fid, name, ipc in fot_rows:
            w.writerow([fid, name, ipc])

    logger.info("Generated simple FOT at %s with patents=%d fots=%d", out_dir, len(patent_rows), len(fot_rows))
    return str(out_dir)


def _generate_ner_fot(
    titles_tsv: str,
    ipc_tsv: str,
    chunks_dir: str,
    model_path: str,
    config: Dict
) -> str:
    """Full NER-based FOT generation (real mode).

    Uses trained BertBiLSTMCRF model with:
    - GPU acceleration (if available)
    - SpaCy POS tagging and lemmatization
    - 74 special symbol FOT preservation
    - Rule-based filtering
    - 20-chunk processing with checkpoints
    """
    # Import heavy dependencies only when needed
    try:
        from ..models.ner.bert_crf import BertBiLSTMCRF
        from transformers import AutoTokenizer
        from .ner_dataset import create_inference_dataset, custom_collate_fn
        from .fot_extractor import process_batch_predictions
    except ImportError as e:
        logger.error(f"Failed to import NER dependencies: {e}")
        logger.error("Please install: pip install transformers spacy torch")
        raise

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load configuration
    batch_size = config.get("batch_size", 128)
    max_len = config.get("max_len", 48)
    num_chunks = config.get("num_chunks", 20)
    use_amp = config.get("use_amp", True) and torch.cuda.is_available()

    # POS weights for CRF
    pos_weight_dict = config.get("pos_weight_dict", {
        'NOUN': 1.3, 'PROPN': 1.3, 'ADJ': 1.1, 'VERB': 0.9, 'NUM': 0.8,
        'ADP': 0.6, 'DET': 0.5, 'CCONJ': 0.6, 'PART': 0.6, 'PRON': 0.5,
        'AUX': 0.5, 'ADV': 0.7, 'SCONJ': 0.6, 'INTJ': 0.4, 'SYM': 0.7, 'X': 0.8,
    })

    # Load model
    logger.info(f"Loading NER model from {model_path}")
    model, tokenizer, idx2tag = _load_ner_model(model_path, pos_weight_dict, device)

    # Load data
    logger.info("Loading patent titles and IPC codes")
    data, ipc_codes = _load_data_for_ner(titles_tsv, ipc_tsv)
    total_patents = len(data)
    logger.info(f"Loaded {total_patents} patents")

    if total_patents == 0:
        logger.error("No data loaded. Exiting.")
        return str(Path(chunks_dir) / "chunk_000")

    # Setup output directories
    out_dir = Path(chunks_dir) / "chunk_files"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_save_dir = out_dir / "dataset_checkpoints"
    dataset_save_dir.mkdir(exist_ok=True)

    # Initialize FOT tracking
    fot_id_counter = 1
    fot_id_mapping = {}
    all_output_data = []
    all_fot_id_data = []

    # Check for existing checkpoint
    latest_checkpoint = _get_latest_checkpoint(str(dataset_save_dir))
    start_index = 0
    if latest_checkpoint:
        logger.info(f"Found checkpoint: {latest_checkpoint}")
        processed_data = _load_checkpoint(latest_checkpoint)
        start_index = len(processed_data)
        logger.info(f"Resuming from patent {start_index + 1}")

    # Process data
    remaining_data = data[start_index:]
    logger.info("Creating inference dataset with SpaCy preprocessing")
    inference_dataset = create_inference_dataset(
        remaining_data, tokenizer, max_len, pos_weight_dict, save_dir=str(dataset_save_dir)
    )

    dataloader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    save_interval = total_patents // 20  # Every 5%
    processed_patents = start_index

    # Process batches with progress bar
    logger.info("Starting NER inference")
    with tqdm(total=total_patents, initial=processed_patents, desc="Processing patents") as pbar:
        for batch in dataloader:
            # Run model inference
            results = _process_batch_ner(
                batch, model, device, idx2tag, fot_id_counter, fot_id_mapping, use_amp
            )

            # Collect results
            for patent_id, title, fot_sequence, fot_details in results:
                all_output_data.append([patent_id, title, fot_sequence])
                for fot_id, fot_name, ipc_code in fot_details:
                    all_fot_id_data.append([fot_id, fot_name, ipc_code])

            # Update counter
            if fot_id_mapping:
                fot_id_counter = max(fot_id_mapping.values()) + 1

            batch_size_actual = len(batch['patent_id'])
            processed_patents += batch_size_actual
            pbar.update(batch_size_actual)

            # Memory management every 5%
            if processed_patents % save_interval < batch_size_actual:
                logger.info(f"Processed {processed_patents}/{total_patents} ({processed_patents/total_patents*100:.1f}%)")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Write results
    logger.info("Writing results to files")
    output_file = out_dir / "patent_title_with_FOT.txt"
    fot_id_file = out_dir / "fot_id_mapping.txt"

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['patentid', 'title', 'FOT'])
        writer.writerows(all_output_data)

    with open(fot_id_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['fot_id', 'fot_name', 'ipc_code'])
        writer.writerows(all_fot_id_data)

    logger.info(f"NER FOT extraction complete: {len(all_output_data)} patents, {len(set(fot_id_mapping.values()))} unique FOTs")
    logger.info(f"Results saved to {out_dir}")

    return str(out_dir)


def _load_ner_model(model_path: str, pos_weight_dict: Dict, device: torch.device):
    """Load trained NER model from checkpoint."""
    from ..models.ner.bert_crf import build_model
    from transformers import AutoTokenizer, AutoModel

    # PyTorch 2.6+ changed weights_only default to True, but our checkpoints contain numpy objects
    # Set weights_only=False for backward compatibility with checkpoints saved in PyTorch < 2.6
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Tag mappings
    tag2idx = {"O": 0, "B-FOT": 1, "I-FOT": 2}
    idx2tag = {v: k for k, v in tag2idx.items()}
    num_tags = len(tag2idx)

    # POS mappings
    idx2pos = {i: pos for i, pos in enumerate(['PAD'] + list(pos_weight_dict.keys()))}

    # Determine model path - check config or use default SciBERT
    bert_model_path = "files/scibert_scivocab_uncased"

    # Build model with backbone
    model = build_model(
        pretrained_name_or_path=bert_model_path,
        num_tags=num_tags,
        pos_weight_dict=pos_weight_dict,
        idx2pos=idx2pos,
        lstm_hidden_dim=256,
        fot_weight=0.8,
        num_hidden_layers=4,
        tag2idx=tag2idx
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    # Load tokenizer (SciBERT)
    tokenizer = AutoTokenizer.from_pretrained(bert_model_path)

    logger.info("NER model loaded successfully")
    return model, tokenizer, idx2tag


def _load_data_for_ner(titles_tsv: str, ipc_tsv: str) -> Tuple[List[Tuple], Dict]:
    """Load patent titles and IPC codes for NER processing."""
    # Load IPC codes first
    ipc_codes = {}
    with open(ipc_tsv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                patent_id, ipc_code_str = row[0], row[1]
                ipc_codes[patent_id] = ipc_code_str

    # Load titles
    data = []
    with open(titles_tsv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            patent_id = row.get('patent_id', '')
            title = row.get('title', '')
            ipc_code = ipc_codes.get(patent_id, '')
            if patent_id and title:
                data.append((patent_id, title, ipc_code))

    return data, ipc_codes


def _process_batch_ner(batch, model, device, idx2tag, fot_id_counter, fot_id_mapping, use_amp):
    """Process single batch with NER model."""
    from .fot_extractor import process_batch_predictions

    # Move to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    pos_ids = batch['pos_ids'].to(device)

    # Run inference
    with torch.no_grad():
        if use_amp:
            with autocast():
                preds = model(input_ids, attention_mask, pos_ids=pos_ids, tokens=batch['tokens'])
        else:
            preds = model(input_ids, attention_mask, pos_ids=pos_ids, tokens=batch['tokens'])

    # Handle tuple output
    if isinstance(preds, tuple):
        preds = preds[0]

    # Move to CPU for post-processing
    preds = preds.cpu()

    # Extract FOTs with rule-based filtering
    results, fot_id_counter, fot_id_mapping = process_batch_predictions(
        batch, preds, idx2tag, fot_id_mapping, fot_id_counter
    )

    return results


def _get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find latest checkpoint file in directory."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'dataset_checkpoint_*.pkl'))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)


def _load_checkpoint(checkpoint_path: str):
    """Load checkpoint data."""
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)