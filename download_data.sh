#!/bin/bash

# FOT-Storage Data Download Script
# Downloads datasets and models from Hugging Face to the correct local directories.

# Base URL for your Hugging Face dataset (using resolve/main for direct file access)
HF_BASE="https://huggingface.co/datasets/tim-berg/fot-storage-dataset/resolve/main"

echo "======================================================="
echo "Starting download for FOT Storage Project..."
echo "======================================================="

# --- 1. Setup Directory Structure ---
echo "[1/6] Creating local directory structure..."
mkdir -p files/distilbert-base-uncased
mkdir -p files/scibert_scivocab_uncased
mkdir -p data/raw
mkdir -p data/interim
mkdir -p data/processed
mkdir -p artifacts/models
mkdir -p src/BLINK-main/models
echo "Done."

# --- 2. Common & External Files ---
echo "[2/6] Downloading Common & External Knowledge Bases..."

# Entity Catalog (Required by Stage 1 & BLINK) -> files/entity.jsonl
wget -c -O files/entity.jsonl "$HF_BASE/common/entity.jsonl"

# MAG Data (Required by Stage 1 GAT) -> data/raw/15.FieldsOfStudy.nt
wget -c -O data/raw/15.FieldsOfStudy.nt "$HF_BASE/common/15.FieldsOfStudy.nt"

# --- NEW: Base Models (Required by Stage 2 & 3) ---
echo "Downloading Base Models to files/..."

# 1. DistilBERT
echo "  - distilbert-base-uncased..."
DB_BASE="$HF_BASE/base_models/distilbert-base-uncased"
DB_DEST="files/distilbert-base-uncased"
wget -c -O "$DB_DEST/config.json" "$DB_BASE/config.json"
wget -c -O "$DB_DEST/pytorch_model.bin" "$DB_BASE/pytorch_model.bin"
wget -c -O "$DB_DEST/vocab.txt" "$DB_BASE/vocab.txt"
# Optional: tokenizer files if you uploaded them (recommended)
wget -c -O "$DB_DEST/tokenizer.json" "$DB_BASE/tokenizer.json" || true
wget -c -O "$DB_DEST/tokenizer_config.json" "$DB_BASE/tokenizer_config.json" || true

# 2. SciBERT
echo "  - scibert_scivocab_uncased..."
SB_BASE="$HF_BASE/base_models/scibert_scivocab_uncased"
SB_DEST="files/scibert_scivocab_uncased"
wget -c -O "$SB_DEST/config.json" "$SB_BASE/config.json"
wget -c -O "$SB_DEST/pytorch_model.bin" "$SB_BASE/pytorch_model.bin"
wget -c -O "$SB_DEST/vocab.txt" "$SB_BASE/vocab.txt"

# --- 3. BLINK Models ---
echo "[3/6] Downloading BLINK Models..."
# Required by Stage 1
wget -c -O src/BLINK-main/models/biencoder_wiki_large.bin "$HF_BASE/BLINK-main/biencoder_wiki_large.bin"
wget -c -O src/BLINK-main/models/biencoder_wiki_large.json "$HF_BASE/BLINK-main/biencoder_wiki_large.json"
wget -c -O src/BLINK-main/models/crossencoder_wiki_large.bin "$HF_BASE/BLINK-main/crossencoder_wiki_large.bin"
wget -c -O src/BLINK-main/models/crossencoder_wiki_large.json "$HF_BASE/BLINK-main/crossencoder_wiki_large.json"
# Symlink entity.jsonl for BLINK
ln -sf ../../../files/entity.jsonl src/BLINK-main/models/entity.jsonl

# --- 4. Stage 1: Static Ontology ---
echo "[4/6] Downloading Stage 1 Data..."

# Input Seeds
wget -c -O data/processed/floor_1.csv "$HF_BASE/stage1/floor_1.csv"
wget -c -O data/processed/floor_2.csv "$HF_BASE/stage1/floor_2.csv"
wget -c -O data/processed/floor_3.csv "$HF_BASE/stage1/floor_3.csv"

# Intermediate Outputs
wget -c -O data/interim/blink_candidates_l3.json "$HF_BASE/stage1/blink_candidates_l3.json"
wget -c -O data/interim/blink_candidates_l1l2l3.json "$HF_BASE/stage1/blink_candidates_l1l2l3.json"
# Note: fot_level_12.txt is often used in Stage 3 but produced in Stage 1 context
wget -c -O data/interim/fot_level_12.txt "$HF_BASE/stage3/fot_level_12.txt"

# Final Outputs
wget -c -O data/processed/l3_dedup.json "$HF_BASE/stage1/l3_dedup.json"
wget -c -O data/processed/l3_with_parents.json "$HF_BASE/stage1/l3_with_parents.json"
wget -c -O data/processed/gat_l3_predictions.json "$HF_BASE/stage1/gat_l3_predictions.json"

# --- 5. Stage 2: Dynamic Ontology (NER) ---
echo "[5/6] Downloading Stage 2 Data & Models..."

# Intermediate Search Data
wget -c -O data/interim/mag_entities.json "$HF_BASE/stage2/mag_entities.json"
wget -c -O data/interim/third_entities.json "$HF_BASE/stage2/third_entities.json"
wget -c -O data/interim/mag1_tagged_searched_sentences.json "$HF_BASE/stage2/mag1_tagged_searched_sentences.json"
wget -c -O data/interim/mag2_tagged_searched_sentences.json "$HF_BASE/stage2/mag2_tagged_searched_sentences.json"
wget -c -O data/interim/FOT1_tagged_searched_sentences.json "$HF_BASE/stage2/FOT1_tagged_searched_sentences.json"
wget -c -O data/interim/FOT2_tagged_searched_sentences.json "$HF_BASE/stage2/FOT2_tagged_searched_sentences.json"

# Cleaned Training Data
wget -c -O data/processed/cleaned_mag1_tagged_searched_sentences_with_entity.json "$HF_BASE/stage2/cleaned_mag1_tagged_searched_sentences_with_entity.json"
wget -c -O data/processed/cleaned_mag2_tagged_searched_sentences_with_entity.json "$HF_BASE/stage2/cleaned_mag2_tagged_searched_sentences_with_entity.json"
wget -c -O data/processed/cleaned_FOT1_tagged_searched_sentences_with_entity.json "$HF_BASE/stage2/cleaned_FOT1_tagged_searched_sentences_with_entity.json"
wget -c -O data/processed/cleaned_FOT2_tagged_searched_sentences_with_entity.json "$HF_BASE/stage2/cleaned_FOT2_tagged_searched_sentences_with_entity.json"

# Trained Models (Output of Stage 2)
wget -c -O artifacts/models/NER_pretrain_model_best.pth "$HF_BASE/stage2/NER_pretrain_model_best.pth"
wget -c -O artifacts/models/NER_finetune_model_best.pth "$HF_BASE/stage2/NER_finetune_model_best.pth"

# --- 6. Stage 3: Patents & Extraction ---
echo "[6/6] Downloading Stage 3 Inputs & Outputs..."

# Processed Inputs
wget -c -O data/processed/titles.txt "$HF_BASE/stage3/titles.txt"
wget -c -O data/processed/publication_number.txt "$HF_BASE/stage3/publication_number.txt"
wget -c -O data/processed/ipc_codes.txt "$HF_BASE/stage3/ipc_codes.txt"

# Intermediate
wget -c -O data/interim/fot_level_3.txt "$HF_BASE/stage3/fot_level_3.txt"

# Embedding Cache (Optional but recommended)
wget -c -O artifacts/embeddings/hierarchy_embeddings.npz "$HF_BASE/stage3/hierarchy_embeddings.npz" || true

# Final Outputs
wget -c -O data/processed/new_total_merged_patent_title_with_FOT.txt "$HF_BASE/stage3/new_total_merged_patent_title_with_FOT.txt"
wget -c -O data/processed/new_total_merged_fot_mapping.txt "$HF_BASE/stage3/new_total_merged_fot_mapping.txt"
wget -c -O data/processed/new_fot_hierarchy_dynamic.txt "$HF_BASE/stage3/new_fot_hierarchy_dynamic.txt"
wget -c -O data/processed/full_fot_library.txt "$HF_BASE/stage3/full_fot_library.txt"

echo "======================================================="
echo "All downloads complete!"
echo "You can now run the pipeline stages."
echo "======================================================="
