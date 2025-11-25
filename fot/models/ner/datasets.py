"""
Complete dataset handling with spaCy POS integration and dynamic balanced sampling.

This module implements all the advanced dataset functionality from the original scripts:
- SpaCy-based POS tag processing
- Dynamic balanced batch sampling
- Entity-aware dataset balancing
- Multi-processing POS tag extraction
- Custom collate functions
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional
from multiprocessing import Pool, cpu_count

import torch
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer
from tqdm import tqdm


# Global spaCy model - will be loaded when needed
nlp = None


def _load_spacy_model():
    """Load spaCy model for POS tagging."""
    global nlp
    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
        except OSError:
            # Fallback to smaller model if large model not available
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except OSError:
                # If no spacy model available, use dummy processing
                nlp = "dummy"
    return nlp


def process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of samples to add POS tags using spaCy.

    Args:
        batch: List of samples with 'tokens' field

    Returns:
        List of samples with added 'pos_tags' field
    """
    nlp_model = _load_spacy_model()

    if nlp_model == "dummy":
        # Fallback: assign default POS tags
        for item in batch:
            item['pos_tags'] = ['NOUN'] * len(item.get('tokens', []))
        return batch

    # Process with spaCy
    docs = list(nlp_model.pipe([" ".join(item['tokens']) for item in batch]))
    for item, doc in zip(batch, docs):
        item['pos_tags'] = [token.pos_ for token in doc]

        # Ensure POS tags match token length
        tokens_len = len(item.get('tokens', []))
        pos_len = len(item['pos_tags'])

        if pos_len < tokens_len:
            # Pad with NOUN if spaCy produced fewer tags
            item['pos_tags'].extend(['NOUN'] * (tokens_len - pos_len))
        elif pos_len > tokens_len:
            # Truncate if spaCy produced more tags
            item['pos_tags'] = item['pos_tags'][:tokens_len]

    return batch


def preprocess_with_pos(data: List[Dict[str, Any]], batch_size: int = 1000,
                       num_processes: int = 6) -> List[Dict[str, Any]]:
    """Add POS tags to data using multiprocessing.

    Args:
        data: List of samples with 'tokens' field
        batch_size: Size of batches for processing
        num_processes: Number of processes for parallel processing

    Returns:
        List of samples with added 'pos_tags' field
    """
    if not data:
        return data

    # Use fewer processes if data is small
    actual_processes = min(num_processes, max(1, len(data) // batch_size))

    if actual_processes <= 1:
        # Single process for small datasets
        return process_batch(data)

    # Multi-process for large datasets
    batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    with Pool(processes=actual_processes) as pool:
        results = []
        for batch in tqdm(pool.imap(process_batch, batches),
                         total=len(batches), desc="Adding POS tags"):
            results.extend(batch)

    return results


def preprocess_balanced_dataset(data: List[Dict[str, Any]], fot_entities: List[str],
                               negative_ratio: int = 2) -> List[Dict[str, Any]]:
    """Balance dataset based on FOT entity matching - original script logic.

    This function implements the ORIGINAL balanced sampling strategy:
    1. Separate positive (has FOT tags) and negative (all O tags) samples
    2. Filter positive samples to include only those matching FOT entities
    3. Add negative samples at the specified ratio
    4. Deduplicate and shuffle

    Args:
        data: List of samples with 'tokens' and 'tags' fields
        fot_entities: List of FOT entity strings for matching
        negative_ratio: Ratio of negative to positive samples (default: 2)

    Returns:
        Balanced list of samples with entity-matched positives
    """
    # Separate positive and negative samples
    positive_samples = []
    negative_samples = []

    for sample in data:
        if 'tags' in sample and any(tag in ['B-FOT', 'I-FOT'] for tag in sample['tags']):
            positive_samples.append(sample)
        else:
            negative_samples.append(sample)

    print(f"Original dataset: {len(data)} samples")
    print(f"  - Positive (has FOT): {len(positive_samples)}")
    print(f"  - Negative (no FOT): {len(negative_samples)}")

    # Entity-based filtering (ORIGINAL LOGIC)
    balanced_dataset = set()

    def make_hashable(d: Dict[str, Any]) -> str:
        """Convert dict to hashable string for deduplication."""
        return json.dumps(d, sort_keys=True)

    # Filter positive samples by FOT entity matching
    matched_entities = 0
    total_matches = 0

    for fot in fot_entities:
        fot_lower = fot.lower()
        entity_matches = 0

        for s in positive_samples:
            tokens_text = ' '.join(s.get('tokens', [])).lower()
            if fot_lower in tokens_text:
                balanced_dataset.add(make_hashable(s))
                entity_matches += 1

        if entity_matches > 0:
            matched_entities += 1
            total_matches += entity_matches

    total_positive = len(balanced_dataset)

    print(f"Entity matching:")
    print(f"  - Entities tested: {len(fot_entities)}")
    print(f"  - Entities with matches: {matched_entities}")
    print(f"  - Total positive samples matched: {total_matches}")
    print(f"  - Unique positive samples selected: {total_positive}")
    print(f"  - Match rate: {matched_entities/len(fot_entities)*100 if fot_entities else 0:.1f}%")

    # Add negative samples (negative_ratio x the positive samples)
    if total_positive > 0:
        num_neg_samples = min(total_positive * negative_ratio, len(negative_samples))
        if num_neg_samples > 0:
            selected_neg_samples = random.sample(negative_samples, num_neg_samples)
            for sample in selected_neg_samples:
                balanced_dataset.add(make_hashable(sample))
            print(f"  - Negative samples added: {num_neg_samples}")
    else:
        num_neg_samples = 0
        print(f"  - WARNING: No positive samples matched, no negatives added!")

    # Convert back to list and shuffle
    balanced_list = [json.loads(item) for item in balanced_dataset]
    random.shuffle(balanced_list)

    print(f"Final balanced dataset: {len(balanced_list)} samples")
    if total_positive > 0:
        actual_ratio = num_neg_samples / total_positive if total_positive > 0 else 0
        print(f"  - Actual ratio: 1:{actual_ratio:.2f} (positive:negative)")

    return balanced_list


class NERDataset(Dataset):
    """Enhanced NER dataset with POS tag support and comprehensive processing.

    This dataset implementation includes all features from the original script:
    - Automatic POS tag generation using spaCy
    - Proper subword alignment for BERT tokenizers
    - Support for variable length sequences
    - Comprehensive tag mappings
    """

    def __init__(self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer,
                 max_len: int = 128, tag2idx: Optional[Dict[str, int]] = None,
                 add_pos_tags: bool = True):
        self.items = data
        self.tok = tokenizer
        self.max_len = max_len
        self.tag2idx = tag2idx or {"O": 0, "B-FOT": 1, "I-FOT": 2}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        # Add POS tags if requested and not already present
        if add_pos_tags and data and 'pos_tags' not in data[0]:
            print("Adding POS tags to dataset...")
            self.items = preprocess_with_pos(data)

        # Collect all unique POS tags and create mappings
        all_pos_tags = set()
        for item in self.items:
            all_pos_tags.update(item.get('pos_tags', []))

        # Create pos2idx mapping including PAD
        self.pos2idx = {pos: idx for idx, pos in enumerate(['PAD'] + sorted(list(all_pos_tags)))}
        self.idx2pos = {v: k for k, v in self.pos2idx.items()}

        print(f"Dataset created with {len(self.items)} samples")
        print(f"Tag mappings: {self.tag2idx}")
        print(f"POS mappings: {len(self.pos2idx)} unique POS tags")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        tokens = item.get("tokens", [])
        labels = item.get("labels", item.get("tags", []))  # Support both field names
        pos_tags = item.get("pos_tags", [])

        # Ensure all sequences have same length
        min_len = min(len(tokens), len(labels), len(pos_tags)) if pos_tags else min(len(tokens), len(labels))
        tokens = tokens[:min_len]
        labels = labels[:min_len]
        pos_tags = pos_tags[:min_len] if pos_tags else ['NOUN'] * min_len

        # Tokenize with subword alignment
        enc = self.tok(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Align labels with subwords - CRITICAL FIX for subword continuation
        # Only train on first subword of each word; mark continuations and special tokens as -100
        word_ids = enc.word_ids(batch_index=0)
        aligned_labels = []
        aligned_pos_ids = []
        valid_mask = []  # 1 for valid first-subword positions, 0 for ignored positions
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens ([CLS], [SEP], [PAD]) - IGNORE in loss
                aligned_labels.append(-100)
                aligned_pos_ids.append(self.pos2idx.get('PAD', 0))
                valid_mask.append(0)
            elif word_id == prev_word_id:
                # Continuation of previous word (subword split) - IGNORE in loss
                # This prevents training on ##continuation tokens
                aligned_labels.append(-100)
                aligned_pos_ids.append(self.pos2idx.get('PAD', 0))
                valid_mask.append(0)
            else:
                # First subword of a word - ONLY position used for training/inference
                if word_id < len(labels):
                    aligned_labels.append(self.tag2idx.get(labels[word_id], 0))
                    aligned_pos_ids.append(self.pos2idx.get(pos_tags[word_id], self.pos2idx.get('NOUN', 0)))
                    valid_mask.append(1)
                else:
                    aligned_labels.append(-100)  # Out of range
                    aligned_pos_ids.append(self.pos2idx.get('PAD', 0))
                    valid_mask.append(0)
            prev_word_id = word_id

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
            "pos_ids": torch.tensor(aligned_pos_ids, dtype=torch.long),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),  # NEW: mask for valid positions
            "tokens": tokens + ['[PAD]'] * (self.max_len - len(tokens))  # Padded tokens for debugging
        }

    def save(self, path: str) -> None:
        """Save dataset to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'items': self.items,
                'tokenizer': self.tok,
                'max_len': self.max_len,
                'tag2idx': self.tag2idx,
                'idx2tag': self.idx2tag,
                'pos2idx': self.pos2idx,
                'idx2pos': self.idx2pos
            }, f)

    @classmethod
    def load(cls, path: str) -> 'NERDataset':
        """Load dataset from file."""
        import pickle
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)

        dataset = cls.__new__(cls)
        dataset.items = data_dict['items']
        dataset.tok = data_dict['tokenizer']
        dataset.max_len = data_dict['max_len']
        dataset.tag2idx = data_dict['tag2idx']
        dataset.idx2tag = data_dict['idx2tag']
        dataset.pos2idx = data_dict['pos2idx']
        dataset.idx2pos = data_dict['idx2pos']
        return dataset


class DynamicBalancedBatchSampler(Sampler[List[int]]):
    """Dynamic batch sampler that adjusts positive/negative ratio during training.

    This implements the sophisticated sampling strategy from the original script:
    - Starts with high positive ratio for learning FOT patterns
    - Gradually reduces positive ratio to improve generalization
    - Maintains balanced batches throughout training
    """

    def __init__(self, dataset: Dataset, batch_size: int,
                 initial_positive_ratio: float = 0.5,
                 final_positive_ratio: float = 0.3,
                 epochs: int = 10):
        self.dataset = dataset
        self.batch_size = batch_size
        self.initial_positive_ratio = initial_positive_ratio
        self.final_positive_ratio = final_positive_ratio
        self.epochs = epochs
        self.current_epoch = 0

        # Identify positive and negative samples
        self.positive_indices = []
        self.negative_indices = []

        for i, item in enumerate(dataset):
            labels = item.get('labels', torch.tensor([]))
            if isinstance(labels, torch.Tensor):
                # Check if any non-padding label is positive (B-FOT or I-FOT)
                valid_labels = labels[labels != -100]
                if torch.any((valid_labels == 1) | (valid_labels == 2)):
                    self.positive_indices.append(i)
                else:
                    self.negative_indices.append(i)
            else:
                # Fallback for non-tensor labels
                if any(label in [1, 2] for label in labels if label != -100):
                    self.positive_indices.append(i)
                else:
                    self.negative_indices.append(i)

        print(f"DynamicBalancedBatchSampler: {len(self.positive_indices)} positive, "
              f"{len(self.negative_indices)} negative samples")

    def __iter__(self) -> Iterable[List[int]]:
        # Calculate current positive ratio
        if self.epochs > 1:
            progress = self.current_epoch / (self.epochs - 1)
            current_positive_ratio = (self.initial_positive_ratio -
                                    (self.initial_positive_ratio - self.final_positive_ratio) * progress)
        else:
            current_positive_ratio = self.initial_positive_ratio

        num_positive = int(self.batch_size * current_positive_ratio)
        num_negative = self.batch_size - num_positive

        # Generate batches
        total_batches = len(self.dataset) // self.batch_size
        all_batches = []

        for _ in range(total_batches):
            batch = []

            # Sample positive examples
            if self.positive_indices and num_positive > 0:
                sampled_positive = random.choices(
                    self.positive_indices,
                    k=min(num_positive, len(self.positive_indices))
                )
                batch.extend(sampled_positive)

            # Sample negative examples
            if self.negative_indices and num_negative > 0:
                sampled_negative = random.choices(
                    self.negative_indices,
                    k=min(num_negative, len(self.negative_indices))
                )
                batch.extend(sampled_negative)

            # Pad batch if needed
            while len(batch) < self.batch_size:
                if self.positive_indices:
                    batch.append(random.choice(self.positive_indices))
                elif self.negative_indices:
                    batch.append(random.choice(self.negative_indices))
                else:
                    break

            random.shuffle(batch)
            all_batches.append(batch[:self.batch_size])

        self.current_epoch += 1
        return iter(all_batches)

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for NER batches.

    Handles variable length sequences and ensures proper padding/stacking.
    """
    if not batch:
        return {}

    # Get the first item to determine keys
    elem = batch[0]
    collated = {}

    for key in elem.keys():
        if key == "tokens":
            # Handle tokens separately (list of strings)
            max_len = max(len(item[key]) for item in batch)
            padded_tokens = []
            for item in batch:
                tokens = item[key]
                padded = tokens + ['[PAD]'] * (max_len - len(tokens))
                padded_tokens.append(padded)
            collated[key] = padded_tokens
        else:
            # Stack tensors
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values

    return collated


def create_dataset(data: List[Dict[str, Any]], tokenizer: AutoTokenizer,
                  max_len: int = 128, **kwargs) -> NERDataset:
    """Factory function to create NER dataset."""
    return NERDataset(data, tokenizer, max_len, **kwargs)


# For backward compatibility
TokenLabelDataset = NERDataset