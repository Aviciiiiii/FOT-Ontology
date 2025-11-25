"""
NER Inference Dataset for FOT Extraction

Ported from /src/FOT_generator.py lines 40-114.
This dataset handles SpaCy preprocessing, POS tagging, and checkpoint saving.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import spacy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class InferenceNERDataset(Dataset):
    """Dataset for NER inference on patent titles.

    Features:
    - SpaCy preprocessing with POS tagging
    - Checkpoint saving every 5% of data
    - Resume capability from checkpoints
    - Handles patent_id, title, and IPC code triplets
    """

    def __init__(
        self,
        data: List[Tuple[str, str, str]],
        tokenizer,
        max_len: int,
        pos_weight_dict: Dict[str, float],
        save_dir: Optional[str] = None
    ):
        """Initialize dataset.

        Args:
            data: List of (patent_id, title, ipc_code) tuples
            tokenizer: Hugging Face tokenizer (SciBERT)
            max_len: Maximum sequence length
            pos_weight_dict: POS tag weights for CRF
            save_dir: Directory to save checkpoints (if None, no saving)
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2idx = {'O': 0, 'B-FOT': 1, 'I-FOT': 2}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        self.pos2idx = {pos: idx for idx, pos in enumerate(['PAD'] + list(pos_weight_dict.keys()))}
        self.idx2pos = {v: k for k, v in self.pos2idx.items()}

        # Load SpaCy model
        try:
            self.nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
            logger.info("Loaded SpaCy en_core_web_lg for POS tagging")
        except OSError:
            logger.warning("en_core_web_lg not found, falling back to en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.data = self.process_data(data)

    def process_data(self, data: List[Tuple[str, str, str]]) -> List[Dict]:
        """Process raw data with SpaCy and tokenization.

        Saves checkpoints every 5% of data for resume capability.
        """
        processed_data = []
        total = len(data)

        if total == 0:
            logger.warning("No data to process in InferenceNERDataset")
            return processed_data

        save_interval = max(1, total // 20)  # Every 5%

        for i, item in enumerate(tqdm(data, desc="Processing NER dataset")):
            processed_item = self.process_item(item)
            processed_data.append(processed_item)

            # Save checkpoint every 5%
            if self.save_dir and (i + 1) % save_interval == 0:
                save_path = os.path.join(self.save_dir, f"dataset_checkpoint_{i+1}.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(processed_data, f)
                logger.info(f"Saved dataset checkpoint at {i+1}/{total} items ({(i+1)/total*100:.1f}%)")

        return processed_data

    def process_item(self, item: Tuple[str, str, str]) -> Dict:
        """Process single item: SpaCy POS tagging + SciBERT tokenization.

        Args:
            item: (patent_id, title, ipc_code) tuple

        Returns:
            Dictionary with input_ids, attention_mask, pos_ids, tokens, etc.
        """
        patent_id, title, ipc_code = item

        # SpaCy processing for POS tags
        doc = self.nlp(title)
        tokens = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]

        # Truncate to max_len
        tokens = tokens[:self.max_len]
        pos_tags = pos_tags[:self.max_len]

        # Pad to max_len
        padding_length = self.max_len - len(tokens)
        tokens += ['[PAD]'] * padding_length
        pos_tags += ['PAD'] * padding_length

        # SciBERT tokenization
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

        # Convert POS tags to indices
        pos_ids = [self.pos2idx.get(pos, self.pos2idx['PAD']) for pos in pos_tags]

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'pos_ids': torch.tensor(pos_ids),
            'tokens': tokens,
            'patent_id': patent_id,
            'ipc_code': ipc_code
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """Get preprocessed item at index."""
        item = self.data[idx]
        tokens = item['tokens'][:self.max_len] + ['[PAD]'] * (self.max_len - len(item['tokens']))
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'pos_ids': item['pos_ids'],
            'tokens': tokens,
            'patent_id': item['patent_id'],
            'ipc_code': item['ipc_code']
        }


def create_inference_dataset(
    data: List[Tuple[str, str, str]],
    tokenizer,
    max_len: int,
    pos_weight_dict: Dict[str, float],
    save_dir: Optional[str] = None
) -> InferenceNERDataset:
    """Factory function for creating inference dataset.

    Args:
        data: List of (patent_id, title, ipc_code) tuples
        tokenizer: Hugging Face tokenizer
        max_len: Maximum sequence length
        pos_weight_dict: POS tag weights
        save_dir: Checkpoint directory (optional)

    Returns:
        InferenceNERDataset instance
    """
    return InferenceNERDataset(data, tokenizer, max_len, pos_weight_dict, save_dir)


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for DataLoader.

    Handles variable-length sequences and stacks tensors.
    """
    max_len = max(len(item['tokens']) for item in batch)
    for item in batch:
        item['tokens'] = item['tokens'] + ['[PAD]'] * (max_len - len(item['tokens']))

    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'pos_ids': torch.stack([item['pos_ids'] for item in batch]),
        'tokens': [item['tokens'] for item in batch],
        'patent_id': [item['patent_id'] for item in batch],
        'ipc_code': [item['ipc_code'] for item in batch]
    }