"""
MAG data cleaning module with comprehensive text processing and entity tagging.

This module implements the complete text cleaning pipeline for MAG (Microsoft Academic Graph)
data, including Unicode normalization, contraction expansion, Trie-based entity matching,
and BIO tagging for Named Entity Recognition.
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from ...utils.logging import get_logger
from ...utils.trie import build_fot_trie


# Special cases that should always be preserved
SPECIAL_CASES = {
    "x ray", "x-ray", "t cell", "t-cell", "b cell", "b-cell", "3d printing",
    "5g network", "e-commerce", "a/b testing", "c++", "r", "k-means", "n-gram",
    "p value", "q factor", "s wave", "v chip", "h-index", "i beam", "o ring", "z score"
}

# Words that are allowed at the beginning of FOT entities
ALLOWED_WORDS = {'and', 'of', 'for', 'in', 'to'}

# Common prepositions and conjunctions that should be filtered
COMMON_PREPOSITIONS = {
    'with', 'on', 'at', 'from', 'by', 'about', 'as', 'into', 'like', 'through',
    'after', 'over', 'between', 'out', 'against', 'during', 'without', 'before',
    'under', 'around', 'among'
}

COMMON_CONJUNCTIONS = {
    'but', 'or', 'yet', 'so', 'nor', 'because', 'although', 'since', 'unless',
    'while', 'where', 'if'
}

DISALLOWED_WORDS = COMMON_PREPOSITIONS | COMMON_CONJUNCTIONS

# Special characters handling
SPECIAL_CHARS = ['[', ']', '(', ')', '{', '}', '<', '>', '...', '…']
REMOVE_CHARS = set(string.punctuation) - set(SPECIAL_CHARS) - {"'"}

# Contractions mapping
CONTRACTIONS = {
    "'s": "is",
    "n't": "not",
    "'m": "am",
    "'re": "are",
    "'ve": "have",
    "'ll": "will"
}


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to ASCII equivalents.

    Args:
        text: Input text with potential Unicode characters

    Returns:
        Text with normalized characters
    """
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')

    replacements = {
        '\u2026': '...',  # Ellipsis
        '\u2122': '(TM)',  # Trademark symbol
        '\u00ae': '(R)',   # Registered trademark
        '\u2019': "'",     # Right single quotation mark
        '\u2018': "'",     # Left single quotation mark
        '\u201c': '"',     # Left double quotation mark
        '\u201d': '"',     # Right double quotation mark
        '\u2013': '-',     # En dash
        '\u2014': '--',    # Em dash
    }

    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)

    return text


def clean_text(text: str) -> str:
    """Clean and normalize text for processing.

    Args:
        text: Raw input text

    Returns:
        Cleaned and normalized text
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Normalize Unicode characters
    text = normalize_unicode(text)

    # Remove specified punctuation
    for char in REMOVE_CHARS:
        text = text.replace(char, ' ')

    # Ensure special characters are surrounded by spaces
    for char in SPECIAL_CHARS:
        text = text.replace(char, f' {char} ')

    # Replace multiple consecutive dots with a single space
    text = re.sub(r'\.{2,}', ' ', text)

    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def expand_contractions(tokens: List[str]) -> List[str]:
    """Expand English contractions in token list.

    Args:
        tokens: List of tokens that may contain contractions

    Returns:
        List of tokens with contractions expanded
    """
    expanded_tokens = []
    i = 0

    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == "'" and tokens[i+1].lower() == "s":
            expanded_tokens.append("is")
            i += 2
        elif tokens[i] in CONTRACTIONS:
            expanded_tokens.append(CONTRACTIONS[tokens[i]])
            i += 1
        elif i < len(tokens) - 1 and tokens[i] + tokens[i+1] in CONTRACTIONS:
            expanded_tokens.append(CONTRACTIONS[tokens[i] + tokens[i+1]])
            i += 2
        else:
            expanded_tokens.append(tokens[i])
            i += 1

    return expanded_tokens


def should_keep_fot(fot_tokens: List[str]) -> bool:
    """Determine if a FOT entity should be kept based on filtering rules.

    Args:
        fot_tokens: List of tokens forming the FOT entity

    Returns:
        True if the FOT should be kept, False otherwise
    """
    fot = " ".join(fot_tokens).lower()

    # Check if it's a special case
    if fot in SPECIAL_CASES:
        return True

    # Check if FOT starts with disallowed words
    if fot_tokens[0].lower() in ALLOWED_WORDS:
        return False

    # Check if FOT contains disallowed words
    for token in fot_tokens:
        if token.lower() in DISALLOWED_WORDS:
            return False

    # Check FOT length
    if len(fot_tokens) < 2:
        return False

    return True


def remove_duplicates(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate entries from the dataset.

    Args:
        data: List of data items with 'tokens' and 'tags' fields

    Returns:
        List with duplicates removed
    """
    seen = set()
    unique_data = []

    for item in data:
        # Use frozenset to represent dictionary, ignoring key order
        item_set = frozenset((k, tuple(v) if isinstance(v, list) else v) for k, v in item.items())
        if item_set not in seen:
            seen.add(item_set)
            unique_data.append(item)

    return unique_data


def clean_dataset(data: List[Dict[str, Any]], fot_trie) -> List[Dict[str, Any]]:
    """Apply complete cleaning pipeline to a dataset.

    Args:
        data: List of data items with 'tokens' and optional 'tags' fields
        fot_trie: Trie structure containing FOT entities for matching

    Returns:
        Cleaned dataset with proper BIO tagging
    """
    cleaned_data = []
    positive_samples = 0
    negative_samples = 0

    for item in tqdm(data, desc="Cleaning dataset"):
        tokens = item.get('tokens', [])

        # Clean and normalize text
        cleaned_text = clean_text(" ".join(tokens))
        cleaned_tokens = cleaned_text.split()
        expanded_tokens = expand_contractions(cleaned_tokens)

        # Initialize output tokens and tags
        final_tokens = []
        final_tags = []

        i = 0
        has_fot = False

        while i < len(expanded_tokens):
            # Find the longest matching FOT entity
            max_match = None
            max_match_length = 0

            for j in range(i, len(expanded_tokens)):
                potential_fot = " ".join(expanded_tokens[i:j+1]).lower()
                fot_match = fot_trie.search_exact(potential_fot.split())
                if fot_match and len(potential_fot.split()) > max_match_length:
                    max_match = fot_match
                    max_match_length = len(potential_fot.split())

            if max_match:
                fot_tokens = expanded_tokens[i:i+max_match_length]
                if should_keep_fot(fot_tokens):
                    # Add B-FOT and I-FOT tags
                    for j in range(max_match_length):
                        final_tokens.append(expanded_tokens[i+j])
                        final_tags.append('B-FOT' if j == 0 else 'I-FOT')
                    has_fot = True
                else:
                    # Mark as O (outside)
                    for j in range(max_match_length):
                        final_tokens.append(expanded_tokens[i+j])
                        final_tags.append('O')
                i += max_match_length
            else:
                final_tokens.append(expanded_tokens[i])
                final_tags.append('O')
                i += 1

        if has_fot:
            positive_samples += 1
        else:
            negative_samples += 1

        # Preserve source_id (entity) field for entity-level train/test split
        result = {"tokens": final_tokens, "tags": final_tags}
        if "source_id" in item:
            result["source_id"] = item["source_id"]
        elif "entity" in item:
            result["source_id"] = item["entity"]
        cleaned_data.append(result)

    # Remove duplicates
    cleaned_data = remove_duplicates(cleaned_data)

    return cleaned_data


def _load_list(path: str) -> List[dict]:
    """Load data from JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_mag_entities(vocab_path: str) -> List[str]:
    """Load MAG entities from vocab file.

    Args:
        vocab_path: Path to MAG entities file

    Returns:
        List of entity names
    """
    try:
        data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        if isinstance(data, list) and data and isinstance(data[0], str):
            return data
        else:
            # Handle case where data is list of dicts with 'name' field
            return [item['name'] if isinstance(item, dict) else str(item) for item in data]
    except Exception:
        return []


def clean_mag(in1: str, in2: str, vocab_path: str, out1: str, out2: str) -> Tuple[str, str]:
    """Clean MAG datasets with complete text processing pipeline.

    Args:
        in1: Path to first input dataset
        in2: Path to second input dataset
        vocab_path: Path to MAG entities vocabulary
        out1: Path for first output dataset
        out2: Path for second output dataset

    Returns:
        Tuple of output file paths
    """
    logger = get_logger("mag_cleaner")

    # Load data
    s1 = _load_list(in1)
    s2 = _load_list(in2)

    # Load MAG entities and build Trie
    mag_entities = _load_mag_entities(vocab_path)
    fot_trie = build_fot_trie(mag_entities)

    logger.info(f"Loaded {len(mag_entities)} MAG entities for matching")

    # Clean datasets
    c1 = clean_dataset(s1, fot_trie)
    c2 = clean_dataset(s2, fot_trie)

    # Write outputs
    p1 = Path(out1)
    p2 = Path(out2)
    p1.parent.mkdir(parents=True, exist_ok=True)
    p2.parent.mkdir(parents=True, exist_ok=True)

    p1.write_text(json.dumps(c1, ensure_ascii=False, indent=2), encoding="utf-8")
    p2.write_text(json.dumps(c2, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("MAG cleaned: in1=%d in2=%d -> out1=%d out2=%d", len(s1), len(s2), len(c1), len(c2))

    return str(p1), str(p2)