"""
FOT Extraction from NER Predictions

Ported from /src/FOT_generator.py lines 298-423.
Implements rule-based FOT extraction with SpaCy POS filtering.
"""

import logging
import re
from typing import List, Tuple

import spacy

logger = logging.getLogger(__name__)

# 68 special symbol FOTs that should be preserved (original script lines 307-374)
COMMON_SYMBOL_FOTS = {
    'x-ray',
    'e-mail',
    'wi-fi',
    'e-commerce',
    'e-learning',
    'e-book',
    'i-beam',
    't-shirt',
    'u-bolt',
    'v-belt',
    'z-wave',
    'k-means',
    'n-gram',
    'p-value',
    't-test',
    'f-score',
    'c-section',
    'h-bridge',
    'q-factor',
    's-curve',
    'y-axis',
    'b-tree',
    'k-nearest neighbor',
    'l-system',
    'r-value',
    'g-force',
    'o-ring',
    'u-value',
    't-junction',
    'v-chip',
    'x-axis',
    'gamma-ray',
    'beta-carotene',
    'alpha-particle',
    'omega-3',
    'micro-controller',
    'nano-technology',
    'bio-fuel',
    'cryo-surgery',
    'photo-voltaic',
    'thermo-electric',
    'electro-magnetic',
    'hydro-electric',
    'piezo-electric',
    'magneto-resistive',
    'opto-electronic',
    'ferro-magnetic',
    'quasi-periodic',
    'semi-conductor',
    'ultra-sonic',
    'infra-red',
    'ultra-violet',
    'multi-layer',
    'co-processor',
    'sub-system',
    'inter-connect',
    'cross-platform',
    'self-assembly',
    'non-linear',
    'anti-aliasing',
    'pre-processing',
    'post-processing',
    'real-time',
    'feed-forward',
    'back-propagation',
    'machine-learning',
    'deep-learning',
    'blockchain'
}

# Invalid POS tags for FOT start/end (original script line 385)
INVALID_START_END_POS = {'ADP', 'DET', 'CCONJ', 'SCONJ', 'PRON', 'AUX'}

# Global SpaCy model (lazy loaded)
_nlp_sm = None


def get_spacy_model():
    """Lazy load SpaCy small model for POS tagging."""
    global _nlp_sm
    if _nlp_sm is None:
        try:
            _nlp_sm = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            logger.info("Loaded SpaCy en_core_web_sm for FOT extraction")
        except OSError:
            logger.error("SpaCy en_core_web_sm not found. Please install: python -m spacy download en_core_web_sm")
            raise
    return _nlp_sm


def extract_fot(words: List[str], pred: List[int], idx2tag: dict) -> List[Tuple[str, int, int]]:
    """Extract FOT (Field of Technology) phrases from BIO predictions.

    This function implements the complete rule-based extraction logic from the original script:
    1. Parse BIO tags to identify FOT spans
    2. Apply SpaCy POS tagging and lemmatization
    3. Filter by POS rules (remove invalid start/end POS tags)
    4. Preserve 74 special symbol FOTs (e.g., x-ray, e-mail)
    5. Remove single-word FOTs
    6. Clean punctuation (except for special FOTs)

    Args:
        words: List of tokens from patent title
        pred: List of predicted tag indices (0=O, 1=B-FOT, 2=I-FOT)
        idx2tag: Mapping from index to tag name

    Returns:
        List of (fot_string, start_idx, end_idx) tuples
    """
    fot_words = []
    current_fot = []
    start_index = -1

    # SpaCy processing for POS tags and lemmatization
    nlp_sm = get_spacy_model()
    doc = nlp_sm(" ".join(words))

    # Parse BIO tags to extract FOT spans
    for i, (word, tag_idx, token) in enumerate(zip(words, pred, doc)):
        # Convert tensor to int if needed
        tag = idx2tag[tag_idx if isinstance(tag_idx, int) else tag_idx.item()]
        lemma = token.lemma_.lower()
        pos = token.pos_

        logger.debug(f"Token: {word}, POS: {pos}, Lemma: {lemma}, Tag: {tag}")

        if tag == 'B-FOT':
            # Start new FOT
            if current_fot:
                fot_words.append((current_fot, start_index, i - 1))
            current_fot = [(lemma, pos, i)]
            start_index = i
        elif tag == 'I-FOT' and current_fot:
            # Continue current FOT
            current_fot.append((lemma, pos, i))
        else:
            # End current FOT
            if current_fot:
                fot_words.append((current_fot, start_index, i - 1))
                current_fot = []
                start_index = -1

    # Add final FOT if exists
    if current_fot:
        fot_words.append((current_fot, start_index, len(words) - 1))

    # Apply filtering rules
    processed_fot_words = []
    for fot, start, end in fot_words:
        # Rule 1: Remove FOTs with invalid start/end POS tags
        if fot[0][1] in INVALID_START_END_POS or fot[-1][1] in INVALID_START_END_POS:
            logger.debug(f"Filtered FOT (invalid POS): {[w for w, _, _ in fot]}")
            continue

        # Construct FOT string from lemmas
        fot_string = ' '.join([word for word, _, _ in fot])

        # Rule 2: Clean punctuation (except for special symbol FOTs)
        if fot_string.lower() not in COMMON_SYMBOL_FOTS:
            fot_string = re.sub(r'[^\w\s]', '', fot_string)

        # Normalize whitespace
        fot_string = ' '.join(fot_string.split())

        # Rule 3: Remove single-word FOTs
        if len(fot_string.split()) <= 1:
            logger.debug(f"Filtered FOT (single word): {fot_string}")
            continue

        # Use original token indices (not string matching)
        new_start = fot[0][2]  # First token's original index
        new_end = fot[-1][2]   # Last token's original index

        processed_fot_words.append((fot_string, new_start, new_end))

    logger.debug(f"Extracted {len(processed_fot_words)} FOTs: {processed_fot_words}")
    return processed_fot_words


def format_fot_sequence(fot_list: List[Tuple[str, int, int]]) -> str:
    """Format FOT list as colon-separated FOT IDs.

    Args:
        fot_list: List of (fot_id, fot_name, ipc_code) tuples

    Returns:
        Formatted string like "1:0:5:0:12" (FOT IDs separated by ':0:')
    """
    if not fot_list:
        return ""

    fot_ids = [str(fot_id) for fot_id, _, _ in fot_list]
    return ":0:".join(fot_ids)


def process_batch_predictions(
    batch: dict,
    preds: List[List[int]],
    idx2tag: dict,
    fot_id_mapping: dict,
    fot_id_counter: int
) -> Tuple[List[Tuple], int, dict]:
    """Process batch of NER predictions to extract FOTs.

    Args:
        batch: Batch dictionary with tokens, patent_id, ipc_code
        preds: Predicted tag indices for each sequence
        idx2tag: Tag index to name mapping
        fot_id_mapping: Existing FOT name -> ID mapping
        fot_id_counter: Current FOT ID counter

    Returns:
        Tuple of (results, updated_counter, updated_mapping)
        - results: List of (patent_id, title, fot_sequence, fot_details) tuples
        - updated_counter: New FOT ID counter value
        - updated_mapping: Updated FOT name -> ID mapping
    """
    results = []

    for i in range(len(batch['patent_id'])):
        patent_id = batch['patent_id'][i]
        tokens = batch['tokens'][i]
        ipc_code = batch['ipc_code'][i]

        # Filter out padding
        words = [token for token in tokens if token != '[PAD]']
        pred = preds[i][:len(words)]

        # Extract FOTs using rule-based logic
        fot_spans = extract_fot(words, pred, idx2tag)

        # Assign FOT IDs and track mapping
        fot_details = []
        for fot_name, start_idx, end_idx in fot_spans:
            if fot_name not in fot_id_mapping:
                fot_id_mapping[fot_name] = fot_id_counter
                fot_id_counter += 1
            fot_id = fot_id_mapping[fot_name]
            fot_details.append((fot_id, fot_name, ipc_code))

        # Format FOT sequence as "1:0:5:0:12"
        fot_sequence = format_fot_sequence(fot_details)

        # Reconstruct title
        title = ' '.join(words)

        results.append((patent_id, title, fot_sequence, fot_details))

    return results, fot_id_counter, fot_id_mapping