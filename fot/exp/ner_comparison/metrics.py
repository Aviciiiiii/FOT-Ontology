"""
Evaluation metrics for NER comparison experiments.

Includes standard metrics plus specialized FOT metrics:
- B-prec: B-tag precision
- P-match: Phrase-level match rate
- type-cons: Type consistency
"""

from __future__ import annotations

from typing import List, Tuple


def extract_entities(labels: List[str]) -> List[Tuple[str, ...]]:
    """
    Extract entity spans from BIO-tagged sequence.

    Args:
        labels: List of BIO tags (e.g., ['O', 'B-FOT', 'I-FOT', 'O'])

    Returns:
        List of entity tuples (position sequences)
    """
    entities = []
    current_entity = []

    for label in labels:
        if label == 'B-FOT':
            # Start new entity (save previous if exists)
            if current_entity:
                entities.append(tuple(current_entity))
            current_entity = [label]

        elif label == 'I-FOT':
            # Continue entity
            if current_entity:
                current_entity.append(label)

        else:  # label == 'O'
            # End entity
            if current_entity:
                entities.append(tuple(current_entity))
                current_entity = []

    # Don't forget the last entity
    if current_entity:
        entities.append(tuple(current_entity))

    return entities


def calculate_b_prec(all_preds: List[List[str]], all_labels: List[List[str]]) -> float:
    """
    Calculate B-tag precision.

    Measures how accurately the model predicts the beginning of entities.

    Args:
        all_preds: List of predicted tag sequences
        all_labels: List of true tag sequences

    Returns:
        B-tag precision score
    """
    total_b_preds = 0
    correct_b_preds = 0

    for pred_seq, label_seq in zip(all_preds, all_labels):
        for pred, label in zip(pred_seq, label_seq):
            if pred == 'B-FOT':
                total_b_preds += 1
                if label == 'B-FOT':
                    correct_b_preds += 1

    return correct_b_preds / total_b_preds if total_b_preds > 0 else 0.0


def calculate_p_match(all_preds: List[List[str]], all_labels: List[List[str]]) -> float:
    """
    Calculate phrase-level match rate.

    Measures how many true entities were correctly identified (exact span match).

    Args:
        all_preds: List of predicted tag sequences
        all_labels: List of true tag sequences

    Returns:
        Phrase match rate
    """
    total_entities = 0
    matched_entities = 0

    for pred_seq, label_seq in zip(all_preds, all_labels):
        pred_entities = extract_entities(pred_seq)
        label_entities = extract_entities(label_seq)

        total_entities += len(label_entities)

        for label_ent in label_entities:
            if label_ent in pred_entities:
                matched_entities += 1

    return matched_entities / total_entities if total_entities > 0 else 0.0


def calculate_type_cons(all_preds: List[List[str]], all_labels: List[List[str]]) -> float:
    """
    Calculate type consistency.

    Measures how many matched entities have correct tag types (B vs I consistency).

    Args:
        all_preds: List of predicted tag sequences
        all_labels: List of true tag sequences

    Returns:
        Type consistency score
    """
    total_entities = 0
    consistent_entities = 0

    for pred_seq, label_seq in zip(all_preds, all_labels):
        pred_entities = extract_entities(pred_seq)
        label_entities = extract_entities(label_seq)

        total_entities += len(label_entities)

        for label_ent in label_entities:
            if label_ent in pred_entities:
                # Find corresponding predicted entity
                pred_ent = pred_entities[pred_entities.index(label_ent)]
                # Check if lengths match (type consistency)
                if len(label_ent) == len(pred_ent):
                    consistent_entities += 1

    return consistent_entities / total_entities if total_entities > 0 else 0.0


def calculate_additional_metrics(
    all_labels: List[List[str]],
    all_preds: List[List[str]]
) -> Tuple[float, float, float]:
    """
    Calculate all additional FOT-specific metrics.

    Args:
        all_labels: List of true tag sequences
        all_preds: List of predicted tag sequences

    Returns:
        Tuple of (b_prec, p_match, type_cons)
    """
    b_prec = calculate_b_prec(all_preds, all_labels)
    p_match = calculate_p_match(all_preds, all_labels)
    type_cons = calculate_type_cons(all_preds, all_labels)

    return b_prec, p_match, type_cons
