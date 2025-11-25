"""Entity-level train/test split to prevent data leakage.

This module provides functions to split datasets by entity (source_id) rather than by sample,
ensuring that all samples from the same entity are either in training or testing set, but not both.
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import random


def split_by_entity(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    entity_key: str = "source_id",
    unknown_id: str = "unknown_entity"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split dataset by entity to prevent data leakage.

    CRITICAL FIX: Handles 'unknown_entity' separately to avoid data leakage.
    Unknown samples are stratified by FOT presence and distributed across splits.

    Args:
        data: List of samples, each containing entity_key field
        train_ratio: Ratio of entities for training (default: 0.8)
        val_ratio: Ratio of entities for validation (default: 0.1)
        test_ratio: Ratio of entities for testing (default: 0.1)
        random_seed: Random seed for reproducibility
        entity_key: Key name for entity identifier (default: "source_id")
        unknown_id: Special entity ID for unknown/negative samples (default: "unknown_entity")

    Returns:
        Tuple of (train_samples, val_samples, test_samples)

    Raises:
        ValueError: If ratios don't sum to 1.0 or data lacks entity_key
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Group samples by entity, but separate unknown_entity samples
    entity_to_samples = defaultdict(list)
    unknown_samples = []  # NEW: separate unknown samples
    samples_without_entity = []

    for sample in data:
        if entity_key in sample and sample[entity_key]:
            # CRITICAL: Separate unknown_entity from known entities
            if sample[entity_key] == unknown_id:
                unknown_samples.append(sample)
            else:
                entity_to_samples[sample[entity_key]].append(sample)
        else:
            samples_without_entity.append(sample)

    if samples_without_entity:
        print(f"Warning: {len(samples_without_entity)} samples lack '{entity_key}' field")

    if not entity_to_samples and not unknown_samples:
        raise ValueError(f"No samples found with '{entity_key}' field")

    # Get list of unique known entities (excluding unknown_entity)
    entities = list(entity_to_samples.keys())

    # Shuffle entities deterministically
    random.seed(random_seed)
    random.shuffle(entities)

    # Calculate split points for known entities
    n_entities = len(entities)
    train_end = int(n_entities * train_ratio)
    val_end = train_end + int(n_entities * val_ratio)

    # Split known entities
    train_entities = entities[:train_end] if n_entities > 0 else []
    val_entities = entities[train_end:val_end] if n_entities > 0 else []
    test_entities = entities[val_end:] if n_entities > 0 else []

    # Collect samples for each split from known entities
    train_samples = []
    val_samples = []
    test_samples = []

    for entity in train_entities:
        train_samples.extend(entity_to_samples[entity])

    for entity in val_entities:
        val_samples.extend(entity_to_samples[entity])

    for entity in test_entities:
        test_samples.extend(entity_to_samples[entity])

    # CRITICAL FIX: Stratified splitting of unknown_entity samples
    # This prevents all negative samples from going to one split
    if unknown_samples:
        print(f"\nProcessing {len(unknown_samples)} unknown_entity samples...")

        # Helper function to check if sample contains FOT tags
        def has_fot(sample):
            tags = sample.get("tags", sample.get("labels", []))
            return any(t in ["B-FOT", "I-FOT"] for t in tags)

        # Separate positive (has FOT) and negative (no FOT) unknown samples
        pos_unknown = [s for s in unknown_samples if has_fot(s)]
        neg_unknown = [s for s in unknown_samples if not has_fot(s)]

        print(f"  - Positive unknown samples (has FOT): {len(pos_unknown)}")
        print(f"  - Negative unknown samples (no FOT): {len(neg_unknown)}")

        # Shuffle and split positive unknown samples by ratio
        random.shuffle(pos_unknown)
        n_pos = len(pos_unknown)
        pos_train_end = int(n_pos * train_ratio)
        pos_val_end = pos_train_end + int(n_pos * val_ratio)

        train_samples.extend(pos_unknown[:pos_train_end])
        val_samples.extend(pos_unknown[pos_train_end:pos_val_end])
        test_samples.extend(pos_unknown[pos_val_end:])

        # Shuffle and split negative unknown samples by ratio
        random.shuffle(neg_unknown)
        n_neg = len(neg_unknown)
        neg_train_end = int(n_neg * train_ratio)
        neg_val_end = neg_train_end + int(n_neg * val_ratio)

        train_samples.extend(neg_unknown[:neg_train_end])
        val_samples.extend(neg_unknown[neg_train_end:neg_val_end])
        test_samples.extend(neg_unknown[neg_val_end:])

        print(f"  - Distributed to train/val/test: {pos_train_end + neg_train_end}/{pos_val_end - pos_train_end + neg_val_end - neg_train_end}/{len(pos_unknown) - pos_val_end + len(neg_unknown) - neg_val_end}")

    # Add samples without entity to training set (fallback)
    if samples_without_entity:
        train_samples.extend(samples_without_entity)

    # Shuffle samples within each split
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    # Print statistics
    print(f"\n{'='*80}")
    print(f"Entity-Level Split Statistics")
    print(f"{'='*80}")
    print(f"Total entities: {n_entities}")
    print(f"Train entities: {len(train_entities)} ({len(train_entities)/n_entities*100:.1f}%)")
    print(f"Val entities:   {len(val_entities)} ({len(val_entities)/n_entities*100:.1f}%)")
    print(f"Test entities:  {len(test_entities)} ({len(test_entities)/n_entities*100:.1f}%)")
    print()
    print(f"Total samples: {len(data)}")
    print(f"Train samples: {len(train_samples)} ({len(train_samples)/len(data)*100:.1f}%)")
    print(f"Val samples:   {len(val_samples)} ({len(val_samples)/len(data)*100:.1f}%)")
    print(f"Test samples:  {len(test_samples)} ({len(test_samples)/len(data)*100:.1f}%)")

    # Verify no entity overlap
    train_entity_set = set(train_entities)
    val_entity_set = set(val_entities)
    test_entity_set = set(test_entities)

    overlap_train_val = train_entity_set & val_entity_set
    overlap_train_test = train_entity_set & test_entity_set
    overlap_val_test = val_entity_set & test_entity_set

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print(f"\n⚠️  WARNING: Entity overlap detected!")
        if overlap_train_val:
            print(f"  - Train-Val overlap: {len(overlap_train_val)} entities")
        if overlap_train_test:
            print(f"  - Train-Test overlap: {len(overlap_train_test)} entities")
        if overlap_val_test:
            print(f"  - Val-Test overlap: {len(overlap_val_test)} entities")
    else:
        print(f"\n✅ No entity overlap - clean split!")
    print(f"{'='*80}\n")

    return train_samples, val_samples, test_samples


def verify_no_overlap(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    entity_key: str = "source_id"
) -> bool:
    """Verify that train/val/test sets have no overlapping entities.

    Args:
        train_data: Training samples
        val_data: Validation samples
        test_data: Test samples
        entity_key: Key name for entity identifier

    Returns:
        True if no overlap, False otherwise
    """
    # Extract entities from each set
    train_entities = {s[entity_key] for s in train_data if entity_key in s and s[entity_key]}
    val_entities = {s[entity_key] for s in val_data if entity_key in s and s[entity_key]}
    test_entities = {s[entity_key] for s in test_data if entity_key in s and s[entity_key]}

    # Check for overlaps
    overlap_train_val = train_entities & val_entities
    overlap_train_test = train_entities & test_entities
    overlap_val_test = val_entities & test_entities

    has_overlap = bool(overlap_train_val or overlap_train_test or overlap_val_test)

    print(f"\n{'='*80}")
    print(f"Entity Overlap Verification")
    print(f"{'='*80}")
    print(f"Train entities: {len(train_entities)}")
    print(f"Val entities:   {len(val_entities)}")
    print(f"Test entities:  {len(test_entities)}")
    print()

    if has_overlap:
        print(f"❌ OVERLAP DETECTED:")
        if overlap_train_val:
            print(f"  - Train-Val: {len(overlap_train_val)} entities")
            print(f"    Examples: {list(overlap_train_val)[:3]}")
        if overlap_train_test:
            print(f"  - Train-Test: {len(overlap_train_test)} entities")
            print(f"    Examples: {list(overlap_train_test)[:3]}")
        if overlap_val_test:
            print(f"  - Val-Test: {len(overlap_val_test)} entities")
            print(f"    Examples: {list(overlap_val_test)[:3]}")
    else:
        print(f"✅ NO OVERLAP - Clean entity-level split!")

    print(f"{'='*80}\n")

    return not has_overlap
