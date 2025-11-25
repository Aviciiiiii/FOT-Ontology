"""
Enhanced BERT-BiLSTM-CRF model with complete functionality from original scripts.

This module implements the full NER architecture including:
- SciBERT/DistilBERT backbone support
- BiLSTM sequence modeling
- CustomCRF with constraint rules
- Multiple loss functions (CRF + Focal + Custom + L2)
- POS tag integration
- Dynamic FOT weight adjustment
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# CustomCRF is now fully self-implemented without external dependencies


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in NER with ignore_index support."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        CRITICAL FIX: Ignore positions marked with ignore_index (-100).
        This ensures FocalLoss only trains on valid first-subword positions,
        avoiding massive negative samples from [CLS]/[SEP]/subword continuations.
        """
        # Use ignore_index in cross_entropy to exclude invalid positions
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Only average over valid positions (not ignored)
        valid_mask = (targets != self.ignore_index).float()
        valid_count = valid_mask.sum().clamp(min=1.0)
        return (F_loss * valid_mask).sum() / valid_count


class CustomCRF(nn.Module):
    """Custom CRF implementation copied exactly from original script /src/train_NER.py lines 312-510.

    This is the simple, working version that achieved 0.6+ F1 score.
    """

    def __init__(self, num_tags, pos_weight_dict, idx2pos, fot_weight=1.8, batch_first=True):
        super(CustomCRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.pos_weight_dict = pos_weight_dict
        self.idx2pos = idx2pos
        self.fot_weight = fot_weight

        # Define tag to index mapping
        self.tag2idx = {"O": 0, "B-FOT": 1, "I-FOT": 2}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        # Initialize transition matrices and start/end transitions
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags)*0.1)
        self.start_transitions = nn.Parameter(torch.randn(num_tags)*0.1)
        self.end_transitions = nn.Parameter(torch.randn(num_tags)*0.1)

        # Apply prior knowledge to transition matrix
        with torch.no_grad():
            # Increase B-FOT -> I-FOT transition probability
            self.transitions[self.tag2idx['B-FOT'], self.tag2idx['I-FOT']] = 2.0
            # Decrease B-FOT -> O transition probability
            self.transitions[self.tag2idx['B-FOT'], self.tag2idx['O']] = -1.0
            # Decrease I-FOT -> B-FOT transition probability
            self.transitions[self.tag2idx['I-FOT'], self.tag2idx['B-FOT']] = -1.0

    def forward(self, emissions, tags=None, mask=None, pos_ids=None, tokens=None):
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if tags is not None:
                tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
            pos_ids = pos_ids.transpose(0, 1)

        # Apply POS weights and FOT weights
        emissions = self._apply_weights(emissions, pos_ids)

        if tags is not None:
            # Compute CRF loss
            crf_loss = self._compute_crf_loss(emissions, tags, mask)

            # Compute constraint loss
            constraint_loss = self._apply_constraints(tags, pos_ids, tokens, mask)

            # Total loss
            total_loss = crf_loss + 0.1 * constraint_loss

            return total_loss
        else:
            # Inference mode
            return self._viterbi_decode(emissions, mask)

    def _apply_weights(self, emissions, pos_ids):
        seq_length, batch_size, _ = emissions.size()
        pos_weights = torch.tensor([[self.pos_weight_dict.get(self.idx2pos.get(pos.item(), 'UNK'), 1.0) for pos in seq] for seq in pos_ids]).to(emissions.device)
        emissions = emissions * pos_weights.unsqueeze(-1)
        emissions[:, :, 1:] *= self.fot_weight  # Increase B-FOT and I-FOT weights
        return emissions

    def _compute_crf_loss(self, emissions, tags, mask):
        score = self._compute_score(emissions, tags, mask)
        partition = self._compute_log_partition(emissions, mask)
        return (partition - score).mean()

    def _apply_constraints(self, tags, pos_ids, tokens, mask):
        constraint_loss = 0.0
        seq_length, batch_size = tags.size()

        for i in range(batch_size):
            for j in range(1, seq_length):
                if mask[j, i]:
                    prev_tag = tags[j-1, i].item()
                    current_tag = tags[j, i].item()
                    current_pos = self.idx2pos.get(pos_ids[j, i].item(), 'UNK')
                    current_token = tokens[i][j].lower() if j < len(tokens[i]) else '[PAD]'

                    # Encourage B-FOT followed by I-FOT
                    if prev_tag == self.tag2idx['B-FOT'] and current_tag != self.tag2idx['I-FOT']:
                        constraint_loss += 0.5

                    # Punish I-FOT not preceded by B-FOT or I-FOT
                    if current_tag == self.tag2idx['I-FOT'] and prev_tag not in [self.tag2idx['B-FOT'], self.tag2idx['I-FOT']]:
                        constraint_loss += 1.0

                    # Rule 1: Prepositions, conjunctions usually shouldn't be B-FOT
                    if current_pos in ['ADP', 'CCONJ'] and current_tag == self.tag2idx['B-FOT']:
                        constraint_loss += 0.8

                    # Rule 2: Articles usually shouldn't be FOT (except 'the')
                    if current_pos == 'DET' and current_tag in [self.tag2idx['B-FOT'], self.tag2idx['I-FOT']]:
                        if current_token not in ['the']:  # Allow 'the' as part of FOT
                            constraint_loss += 0.8

                    # Rule 3: Auxiliary verbs shouldn't be FOT
                    if current_pos == 'AUX' and current_tag in [self.tag2idx['B-FOT'], self.tag2idx['I-FOT']]:
                        constraint_loss += 1.0

                    # Rule 4: Certain adverbs (degree adverbs) shouldn't be B-FOT
                    if current_pos == 'ADV' and current_token in ['very', 'quite', 'rather'] and current_tag == self.tag2idx['B-FOT']:
                        constraint_loss += 0.8

                    # Rule 5: Pronouns shouldn't be FOT
                    if current_pos == 'PRON' and current_tag in [self.tag2idx['B-FOT'], self.tag2idx['I-FOT']]:
                        constraint_loss += 0.9

                    # Rule 6: Numbers can be part of FOT, but shouldn't be standalone
                    if current_pos == 'NUM' and current_tag == self.tag2idx['B-FOT']:
                        if j == seq_length - 1 or tags[j+1, i] == self.tag2idx['O']:
                            constraint_loss += 0.7

                    # Rule 7: For biological names, allow lowercase words as part of FOT
                    if current_tag in [self.tag2idx['B-FOT'], self.tag2idx['I-FOT']] and current_token[0].islower():
                        if current_pos not in ['NOUN', 'PROPN']:
                            constraint_loss += 0.5

        return constraint_loss / batch_size

    def decode(self, emissions, mask=None, pos_ids=None):
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
            if pos_ids is not None:
                pos_ids = pos_ids.transpose(0, 1)

        # 应用 POS 权重和 FOT 权重
        emissions = self._apply_weights(emissions, pos_ids)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions, tags, mask):
        seq_length, batch_size = tags.shape
        mask = mask.float()

        # 初始化得分
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # 转移得分
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            # 发射得分
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # 结束转移
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_log_partition(self, emissions, mask):
        seq_length, batch_size, num_tags = emissions.shape
        mask = mask.bool()

        alphas = self.start_transitions.unsqueeze(0) + emissions[0]

        for i in range(1, seq_length):
            broadcast_alphas = alphas.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_alphas = broadcast_alphas + self.transitions + broadcast_emissions
            next_alphas = torch.logsumexp(next_alphas, dim=1)
            alphas = torch.where(mask[i].unsqueeze(-1), next_alphas, alphas)

        return torch.logsumexp(alphas + self.end_transitions, dim=-1)

    def _viterbi_decode(self, emissions, mask):
        seq_length, batch_size, num_tags = emissions.shape
        mask = mask.bool()

        # Initialize
        viterbi = self.start_transitions.unsqueeze(0) + emissions[0]
        backpointers = torch.zeros(seq_length, batch_size, num_tags, dtype=torch.long)

        for i in range(1, seq_length):
            broadcast_viterbi = viterbi.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_viterbi = broadcast_viterbi + self.transitions + broadcast_emissions
            best_tags = next_viterbi.max(dim=1)[1]
            viterbi = torch.where(mask[i].unsqueeze(-1), next_viterbi.max(dim=1)[0], viterbi)
            backpointers[i] = best_tags

        # End transitions
        viterbi += self.end_transitions
        best_last_tag = viterbi.max(dim=1)[1]

        # Backtrack to find best path
        best_tags = torch.zeros(seq_length, batch_size, dtype=torch.long)
        best_tags[-1] = best_last_tag
        for i in range(seq_length - 2, -1, -1):
            best_tags[i] = backpointers[i + 1].gather(1, best_tags[i + 1].unsqueeze(1)).squeeze(1)

        # CRITICAL FIX: Use mask-based compression instead of length-based slicing
        # This prevents sequence misalignment when special tokens and subword continuations
        # are marked with -100 (they're excluded from mask but still exist in tensor)

        # Move to CPU for indexing (best_tags is on CPU after backtracking)
        mask_cpu = mask.cpu() if mask.is_cuda else mask
        best_tags_cpu = best_tags.cpu() if best_tags.is_cuda else best_tags

        result = []
        for b in range(batch_size):
            # Extract only the positions where mask is True
            mask_b = mask_cpu[:, b].bool()  # [seq_length]
            best_tags_b = best_tags_cpu[:, b]  # [seq_length]
            tags_on_valid = best_tags_b[mask_b]  # [num_valid] - only True positions
            result.append(tags_on_valid.tolist())

        return result


class BertBiLSTMCRF(nn.Module):
    """Complete BERT-BiLSTM-CRF model with all original functionality."""

    def __init__(
        self,
        backbone: nn.Module,
        num_tags: int,
        pos_weight_dict: Dict[str, float],
        idx2pos: Dict[int, str],
        *,
        lstm_hidden_dim: int = 256,
        num_lstm_layers: int = 1,
        dropout: float = 0.3,
        fot_weight: float = 1.8,
        num_hidden_layers: int = 4,
        tag2idx: Optional[Dict[str, int]] = None,
        use_focal: bool = True,
        last_k_layers: int = 4,
        l2_lambda: float = 1e-5,
        # Loss weights
        crf_weight: float = 0.5,
        focal_weight: float = 0.1,
        custom_weight: float = 0.5,
        # Rule loss weights
        rule_w_oi: float = 0.9,
        rule_w_bi_break: float = 0.5,
        rule_w_i_bad_prev: float = 0.9,
        rule_w_bad_pos: float = 0.8,
        rule_w_no_seq_fot: float = 2.0,
        rule_coef: float = 0.5,
        focal_coef: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = getattr(backbone.config, "hidden_size", 768)
        self.last_k_layers = max(1, int(last_k_layers))
        self.num_hidden_layers = num_hidden_layers

        # Loss configuration
        self.l2_lambda = float(l2_lambda)
        self.crf_weight = float(crf_weight)
        self.focal_weight = float(focal_weight)
        self.custom_weight = float(custom_weight)

        # Rule weights (from original script)
        self.rule_w_oi = float(rule_w_oi)
        self.rule_w_bi_break = float(rule_w_bi_break)
        self.rule_w_i_bad_prev = float(rule_w_i_bad_prev)
        self.rule_w_bad_pos = float(rule_w_bad_pos)
        self.rule_w_no_seq_fot = float(rule_w_no_seq_fot)
        self.rule_coef = float(rule_coef)
        self.focal_coef = float(focal_coef)

        # Architecture components
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_size * self.last_k_layers,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(lstm_hidden_dim * 2, num_tags)

        # Store mappings
        self.tag2idx = tag2idx or {"O": 0, "B-FOT": 1, "I-FOT": 2}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        self.pos_weight_dict = pos_weight_dict
        self.idx2pos = idx2pos
        self.fot_weight = fot_weight

        # Loss functions
        self.crf = CustomCRF(num_tags, pos_weight_dict, idx2pos, fot_weight, batch_first=True)
        if use_focal:
            self.focal_loss = FocalLoss()
        else:
            self.focal_loss = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None, pos_ids: Optional[torch.Tensor] = None,
                tokens: Optional[List[List[str]]] = None, valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:

        # Get BERT outputs with multiple hidden states
        outputs = self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Concatenate last k hidden layers (following original script)
        sequence_output = torch.cat([
            outputs.hidden_states[-i] for i in range(1, self.last_k_layers + 1)
        ], dim=-1)

        sequence_output = self.dropout(sequence_output)

        # BiLSTM processing
        lstm_output, _ = self.lstm(sequence_output)

        # Linear projection to tag space
        emissions = self.linear(lstm_output)

        # CRITICAL FIX: Only use valid first-subword positions for training/inference
        # Mask out subword continuations and special tokens (marked as -100 in labels)
        if labels is not None:
            # Training: mask = attention_mask AND (labels != -100)
            # This ensures we only train on first subwords of each word
            mask = (attention_mask.bool() & (labels != -100))
        else:
            # Inference: use valid_mask if provided, otherwise fall back to attention_mask
            if valid_mask is not None:
                mask = (attention_mask.bool() & valid_mask)
            else:
                mask = attention_mask.bool()

        if labels is not None:
            # Training mode - compute all losses

            # CRITICAL FIX: Replace -100 with 0 before CRF computation
            # The mask already excludes these positions, so using 0 (O tag) is safe
            # This prevents CUDA device-side assert when -100 is used as tensor index
            labels_clamped = labels.clone()
            labels_clamped[labels == -100] = 0

            # 1. CRF Loss (only on valid positions, needs clamped labels)
            crf_loss = self.crf(emissions, labels_clamped, mask=mask, pos_ids=pos_ids, tokens=tokens)

            # 2. Focal Loss (optional)
            # CRITICAL FIX: Pass original labels with -100, FocalLoss will ignore them
            # This prevents training on massive [CLS]/[SEP]/subword continuation positions
            focal_loss = torch.tensor(0.0, device=emissions.device)
            if self.focal_loss is not None:
                focal_loss = self.focal_loss(
                    emissions.view(-1, emissions.size(-1)),
                    labels.view(-1)  # Original labels with -100, not clamped
                )

            # 3. Custom constraint loss (computed inside CRF, needs clamped labels)
            custom_loss = self._compute_custom_loss(emissions, labels_clamped, pos_ids, mask, tokens)

            # 4. L2 regularization
            l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())

            # Combined loss (following original script weighting)
            total_loss = (self.crf_weight * crf_loss +
                         self.focal_weight * focal_loss +
                         self.custom_weight * custom_loss +
                         self.l2_lambda * l2_reg)

            return total_loss, emissions, crf_loss, focal_loss, custom_loss, l2_reg
        else:
            # Inference mode
            predictions = self.crf.decode(emissions, mask=mask, pos_ids=pos_ids)
            l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
            return predictions, emissions, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), l2_reg

    def _compute_custom_loss(self, emissions: torch.Tensor, labels: torch.Tensor,
                           pos_ids: Optional[torch.Tensor], mask: torch.Tensor,
                           tokens: Optional[List[List[str]]]) -> torch.Tensor:
        """Compute custom constraint loss following original script logic."""
        if pos_ids is None or tokens is None:
            return torch.tensor(0.0, device=emissions.device)

        custom_loss = 0.0
        batch_size, seq_length, _ = emissions.size()

        for i in range(batch_size):
            for j in range(1, seq_length):
                if mask[i, j]:
                    prev_tag = labels[i, j-1].item()
                    current_tag = labels[i, j].item()
                    current_pos = self.idx2pos.get(pos_ids[i, j].item(), 'UNK')

                    # Rule 1: Encourage B-FOT followed by I-FOT
                    if prev_tag == self.tag2idx['B-FOT'] and current_tag != self.tag2idx['I-FOT']:
                        custom_loss += self.rule_w_bi_break

                    # Rule 2: Punish I-FOT not preceded by B-FOT or I-FOT
                    if current_tag == self.tag2idx['I-FOT'] and prev_tag not in [self.tag2idx['B-FOT'], self.tag2idx['I-FOT']]:
                        custom_loss += self.rule_w_i_bad_prev

                    # Rule 3: POS-based constraints
                    if current_pos in ['ADP', 'DET', 'PRON'] and current_tag in [self.tag2idx['B-FOT'], self.tag2idx['I-FOT']]:
                        custom_loss += self.rule_w_bad_pos

                    # Rule 4: Punish isolated I-FOT
                    if current_tag == self.tag2idx['I-FOT'] and j > 0 and labels[i, j-1] != self.tag2idx['B-FOT']:
                        custom_loss += 1.0

                    # Rule 5: Encourage FOT continuity
                    if current_tag in [self.tag2idx['B-FOT'], self.tag2idx['I-FOT']] and j < seq_length - 1:
                        if labels[i, j+1] not in [self.tag2idx['B-FOT'], self.tag2idx['I-FOT']]:
                            custom_loss += 0.8

            # Rule 6: Punish sequences with no FOT tags
            if torch.sum((labels[i] == self.tag2idx['B-FOT']) | (labels[i] == self.tag2idx['I-FOT'])) == 0:
                custom_loss += self.rule_w_no_seq_fot

        return torch.tensor(custom_loss / batch_size, device=emissions.device)


def adjust_fot_weight(f1_score: float, current_weight: float,
                     min_weight: float = 0.5, max_weight: float = 2.0) -> float:
    """Dynamically adjust FOT weight based on F1 score (from original script)."""
    if f1_score < 0.5:
        return min(current_weight * 1.1, max_weight)
    elif f1_score > 0.8:
        return max(current_weight * 0.9, min_weight)
    return current_weight


def build_model(
    num_tags: int,
    pos_weight_dict: Dict[str, float],
    idx2pos: Dict[int, str],
    *,
    pretrained_name_or_path: str = "distilbert-base-uncased",
    lstm_hidden_dim: int = 256,
    fot_weight: float = 1.8,
    num_hidden_layers: int = 4,
    use_focal: bool = True,
    last_k_layers: int = 4,
    l2_lambda: float = 1e-5,
    **kwargs: Any
) -> BertBiLSTMCRF:
    """Build the complete BERT-BiLSTM-CRF model."""
    try:
        from transformers import AutoModel  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required for model building.") from e

    # Support both SciBERT and DistilBERT
    if "scibert" in pretrained_name_or_path.lower():
        # Special handling for SciBERT paths
        backbone = AutoModel.from_pretrained(pretrained_name_or_path)
    else:
        backbone = AutoModel.from_pretrained(pretrained_name_or_path)

    # Extract tag2idx from kwargs to avoid duplication
    kwargs_copy = kwargs.copy()
    tag2idx = kwargs_copy.pop("tag2idx", {"O": 0, "B-FOT": 1, "I-FOT": 2})

    return BertBiLSTMCRF(
        backbone=backbone,
        num_tags=num_tags,
        pos_weight_dict=pos_weight_dict,
        idx2pos=idx2pos,
        lstm_hidden_dim=lstm_hidden_dim,
        fot_weight=fot_weight,
        num_hidden_layers=num_hidden_layers,
        tag2idx=tag2idx,
        use_focal=use_focal,
        last_k_layers=last_k_layers,
        l2_lambda=l2_lambda,
        **kwargs_copy
    )


def save_checkpoint(path: str, checkpoint_data: Dict[str, Any]) -> None:
    """Save model checkpoint."""
    torch.save(checkpoint_data, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load model checkpoint."""
    # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
    return torch.load(path, map_location="cpu", weights_only=False)