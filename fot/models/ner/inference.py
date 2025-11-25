from __future__ import annotations

from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer

from .bert_crf import build_model, load_checkpoint


def predict(tokens: List[str], checkpoint_path: str) -> List[str]:
    """Run inference with BertBiLSTMCRF and return BIO tags per input token.

    - tokens: pre-tokenized words
    - checkpoint_path: path to saved .pth with meta
    """
    ckpt = load_checkpoint(checkpoint_path)
    pretrained = ckpt.get("pretrained_name", "distilbert-base-uncased")
    num_hidden_layers = int(ckpt.get("num_hidden_layers", 4))
    lstm_hidden_dim = int(ckpt.get("lstm_hidden_dim", 256))
    fot_weight = float(ckpt.get("fot_weight", 1.0))
    last_k_layers = int(ckpt.get("last_k_layers", 4))
    l2_lambda = float(ckpt.get("l2_lambda", 1e-6))
    use_focal = bool(ckpt.get("use_focal", False))
    pos_weight_dict = ckpt.get("pos_weight_dict", {"PAD": 1.0})
    idx2pos = ckpt.get("idx2pos", {0: "PAD"})
    rl = ckpt.get("rule_loss", {}) or {}
    tag2idx: Dict[str, int] = ckpt.get("tag2idx", {"O": 0, "B-FOT": 1, "I-FOT": 2})
    idx2tag = {v: k for k, v in tag2idx.items()}

    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=128)
    word_ids = enc.word_ids(batch_index=0)
    # valid mask: only first subword of each word
    valid_mask = []
    prev = None
    for wi in word_ids:
        if wi is None:
            valid_mask.append(0)
        elif wi == prev:
            valid_mask.append(0)
        else:
            valid_mask.append(1)
        prev = wi
    valid_mask_t = torch.tensor([valid_mask], dtype=torch.uint8)

    model = build_model(
        len(tag2idx),
        pos_weight_dict=pos_weight_dict,
        idx2pos=idx2pos,
        lstm_hidden_dim=lstm_hidden_dim,
        fot_weight=fot_weight,
        num_hidden_layers=num_hidden_layers,
        pretrained_name_or_path=pretrained,
        last_k_layers=last_k_layers,
        l2_lambda=l2_lambda,
        use_focal=use_focal,
        rule_w_oi=float(rl.get("w_oi", 0.9)),
        rule_w_bi_break=float(rl.get("w_bi_break", 0.5)),
        rule_w_i_bad_prev=float(rl.get("w_i_bad_prev", 0.9)),
        rule_w_bad_pos=float(rl.get("w_bad_pos", 0.8)),
        rule_w_no_seq_fot=float(rl.get("w_no_seq_fot", 2.0)),
        rule_coef=float(rl.get("coef", 0.5)),
        focal_coef=float(rl.get("focal_coef", 0.1)),
    )
    model.load_state_dict(ckpt.get("state_dict", {}), strict=False)
    model.eval()
    with torch.no_grad():
        # forward to get emissions, decode CRF using attention_mask for stability
        _, emissions, *_ = model(enc["input_ids"], enc["attention_mask"], labels=None, pos_ids=None, valid_mask=valid_mask_t)
        preds = model.crf.decode(emissions, mask=enc["attention_mask"].bool())

    # Map decoded compressed indices back to word-level tags
    attn_mask_list = [int(x) for x in enc["attention_mask"].squeeze(0).tolist()]
    valid_pos = [i for i, m in enumerate(attn_mask_list) if m == 1]
    seq_decoded = preds[0] if preds else []
    # absolute index -> compressed index
    pos_to_compressed = {abs_idx: comp_idx for comp_idx, abs_idx in enumerate(valid_pos)}
    # word_id -> first absolute position
    wi_to_first_abs = {}
    for abs_idx, wi in enumerate(word_ids):
        if wi is not None and abs_idx in pos_to_compressed and wi not in wi_to_first_abs:
            wi_to_first_abs[wi] = abs_idx

    tags: List[str] = []
    max_wi = max([wi for wi in wi_to_first_abs.keys()] or [-1])
    for wi in range(max_wi + 1):
        abs_idx = wi_to_first_abs.get(wi)
        if abs_idx is None:
            tags.append("O")
            continue
        comp_idx = pos_to_compressed.get(abs_idx)
        if comp_idx is None or comp_idx >= len(seq_decoded):
            tags.append("O")
        else:
            tags.append(idx2tag.get(seq_decoded[comp_idx], "O"))
    return tags
