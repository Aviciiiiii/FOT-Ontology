"""
Baseline NER models for comparison experiments.

Includes:
- BiLSTM-CRF baseline (with STANDARD CRF - no POS weights, no constraints)
- BERT/SciBERT-CRF baselines (with STANDARD CRF - no POS weights, no constraints)
- SpaCy NER wrapper
- Stanford NER wrapper

CRITICAL: Baseline models use StandardCRF (vanilla Viterbi), NOT CustomCRF.
Only PatentNER uses CustomCRF with POS weights and constraint rules.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn

from ...utils.logging import get_logger

logger = get_logger("ner_models")


class StandardCRF(nn.Module):
    """
    Standard CRF implementation for baseline models.

    CRITICAL DIFFERENCE from CustomCRF:
    - NO POS-aware weights
    - NO constraint rules
    - NO FOT weight boosting
    - Pure vanilla Viterbi algorithm

    This ensures fair comparison with PatentNER's innovations.
    """

    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        # Learnable transition parameters (standard CRF)
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor,
                mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute CRF negative log-likelihood loss (standard implementation).

        Args:
            emissions: [batch, seq_len, num_tags] or [seq_len, batch, num_tags]
            tags: [batch, seq_len] or [seq_len, batch]
            mask: [batch, seq_len] or [seq_len, batch]
            **kwargs: Ignored (pos_ids, tokens) - not used in standard CRF

        Returns:
            Negative log-likelihood loss (scalar or [batch])
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)  # [seq_len, batch, num_tags]
            tags = tags.transpose(0, 1)  # [seq_len, batch]
            mask = mask.transpose(0, 1)  # [seq_len, batch]

        # Compute score of the gold path
        score = self._compute_score(emissions, tags, mask)

        # Compute log partition (normalization)
        log_partition = self._compute_log_partition(emissions, mask)

        # NLL loss
        return (log_partition - score).mean()

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        """Compute score of gold tag sequence (standard CRF)."""
        seq_length, batch_size = tags.shape
        mask = mask.bool()

        # Start transition
        score = self.start_transitions[tags[0]]

        # Emissions + transitions
        for i in range(seq_length - 1):
            score += self.transitions[tags[i], tags[i + 1]] * mask[i + 1]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # Last emission
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += emissions[seq_ends, torch.arange(batch_size), last_tags]

        # End transition
        score += self.end_transitions[last_tags]

        return score

    def _compute_log_partition(self, emissions: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
        """Compute log partition function (standard forward algorithm)."""
        seq_length, batch_size, num_tags = emissions.shape
        mask = mask.bool()

        # Initialize with start transitions
        alphas = self.start_transitions.unsqueeze(0) + emissions[0]

        # Forward pass
        for i in range(1, seq_length):
            broadcast_alphas = alphas.unsqueeze(2)  # [batch, num_tags, 1]
            broadcast_emissions = emissions[i].unsqueeze(1)  # [batch, 1, num_tags]
            next_alphas = broadcast_alphas + self.transitions + broadcast_emissions
            next_alphas = torch.logsumexp(next_alphas, dim=1)
            alphas = torch.where(mask[i].unsqueeze(-1), next_alphas, alphas)

        # Add end transitions
        return torch.logsumexp(alphas + self.end_transitions, dim=-1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor,
               **kwargs) -> list:
        """
        Viterbi decoding (standard algorithm).

        Args:
            emissions: [batch, seq_len, num_tags] or [seq_len, batch, num_tags]
            mask: [batch, seq_len] or [seq_len, batch]
            **kwargs: Ignored (pos_ids) - not used in standard CRF

        Returns:
            List of decoded tag sequences (list of lists)
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list:
        """Standard Viterbi algorithm for decoding."""
        seq_length, batch_size, num_tags = emissions.shape
        mask = mask.bool()

        # Initialize
        viterbi = self.start_transitions.unsqueeze(0) + emissions[0]
        backpointers = torch.zeros(seq_length, batch_size, num_tags, dtype=torch.long)

        # Forward pass
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

        # Backtrack
        best_tags = torch.zeros(seq_length, batch_size, dtype=torch.long)
        best_tags[-1] = best_last_tag
        for i in range(seq_length - 2, -1, -1):
            best_tags[i] = backpointers[i + 1].gather(1, best_tags[i + 1].unsqueeze(1)).squeeze(1)

        # Extract valid positions using mask (same as CustomCRF)
        mask_cpu = mask.cpu() if mask.is_cuda else mask
        best_tags_cpu = best_tags.cpu() if best_tags.is_cuda else best_tags

        result = []
        for b in range(batch_size):
            mask_b = mask_cpu[:, b].bool()
            best_tags_b = best_tags_cpu[:, b]
            tags_on_valid = best_tags_b[mask_b]
            result.append(tags_on_valid.tolist())

        return result


class BiLSTM_CRF(nn.Module):
    """
    Baseline BiLSTM-CRF model with STANDARD CRF.

    CRITICAL: Uses StandardCRF (vanilla Viterbi) for fair comparison.
    Does NOT use CustomCRF with POS weights or constraint rules.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels,
                 pos_weight_dict, idx2pos, fot_weight=1.8):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.hidden2label = nn.Linear(hidden_dim * 2, num_labels)

        # CRITICAL FIX: Use StandardCRF (no POS weights, no constraints)
        # Parameters pos_weight_dict, idx2pos, fot_weight are kept for API compatibility but NOT used
        self.crf = StandardCRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, pos_ids=None, tokens=None, valid_mask=None):
        """
        Forward pass with optional valid_mask for subword handling.

        Args:
            valid_mask: Optional boolean mask for valid first-subword positions.
                       If provided, will be used for CRF decoding.
        """
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        emission = self.hidden2label(lstm_out)

        # CRITICAL FIX: Use valid_mask for proper alignment with subword tokens
        if labels is not None:
            mask = (attention_mask.bool() & (labels != -100))
        else:
            if valid_mask is not None:
                mask = (attention_mask.bool() & valid_mask)
            else:
                mask = attention_mask.bool()

        if labels is not None:
            # Clamp labels to avoid -100 index errors in CRF
            labels_clamped = labels.clone()
            labels_clamped[labels == -100] = 0
            loss = self.crf(emission, labels_clamped, mask=mask, pos_ids=pos_ids, tokens=tokens)
            return loss, emission, 0, 0, 0, 0
        else:
            preds = self.crf.decode(emission, mask=mask, pos_ids=pos_ids)
            return preds, emission, 0, 0, 0, 0


class SciBERT_CRF(nn.Module):
    """
    BERT/SciBERT baseline with STANDARD CRF.

    CRITICAL: Uses StandardCRF (vanilla Viterbi) for fair comparison.
    Does NOT use CustomCRF with POS weights or constraint rules.
    """

    def __init__(self, num_labels, pos_weight_dict, idx2pos,
                 pretrained_model_name='files/scibert_scivocab_uncased', fot_weight=1.8):
        super(SciBERT_CRF, self).__init__()

        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.hidden2label = nn.Linear(self.hidden_size, num_labels)

        # CRITICAL FIX: Use StandardCRF (no POS weights, no constraints)
        # Parameters pos_weight_dict, idx2pos, fot_weight are kept for API compatibility but NOT used
        self.crf = StandardCRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, pos_ids=None, tokens=None, valid_mask=None):
        """
        Forward pass with optional valid_mask for subword handling.

        Args:
            valid_mask: Optional boolean mask for valid first-subword positions.
                       If provided, will be used for CRF decoding.
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        emission = self.hidden2label(sequence_output)

        # CRITICAL FIX: Use valid_mask for proper alignment with subword tokens
        if labels is not None:
            mask = (attention_mask.bool() & (labels != -100))
        else:
            if valid_mask is not None:
                mask = (attention_mask.bool() & valid_mask)
            else:
                mask = attention_mask.bool()

        if labels is not None:
            # Clamp labels to avoid -100 index errors in CRF
            labels_clamped = labels.clone()
            labels_clamped[labels == -100] = 0
            loss = self.crf(emission, labels_clamped, mask=mask, pos_ids=pos_ids, tokens=tokens)
            return loss, emission, 0, 0, 0, 0
        else:
            preds = self.crf.decode(emission, mask=mask, pos_ids=pos_ids)
            return preds, emission, 0, 0, 0, 0


class SpacyNERWrapper:
    """Wrapper for SpaCy NER system."""

    def __init__(self):
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_lg")
                logger.info("Loaded SpaCy large model (en_core_web_lg)")
            except:
                try:
                    self.nlp = spacy.load("en_core_web_md")
                    logger.info("Loaded SpaCy medium model (en_core_web_md)")
                except:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded SpaCy small model (en_core_web_sm)")
        except ImportError:
            logger.error("SpaCy not installed. Install with: pip install spacy")
            raise
        except Exception as e:
            logger.error(f"Failed to load SpaCy model: {e}")
            raise

        self.tag2idx = {"O": 0, "B-FOT": 1, "I-FOT": 2}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

    def to(self, device):
        """No-op for compatibility."""
        return self

    def eval(self):
        """No-op for compatibility."""
        return self

    def train(self, mode=True):
        """No-op for compatibility."""
        return self

    def __call__(self, input_ids, attention_mask, labels=None, pos_ids=None, tokens=None):
        return self.forward(input_ids, attention_mask, labels, pos_ids, tokens)

    def forward(self, input_ids, attention_mask, labels=None, pos_ids=None, tokens=None):
        """Run SpaCy NER on input tokens."""
        predictions = []
        batch_size = len(tokens)

        for i in range(batch_size):
            # Reconstruct text from tokens
            text = ' '.join([t for t in tokens[i] if t != '[PAD]'])
            doc = self.nlp(text)

            # Convert SpaCy entities to BIO tags
            tags = []
            for token in doc:
                if token.ent_type_:  # Token is part of an entity
                    if not tags or tags[-1] == 0:
                        tags.append(1)  # B-FOT
                    else:
                        tags.append(2)  # I-FOT
                else:
                    tags.append(0)  # O

            # Pad or truncate to match input length
            if len(tags) < len(tokens[i]):
                tags.extend([0] * (len(tokens[i]) - len(tags)))
            else:
                tags = tags[:len(tokens[i])]

            predictions.append(tags)

        return torch.tensor(predictions), None, 0, 0, 0, 0


class StanfordNERWrapper:
    """Wrapper for Stanford NER system."""

    def __init__(self):
        try:
            from nltk.tag import StanfordNERTagger

            # Java setup
            java_path = "/usr/lib/jvm/java-11-openjdk-amd64/bin/java/"
            os.environ['JAVA_HOME'] = java_path

            # Stanford NER paths
            jar_path = os.path.join(os.getcwd(), 'stanford-ner', 'stanford-ner.jar')
            model_path = os.path.join(
                os.getcwd(), 'stanford-ner', 'classifiers',
                'english.all.3class.distsim.crf.ser.gz'
            )

            if not os.path.exists(jar_path):
                raise FileNotFoundError(f"Stanford NER JAR not found: {jar_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Stanford NER model not found: {model_path}")

            self.tagger = StanfordNERTagger(
                model_path, jar_path,
                encoding='utf-8',
                java_options='-mx4g'
            )
            logger.info("Successfully loaded Stanford NER tagger")

        except ImportError:
            logger.error("NLTK not installed or StanfordNERTagger not available")
            self.tagger = None
        except Exception as e:
            logger.error(f"Error loading Stanford NER: {str(e)}")
            self.tagger = None

        self.tag2idx = {"O": 0, "B-FOT": 1, "I-FOT": 2}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

    def to(self, device):
        """No-op for compatibility."""
        return self

    def eval(self):
        """No-op for compatibility."""
        return self

    def train(self, mode=True):
        """No-op for compatibility."""
        return self

    def __call__(self, input_ids, attention_mask, labels=None, pos_ids=None, tokens=None):
        return self.forward(input_ids, attention_mask, labels, pos_ids, tokens)

    def forward(self, input_ids, attention_mask, labels=None, pos_ids=None, tokens=None):
        """Run Stanford NER on input tokens."""
        device = attention_mask.device if isinstance(attention_mask, torch.Tensor) else 'cpu'
        predictions = []

        for token_list in tokens:
            text = ' '.join([t for t in token_list if t != '[PAD]'])

            if self.tagger is not None:
                try:
                    tagged = self.tagger.tag(text.split())
                    tags = []
                    prev_tag = 'O'

                    for word, tag in tagged:
                        if tag != 'O':
                            tags.append(1 if prev_tag == 'O' else 2)  # B-FOT or I-FOT
                        else:
                            tags.append(0)  # O
                        prev_tag = tag

                except Exception as e:
                    logger.error(f"Stanford NER tagging error: {str(e)}")
                    tags = [0] * len(text.split())
            else:
                # Fallback: all O tags
                tags = [0] * len(text.split())

            # Pad or truncate
            if len(tags) < len(token_list):
                tags.extend([0] * (len(token_list) - len(tags)))
            else:
                tags = tags[:len(token_list)]

            predictions.append(tags)

        return torch.tensor(predictions).to(device), None, 0, 0, 0, 0
