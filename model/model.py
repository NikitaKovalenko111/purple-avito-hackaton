from __future__ import annotations

import ast
import csv
import copy
from dataclasses import dataclass
import importlib
import itertools
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request


def _optional_import(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


torch = _optional_import("torch")
nn = _optional_import("torch.nn")
_sklearn_text = _optional_import("sklearn.feature_extraction.text")
TfidfVectorizer = getattr(_sklearn_text, "TfidfVectorizer", None)
_transformers = _optional_import("transformers")
AutoModel = getattr(_transformers, "AutoModel", None)
AutoTokenizer = getattr(_transformers, "AutoTokenizer", None)
_BaseModule = nn.Module if nn is not None else object


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required. Install with: pip install torch")


def _require_sklearn() -> None:
    if TfidfVectorizer is None:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")


def _require_transformers() -> None:
    if AutoModel is None or AutoTokenizer is None:
        raise ImportError("transformers is required. Install with: pip install transformers")


@dataclass
class MicroCategory:
    mc_id: int
    mc_title: str
    key_phrases: List[str]


@dataclass
class Item:
    item_id: int
    mc_id: int
    mc_title: str
    description: str


@dataclass
class LabeledItem(Item):
    target_detected_mc_ids: List[int]
    target_split_mc_ids: List[int]
    should_split: bool
    case_type: str
    split: str


@dataclass
class Draft:
    mc_id: int
    mc_title: str
    text: str


@dataclass
class PredictionResult:
    detected_mc_ids: List[int]
    should_split: bool
    drafts: List[Draft]
    probabilities: Dict[int, float]


@dataclass
class PipelineSettings:
    transformer_name: str = "DeepPavlov/rubert-base-cased"
    tfidf_ngram_range: Tuple[int, int] = (2, 3)
    tfidf_threshold: float = 0.05
    tfidf_top_k: int = 11
    prob_threshold: float = 0.08
    split_threshold: float = 0.5
    max_drafts: int = 0
    score_margin: float = 1.0
    relative_ratio: float = 0.0
    score_blend_alpha: float = 1.0
    device: Optional[str] = None
    max_length: int = 256
    use_llm_drafts: bool = False
    openrouter_model: str = "qwen/qwen3.6-plus:free"
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_timeout_sec: float = 30.0
    openrouter_site_url: Optional[str] = None
    openrouter_app_name: str = "purple-avito-hackaton"
    normalize_text: bool = True
    noise_min_words: int = 3
    noise_min_unique_ratio: float = 0.35
    collapse_repeated_words: bool = True
    sentence_head_count: int = 1
    sentence_tail_count: int = 1
    sentence_top_k: int = 7
    long_text_mode: str = "head"
    long_text_window_tokens: int = 256
    long_text_stride_tokens: int = 192
    long_text_max_windows: int = 4
    top_k_drafts: int = 5
    use_cross_encoder: bool = False
    cross_encoder_alpha: float = 0.5
    cross_encoder_loss_weight: float = 0.5


@dataclass
class TrainingSettings:
    epochs: int = 5
    batch_size: int = 8
    lr: float = 3e-5
    weight_decay: float = 0.01
    split_loss_weight: float = 0.5
    split_pos_weight: float = 1.0
    patience: int = 2
    min_delta: float = 1e-4
    threshold_grid: Tuple[float, ...] = (0.08, 0.12, 0.16, 0.20, 0.24)
    split_threshold_grid: Tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    optimize_for: str = "f1"
    min_recall: float = 0.0
    cross_encoder_loss_weight: float = 0.5


def _parse_int_list(raw_value: Any) -> List[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [int(v) for v in raw_value]

    text = str(raw_value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [int(v) for v in parsed]
    except (SyntaxError, ValueError):
        pass

    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    return [int(p) for p in parts if p and p.isdigit()]


def _normalize_text_content(text: str) -> str:
    """Normalize noisy user-generated text while preserving semantics."""
    if not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", str(text))
    # Some dataset rows contain literal escape sequences (e.g. "\\u2028") as plain text.
    cleaned = re.sub(r"\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}", " ", cleaned)
    cleaned = cleaned.replace("\u00A0", " ")
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    # Remove most emoji and pictographic symbols that add noise for retrieval/encoder.
    cleaned = re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U000024C2-\U0001F251]", " ", cleaned)
    # Remove remaining unicode symbol noise (decorative blocks, dingbats, variation selectors).
    cleaned = "".join(ch if unicodedata.category(ch) not in {"So", "Sk", "Cs", "Cf"} else " " for ch in cleaned)
    cleaned = re.sub(r"[~]{3,}", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[\w\-]+", text.lower(), flags=re.UNICODE)


def _collapse_repeated_words(text: str, max_repeat: int = 2) -> str:
    words = str(text).split()
    if not words:
        return ""

    collapsed: List[str] = []
    prev = ""
    run = 0
    for word in words:
        norm = word.casefold()
        if norm == prev:
            run += 1
        else:
            prev = norm
            run = 1
        if run <= max_repeat:
            collapsed.append(word)
    return " ".join(collapsed)


def _split_into_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|[\n\r;]+", str(text))
    return [p.strip() for p in parts if p and p.strip()]


def _sentence_noise_score(sentence: str, min_words: int, min_unique_ratio: float) -> bool:
    words = _tokenize_words(sentence)
    if len(words) < max(1, int(min_words)):
        return True
    unique_ratio = len(set(words)) / max(len(words), 1)
    return unique_ratio < float(min_unique_ratio)


def load_microcategories_from_csv(csv_path: str) -> List[MicroCategory]:
    microcategories: List[MicroCategory] = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phrases = [p.strip() for p in str(row.get("keyPhrases", "")).split(";") if p.strip()]
            microcategories.append(
                MicroCategory(
                    mc_id=int(row["mcId"]),
                    mc_title=str(row.get("mcTitle", "")).strip(),
                    key_phrases=phrases,
                )
            )
    if not microcategories:
        raise ValueError(f"No microcategories loaded from {csv_path}")
    return microcategories


def load_labeled_items_csv(csv_path: str) -> List[LabeledItem]:
    items: List[LabeledItem] = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(
                LabeledItem(
                    item_id=int(row["itemId"]),
                    mc_id=int(row["sourceMcId"]),
                    mc_title=str(row.get("sourceMcTitle", "")).strip(),
                    description=str(row.get("description", "")).strip(),
                    target_detected_mc_ids=_parse_int_list(row.get("targetDetectedMcIds", "[]")),
                    target_split_mc_ids=_parse_int_list(row.get("targetSplitMcIds", "[]")),
                    should_split=str(row.get("shouldSplit", "")).strip().lower() == "true",
                    case_type=str(row.get("caseType", "")).strip(),
                    split=str(row.get("split", "")).strip(),
                )
            )
    if not items:
        raise ValueError(f"No dataset rows loaded from {csv_path}")
    return items


def load_labeled_items_jsonl(jsonl_path: str) -> List[LabeledItem]:
    items: List[LabeledItem] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                LabeledItem(
                    item_id=int(row["itemId"]),
                    mc_id=int(row["sourceMcId"]),
                    mc_title=str(row.get("sourceMcTitle", "")).strip(),
                    description=str(row.get("description", "")).strip(),
                    target_detected_mc_ids=_parse_int_list(row.get("targetDetectedMcIds", [])),
                    target_split_mc_ids=_parse_int_list(row.get("targetSplitMcIds", [])),
                    should_split=bool(row.get("shouldSplit", False)),
                    case_type=str(row.get("caseType", "")).strip(),
                    split=str(row.get("split", "")).strip(),
                )
            )
    if not items:
        raise ValueError(f"No dataset rows loaded from {jsonl_path}")
    return items


def split_dataset(items: Sequence[LabeledItem]) -> Dict[str, List[LabeledItem]]:
    buckets: Dict[str, List[LabeledItem]] = {"train": [], "val": [], "test": []}
    for item in items:
        bucket = item.split.lower()
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(item)
    return buckets


class TfidfMicroCategoryRetriever:
    """TF-IDF retriever over key phrases with bi/tri-grams and optional char n-grams for candidate filtering."""

    def __init__(
            self,
            ngram_range: Tuple[int, int] = (2, 3),
            min_similarity: float = 0.05,
            top_k: int = 11,
            use_char_ngrams: bool = True,
            char_ngram_range: Tuple[int, int] = (2, 4),
            word_weight: float = 0.85,
            char_weight: float = 0.15,
    ) -> None:
        _require_sklearn()
        self.ngram_range = ngram_range
        self.min_similarity = min_similarity
        self.top_k = top_k
        self.use_char_ngrams = use_char_ngrams
        self.char_ngram_range = char_ngram_range
        self.word_weight = word_weight
        self.char_weight = char_weight
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            lowercase=True,
        )
        self.char_vectorizer = None
        if use_char_ngrams:
            self.char_vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=char_ngram_range,
                lowercase=True,
            )
        self._id_to_index: Dict[int, int] = {}
        self._index_to_id: Dict[int, int] = {}
        self._index_to_title: Dict[int, str] = {}
        self._mc_matrix = None
        self._mc_matrix_char = None

    @staticmethod
    def _mc_document(mc: MicroCategory) -> str:
        # Duplicate key phrases in document to strengthen category-specific terms.
        phrases = [p.strip() for p in mc.key_phrases if p and p.strip()]
        return " ; ".join(phrases + phrases)

    def fit(self, microcategories: Sequence[MicroCategory]) -> None:
        docs: List[str] = []
        self._id_to_index.clear()
        self._index_to_id.clear()
        self._index_to_title.clear()

        for idx, mc in enumerate(microcategories):
            self._id_to_index[mc.mc_id] = idx
            self._index_to_id[idx] = mc.mc_id
            self._index_to_title[idx] = mc.mc_title
            docs.append(self._mc_document(mc))

        if not docs:
            raise ValueError("At least one microcategory is required for retrieval fit().")

        self._mc_matrix = self.vectorizer.fit_transform(docs)
        if self.use_char_ngrams and self.char_vectorizer is not None:
            self._mc_matrix_char = self.char_vectorizer.fit_transform(docs)

    def detect(self, text: str) -> Tuple[List[int], Dict[int, float]]:
        if self._mc_matrix is None:
            raise RuntimeError("Retriever is not fitted. Call fit() first.")

        query_vec = self.vectorizer.transform([text])
        sims = (query_vec @ self._mc_matrix.T).toarray().ravel()

        # If char n-grams enabled, blend word and char similarities
        if self.use_char_ngrams and self._mc_matrix_char is not None:
            query_vec_char = self.char_vectorizer.transform([text])
            sims_char = (query_vec_char @ self._mc_matrix_char.T).toarray().ravel()
            sims = self.word_weight * sims + self.char_weight * sims_char

        scored = [
            (self._index_to_id[i], float(score))
            for i, score in enumerate(sims)
            if score >= self.min_similarity
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[: self.top_k]

        detected = [mc_id for mc_id, _ in scored]
        score_map = {mc_id: score for mc_id, score in scored}
        return detected, score_map


class TransformerSoftmaxSplitModel(_BaseModule):
    """
    Softmax model:
    - Transformer encodes ad text.
    - Trainable embedding table encodes microcategories.
    - Dot-product logits -> probability distribution over candidate microcategories.
    """

    def __init__(
            self,
            model_name: str,
            num_microcategories: int,
            hidden_dim: Optional[int] = None,
            dropout: float = 0.1,
            max_length: int = 256,
            long_text_mode: str = "head",
            long_text_window_tokens: int = 256,
            long_text_stride_tokens: int = 192,
            long_text_max_windows: int = 4,
    ) -> None:
        _require_torch()
        _require_transformers()
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        encoder_hidden = self.text_encoder.config.hidden_size
        self.hidden_dim = hidden_dim or encoder_hidden
        self.max_length = max_length
        self.long_text_mode = str(long_text_mode).strip().lower() or "head"
        if self.long_text_mode not in {"head", "head_tail", "chunks"}:
            self.long_text_mode = "head"
        self.long_text_window_tokens = max(32, int(long_text_window_tokens))
        self.long_text_stride_tokens = max(8, int(long_text_stride_tokens))
        self.long_text_max_windows = max(1, int(long_text_max_windows))

        self.projection = nn.Linear(encoder_hidden, self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.mc_embeddings = nn.Embedding(num_microcategories, self.hidden_dim)
        self.split_head = nn.Linear(self.hidden_dim, 1)
        self.cross_head = nn.Linear(self.hidden_dim, 1)
        self.temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def _pool(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom

    def _available_tokens_without_specials(self) -> int:
        return max(8, int(self.max_length) - 2)

    def _build_text_windows(self, token_ids: List[int]) -> List[List[int]]:
        available = self._available_tokens_without_specials()
        if not token_ids:
            fallback = self.tokenizer.unk_token_id
            if fallback is None:
                fallback = self.tokenizer.pad_token_id
            if fallback is None:
                fallback = 0
            token_ids = [int(fallback)]

        mode = self.long_text_mode
        if mode == "head":
            return [token_ids[:available]]

        if mode == "head_tail":
            if len(token_ids) <= available:
                return [token_ids]
            head_len = available // 2
            tail_len = available - head_len
            return [token_ids[:head_len] + token_ids[-tail_len:]]

        window_size = min(self.long_text_window_tokens, available)
        stride = max(1, min(self.long_text_stride_tokens, window_size))
        windows: List[List[int]] = []
        for start in range(0, len(token_ids), stride):
            chunk = token_ids[start:start + window_size]
            if not chunk:
                continue
            windows.append(chunk)
            if len(windows) >= self.long_text_max_windows:
                break
            if start + window_size >= len(token_ids):
                break
        if not windows:
            windows = [token_ids[:window_size]]
        return windows

    def _encode_text_with_windows(self, text: str, device: Any) -> Any:
        token_ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        windows = self._build_text_windows(token_ids)
        input_ids: List[List[int]] = []
        attention_masks: List[List[int]] = []
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        specials = int(cls_token_id is not None) + int(sep_token_id is not None)
        max_chunk_len = max(1, int(self.max_length) - specials)
        for chunk in windows:
            chunk = list(chunk[:max_chunk_len])
            seq: List[int] = []
            if cls_token_id is not None:
                seq.append(int(cls_token_id))
            seq.extend(chunk)
            if sep_token_id is not None:
                seq.append(int(sep_token_id))
            input_ids.append(seq)
            attention_masks.append([1] * len(seq))

        max_seq_len = max(len(seq) for seq in input_ids)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        padded_ids = [seq + [pad_token_id] * (max_seq_len - len(seq)) for seq in input_ids]
        padded_masks = [mask + [0] * (max_seq_len - len(mask)) for mask in attention_masks]
        batch = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long, device=device),
        }

        outputs = self.text_encoder(**batch)
        pooled = self._pool(outputs.last_hidden_state, batch["attention_mask"])
        projected = self.dropout(self.projection(pooled))

        if projected.size(0) == 1:
            return projected[0]
        return projected.max(dim=0).values

    def encode_text(self, texts: Sequence[str], device: Any) -> Any:
        if self.long_text_mode == "head":
            batch = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.text_encoder(**batch)
            pooled = self._pool(outputs.last_hidden_state, batch["attention_mask"])
            return self.dropout(self.projection(pooled))

        embeddings = [self._encode_text_with_windows(text, device) for text in texts]
        return torch.stack(embeddings, dim=0)

    def forward(self, text_embeddings: Any, mc_indices: Any) -> Any:
        mc_vecs = self.mc_embeddings(mc_indices)
        logits = torch.matmul(text_embeddings, mc_vecs.transpose(0, 1))
        temp = torch.clamp(self.temperature, min=0.05)
        return logits / temp

    def split_logits(self, text_embeddings: Any) -> Any:
        return self.split_head(text_embeddings).squeeze(-1)

    def cross_logits(self, texts: Sequence[str], candidate_texts: Sequence[str], device: Any) -> Any:
        if len(texts) != len(candidate_texts):
            raise ValueError("texts and candidate_texts must have equal lengths")
        batch = self.tokenizer(
            list(texts),
            list(candidate_texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = self.text_encoder(**batch)
        pooled = self._pool(outputs.last_hidden_state, batch["attention_mask"])
        pair_embeddings = self.dropout(self.projection(pooled))
        return self.cross_head(pair_embeddings).squeeze(-1)


class DraftSplitPipeline:
    """End-to-end pipeline for category detection and draft split prediction."""

    def __init__(
            self,
            microcategories: Sequence[MicroCategory],
            transformer_name: str = "DeepPavlov/rubert-base-cased",
            tfidf_ngram_range: Tuple[int, int] = (2, 3),
            tfidf_threshold: float = 0.08,
            tfidf_top_k: int = 11,
            prob_threshold: float = 0.30,
            split_threshold: float = 0.5,
            max_drafts: int = 0,
            score_margin: float = 1.0,
            relative_ratio: float = 0.0,
            score_blend_alpha: float = 1.0,
            max_length: int = 256,
            long_text_mode: str = "head",
            long_text_window_tokens: int = 256,
            long_text_stride_tokens: int = 192,
            long_text_max_windows: int = 4,
            top_k_drafts: int = 5,
            use_cross_encoder: bool = False,
            cross_encoder_alpha: float = 0.5,
            cross_encoder_loss_weight: float = 0.5,
            device: Optional[str] = None,
            use_char_ngrams: bool = True,
            use_llm_drafts: bool = False,
            openrouter_model: str = "qwen/qwen3.6-plus:free",
            openrouter_api_key: Optional[str] = None,
            openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions",
            openrouter_timeout_sec: float = 30.0,
            openrouter_site_url: Optional[str] = None,
            openrouter_app_name: str = "purple-avito-hackaton",
            normalize_text: bool = True,
                noise_min_words: int = 3,
                noise_min_unique_ratio: float = 0.35,
                collapse_repeated_words: bool = True,
                sentence_head_count: int = 1,
                sentence_tail_count: int = 1,
                sentence_top_k: int = 7,
            settings: Optional[PipelineSettings] = None,
    ) -> None:
        _require_torch()
        self.microcategories = list(microcategories)
        if not self.microcategories:
            raise ValueError("microcategories must not be empty")

        if settings is not None:
            transformer_name = settings.transformer_name
            tfidf_ngram_range = settings.tfidf_ngram_range
            tfidf_threshold = settings.tfidf_threshold
            tfidf_top_k = settings.tfidf_top_k
            prob_threshold = settings.prob_threshold
            split_threshold = settings.split_threshold
            max_drafts = settings.max_drafts
            score_margin = settings.score_margin
            relative_ratio = settings.relative_ratio
            score_blend_alpha = settings.score_blend_alpha
            max_length = settings.max_length
            long_text_mode = settings.long_text_mode
            long_text_window_tokens = settings.long_text_window_tokens
            long_text_stride_tokens = settings.long_text_stride_tokens
            long_text_max_windows = settings.long_text_max_windows
            top_k_drafts = settings.top_k_drafts
            use_cross_encoder = settings.use_cross_encoder
            cross_encoder_alpha = settings.cross_encoder_alpha
            cross_encoder_loss_weight = settings.cross_encoder_loss_weight
            device = settings.device if settings.device is not None else device
            use_llm_drafts = settings.use_llm_drafts
            openrouter_model = settings.openrouter_model
            openrouter_api_key = settings.openrouter_api_key
            openrouter_base_url = settings.openrouter_base_url
            openrouter_timeout_sec = settings.openrouter_timeout_sec
            openrouter_site_url = settings.openrouter_site_url
            openrouter_app_name = settings.openrouter_app_name
            normalize_text = settings.normalize_text
            noise_min_words = settings.noise_min_words
            noise_min_unique_ratio = settings.noise_min_unique_ratio
            collapse_repeated_words = settings.collapse_repeated_words
            sentence_head_count = settings.sentence_head_count
            sentence_tail_count = settings.sentence_tail_count
            sentence_top_k = settings.sentence_top_k

        self.id_to_title = {mc.mc_id: mc.mc_title for mc in self.microcategories}
        self.id_to_idx = {mc.mc_id: i for i, mc in enumerate(self.microcategories)}

        self.retriever = TfidfMicroCategoryRetriever(
            ngram_range=tfidf_ngram_range,
            min_similarity=tfidf_threshold,
            top_k=tfidf_top_k,
            use_char_ngrams=use_char_ngrams,
            word_weight=0.85,
            char_weight=0.15,
        )
        self.retriever.fit(self.microcategories)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = TransformerSoftmaxSplitModel(
            model_name=transformer_name,
            num_microcategories=len(self.microcategories),
            max_length=max_length,
            long_text_mode=long_text_mode,
            long_text_window_tokens=long_text_window_tokens,
            long_text_stride_tokens=long_text_stride_tokens,
            long_text_max_windows=long_text_max_windows,
        ).to(self.device)
        self.prob_threshold = prob_threshold
        self.split_threshold = split_threshold
        self.class_prob_thresholds: Dict[int, float] = {}
        self.max_drafts = max(0, int(max_drafts))
        self.score_margin = max(0.0, float(score_margin))
        self.relative_ratio = min(1.0, max(0.0, float(relative_ratio)))
        self.score_blend_alpha = min(1.0, max(0.0, float(score_blend_alpha)))
        self.use_llm_drafts = bool(use_llm_drafts)
        self.openrouter_model = str(openrouter_model)
        self.openrouter_base_url = str(openrouter_base_url)
        self.openrouter_timeout_sec = float(openrouter_timeout_sec)
        self.openrouter_site_url = openrouter_site_url
        self.openrouter_app_name = str(openrouter_app_name)
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.normalize_text = bool(normalize_text)
        self.noise_min_words = max(1, int(noise_min_words))
        self.noise_min_unique_ratio = min(1.0, max(0.0, float(noise_min_unique_ratio)))
        self.collapse_repeated_words = bool(collapse_repeated_words)
        self.sentence_head_count = max(0, int(sentence_head_count))
        self.sentence_tail_count = max(0, int(sentence_tail_count))
        self.sentence_top_k = max(0, int(sentence_top_k))
        self.top_k_drafts = max(0, int(top_k_drafts))
        self.use_cross_encoder = bool(use_cross_encoder)
        self.cross_encoder_alpha = min(1.0, max(0.0, float(cross_encoder_alpha)))
        self.cross_encoder_loss_weight = max(0.0, float(cross_encoder_loss_weight))
        self.id_to_candidate_text = {
            mc.mc_id: self.retriever._mc_document(mc) for mc in self.microcategories
        }
        self._phrase_lexicon = {
            p.strip().lower()
            for mc in self.microcategories
            for p in mc.key_phrases
            if p and p.strip()
        }
        self.llm_success_count = 0
        self.llm_fallback_count = 0
        self.last_llm_error: Optional[str] = None

    def _sentence_signal_score(self, sentence: str) -> float:
        words = _tokenize_words(sentence)
        if not words:
            return 0.0
        lower_sentence = sentence.lower()
        phrase_hits = sum(1 for phrase in self._phrase_lexicon if len(phrase) >= 4 and phrase in lower_sentence)
        uniq = len(set(words))
        # Prefer informative sentences with phrase evidence and lexical diversity.
        return float(phrase_hits * 3 + uniq)

    def _select_informative_sentences(self, text: str) -> str:
        sentences = _split_into_sentences(text)
        if not sentences:
            return text

        filtered: List[str] = []
        seen: set[str] = set()
        for sentence in sentences:
            compact = " ".join(sentence.split())
            if self.collapse_repeated_words:
                compact = _collapse_repeated_words(compact)
            if not compact:
                continue
            if _sentence_noise_score(compact, self.noise_min_words, self.noise_min_unique_ratio):
                continue
            dedup_key = compact.casefold()
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            filtered.append(compact)

        if not filtered:
            fallback = " ".join(str(text).split())
            return _collapse_repeated_words(fallback) if self.collapse_repeated_words else fallback

        selected: List[str] = []
        used_indices: set[int] = set()

        for idx in range(min(self.sentence_head_count, len(filtered))):
            selected.append(filtered[idx])
            used_indices.add(idx)

        if self.sentence_tail_count > 0:
            tail_start = max(0, len(filtered) - self.sentence_tail_count)
            for idx in range(tail_start, len(filtered)):
                if idx in used_indices:
                    continue
                selected.append(filtered[idx])
                used_indices.add(idx)

        if self.sentence_top_k > 0:
            scored = [
                (idx, self._sentence_signal_score(sentence))
                for idx, sentence in enumerate(filtered)
                if idx not in used_indices
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            for idx, _ in scored[: self.sentence_top_k]:
                selected.append(filtered[idx])
                used_indices.add(idx)

        if not selected:
            selected = filtered
        return " ".join(selected)

    def _prepare_text(self, text: str) -> str:
        prepared = " ".join(str(text).strip().split())
        if self.normalize_text:
            prepared = _normalize_text_content(prepared)
        if self.collapse_repeated_words:
            prepared = _collapse_repeated_words(prepared)
        prepared = self._select_informative_sentences(prepared)
        return " ".join(prepared.split())

    def train_step(
            self,
            texts: Sequence[str],
            candidate_mc_ids: Sequence[Sequence[int]],
            target_mc_ids: Sequence[Sequence[int]],
            should_split_targets: Sequence[bool],
            source_mc_ids: Sequence[int],
            optimizer: Any,
            split_loss_weight: float = 0.5,
            split_pos_weight: float = 1.0,
                cross_encoder_loss_weight: float = 0.5,
    ) -> float:
        if (
                len(texts) != len(candidate_mc_ids)
                or len(texts) != len(target_mc_ids)
                or len(texts) != len(should_split_targets)
                or len(texts) != len(source_mc_ids)
        ):
            raise ValueError(
                "texts, candidate_mc_ids, target_mc_ids, should_split_targets and source_mc_ids must have equal lengths")

        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        used_examples = 0

        def build_target_distribution(candidates: Sequence[int], targets: Sequence[int], source_mc_id: int) -> Any:
            target_vec = torch.zeros(len(candidates), dtype=torch.float32, device=self.device)
            positive_ids = [mc_id for mc_id in targets if mc_id in candidates and mc_id != source_mc_id]
            if positive_ids:
                weight = 1.0 / len(positive_ids)
                for mc_id in positive_ids:
                    target_vec[candidates.index(mc_id)] = weight
                return target_vec

            if source_mc_id in candidates:
                target_vec[candidates.index(source_mc_id)] = 1.0
            else:
                target_vec.fill_(1.0 / len(candidates))
            return target_vec

        def build_pair_targets(candidates: Sequence[int], targets: Sequence[int], source_mc_id: int) -> Any:
            target_vec = torch.zeros(len(candidates), dtype=torch.float32, device=self.device)
            positive_ids = [mc_id for mc_id in targets if mc_id in candidates and mc_id != source_mc_id]
            for mc_id in positive_ids:
                target_vec[candidates.index(mc_id)] = 1.0
            return target_vec

        for text, candidates, target_mc_list, should_split_target, source_mc_id in zip(
                texts,
                candidate_mc_ids,
                target_mc_ids,
                should_split_targets,
                source_mc_ids,
        ):
            if not candidates:
                continue
            candidates_unique = list(dict.fromkeys(candidates))
            prepared_text = self._prepare_text(text)

            cand_indices = torch.tensor(
                [self.id_to_idx[mc_id] for mc_id in candidates_unique],
                dtype=torch.long,
                device=self.device,
            )
            text_emb = self.model.encode_text([prepared_text], self.device)
            logits = self.model(text_emb, cand_indices).squeeze(0)
            target_distribution = build_target_distribution(candidates_unique, target_mc_list, source_mc_id)
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(logits, dim=-1),
                target_distribution,
                reduction="batchmean",
            )
            split_logit = self.model.split_logits(text_emb).view(-1)
            split_target = torch.tensor([1.0 if should_split_target else 0.0], device=self.device)
            pos_weight = torch.tensor([max(1e-6, float(split_pos_weight))], device=self.device)
            split_loss = nn.functional.binary_cross_entropy_with_logits(
                split_logit,
                split_target,
                pos_weight=pos_weight,
            )
            total_loss = total_loss + loss + split_loss_weight * split_loss

            if self.use_cross_encoder:
                candidate_texts = [self.id_to_candidate_text[mc_id] for mc_id in candidates_unique]
                pair_logits = self.model.cross_logits(
                    [prepared_text] * len(candidates_unique),
                    candidate_texts,
                    self.device,
                )
                pair_targets = build_pair_targets(candidates_unique, target_mc_list, source_mc_id)
                cross_loss = nn.functional.binary_cross_entropy_with_logits(pair_logits, pair_targets)
                total_loss = total_loss + cross_encoder_loss_weight * cross_loss
            used_examples += 1

        if used_examples == 0:
            return 0.0

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        return float(total_loss.detach().cpu().item())

    def build_candidates(self, item: Item, force_include: Optional[Sequence[int]] = None) -> List[int]:
        # In the current problem we only have 11 microcategories, so filtering is not necessary.
        candidates = [mc.mc_id for mc in self.microcategories]
        if force_include:
            for mc_id in force_include:
                if mc_id not in candidates and mc_id in self.id_to_idx:
                    candidates.append(mc_id)
        return candidates

    def train_on_labeled_items(
            self,
            items: Sequence[LabeledItem],
            optimizer: Any,
            batch_size: int = 16,
            epochs: int = 1,
            force_include_targets: bool = False,
            split_loss_weight: float = 0.5,
            split_pos_weight: float = 1.0,
            cross_encoder_loss_weight: float = 0.5,
            verbose: bool = False,
    ) -> List[float]:
        history: List[float] = []
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        total_steps = (len(items) + batch_size - 1) // batch_size if items else 0
        for _ in range(epochs):
            epoch_index = len(history) + 1
            epoch_loss = 0.0
            steps = 0
            for i in range(0, len(items), batch_size):
                batch = items[i: i + batch_size]
                texts = [x.description for x in batch]
                candidates = [
                    self.build_candidates(x, force_include=x.target_split_mc_ids if force_include_targets else None)
                    for x in batch
                ]
                target_mc_ids = [x.target_split_mc_ids for x in batch]
                should_split_targets = [x.should_split for x in batch]
                source_mc_ids = [x.mc_id for x in batch]
                loss = self.train_step(
                    texts,
                    candidates,
                    target_mc_ids,
                    should_split_targets,
                    source_mc_ids,
                    optimizer,
                    split_loss_weight=split_loss_weight,
                    split_pos_weight=split_pos_weight,
                    cross_encoder_loss_weight=cross_encoder_loss_weight,
                )
                epoch_loss += loss
                steps += 1
                if verbose and total_steps > 0:
                    avg_loss = epoch_loss / max(steps, 1)
                    progress = (steps / total_steps) * 100.0
                    print(
                        f"Epoch {epoch_index}/{epochs} | "
                        f"step {steps}/{total_steps} ({progress:5.1f}%) | "
                        f"avg_loss={avg_loss:.4f}",
                        end="\r",
                        flush=True,
                    )
            epoch_avg_loss = epoch_loss / max(steps, 1)
            history.append(epoch_avg_loss)
            if verbose:
                if total_steps > 0:
                    print()
                print(f"Epoch {epoch_index}/{epochs} | train_loss={epoch_avg_loss:.4f}")
        return history

    def fit_with_early_stopping(
            self,
            train_items: Sequence[LabeledItem],
            val_items: Sequence[LabeledItem],
            optimizer: Any,
            training_settings: Optional[TrainingSettings] = None,
            verbose: bool = True,
    ) -> Dict[str, Any]:
        settings = training_settings or TrainingSettings()
        best_state = None
        best_metrics: Optional[Dict[str, float]] = None
        best_threshold = self.prob_threshold
        best_split_threshold = self.split_threshold
        best_score = -1.0
        patience_left = settings.patience
        history: List[Dict[str, Any]] = []

        for epoch in range(1, settings.epochs + 1):
            loss_history = self.train_on_labeled_items(
                train_items,
                optimizer=optimizer,
                batch_size=settings.batch_size,
                epochs=1,
                split_loss_weight=settings.split_loss_weight,
                split_pos_weight=settings.split_pos_weight,
                cross_encoder_loss_weight=settings.cross_encoder_loss_weight,
                verbose=verbose,
            )
            loss_value = loss_history[-1] if loss_history else 0.0
            threshold_report = search_best_probability_threshold(
                self,
                val_items,
                settings.threshold_grid,
                optimize_for=settings.optimize_for,
                min_recall=settings.min_recall,
            )
            self.prob_threshold = threshold_report["threshold"]
            split_threshold_report = evaluate_split_probability_threshold(
                self,
                val_items,
                settings.split_threshold_grid,
                optimize_for=settings.optimize_for,
                min_recall=settings.min_recall,
            )
            self.split_threshold = split_threshold_report["threshold"]
            val_metrics = evaluate_split_quality(self, val_items)
            current_score = _selection_score(
                val_metrics,
                optimize_for=settings.optimize_for,
                min_recall=settings.min_recall,
            )

            if verbose:
                print(
                    f"Epoch {epoch}/{settings.epochs} | "
                    f"train_loss={loss_value:.4f} | "
                    f"val_f1={val_metrics['f1_micro']:.4f} | "
                    f"val_precision={val_metrics['precision_micro']:.4f} | "
                    f"val_recall={val_metrics['recall_micro']:.4f} | "
                    f"shouldSplit_acc={val_metrics['should_split_accuracy']:.4f} | "
                    f"threshold={threshold_report['threshold']:.2f} | "
                    f"split_threshold={split_threshold_report['threshold']:.2f}"
                )

            history.append(
                {
                    "epoch": epoch,
                    "loss": loss_value,
                    "threshold": threshold_report["threshold"],
                    "split_threshold": split_threshold_report["threshold"],
                    "metrics": val_metrics,
                }
            )

            if current_score > best_score + settings.min_delta:
                best_score = current_score
                best_threshold = threshold_report["threshold"]
                best_split_threshold = split_threshold_report["threshold"]
                best_metrics = val_metrics
                best_state = copy.deepcopy(self.model.state_dict())
                patience_left = settings.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.prob_threshold = best_threshold
        self.split_threshold = best_split_threshold

        return {
            "history": history,
            "best_threshold": best_threshold,
            "best_split_threshold": best_split_threshold,
            "best_metrics": best_metrics,
            "best_score": best_score,
            "optimize_for": settings.optimize_for,
            "min_recall": float(settings.min_recall),
        }

    def score_candidates(self, description: str, candidate_mc_ids: Sequence[int]) -> Dict[int, float]:
        if not candidate_mc_ids:
            return {}

        prepared_description = self._prepare_text(description)
        self.model.eval()
        with torch.no_grad():
            cand_indices = torch.tensor(
                [self.id_to_idx[mc_id] for mc_id in candidate_mc_ids],
                dtype=torch.long,
                device=self.device,
            )
            text_emb = self.model.encode_text([prepared_description], self.device)
            logits = self.model(text_emb, cand_indices).squeeze(0)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            return {mc_id: float(prob) for mc_id, prob in zip(candidate_mc_ids, probs)}

    def set_class_prob_thresholds(self, thresholds: Dict[int, float]) -> None:
        self.class_prob_thresholds = {int(mc_id): float(threshold) for mc_id, threshold in thresholds.items()}

    def get_class_prob_threshold(self, mc_id: int) -> float:
        return float(self.class_prob_thresholds.get(mc_id, self.prob_threshold))

    def set_reranking_controls(
            self,
            max_drafts: Optional[int] = None,
            score_margin: Optional[float] = None,
            relative_ratio: Optional[float] = None,
            score_blend_alpha: Optional[float] = None,
    ) -> None:
        if max_drafts is not None:
            self.max_drafts = max(0, int(max_drafts))
        if score_margin is not None:
            self.score_margin = max(0.0, float(score_margin))
        if relative_ratio is not None:
            self.relative_ratio = min(1.0, max(0.0, float(relative_ratio)))
        if score_blend_alpha is not None:
            self.score_blend_alpha = min(1.0, max(0.0, float(score_blend_alpha)))

    def _blend_scores(
            self,
            prob_map: Dict[int, float],
            tfidf_scores: Dict[int, float],
            candidate_mc_ids: Sequence[int],
    ) -> Dict[int, float]:
        if not candidate_mc_ids:
            return {}

        tfidf_max = max([float(tfidf_scores.get(mc_id, 0.0)) for mc_id in candidate_mc_ids], default=0.0)
        blended: Dict[int, float] = {}
        for mc_id in candidate_mc_ids:
            model_score = float(prob_map.get(mc_id, 0.0))
            tfidf_score = float(tfidf_scores.get(mc_id, 0.0))
            if tfidf_max > 0.0:
                tfidf_score = tfidf_score / tfidf_max
            blended[mc_id] = self.score_blend_alpha * model_score + (1.0 - self.score_blend_alpha) * tfidf_score
        return blended

    def _apply_reranking_controls(self, scored_items: List[Tuple[int, float]]) -> List[int]:
        if not scored_items:
            return []

        if self.top_k_drafts > 0:
            scored_items = scored_items[: self.top_k_drafts]

        top_score = float(scored_items[0][1])
        selected: List[int] = []
        for mc_id, score in scored_items:
            within_margin = (top_score - float(score)) <= self.score_margin
            within_ratio = True if self.relative_ratio <= 0.0 else float(score) >= top_score * self.relative_ratio
            if within_margin and within_ratio:
                selected.append(mc_id)

        if self.max_drafts > 0:
            selected = selected[: self.max_drafts]
        return selected

    def _generate_template_draft_text(self, mc_title: str, description: str) -> str:
        short = " ".join(description.strip().split())
        short = short[:220] + ("..." if len(short) > 220 else "")
        return f"Выполняем работы по направлению \"{mc_title}\". Опыт и аккуратное исполнение. {short}"

    def _build_llm_draft_prompt(self, mc_title: str, description: str) -> str:
        source = " ".join(description.strip().split())
        source = source[:1000]
        return (
            "Сгенерируй короткий продающий черновик объявления на русском языке.\\n"
            "Ограничения:\\n"
            "- 1-2 предложения, до 260 символов.\\n"
            "- Без markdown, без списков, без кавычек-елочек.\\n"
            "- Только конкретика по категории, без выдумывания услуг, которых нет в тексте.\\n"
            "- Нейтрально-деловой тон.\\n"
            f"Категория: {mc_title}\\n"
            f"Исходный текст: {source}\\n"
            "Верни только готовый текст черновика."
        )

    def _generate_llm_draft_text(self, mc_title: str, description: str) -> Optional[str]:
        if not self.use_llm_drafts or not self.openrouter_api_key:
            if self.use_llm_drafts and not self.openrouter_api_key:
                self.last_llm_error = "OPENROUTER_API_KEY is missing"
            return None

        payload = {
            "model": self.openrouter_model,
            "temperature": 0.35,
            "max_tokens": 180,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты копирайтер для объявлений о ремонте. Пиши кратко, конкретно и безопасно.",
                },
                {
                    "role": "user",
                    "content": self._build_llm_draft_prompt(mc_title, description),
                },
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.openrouter_site_url:
            headers["HTTP-Referer"] = self.openrouter_site_url
        if self.openrouter_app_name:
            headers["X-Title"] = self.openrouter_app_name

        request = urllib_request.Request(
            self.openrouter_base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=self.openrouter_timeout_sec) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            content = (
                parsed.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if not isinstance(content, str):
                self.last_llm_error = "OpenRouter response has no text content"
                return None
            cleaned = " ".join(content.strip().split())
            if cleaned:
                self.last_llm_error = None
            return cleaned[:260] if cleaned else None
        except urllib_error.HTTPError as exc:
            details = ""
            try:
                details = exc.read().decode("utf-8")[:240]
            except Exception:
                details = ""
            self.last_llm_error = f"HTTP {exc.code}: {details or exc.reason}"
            return None
        except urllib_error.URLError as exc:
            self.last_llm_error = f"URL error: {exc.reason}"
            return None
        except TimeoutError:
            self.last_llm_error = "Timeout while calling OpenRouter"
            return None
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            self.last_llm_error = f"Invalid OpenRouter response: {exc}"
            return None

    def _generate_draft_text(self, mc_title: str, description: str) -> str:
        llm_text = self._generate_llm_draft_text(mc_title, description)
        if llm_text:
            self.llm_success_count += 1
            return llm_text
        if self.use_llm_drafts:
            self.llm_fallback_count += 1
        return self._generate_template_draft_text(mc_title, description)

    def get_llm_diagnostics(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.use_llm_drafts),
            "success_count": int(self.llm_success_count),
            "fallback_count": int(self.llm_fallback_count),
            "last_error": self.last_llm_error,
            "model": self.openrouter_model,
            "api_key_present": bool(self.openrouter_api_key),
        }

    def predict(self, item: Item) -> PredictionResult:
        prepared_description = self._prepare_text(item.description)
        detected, tfidf_scores = self.retriever.detect(prepared_description)
        detected_wo_source = [mc_id for mc_id in detected if mc_id != item.mc_id]
        prob_map = self.score_candidates(prepared_description, detected_wo_source)
        blended_scores = self._blend_scores(prob_map, tfidf_scores, detected_wo_source)
        self.model.eval()
        with torch.no_grad():
            text_emb = self.model.encode_text([prepared_description], self.device)
            split_prob = torch.sigmoid(self.model.split_logits(text_emb).view(-1)).item()

        selected = [
            (mc_id, score)
            for mc_id, score in blended_scores.items()
            if score >= self.get_class_prob_threshold(mc_id)
        ]
        selected.sort(key=lambda x: x[1], reverse=True)
        target_mc_ids = self._apply_reranking_controls(selected)
        should_split = split_prob >= self.split_threshold
        if not should_split:
            target_mc_ids = []

        if self.use_cross_encoder and target_mc_ids:
            candidate_texts = [self.id_to_candidate_text[mc_id] for mc_id in target_mc_ids]
            cross_logits = self.model.cross_logits(
                [prepared_description] * len(target_mc_ids),
                candidate_texts,
                self.device,
            )
            cross_probs = torch.sigmoid(cross_logits).detach().cpu().tolist()
            cross_map = {mc_id: float(prob) for mc_id, prob in zip(target_mc_ids, cross_probs)}
            reranked_scores: Dict[int, float] = {}
            for mc_id in target_mc_ids:
                base_score = float(blended_scores.get(mc_id, 0.0))
                cross_score = float(cross_map.get(mc_id, 0.0))
                reranked_scores[mc_id] = (
                    (1.0 - self.cross_encoder_alpha) * base_score + self.cross_encoder_alpha * cross_score
                )
            target_mc_ids = [mc_id for mc_id, _ in sorted(reranked_scores.items(), key=lambda x: x[1], reverse=True)]

        drafts = [
            Draft(
                mc_id=mc_id,
                mc_title=self.id_to_title[mc_id],
                text=self._generate_draft_text(self.id_to_title[mc_id], prepared_description),
            )
            for mc_id in target_mc_ids
        ]

        merged_probs = blended_scores

        return PredictionResult(
            detected_mc_ids=detected,
            should_split=should_split,
            drafts=drafts,
            probabilities=merged_probs,
        )


def to_response_json(result: PredictionResult) -> Dict[str, object]:
    """Convert internal output to the hackathon response schema."""
    return {
        "detectedMcIds": result.detected_mc_ids,
        "shouldSplit": result.should_split,
        "drafts": [
            {
                "mcId": d.mc_id,
                "mcTitle": d.mc_title,
                "text": d.text,
            }
            for d in result.drafts
        ],
    }


def evaluate_split_quality(
        pipeline: DraftSplitPipeline,
        items: Sequence[LabeledItem],
) -> Dict[str, float]:
    """Compute micro Precision/Recall/F1 for targetSplitMcIds and shouldSplit accuracy."""
    tp = 0
    fp = 0
    fn = 0
    correct_split = 0

    for item in items:
        pred = pipeline.predict(item)
        pred_set = {d.mc_id for d in pred.drafts}
        gold_set = set(item.target_split_mc_ids)

        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
        correct_split += int(pred.should_split == item.should_split)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = correct_split / len(items) if items else 0.0

    return {
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": f1,
        "should_split_accuracy": accuracy,
    }


def evaluate_detect_quality(
        pipeline: DraftSplitPipeline,
        items: Sequence[LabeledItem],
) -> Dict[str, float]:
    """Compute micro Precision/Recall/F1 for detected categories against targetDetectedMcIds."""
    tp = 0
    fp = 0
    fn = 0

    for item in items:
        pred = pipeline.predict(item)
        pred_set = {mc_id for mc_id in pred.detected_mc_ids if mc_id != item.mc_id}
        gold_set = set(item.target_detected_mc_ids)

        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": f1,
    }


def _selection_score(metrics: Dict[str, float], optimize_for: str, min_recall: float) -> float:
    opt = optimize_for.strip().lower()
    precision = float(metrics.get("precision_micro", 0.0))
    recall = float(metrics.get("recall_micro", 0.0))
    f1 = float(metrics.get("f1_micro", 0.0))

    if recall < float(min_recall):
        return -1.0

    if opt == "precision":
        return precision
    if opt == "recall":
        return recall
    return f1


def search_best_probability_threshold(
        pipeline: DraftSplitPipeline,
        items: Sequence[LabeledItem],
        thresholds: Sequence[float],
        optimize_for: str = "f1",
        min_recall: float = 0.0,
) -> Dict[str, Any]:
    if not thresholds:
        raise ValueError("thresholds must not be empty")

    original_threshold = pipeline.prob_threshold
    best_threshold = original_threshold
    best_metrics: Optional[Dict[str, float]] = None
    best_score = -1.0
    results: List[Dict[str, Any]] = []

    for threshold in thresholds:
        pipeline.prob_threshold = float(threshold)
        metrics = evaluate_split_quality(pipeline, items)
        score = _selection_score(metrics, optimize_for=optimize_for, min_recall=min_recall)
        results.append({"threshold": float(threshold), "metrics": metrics, "score": score})
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    pipeline.prob_threshold = original_threshold
    return {
        "threshold": best_threshold,
        "optimize_for": optimize_for,
        "min_recall": float(min_recall),
        "best_score": best_score,
        "metrics": best_metrics or evaluate_split_quality(pipeline, items),
        "results": results,
    }


def evaluate_split_probability_threshold(
        pipeline: DraftSplitPipeline,
        items: Sequence[LabeledItem],
        thresholds: Sequence[float],
    optimize_for: str = "f1",
    min_recall: float = 0.0,
) -> Dict[str, Any]:
    if not thresholds:
        raise ValueError("thresholds must not be empty")

    original_threshold = pipeline.split_threshold
    best_threshold = original_threshold
    best_metrics: Optional[Dict[str, float]] = None
    best_score = -1.0
    results: List[Dict[str, Any]] = []

    for threshold in thresholds:
        pipeline.split_threshold = float(threshold)
        metrics = evaluate_split_quality(pipeline, items)
        score = _selection_score(metrics, optimize_for=optimize_for, min_recall=min_recall)
        results.append({"threshold": float(threshold), "metrics": metrics, "score": score})
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    pipeline.split_threshold = original_threshold
    return {
        "threshold": best_threshold,
        "optimize_for": optimize_for,
        "min_recall": float(min_recall),
        "best_score": best_score,
        "metrics": best_metrics or evaluate_split_quality(pipeline, items),
        "results": results,
    }


def search_best_class_probability_thresholds(
        pipeline: DraftSplitPipeline,
        items: Sequence[LabeledItem],
        thresholds: Sequence[float],
        optimize_for: str = "f1",
        min_recall: float = 0.0,
) -> Dict[str, Any]:
    if not thresholds:
        raise ValueError("thresholds must not be empty")

    original_thresholds = dict(pipeline.class_prob_thresholds)
    class_support: Dict[int, int] = {}
    for item in items:
        for mc_id in item.target_split_mc_ids:
            class_support[mc_id] = class_support.get(mc_id, 0) + 1

    search_order = [mc_id for mc_id, _ in sorted(class_support.items(), key=lambda kv: (-kv[1], kv[0]))]
    for mc_id in pipeline.id_to_title:
        if mc_id not in search_order:
            search_order.append(mc_id)

    best_metrics = evaluate_split_quality(pipeline, items)
    best_score = _selection_score(best_metrics, optimize_for=optimize_for, min_recall=min_recall)
    per_class_results: List[Dict[str, Any]] = []
    current_thresholds = dict(pipeline.class_prob_thresholds)

    for mc_id in search_order:
        base_threshold = current_thresholds.get(mc_id, pipeline.prob_threshold)
        best_class_threshold = base_threshold
        best_class_metrics = best_metrics
        best_class_score = best_score

        for threshold in thresholds:
            pipeline.class_prob_thresholds[mc_id] = float(threshold)
            metrics = evaluate_split_quality(pipeline, items)
            score = _selection_score(metrics, optimize_for=optimize_for, min_recall=min_recall)
            per_class_results.append(
                {
                    "mcId": mc_id,
                    "mcTitle": pipeline.id_to_title[mc_id],
                    "threshold": float(threshold),
                    "metrics": metrics,
                    "score": score,
                }
            )
            if score > best_class_score:
                best_class_score = score
                best_class_threshold = float(threshold)
                best_class_metrics = metrics

        pipeline.class_prob_thresholds[mc_id] = best_class_threshold
        current_thresholds[mc_id] = best_class_threshold
        best_metrics = best_class_metrics
        best_score = best_class_score

    pipeline.class_prob_thresholds = current_thresholds
    if not current_thresholds:
        pipeline.class_prob_thresholds = original_thresholds

    return {
        "class_thresholds": dict(sorted(current_thresholds.items())),
        "optimize_for": optimize_for,
        "min_recall": float(min_recall),
        "best_score": best_score,
        "metrics": best_metrics,
        "best_f1": best_metrics["f1_micro"],
        "results": per_class_results,
    }


def search_best_reranking_controls(
        pipeline: DraftSplitPipeline,
        items: Sequence[LabeledItem],
        top_k_grid: Sequence[int],
        max_drafts_grid: Sequence[int],
        margin_grid: Sequence[float],
        relative_ratio_grid: Sequence[float],
        blend_alpha_grid: Sequence[float],
        optimize_for: str = "f1",
        min_recall: float = 0.0,
) -> Dict[str, Any]:
    if not top_k_grid:
        raise ValueError("top_k_grid must not be empty")
    if not max_drafts_grid:
        raise ValueError("max_drafts_grid must not be empty")
    if not margin_grid:
        raise ValueError("margin_grid must not be empty")
    if not relative_ratio_grid:
        raise ValueError("relative_ratio_grid must not be empty")
    if not blend_alpha_grid:
        raise ValueError("blend_alpha_grid must not be empty")

    orig_top_k = pipeline.top_k_drafts
    orig_max_drafts = pipeline.max_drafts
    orig_margin = pipeline.score_margin
    orig_relative_ratio = pipeline.relative_ratio
    orig_blend_alpha = pipeline.score_blend_alpha

    best_metrics: Optional[Dict[str, float]] = None
    best_score = -1.0
    best_cfg = {
        "top_k_drafts": orig_top_k,
        "max_drafts": orig_max_drafts,
        "score_margin": orig_margin,
        "relative_ratio": orig_relative_ratio,
        "score_blend_alpha": orig_blend_alpha,
    }
    results: List[Dict[str, Any]] = []

    for top_k, max_drafts, margin, ratio, blend_alpha in itertools.product(
            top_k_grid,
            max_drafts_grid,
            margin_grid,
            relative_ratio_grid,
            blend_alpha_grid,
    ):
        pipeline.top_k_drafts = max(0, int(top_k))
        pipeline.set_reranking_controls(
            max_drafts=int(max_drafts),
            score_margin=float(margin),
            relative_ratio=float(ratio),
            score_blend_alpha=float(blend_alpha),
        )
        metrics = evaluate_split_quality(pipeline, items)
        score = _selection_score(metrics, optimize_for=optimize_for, min_recall=min_recall)
        results.append(
            {
                "top_k_drafts": int(top_k),
                "max_drafts": int(max_drafts),
                "score_margin": float(margin),
                "relative_ratio": float(ratio),
                "score_blend_alpha": float(blend_alpha),
                "metrics": metrics,
                "score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_cfg = {
                "top_k_drafts": int(top_k),
                "max_drafts": int(max_drafts),
                "score_margin": float(margin),
                "relative_ratio": float(ratio),
                "score_blend_alpha": float(blend_alpha),
            }

    pipeline.top_k_drafts = best_cfg["top_k_drafts"]
    pipeline.set_reranking_controls(
        max_drafts=best_cfg["max_drafts"],
        score_margin=best_cfg["score_margin"],
        relative_ratio=best_cfg["relative_ratio"],
        score_blend_alpha=best_cfg["score_blend_alpha"],
    )

    if best_metrics is None:
        pipeline.top_k_drafts = orig_top_k
        pipeline.set_reranking_controls(
            max_drafts=orig_max_drafts,
            score_margin=orig_margin,
            relative_ratio=orig_relative_ratio,
            score_blend_alpha=orig_blend_alpha,
        )
        best_metrics = evaluate_split_quality(pipeline, items)

    return {
        "optimize_for": optimize_for,
        "min_recall": float(min_recall),
        "best_score": best_score,
        "best_controls": best_cfg,
        "metrics": best_metrics,
        "results": results,
    }
def evaluate_retrieval_recall(
        pipeline: DraftSplitPipeline,
        items: Sequence[LabeledItem],
) -> float:
    # With full candidate set this becomes a sanity metric rather than a filter metric.
    total = 0
    hit = 0
    for item in items:
        detected, _ = pipeline.retriever.detect(item.description)
        pred_set = {mc_id for mc_id in detected if mc_id != item.mc_id}
        gold_set = set(item.target_split_mc_ids)
        total += len(gold_set)
        hit += len(pred_set & gold_set)
    return hit / total if total else 0.0

