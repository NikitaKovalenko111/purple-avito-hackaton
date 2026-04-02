from __future__ import annotations

import ast
import csv
import copy
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
    device: Optional[str] = None
    max_length: int = 256


@dataclass
class TrainingSettings:
    epochs: int = 10
    batch_size: int = 8
    lr: float = 3e-5
    weight_decay: float = 0.01
    split_loss_weight: float = 0.5
    patience: int = 2
    min_delta: float = 1e-4
    threshold_grid: Tuple[float, ...] = (0.08, 0.12, 0.16, 0.20, 0.24)


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


class TransformerSoftmaxSplitModel(nn.Module):
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
    ) -> None:
        _require_torch()
        _require_transformers()
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        encoder_hidden = self.text_encoder.config.hidden_size
        self.hidden_dim = hidden_dim or encoder_hidden
        self.max_length = max_length

        self.projection = nn.Linear(encoder_hidden, self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.mc_embeddings = nn.Embedding(num_microcategories, self.hidden_dim)
        self.split_head = nn.Linear(self.hidden_dim, 1)
        self.temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def _pool(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom

    def encode_text(self, texts: Sequence[str], device: Any) -> Any:
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

    def forward(self, text_embeddings: Any, mc_indices: Any) -> Any:
        mc_vecs = self.mc_embeddings(mc_indices)
        logits = torch.matmul(text_embeddings, mc_vecs.transpose(0, 1))
        temp = torch.clamp(self.temperature, min=0.05)
        return logits / temp

    def split_logits(self, text_embeddings: Any) -> Any:
        return self.split_head(text_embeddings).squeeze(-1)


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
            max_length: int = 256,
            device: Optional[str] = None,
            use_char_ngrams: bool = True,
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
            max_length = settings.max_length
            device = settings.device if settings.device is not None else device

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
        ).to(self.device)
        self.prob_threshold = prob_threshold
        self.split_threshold = split_threshold

    def train_step(
            self,
            texts: Sequence[str],
            candidate_mc_ids: Sequence[Sequence[int]],
            target_mc_ids: Sequence[Sequence[int]],
            should_split_targets: Sequence[bool],
            source_mc_ids: Sequence[int],
            optimizer: Any,
            split_loss_weight: float = 0.5,
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

            cand_indices = torch.tensor(
                [self.id_to_idx[mc_id] for mc_id in candidates_unique],
                dtype=torch.long,
                device=self.device,
            )
            text_emb = self.model.encode_text([text], self.device)
            logits = self.model(text_emb, cand_indices).squeeze(0)
            target_distribution = build_target_distribution(candidates_unique, target_mc_list, source_mc_id)
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(logits, dim=-1),
                target_distribution,
                reduction="batchmean",
            )
            split_logit = self.model.split_logits(text_emb).view(-1)
            split_target = torch.tensor([1.0 if should_split_target else 0.0], device=self.device)
            split_loss = nn.functional.binary_cross_entropy_with_logits(split_logit, split_target)
            total_loss = total_loss + loss + split_loss_weight * split_loss
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
            verbose: bool = False,
    ) -> List[float]:
        history: List[float] = []
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

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
                )
                epoch_loss += loss
                steps += 1
            epoch_avg_loss = epoch_loss / max(steps, 1)
            history.append(epoch_avg_loss)
            if verbose:
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
        best_f1 = -1.0
        patience_left = settings.patience
        history: List[Dict[str, Any]] = []

        for epoch in range(1, settings.epochs + 1):
            loss_history = self.train_on_labeled_items(
                train_items,
                optimizer=optimizer,
                batch_size=settings.batch_size,
                epochs=1,
                split_loss_weight=settings.split_loss_weight,
                verbose=False,
            )
            loss_value = loss_history[-1] if loss_history else 0.0
            threshold_report = search_best_probability_threshold(self, val_items, settings.threshold_grid)
            val_metrics = threshold_report["metrics"]
            current_f1 = val_metrics["f1_micro"]

            if verbose:
                print(
                    f"Epoch {epoch}/{settings.epochs} | "
                    f"train_loss={loss_value:.4f} | "
                    f"val_f1={current_f1:.4f} | "
                    f"val_precision={val_metrics['precision_micro']:.4f} | "
                    f"val_recall={val_metrics['recall_micro']:.4f} | "
                    f"shouldSplit_acc={val_metrics['should_split_accuracy']:.4f} | "
                    f"threshold={threshold_report['threshold']:.2f}"
                )

            history.append(
                {
                    "epoch": epoch,
                    "loss": loss_value,
                    "threshold": threshold_report["threshold"],
                    "metrics": val_metrics,
                }
            )

            if current_f1 > best_f1 + settings.min_delta:
                best_f1 = current_f1
                best_threshold = threshold_report["threshold"]
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

        return {
            "history": history,
            "best_threshold": best_threshold,
            "best_metrics": best_metrics,
            "best_f1": best_f1,
        }

    def score_candidates(self, description: str, candidate_mc_ids: Sequence[int]) -> Dict[int, float]:
        if not candidate_mc_ids:
            return {}

        self.model.eval()
        with torch.no_grad():
            cand_indices = torch.tensor(
                [self.id_to_idx[mc_id] for mc_id in candidate_mc_ids],
                dtype=torch.long,
                device=self.device,
            )
            text_emb = self.model.encode_text([description], self.device)
            logits = self.model(text_emb, cand_indices).squeeze(0)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            return {mc_id: float(prob) for mc_id, prob in zip(candidate_mc_ids, probs)}

    def _generate_draft_text(self, mc_title: str, description: str) -> str:
        short = " ".join(description.strip().split())
        short = short[:220] + ("..." if len(short) > 220 else "")
        return f"Выполняем работы по направлению \"{mc_title}\". Опыт и аккуратное исполнение. {short}"

    def predict(self, item: Item) -> PredictionResult:
        detected, tfidf_scores = self.retriever.detect(item.description)
        detected_wo_source = [mc_id for mc_id in detected if mc_id != item.mc_id]
        prob_map = self.score_candidates(item.description, detected_wo_source)
        self.model.eval()
        with torch.no_grad():
            text_emb = self.model.encode_text([item.description], self.device)
            split_prob = torch.sigmoid(self.model.split_logits(text_emb).view(-1)).item()

        target_mc_ids = [
            mc_id for mc_id, p in prob_map.items() if p >= self.prob_threshold and mc_id != item.mc_id
        ]
        should_split = split_prob >= self.split_threshold

        drafts = [
            Draft(
                mc_id=mc_id,
                mc_title=self.id_to_title[mc_id],
                text=self._generate_draft_text(self.id_to_title[mc_id], item.description),
            )
            for mc_id in target_mc_ids
        ]

        merged_probs = {mc_id: float(tfidf_scores.get(mc_id, 0.0)) for mc_id in detected_wo_source}
        merged_probs.update(prob_map)

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


def search_best_probability_threshold(
        pipeline: DraftSplitPipeline,
        items: Sequence[LabeledItem],
        thresholds: Sequence[float],
) -> Dict[str, Any]:
    if not thresholds:
        raise ValueError("thresholds must not be empty")

    original_threshold = pipeline.prob_threshold
    best_threshold = original_threshold
    best_metrics: Optional[Dict[str, float]] = None
    best_f1 = -1.0
    results: List[Dict[str, Any]] = []

    for threshold in thresholds:
        pipeline.prob_threshold = float(threshold)
        metrics = evaluate_split_quality(pipeline, items)
        results.append({"threshold": float(threshold), "metrics": metrics})
        if metrics["f1_micro"] > best_f1:
            best_f1 = metrics["f1_micro"]
            best_threshold = float(threshold)
            best_metrics = metrics

    pipeline.prob_threshold = original_threshold
    return {
        "threshold": best_threshold,
        "metrics": best_metrics or evaluate_split_quality(pipeline, items),
        "results": results,
    }


def evaluate_split_probability_threshold(
        pipeline: DraftSplitPipeline,
        items: Sequence[LabeledItem],
        thresholds: Sequence[float],
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
        results.append({"threshold": float(threshold), "metrics": metrics})
        if metrics["should_split_accuracy"] > best_score:
            best_score = metrics["should_split_accuracy"]
            best_threshold = float(threshold)
            best_metrics = metrics

    pipeline.split_threshold = original_threshold
    return {
        "threshold": best_threshold,
        "metrics": best_metrics or evaluate_split_quality(pipeline, items),
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

