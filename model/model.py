from __future__ import annotations

import ast
import csv
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
	"""TF-IDF retriever over key phrases with bi/tri-grams for candidate filtering."""

	def __init__(
		self,
		ngram_range: Tuple[int, int] = (2, 3),
		min_similarity: float = 0.08,
		top_k: int = 8,
	) -> None:
		_require_sklearn()
		self.ngram_range = ngram_range
		self.min_similarity = min_similarity
		self.top_k = top_k
		self.vectorizer = TfidfVectorizer(
			analyzer="word",
			ngram_range=ngram_range,
			lowercase=True,
		)
		self._id_to_index: Dict[int, int] = {}
		self._index_to_id: Dict[int, int] = {}
		self._index_to_title: Dict[int, str] = {}
		self._mc_matrix = None

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

	def detect(self, text: str) -> Tuple[List[int], Dict[int, float]]:
		if self._mc_matrix is None:
			raise RuntimeError("Retriever is not fitted. Call fit() first.")

		query_vec = self.vectorizer.transform([text])
		sims = (query_vec @ self._mc_matrix.T).toarray().ravel()

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
	) -> None:
		_require_torch()
		_require_transformers()
		super().__init__()
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.text_encoder = AutoModel.from_pretrained(model_name)
		encoder_hidden = self.text_encoder.config.hidden_size
		self.hidden_dim = hidden_dim or encoder_hidden

		self.projection = nn.Linear(encoder_hidden, self.hidden_dim)
		self.dropout = nn.Dropout(dropout)
		self.mc_embeddings = nn.Embedding(num_microcategories, self.hidden_dim)
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
			max_length=256,
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


class DraftSplitPipeline:
	"""End-to-end pipeline for category detection and draft split prediction."""

	def __init__(
		self,
		microcategories: Sequence[MicroCategory],
		transformer_name: str = "cointegrated/rubert-tiny2",
		tfidf_ngram_range: Tuple[int, int] = (2, 3),
		tfidf_threshold: float = 0.08,
		prob_threshold: float = 0.30,
		device: Optional[str] = None,
	) -> None:
		_require_torch()
		self.microcategories = list(microcategories)
		if not self.microcategories:
			raise ValueError("microcategories must not be empty")

		self.id_to_title = {mc.mc_id: mc.mc_title for mc in self.microcategories}
		self.id_to_idx = {mc.mc_id: i for i, mc in enumerate(self.microcategories)}

		self.retriever = TfidfMicroCategoryRetriever(
			ngram_range=tfidf_ngram_range,
			min_similarity=tfidf_threshold,
		)
		self.retriever.fit(self.microcategories)

		self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
		self.model = TransformerSoftmaxSplitModel(
			model_name=transformer_name,
			num_microcategories=len(self.microcategories),
		).to(self.device)
		self.prob_threshold = prob_threshold

	def train_step(
		self,
		texts: Sequence[str],
		candidate_mc_ids: Sequence[Sequence[int]],
		target_mc_ids: Sequence[Sequence[int]],
		source_mc_ids: Sequence[int],
		optimizer: Any,
	) -> float:
		if len(texts) != len(candidate_mc_ids) or len(texts) != len(target_mc_ids) or len(texts) != len(source_mc_ids):
			raise ValueError("texts, candidate_mc_ids, target_mc_ids and source_mc_ids must have equal lengths")

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

		for text, candidates, target_mc_list, source_mc_id in zip(
			texts,
			candidate_mc_ids,
			target_mc_ids,
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
			total_loss = total_loss + loss
			used_examples += 1

		if used_examples == 0:
			return 0.0

		optimizer.zero_grad(set_to_none=True)
		total_loss.backward()
		optimizer.step()
		return float(total_loss.detach().cpu().item())

	def build_candidates(self, item: Item, force_include: Optional[Sequence[int]] = None) -> List[int]:
		detected, _ = self.retriever.detect(item.description)
		candidates = list(detected)
		if item.mc_id not in candidates:
			candidates.insert(0, item.mc_id)
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
	) -> List[float]:
		history: List[float] = []
		if batch_size <= 0:
			raise ValueError("batch_size must be > 0")

		for _ in range(epochs):
			epoch_loss = 0.0
			steps = 0
			for i in range(0, len(items), batch_size):
				batch = items[i : i + batch_size]
				texts = [x.description for x in batch]
				candidates = [self.build_candidates(x, force_include=x.target_split_mc_ids) for x in batch]
				target_mc_ids = [x.target_split_mc_ids for x in batch]
				source_mc_ids = [x.mc_id for x in batch]
				loss = self.train_step(texts, candidates, target_mc_ids, source_mc_ids, optimizer)
				epoch_loss += loss
				steps += 1
			history.append(epoch_loss / max(steps, 1))
		return history

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

		target_mc_ids = [
			mc_id for mc_id, p in prob_map.items() if p >= self.prob_threshold and mc_id != item.mc_id
		]
		should_split = len(target_mc_ids) > 0

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

