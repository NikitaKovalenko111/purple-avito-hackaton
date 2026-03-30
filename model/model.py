from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

torch = importlib.import_module("torch")
nn = importlib.import_module("torch.nn")
TfidfVectorizer = importlib.import_module("sklearn.feature_extraction.text").TfidfVectorizer
_transformers = importlib.import_module("transformers")
AutoModel = _transformers.AutoModel
AutoTokenizer = _transformers.AutoTokenizer

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


class TfidfMicroCategoryRetriever:
	"""TF-IDF retriever over key phrases with bi/tri-grams for candidate filtering."""

	def __init__(
		self,
		ngram_range: Tuple[int, int] = (2, 3),
		min_similarity: float = 0.08,
		top_k: int = 8,
	) -> None:
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
		target_mc_ids: Sequence[int],
		optimizer: Any,
	) -> float:
		if len(texts) != len(candidate_mc_ids) or len(texts) != len(target_mc_ids):
			raise ValueError("texts, candidate_mc_ids and target_mc_ids must have equal lengths")

		self.model.train()
		total_loss = torch.tensor(0.0, device=self.device)

		for text, candidates, target_mc_id in zip(texts, candidate_mc_ids, target_mc_ids):
			if not candidates:
				continue

			cand_indices = torch.tensor(
				[self.id_to_idx[mc_id] for mc_id in candidates],
				dtype=torch.long,
				device=self.device,
			)
			text_emb = self.model.encode_text([text], self.device)
			logits = self.model(text_emb, cand_indices).squeeze(0)

			if target_mc_id not in candidates:
				continue
			target_idx = torch.tensor(candidates.index(target_mc_id), dtype=torch.long, device=self.device)

			loss = nn.functional.cross_entropy(logits.unsqueeze(0), target_idx.unsqueeze(0))
			total_loss = total_loss + loss

		optimizer.zero_grad(set_to_none=True)
		total_loss.backward()
		optimizer.step()
		return float(total_loss.detach().cpu().item())

	@torch.no_grad()
	def score_candidates(self, description: str, candidate_mc_ids: Sequence[int]) -> Dict[int, float]:
		if not candidate_mc_ids:
			return {}

		self.model.eval()
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

