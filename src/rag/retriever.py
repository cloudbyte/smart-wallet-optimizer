import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from src.rag.ingestion import IssuerDocumentLoader


LOGGER = logging.getLogger(__name__)
MAX_PASSAGE_CHARS = 500


class SmartWalletRAG:
    def __init__(
        self,
        embedder: Optional[Callable[[str], List[float]]] = None,
        sources: Optional[List[Dict[str, object]]] = None,
        documents: Optional[List[Dict[str, object]]] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_seconds: int = 7 * 24 * 60 * 60,
    ):
        self.embedder = embedder
        self.documents = documents if documents is not None else []
        self._explicit_documents = documents is not None
        self._loaded_card_ids = set()
        self.document_loader = IssuerDocumentLoader(
            sources=sources,
            cache_dir=cache_dir,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self._embedding_cache: Dict[str, List[float]] = {}
        self._embedding_cache_loaded = False
        self._embedding_cache_path = self._get_embedding_cache_path(cache_dir)

    def query_benefits(self, card_id: str, query: str) -> str:
        candidates = [
            document
            for document in self._load_documents(card_id)
            if document["card_id"] == card_id
        ]
        if self.embedder:
            ranked = self._mmr_search(query, candidates)
        else:
            ranked = self._lexical_search(query, candidates)
        return "\n\n".join(self._format_citation(document) for document in ranked)

    def _load_documents(self, card_id: str) -> List[Dict[str, object]]:
        if not self._explicit_documents and card_id not in self._loaded_card_ids:
            self.documents.extend(self.document_loader.load_card_documents(card_id))
            self._loaded_card_ids.add(card_id)
        return self.documents

    def _get_embedding_cache_path(self, cache_dir: Optional[Path]) -> Optional[Path]:
        owner = getattr(self.embedder, "__self__", None)
        model_id = getattr(owner, "embedding_model_id", None)
        if not model_id:
            return None
        root = cache_dir or (
            Path(__file__).resolve().parents[2] / ".cache" / "card-benefits"
        )
        model_hash = hashlib.sha256(model_id.encode()).hexdigest()[:16]
        return root / f"embeddings-{model_hash}.json"

    def _load_embedding_cache(self) -> None:
        if self._embedding_cache_loaded:
            return
        self._embedding_cache_loaded = True
        if self._embedding_cache_path is None or not self._embedding_cache_path.exists():
            return
        try:
            with self._embedding_cache_path.open(encoding="utf-8") as cache_file:
                cached = json.load(cache_file)
            if not isinstance(cached, dict) or not all(
                isinstance(key, str)
                and isinstance(vector, list)
                and all(isinstance(value, (int, float)) for value in vector)
                for key, vector in cached.items()
            ):
                raise ValueError("Embedding cache has an invalid structure.")
            self._embedding_cache = cached
        except (OSError, json.JSONDecodeError, ValueError) as error:
            LOGGER.warning(
                "Ignoring invalid local embedding cache %s: %s",
                self._embedding_cache_path,
                error,
            )
            self._embedding_cache = {}

    def _save_embedding_cache(self) -> None:
        if self._embedding_cache_path is None:
            return
        self._embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._embedding_cache_path.with_suffix(".tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as cache_file:
                json.dump(self._embedding_cache, cache_file)
            temporary_path.replace(self._embedding_cache_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _embeddings(self, texts: List[str]) -> List[List[float]]:
        self._load_embedding_cache()
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        missing_by_key = {
            key: text
            for key, text in zip(keys, texts)
            if key not in self._embedding_cache
        }
        missing = list(missing_by_key.items())
        batch_embedder = getattr(
            getattr(self.embedder, "__self__", None), "embed_many", None
        )
        if callable(batch_embedder) and missing:
            vectors = batch_embedder([text for _, text in missing])
        else:
            vectors = [self.embedder(text) for _, text in missing]
        if len(vectors) != len(missing):
            raise RuntimeError("Embedding model returned an unexpected vector count.")
        self._embedding_cache.update(
            (key, vector) for (key, _), vector in zip(missing, vectors)
        )
        if missing:
            self._save_embedding_cache()
        return [self._embedding_cache[key] for key in keys]

    def _mmr_search(
        self, query: str, candidates: List[Dict[str, object]], k: int = 3
    ) -> List[Dict[str, object]]:
        if not candidates:
            return []
        all_vectors = self._embeddings(
            [query, *(str(document["text"]) for document in candidates)]
        )
        query_vector, candidate_vectors = all_vectors[0], all_vectors[1:]
        relevance = [self._cosine(query_vector, vector) for vector in candidate_vectors]
        selected: List[int] = []
        while len(selected) < min(k, len(candidates)):
            best_index = None
            best_score = float("-inf")
            for index in range(len(candidates)):
                if index in selected:
                    continue
                redundancy = max(
                    (
                        self._cosine(candidate_vectors[index], candidate_vectors[chosen])
                        for chosen in selected
                    ),
                    default=0.0,
                )
                score = 0.75 * relevance[index] - 0.25 * redundancy
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index is None or relevance[best_index] <= 0.0:
                break
            selected.append(best_index)
        return [candidates[index] for index in selected]

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _lexical_search(
        self, query: str, candidates: List[Dict[str, object]], k: int = 3
    ) -> List[Dict[str, object]]:
        query_terms = set(self._terms(query))
        matches = []
        for document in candidates:
            document_terms = set(self._terms(str(document["text"])))
            overlap = len(query_terms & document_terms)
            if overlap:
                matches.append((overlap, str(document["document_date"]), document))
        matches.sort(key=lambda match: (match[0], match[1]), reverse=True)
        return [match[2] for match in matches[:k]]

    @staticmethod
    def _format_citation(document: Dict[str, object]) -> str:
        page = f", page {document['page']}" if document["page"] else ""
        date = (
            f"; document date {document['document_date']}"
            if document["document_date"]
            else ""
        )
        text = str(document["text"])
        if len(text) > MAX_PASSAGE_CHARS:
            text = text[:MAX_PASSAGE_CHARS].rsplit(" ", 1)[0].rstrip() + "..."
        return (
            f"[Source: {document['title']}{page}{date}; fetched "
            f"{document['fetched_at']}; {document['url']}]\n{text}"
        )

    @staticmethod
    def _terms(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.casefold())
