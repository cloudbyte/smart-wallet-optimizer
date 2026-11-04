import hashlib
import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, cast

import chromadb

from src.rag.ingestion import SECTION_CATEGORIES, IssuerDocumentLoader


LOGGER = logging.getLogger(__name__)
MAX_PASSAGE_CHARS = 2000
MMR_CANDIDATE_MULTIPLIER = 5


class VectorCollection(Protocol):
    def get(
        self, *, where: Dict[str, str], include: List[str]
    ) -> Dict[str, object]: ...

    def upsert(
        self,
        *,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, object]],
    ) -> None: ...

    def delete(self, *, ids: List[str]) -> None: ...

    def query(
        self,
        *,
        query_embeddings: List[List[float]],
        n_results: int,
        where: Dict[str, str],
        include: List[str],
    ) -> Dict[str, object]: ...


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
        self.cache_dir = cache_dir or (
            Path(__file__).resolve().parents[2] / ".cache" / "card-benefits"
        )
        self._collection: Optional[VectorCollection] = None
        self._collection_name = self._get_collection_name()
        self._persistent_vectors = (
            cache_dir is not None
            or bool(getattr(getattr(embedder, "__self__", None), "embedding_model_id", None))
        )

    def query_benefits(self, card_id: str, query: str) -> str:
        candidates = [
            document
            for document in self._load_documents(card_id)
            if document["card_id"] == card_id
        ]
        if self.embedder:
            ranked = self._vector_search(query, card_id, candidates)
        else:
            ranked = self._lexical_search(query, candidates)
        return "\n\n".join(self._format_citation(document) for document in ranked)

    def _load_documents(self, card_id: str) -> List[Dict[str, object]]:
        if not self._explicit_documents and card_id not in self._loaded_card_ids:
            self.documents.extend(self.document_loader.load_card_documents(card_id))
            self._loaded_card_ids.add(card_id)
        return self.documents

    def _get_collection_name(self) -> str:
        owner = getattr(self.embedder, "__self__", None)
        model_id = getattr(owner, "embedding_model_id", None)
        identity = model_id or (
            f"{type(self.embedder).__module__}.{type(self.embedder).__qualname__}"
        )
        return "card_benefits_" + hashlib.sha256(identity.encode()).hexdigest()[:20]

    def _get_collection(self) -> VectorCollection:
        if self._collection is None:
            if self._persistent_vectors:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(
                    path=str(self.cache_dir / "chroma"),
                    settings=chromadb.config.Settings(anonymized_telemetry=False),
                )
            else:
                client = chromadb.EphemeralClient(
                    settings=chromadb.config.Settings(anonymized_telemetry=False)
                )
            self._collection = cast(
                VectorCollection,
                client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None,
                ),
            )
        return self._collection

    def _vector_search(
        self, query: str, card_id: str, documents: List[Dict[str, object]], k: int = 3
    ) -> List[Dict[str, object]]:
        if not documents:
            return []
        collection = self._get_collection()
        self._index_documents(collection, card_id, documents)
        query_vector = self._embed_texts([query])[0]
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(len(documents), max(k * MMR_CANDIDATE_MULTIPLIER, k)),
            where={"card_id": card_id},
            include=["documents", "embeddings", "metadatas"],
        )
        raw_documents = cast(List[List[Optional[str]]], result["documents"])[0]
        raw_embeddings = cast(List[List[List[float]]], result["embeddings"])[0]
        raw_metadatas = cast(
            List[List[Optional[Dict[str, object]]]], result["metadatas"]
        )[0]
        candidates = [
            (self._document_from_vector_store(text, metadata), embedding)
            for text, metadata, embedding in zip(
                raw_documents, raw_metadatas, raw_embeddings
            )
            if text is not None and metadata is not None
        ]
        return self._mmr_search(query_vector, candidates, k)

    def _index_documents(
        self,
        collection: VectorCollection,
        card_id: str,
        documents: List[Dict[str, object]],
    ) -> None:
        indexed = cast(
            List[str],
            collection.get(
                where={"card_id": card_id}, include=["metadatas"]
            )["ids"],
        )
        prepared = [
            (self._document_id(document), document)
            for document in documents
        ]
        desired_ids = {document_id for document_id, _ in prepared}
        stale_ids = [document_id for document_id in indexed if document_id not in desired_ids]
        if stale_ids:
            collection.delete(ids=stale_ids)

        existing_ids = set(indexed) - set(stale_ids)
        missing = [
            (document_id, document)
            for document_id, document in prepared
            if document_id not in existing_ids
        ]
        if not missing:
            return
        texts = [str(document["text"]) for _, document in missing]
        vectors = self._embed_texts(texts)
        collection.upsert(
            ids=[document_id for document_id, _ in missing],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                self._metadata(document) for _, document in missing
            ],
        )

    @staticmethod
    def _document_id(document: Dict[str, object]) -> str:
        identity = "\0".join(
            str(document.get(field, ""))
            for field in (
                "card_id",
                "section",
                "title",
                "url",
                "format",
                "page",
                "fetched_at",
                "document_date",
                "text",
            )
        )
        return hashlib.sha256(identity.encode()).hexdigest()

    @staticmethod
    def _metadata(document: Dict[str, object]) -> Dict[str, object]:
        metadata: Dict[str, object] = {
            "card_id": str(document["card_id"]),
            "section": str(
                document.get("section")
                if document.get("section") in SECTION_CATEGORIES
                else IssuerDocumentLoader._classify_section(str(document["text"]))
            ),
            "title": str(document.get("title", "")),
            "url": str(document.get("url", "")),
            "format": str(document.get("format", "")),
            "fetched_at": str(document.get("fetched_at", "")),
            "document_date": str(document.get("document_date", "")),
        }
        page = document.get("page")
        if isinstance(page, int):
            metadata["page"] = page
        return metadata

    @staticmethod
    def _document_from_vector_store(
        text: str, metadata: Dict[str, object]
    ) -> Dict[str, object]:
        return {
            **metadata,
            "text": text,
            "page": metadata.get("page"),
        }

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.embedder is None:
            raise RuntimeError("An embedding model is required for vector retrieval.")
        batch_embedder = getattr(
            getattr(self.embedder, "__self__", None), "embed_many", None
        )
        vectors = (
            batch_embedder(texts)
            if callable(batch_embedder)
            else [self.embedder(text) for text in texts]
        )
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding model returned an unexpected vector count.")
        return vectors

    @classmethod
    def _mmr_search(
        cls,
        query_vector: Sequence[float],
        candidates: List[tuple[Dict[str, object], Sequence[float]]],
        k: int,
    ) -> List[Dict[str, object]]:
        relevance = [
            cls._cosine(query_vector, vector) for _, vector in candidates
        ]
        selected: List[int] = []
        while len(selected) < min(k, len(candidates)):
            best_index = None
            best_score = float("-inf")
            for index, (_, vector) in enumerate(candidates):
                if index in selected:
                    continue
                redundancy = max(
                    (
                        cls._cosine(vector, candidates[chosen][1])
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
        return [candidates[index][0] for index in selected]

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
        page = f", page {document['page']}" if document.get("page") else ""
        date = (
            f"; document date {document['document_date']}"
            if document.get("document_date")
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
