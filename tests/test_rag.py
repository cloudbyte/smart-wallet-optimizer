import hashlib
import tiktoken
from src.rag.retriever import SmartWalletRAG
from src.rag.ingestion import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    IssuerDocumentLoader,
)


def test_retriever_filters_by_card_and_returns_no_unverified_fallback():
    retriever = SmartWalletRAG(
        documents=[
            {
                "card_id": "amex_gold",
                "document_date": "2025-01-01",
                "text": "Purchase Protection covers eligible purchases.",
                "title": "Amex test terms",
                "url": "https://example.test/amex.pdf",
                "page": 1,
                "fetched_at": "2026-01-01",
            }
        ]
    )

    evidence = retriever.query_benefits(
        "amex_gold", "electronics purchase protection"
    )
    unknown_card_evidence = retriever.query_benefits(
        "citi_double_cash", "purchase protection"
    )

    assert "Purchase Protection" in evidence
    assert "Extended Warranty" not in evidence
    assert unknown_card_evidence == ""


def test_embedding_retriever_uses_card_filter_and_mmr_ranking():
    class FakeEmbedder:
        vectors = {
            "electronics damage": [1.0, 0.0],
            "Purchase Protection covers new purchases for up to $10,000 per claim.": [
                1.0,
                0.0,
            ],
            "Extended Warranty Protection adds an additional year to the warranty.": [
                0.98,
                0.02,
            ],
        }

        def __call__(self, text):
            return self.vectors[text]

    texts = [
        "Purchase Protection covers new purchases for up to $10,000 per claim.",
        "Extended Warranty Protection adds an additional year to the warranty.",
    ]
    retriever = SmartWalletRAG(
        embedder=FakeEmbedder(),
        documents=[
            {
                "card_id": "chase_sapphire_reserve",
                "document_date": "2025-01-01",
                "text": text,
                "title": "Chase test terms",
                "url": "https://example.test/chase.pdf",
                "page": index,
                "fetched_at": "2026-01-01",
            }
            for index, text in enumerate(texts, start=1)
        ],
    )
    evidence = retriever.query_benefits(
        "chase_sapphire_reserve", "electronics damage"
    )

    assert "Purchase Protection" in evidence
    assert "Extended Warranty" in evidence
    assert "Amex" not in evidence


def test_retriever_ranks_newer_document_first_when_relevance_ties():
    retriever = SmartWalletRAG(
        documents=[
            {
                "card_id": "amex_gold",
                "document_date": "2022-01-01",
                "text": "Old purchase protection terms are deprecated.",
                "title": "Old policy",
                "url": "https://example.test/old.pdf",
                "page": 1,
                "fetched_at": "2026-01-01",
            },
            {
                "card_id": "amex_gold",
                "document_date": "2026-01-01",
                "text": "New purchase protection terms are current.",
                "title": "New policy",
                "url": "https://example.test/new.pdf",
                "page": 1,
                "fetched_at": "2026-01-01",
            },
        ]
    )

    evidence = retriever.query_benefits("amex_gold", "purchase protection")

    assert evidence.index("New purchase protection") < evidence.index(
        "Old purchase protection"
    )


def test_query_returns_source_url_and_page_citation():
    retriever = SmartWalletRAG(
        documents=[
            {
                "card_id": "chase_sapphire_reserve",
                "document_date": "2026-06-11",
                "text": "Purchase Protection coverage details for eligible items.",
                "title": "Chase Sapphire Reserve Guide",
                "url": "https://example.test/chase.pdf",
                "page": 18,
                "fetched_at": "2026-10-31",
            }
        ]
    )

    evidence = retriever.query_benefits(
        "chase_sapphire_reserve", "purchase protection"
    )

    assert "page 18" in evidence
    assert "https://example.test/chase.pdf" in evidence


def test_ingestion_splits_long_documents_into_passage_sized_chunks():
    text = " ".join(
        f"Sentence number {index} describes a distinct coverage term and condition."
        for index in range(80)
    )

    passages = IssuerDocumentLoader._split_passages(text)
    encoding = tiktoken.get_encoding("cl100k_base")
    tokenized = [encoding.encode(passage) for passage in passages]
    overlaps = [
        max(
            (
                len(encoding.encode(left[-overlap:]))
                for overlap in range(min(len(left), len(right)), 0, -1)
                if left[-overlap:] == right[:overlap]
            ),
            default=0,
        )
        for left, right in zip(passages, passages[1:])
    ]

    assert len(passages) > 1
    assert all(len(tokens) <= CHUNK_SIZE_TOKENS for tokens in tokenized)
    assert all(overlap >= CHUNK_OVERLAP_TOKENS - 5 for overlap in overlaps)
    assert "Sentence number 0" in passages[0]


def test_ingestion_classifies_required_vector_metadata_sections():
    assert (
        IssuerDocumentLoader._classify_section(
            "Foreign transaction fees apply to this purchase."
        )
        == "fees"
    )
    assert (
        IssuerDocumentLoader._classify_section(
            "Trip delay reimbursement applies to eligible travel."
        )
        == "travel_insurance"
    )
    assert (
        IssuerDocumentLoader._classify_section(
            "Purchase protection covers eligible items against damage."
        )
        == "purchase_protection"
    )


def test_html_source_is_loaded_from_cache_and_passages_keep_source_metadata(tmp_path):
    source_url = "https://issuer.example.test/benefits"
    source = {
        "card_id": "test_card",
        "title": "Test issuer benefits",
        "url": source_url,
        "allowed_hosts": ["issuer.example.test"],
        "format": "html",
    }
    cache_dir = tmp_path / "source-cache"
    cache_dir.mkdir()
    cache_file = cache_dir / (
        hashlib.sha256(source_url.encode()).hexdigest() + ".html"
    )
    cache_file.write_text(
        "<html><body><h1>Purchase Protection</h1>"
        "<p>Coverage applies to eligible purchases against accidental damage.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    loader = IssuerDocumentLoader(sources=[source], cache_dir=cache_dir)

    documents = loader.load_card_documents("test_card")

    assert documents
    assert documents[0]["card_id"] == "test_card"
    assert documents[0]["url"] == source_url
    assert "Coverage applies" in documents[0]["text"]


def test_local_embedding_cache_is_reused_between_retriever_instances(tmp_path):
    class FakeEmbeddingModel:
        embedding_model_id = "test/embedding-model"

        def __init__(self):
            self.calls = 0

        def embed(self, text):
            self.calls += 1
            return [1.0, 0.0] if "query" in text else [0.9, 0.1]

        def embed_many(self, texts):
            return [self.embed(text) for text in texts]

    documents = [
        {
            "card_id": "test_card",
            "document_date": "",
            "text": "Benefit text for search.",
            "title": "Test guide",
            "url": "https://issuer.example.test/guide.pdf",
            "page": 1,
            "fetched_at": "2026-01-01",
        }
    ]
    first_model = FakeEmbeddingModel()
    first = SmartWalletRAG(
        embedder=first_model.embed,
        documents=documents,
        cache_dir=tmp_path,
    )
    first.query_benefits("test_card", "query benefit")
    first_call_count = first_model.calls

    second_model = FakeEmbeddingModel()
    second = SmartWalletRAG(
        embedder=second_model.embed,
        documents=documents,
        cache_dir=tmp_path,
    )
    evidence = second.query_benefits("test_card", "query benefit")

    assert first_call_count > 0
    assert second_model.calls == 1
    assert (tmp_path / "chroma").is_dir()
    assert "Benefit text for search." in evidence


def test_vector_store_persists_card_and_section_metadata(tmp_path):
    class FakeEmbeddingModel:
        embedding_model_id = "test/persisted-vector-model"

        def embed(self, text):
            return [1.0, 0.0] if "purchase" in text.casefold() else [0.0, 1.0]

    documents = [
        {
            "card_id": "test_card",
            "document_date": "2026-01-01",
            "section": "purchase_protection",
            "text": "Purchase protection covers eligible purchases.",
            "title": "Test guide",
            "url": "https://issuer.example.test/guide.pdf",
            "page": 2,
            "fetched_at": "2026-01-01",
        }
    ]
    retriever = SmartWalletRAG(
        embedder=FakeEmbeddingModel().embed,
        documents=documents,
        cache_dir=tmp_path,
    )

    evidence = retriever.query_benefits("test_card", "purchase protection")

    assert "Purchase protection" in evidence
    assert "page 2" in evidence
    collection = retriever._get_collection()
    stored = collection.get(where={"card_id": "test_card"}, include=["metadatas"])
    assert stored["metadatas"][0]["section"] == "purchase_protection"
