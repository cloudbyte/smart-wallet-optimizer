import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from pypdf import PdfReader
import tiktoken


LOGGER = logging.getLogger(__name__)
MAX_SOURCE_BYTES = 20 * 1024 * 1024
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
SECTION_CATEGORIES = ("purchase_protection", "travel_insurance", "fees")


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, _attrs: list) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif tag in {"br", "p", "div", "li", "h1", "h2", "h3", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in {"p", "div", "li", "h1", "h2", "h3", "section"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            text = data.strip()
            if text:
                self.parts.extend((text, "\n"))

    def get_text(self) -> str:
        return re.sub(r"\n{2,}", "\n", " ".join(self.parts)).strip()


class IssuerDocumentLoader:
    def __init__(
        self,
        sources: Optional[List[Dict[str, object]]] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_seconds: int = 7 * 24 * 60 * 60,
    ):
        self.sources = sources if sources is not None else self._load_source_manifest()
        self.cache_dir = cache_dir or (
            Path(__file__).resolve().parents[2] / ".cache" / "card-benefits"
        )
        self.cache_ttl_seconds = cache_ttl_seconds

    def load_card_documents(self, card_id: str) -> List[Dict[str, object]]:
        documents: List[Dict[str, object]] = []
        for source in self.sources:
            if source["card_id"] != card_id:
                continue
            path, fetched_at = self._get_source_file(source)
            documents.extend(self._extract_chunks(source, path, fetched_at))
        return documents

    @staticmethod
    def _load_source_manifest() -> List[Dict[str, object]]:
        path = Path(__file__).resolve().parents[2] / "data" / "benefit_sources.json"
        with path.open(encoding="utf-8") as source_file:
            sources = json.load(source_file)
        if not isinstance(sources, list):
            raise ValueError(f"Issuer source manifest must contain a list: {path}")
        return sources

    def _get_source_file(self, source: Dict[str, object]) -> Tuple[Path, str]:
        url = str(source["url"])
        allowed_hosts = source["allowed_hosts"]
        requested_host = urlparse(url).hostname
        if (
            urlparse(url).scheme != "https"
            or not isinstance(allowed_hosts, list)
            or requested_host not in allowed_hosts
        ):
            raise ValueError(f"Untrusted issuer source URL: {url}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".pdf" if source["format"] == "pdf" else ".html"
        cache_path = self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}{suffix}"
        if (
            cache_path.exists()
            and time.time() - cache_path.stat().st_mtime < self.cache_ttl_seconds
        ):
            return cache_path, datetime.fromtimestamp(
                cache_path.stat().st_mtime, timezone.utc
            ).isoformat()

        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            try:
                with requests.get(
                    url,
                    headers={"User-Agent": "SmartWalletRewardsOptimizer/1.0"},
                    timeout=(10, 60),
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    final_host = urlparse(response.url).hostname
                    if final_host not in allowed_hosts:
                        raise ValueError(
                            f"Issuer source redirected to an untrusted host: "
                            f"{response.url}"
                        )
                    content_type = response.headers.get("Content-Type", "").casefold()
                    expected_type = (
                        "pdf" if source["format"] == "pdf" else "html"
                    )
                    if expected_type not in content_type:
                        raise ValueError(
                            f"Expected {expected_type} content from {url}, received "
                            f"{content_type or 'unknown content'}."
                        )

                    downloaded = 0
                    with temporary_path.open("wb") as output:
                        for block in response.iter_content(chunk_size=64 * 1024):
                            if not block:
                                continue
                            downloaded += len(block)
                            if downloaded > MAX_SOURCE_BYTES:
                                raise ValueError(
                                    f"Issuer source exceeds the "
                                    f"{MAX_SOURCE_BYTES}-byte limit: {url}"
                                )
                            output.write(block)
                if downloaded == 0:
                    raise ValueError(f"Issuer source returned an empty document: {url}")
                if source["format"] == "pdf":
                    with temporary_path.open("rb") as document_file:
                        if document_file.read(5) != b"%PDF-":
                            raise ValueError(f"Issuer source is not a valid PDF: {url}")
                temporary_path.replace(cache_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
            return cache_path, datetime.now(timezone.utc).isoformat()
        except requests.RequestException as error:
            if cache_path.exists():
                LOGGER.warning(
                    "Could not refresh issuer source %s; using the cached copy: %s",
                    url,
                    error,
                )
                return cache_path, datetime.fromtimestamp(
                    cache_path.stat().st_mtime, timezone.utc
                ).isoformat()
            raise RuntimeError(f"Could not download issuer terms from {url}: {error}") from error

    def _extract_chunks(
        self,
        source: Dict[str, object],
        path: Path,
        fetched_at: str,
    ) -> List[Dict[str, object]]:
        if source["format"] == "pdf":
            reader = PdfReader(str(path))
            pdf_created = (
                reader.metadata.creation_date.isoformat()
                if reader.metadata and reader.metadata.creation_date
                else ""
            )
            pages = [
                (page_number, page.extract_text() or "")
                for page_number, page in enumerate(reader.pages, start=1)
            ]
        elif source["format"] == "html":
            parser = _VisibleTextParser()
            parser.feed(path.read_text(encoding="utf-8", errors="replace"))
            pages = [(None, parser.get_text())]
            pdf_created = ""
        else:
            raise ValueError(f"Unsupported issuer document format: {source['format']}")

        complete_text = " ".join(text for _, text in pages)
        date_match = re.search(
            r"(?:in effect as of|applicable to purchases made on or after)"
            r"\s+(\d{1,2}/\d{1,2}/\d{2,4})",
            complete_text,
            flags=re.IGNORECASE,
        )
        document_date = date_match.group(1) if date_match else pdf_created
        documents: List[Dict[str, object]] = []
        for page_number, text in pages:
            for passage in self._split_passages(text):
                documents.append(
                    {
                        "card_id": source["card_id"],
                        "section": self._classify_section(passage),
                        "title": source["title"],
                        "url": source["url"],
                        "format": source["format"],
                        "page": page_number,
                        "fetched_at": fetched_at,
                        "document_date": document_date,
                        "text": passage,
                    }
                )
        return documents

    @staticmethod
    def _split_passages(text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(normalized)
        stride = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
        passages = [
            encoding.decode(tokens[start : start + CHUNK_SIZE_TOKENS]).strip()
            for start in range(0, len(tokens), stride)
        ]
        return [passage for passage in passages if len(passage) >= 40]

    @staticmethod
    def _classify_section(text: str) -> str:
        normalized = text.casefold()
        if re.search(
            r"\b(foreign transaction|annual fee|late fee|overdraft fee|"
            r"fee schedule|transaction fee)\b",
            normalized,
        ):
            return "fees"
        if re.search(
            r"\b(trip delay|trip cancellation|trip interruption|"
            r"baggage delay|lost luggage|travel accident|rental car)\b",
            normalized,
        ):
            return "travel_insurance"
        return "purchase_protection"
