"""Document parsers for literature-rag-mcp.

All parser implementations live here. Each class implements
:class:`.base.AbstractParser`.

Parsers
-------
PrebuiltMdParser
    Read pre-processed ``.md`` files from a directory tree (MinerU output).
MarkItDownParser
    Convert PDFs on-the-fly using ``pdf2md.markitdown.MarkItDownConverter``.
MinerUParser
    Convert PDFs using the MinerU API via ``pdf2md.converter`` (batch mode).
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

from .base import AbstractParser


# ---------------------------------------------------------------------------
# PrebuiltMdParser  (reads already-processed .md files)
# ---------------------------------------------------------------------------


class PrebuiltMdParser(AbstractParser):
    """Read markdown files from a prebuilt directory structure.

    Expected structure::

        md_root/
            {attachment_key}/
                *.md

    Typical use-case: MinerU output written to disk ahead of time.
    """

    def __init__(self, md_root: str) -> None:
        self.md_root = Path(md_root) if md_root else None

    def get_markdown(self, attachment_key: str) -> str:
        """Read all markdown files for an attachment key."""
        if not self.md_root or not self.md_root.exists():
            return ""

        pattern = str(self.md_root / attachment_key / "*.md")
        md_files = sorted(glob.glob(pattern))

        if not md_files:
            return ""

        parts = []
        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8", errors="ignore") as f:
                    parts.append(f.read())
            except Exception:
                continue

        return "\n\n".join(parts)

    def parse(self, pdf_path: str) -> str:
        """Not applicable — use :class:`MarkItDownParser` or :class:`MinerUParser`."""
        raise NotImplementedError(
            "PrebuiltMdParser reads pre-existing .md files; "
            "use MarkItDownParser or MinerUParser for on-the-fly PDF conversion."
        )


# ---------------------------------------------------------------------------
# MarkItDownParser  (in-process PDF → MD via markitdown)
# ---------------------------------------------------------------------------


class MarkItDownParser(AbstractParser):
    """Convert PDFs to Markdown in-process using :class:`pdf2md.markitdown.MarkItDownConverter`.

    Parameters
    ----------
    storage_path:
        Root of the Zotero storage directory (``~/Zotero/storage`` by default).
    """

    def __init__(self, storage_path: str = "") -> None:
        from pdf2md.markitdown import MarkItDownConverter
        self._converter = MarkItDownConverter(storage_root=storage_path)

    def get_markdown(self, attachment_key: str) -> str:
        """Find the PDF in Zotero storage and convert it to Markdown."""
        return self._converter.convert_attachment(attachment_key)

    def parse(self, pdf_path: str) -> str:
        """Convert the PDF at *pdf_path* to Markdown."""
        return self._converter.convert_file(pdf_path)


# ---------------------------------------------------------------------------
# MinerUParser  (HTTP API PDF → MD via MinerU)
# ---------------------------------------------------------------------------


class MinerUParser(AbstractParser):
    """Convert PDFs to Markdown via the MinerU HTTP API (``pdf2md.converter``).

    Requires a running MinerU server.  See ``pdf2md/converter.py`` and
    ``pdf2md/run.sh`` for setup instructions.

    Parameters
    ----------
    storage_path:
        Root of the Zotero storage directory.
    api_url:
        MinerU API base URL (default: ``http://localhost:8000``).
    lang:
        Language hint passed to the MinerU API (e.g. ``"en"``, ``"ch"``, ``"auto"``).
    """

    def __init__(
        self,
        storage_path: str = "",
        api_url: str = "http://localhost:8000",
        lang: str = "auto",
    ) -> None:
        self.storage_path = Path(storage_path).expanduser() if storage_path else Path.home() / "Zotero" / "storage"
        self.api_url = api_url
        self.lang = lang
        self._client: Any = None  # lazy httpx.Client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise ImportError("httpx is required for MinerUParser: pip install httpx") from exc
            self._client = httpx.Client(timeout=httpx.Timeout(5.0, read=300.0))
        return self._client

    def _find_pdf(self, attachment_key: str) -> Path | None:
        attachment_dir = self.storage_path / attachment_key
        if not attachment_dir.exists():
            return None
        pdf_files = sorted(attachment_dir.glob("*.pdf"))
        return pdf_files[0] if pdf_files else None

    def get_markdown(self, attachment_key: str) -> str:
        """Find the PDF in Zotero storage and convert it via the MinerU API."""
        pdf_path = self._find_pdf(attachment_key)
        if pdf_path is None:
            return ""
        return self.parse(str(pdf_path))

    def parse(self, pdf_path: str) -> str:
        """Send the PDF at *pdf_path* to the MinerU API and return Markdown."""
        from pdf2md.converter import submit_to_mineru
        import logging

        path = Path(pdf_path)
        if not path.exists():
            logging.warning("MinerUParser: PDF not found: %s", pdf_path)
            return ""

        pdf_bytes = path.read_bytes()
        client = self._get_client()
        result = submit_to_mineru(
            pdf_bytes=pdf_bytes,
            filename=path.name,
            api_url=self.api_url,
            lang=self.lang,
            backend="pipeline",
            client=client,
        )
        return result or ""
