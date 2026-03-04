"""MarkItDown-based PDF converter for Zotero local storage.

Provides :class:`MarkItDownConverter` — a standalone helper that locates PDF
files inside Zotero's storage directory layout and converts them to Markdown
using the ``markitdown`` library.

Storage layout assumed::

    {storage_root}/{attachment_key}/{filename}.pdf

This module intentionally has **no dependency** on
``literature_rag_mcp``; it can be used independently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MarkItDownConverter:
    """Convert Zotero PDF attachments to Markdown using MarkItDown.

    Parameters
    ----------
    storage_root:
        Root of the Zotero storage directory, e.g. ``~/Zotero/storage``.
        Defaults to ``~/Zotero/storage`` when empty.
    """

    def __init__(self, storage_root: str = "") -> None:
        if storage_root:
            self.storage_root = Path(storage_root).expanduser()
        else:
            self.storage_root = Path.home() / "Zotero" / "storage"

        self._md: Any = None  # lazy-initialised MarkItDown instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_markitdown(self) -> Any:
        """Return a (cached) MarkItDown instance."""
        if self._md is None:
            try:
                from markitdown import MarkItDown
            except ImportError as exc:
                raise ImportError(
                    "markitdown[pdf] is required. "
                    "Install it with: pip install 'markitdown[pdf]'"
                ) from exc
            self._md = MarkItDown()
        return self._md

    def find_pdf(self, attachment_key: str) -> Path | None:
        """Return the first PDF found under ``storage_root/attachment_key/``.

        If multiple PDFs exist in the directory, the first in alphabetical
        order is returned.
        """
        attachment_dir = self.storage_root / attachment_key
        if not attachment_dir.exists():
            logger.debug("Attachment directory not found: %s", attachment_dir)
            return None

        pdf_files = list(attachment_dir.glob("*.pdf"))
        if not pdf_files:
            logger.debug("No PDF files in: %s", attachment_dir)
            return None

        return sorted(pdf_files)[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert_file(self, pdf_path: str | Path) -> str:
        """Convert the PDF at *pdf_path* to Markdown text.

        Returns an empty string if the file does not exist or conversion
        fails for any reason (exception is logged at WARNING level).
        """
        path = Path(pdf_path)
        if not path.exists():
            logger.warning("PDF not found: %s", pdf_path)
            return ""

        try:
            md = self._get_markitdown()
            result = md.convert(str(path))
            return result.text_content or ""
        except Exception as exc:
            logger.warning("MarkItDown conversion failed for %s: %s", pdf_path, exc)
            return ""

    def convert_attachment(self, attachment_key: str) -> str:
        """Locate the PDF for *attachment_key* and convert it to Markdown.

        Combines :meth:`find_pdf` and :meth:`convert_file`.
        Returns an empty string if the PDF cannot be found.
        """
        pdf_path = self.find_pdf(attachment_key)
        if pdf_path is None:
            return ""
        return self.convert_file(pdf_path)
