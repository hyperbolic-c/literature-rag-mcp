"""pdf2md: Convert Zotero PDF attachments to Markdown.

Provides:
  - :class:`~pdf2md.markitdown.MarkItDownConverter` — in-process PDF→MD via MarkItDown
  - :mod:`~pdf2md.converter` — batch PDF→MD via MinerU API
  - :mod:`~pdf2md.download_pdfs` — download missing PDFs from open-access sources
"""

from pdf2md.markitdown import MarkItDownConverter

__all__ = ["MarkItDownConverter"]
