"""Prebuilt Markdown parser - reads already-processed MD files."""

from pathlib import Path
import glob

from .base import AbstractParser


class PrebuiltMdParser(AbstractParser):
    """Read markdown files from a prebuilt directory structure.

    Expected structure:
        md_root/
            {attachment_key}/
                *.md
    """

    def __init__(self, md_root: str):
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
        """Not implemented for prebuilt - use MinerU parser instead."""
        raise NotImplementedError("Use MinerU parser for PDF conversion")
