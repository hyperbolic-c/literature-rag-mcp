"""Tests for parsers in prebuilt_md.py (MarkItDownParser and PrebuiltMdParser)."""

import pytest
from pathlib import Path
import tempfile


# ---------------------------------------------------------------------------
# PrebuiltMdParser tests (unchanged behaviour)
# ---------------------------------------------------------------------------


def test_prebuilt_md_get_markdown():
    """Test reading markdown files from prebuilt directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        attachment_dir = Path(tmpdir) / "ABC123"
        attachment_dir.mkdir()
        (attachment_dir / "document.md").write_text("# Test Title\n\nTest content.")

        from literature_rag_mcp.parsers.prebuilt_md import PrebuiltMdParser
        parser = PrebuiltMdParser(md_root=tmpdir)
        content = parser.get_markdown("ABC123")

        assert "Test Title" in content
        assert "Test content" in content


def test_prebuilt_md_no_dir():
    """Test when directory doesn't exist."""
    from literature_rag_mcp.parsers.prebuilt_md import PrebuiltMdParser
    parser = PrebuiltMdParser(md_root="/nonexistent/path")
    content = parser.get_markdown("ABC123")
    assert content == ""


# ---------------------------------------------------------------------------
# MarkItDownParser – storage / PDF-lookup logic (no markitdown call needed)
# ---------------------------------------------------------------------------


class TestMarkItDownParserStorageLookup:
    """Test PDF discovery logic without requiring markitdown."""

    def _make_parser(self, storage_path: str):
        from literature_rag_mcp.parsers.prebuilt_md import MarkItDownParser
        return MarkItDownParser(storage_path=storage_path)

    def test_get_markdown_missing_dir_returns_empty(self, tmp_path):
        parser = self._make_parser(str(tmp_path))
        result = parser.get_markdown("NONEXISTENT_KEY")
        assert result == ""

    def test_get_markdown_dir_exists_no_pdf_returns_empty(self, tmp_path):
        attachment_dir = tmp_path / "ABC123"
        attachment_dir.mkdir()
        (attachment_dir / "notes.txt").write_text("not a pdf")

        parser = self._make_parser(str(tmp_path))
        result = parser.get_markdown("ABC123")
        assert result == ""

    def test_find_pdf_returns_first_alphabetically(self, tmp_path):
        attachment_dir = tmp_path / "KEY1"
        attachment_dir.mkdir()
        (attachment_dir / "b_paper.pdf").write_bytes(b"%PDF-1.4 dummy")
        (attachment_dir / "a_paper.pdf").write_bytes(b"%PDF-1.4 dummy")

        parser = self._make_parser(str(tmp_path))
        found = parser._converter.find_pdf("KEY1")
        assert found is not None
        assert found.name == "a_paper.pdf"

    def test_parse_nonexistent_file_returns_empty(self, tmp_path):
        parser = self._make_parser(str(tmp_path))
        result = parser.parse(str(tmp_path / "does_not_exist.pdf"))
        assert result == ""

    def test_auto_detect_storage_path(self):
        """Parser should default to ~/Zotero/storage when no path given."""
        from literature_rag_mcp.parsers.prebuilt_md import MarkItDownParser
        parser = MarkItDownParser()
        expected = Path.home() / "Zotero" / "storage"
        assert parser._converter.storage_root == expected


# ---------------------------------------------------------------------------
# MarkItDownParser – integration tests (require markitdown[pdf])
# ---------------------------------------------------------------------------


class TestMarkItDownParserConversion:
    """Integration tests that actually call MarkItDown."""

    @pytest.fixture(autouse=True)
    def require_markitdown(self):
        pytest.importorskip("markitdown", reason="markitdown not installed")

    def test_parse_minimal_pdf(self, tmp_path):
        """parse() must return a str and never raise."""
        known_good_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
            b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            b"4 0 obj\n<< /Length 44 >>\nstream\n"
            b"BT /F1 12 Tf 72 720 Td (Hello World) Tj ET\n"
            b"endstream\nendobj\n"
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
            b"xref\n0 6\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000068 00000 n \n"
            b"0000000125 00000 n \n"
            b"0000000274 00000 n \n"
            b"0000000373 00000 n \n"
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n450\n%%EOF\n"
        )
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(known_good_pdf)

        from literature_rag_mcp.parsers.prebuilt_md import MarkItDownParser
        parser = MarkItDownParser(storage_path=str(tmp_path))
        result = parser.parse(str(pdf_file))
        assert isinstance(result, str)

    def test_get_markdown_with_pdf_in_storage(self, tmp_path):
        """get_markdown() should locate the PDF and attempt conversion."""
        attachment_dir = tmp_path / "MYKEY"
        attachment_dir.mkdir()
        (attachment_dir / "paper.pdf").write_bytes(b"%PDF-1.4 stub")

        from literature_rag_mcp.parsers.prebuilt_md import MarkItDownParser
        parser = MarkItDownParser(storage_path=str(tmp_path))
        result = parser.get_markdown("MYKEY")
        assert isinstance(result, str)
