import pytest
from pathlib import Path
import tempfile


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
