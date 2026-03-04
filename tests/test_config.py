import pytest
from pathlib import Path


def test_load_default_config():
    from literature_rag_mcp.config import load_config
    config = load_config()
    assert config["source"]["type"] == "zotero_local"
    assert config["embeddings"]["type"] == "sentence_transformers"


def test_load_config_with_overrides():
    from literature_rag_mcp.config import load_config
    config = load_config({"source": {"type": "zotero_api"}})
    assert config["source"]["type"] == "zotero_api"


def test_config_is_fresh_copy():
    """Verify each call returns independent copy, not shared state."""
    from literature_rag_mcp.config import load_config
    config1 = load_config()
    config2 = load_config()
    config1["source"]["type"] = "modified"
    assert config2["source"]["type"] == "zotero_local"
