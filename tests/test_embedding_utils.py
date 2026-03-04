import pytest


def test_resolve_sentence_transformers():
    """Test sentence_transformers type returns model name."""
    from literature_rag_mcp.embedding_utils import resolve_embedding_config

    cfg = {"type": "sentence_transformers", "model": "all-MiniLM-L6-v2"}
    model, config = resolve_embedding_config(cfg)

    assert model == "all-MiniLM-L6-v2"
    assert config == {}


def test_resolve_openai_with_api_key():
    """Test openai type returns model and config with api_key/base_url."""
    from literature_rag_mcp.embedding_utils import resolve_embedding_config

    cfg = {
        "type": "openai",
        "model": "text-embedding-3-large",
        "api_key": "sk-test123",
        "base_url": "https://custom.example.com/v1",
    }
    model, config = resolve_embedding_config(cfg)

    assert model == "openai"
    assert config["model_name"] == "text-embedding-3-large"
    assert config["api_key"] == "sk-test123"
    assert config["base_url"] == "https://custom.example.com/v1"


def test_resolve_gemini_with_api_key():
    """Test gemini type returns model and config with api_key/base_url."""
    from literature_rag_mcp.embedding_utils import resolve_embedding_config

    cfg = {
        "type": "gemini",
        "model": "gemini-embedding-001",
        "api_key": "gemini-key",
        "base_url": "https://generativelanguage.googleapis.com",
    }
    model, config = resolve_embedding_config(cfg)

    assert model == "gemini"
    assert config["model_name"] == "gemini-embedding-001"
    assert config["api_key"] == "gemini-key"
    assert config["base_url"] == "https://generativelanguage.googleapis.com"


def test_resolve_unknown_type_fallback():
    """Test unknown type falls back to sentence_transformers."""
    from literature_rag_mcp.embedding_utils import resolve_embedding_config

    cfg = {"type": "unknown_provider", "model": "custom-model"}
    model, config = resolve_embedding_config(cfg)

    # Unknown type should fall back to sentence_transformers with provided model
    assert model == "custom-model"
    assert config == {}


def test_resolve_defaults_to_sentence_transformers():
    """Test missing type defaults to sentence_transformers."""
    from literature_rag_mcp.embedding_utils import resolve_embedding_config

    cfg = {}
    model, config = resolve_embedding_config(cfg)

    assert model == "all-MiniLM-L6-v2"
    assert config == {}
