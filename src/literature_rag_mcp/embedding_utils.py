"""Shared embedding configuration utilities.

统一 embedding 配置解析逻辑，避免 CLI 和 Retriever 重复代码。
"""

from typing import Any


def resolve_embedding_config(embeddings_cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Resolve embedding configuration for ChromaDB.

    Args:
        embeddings_cfg: The embeddings section of config

    Returns:
        Tuple of (embedding_model, embedding_config) for ChromaClient
    """
    embedding_type = embeddings_cfg.get("type", "sentence_transformers")

    if embedding_type == "sentence_transformers":
        model = embeddings_cfg.get("model", "all-MiniLM-L6-v2")
        return model, {}

    elif embedding_type == "openai":
        return "openai", {
            "model_name": embeddings_cfg.get("model", "text-embedding-3-small"),
            "api_key": embeddings_cfg.get("api_key"),
            "base_url": embeddings_cfg.get("base_url"),
        }

    elif embedding_type == "gemini":
        return "gemini", {
            "model_name": embeddings_cfg.get("model", "gemini-embedding-001"),
            "api_key": embeddings_cfg.get("api_key"),
            "base_url": embeddings_cfg.get("base_url"),
        }

    # Default fallback
    return embeddings_cfg.get("model", "all-MiniLM-L6-v2"), {}
