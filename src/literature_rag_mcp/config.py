"""Configuration management for literature-rag-mcp."""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


def _default_config() -> dict[str, Any]:
    """Return fresh default config dict (call to get fresh copy)."""
    return {
        "source": {
            "type": "zotero_local",
            "zotero_db_path": "",
            "storage_path": "",
        },
        "parser": {
            "type": "prebuilt_md",
            "md_root": "",
        },
        "embeddings": {
            "type": "sentence_transformers",
            "model": "all-MiniLM-L6-v2",
        },
        "chroma_db_path": "~/.config/literature-rag-mcp/chroma_db",
        "rag": {
            "retrieve": {
                "candidate_k": 30,
                "meta_weight": 0.70,
            },
            "chunk": {
                "backend": "langchain",
                "strategy": "markdown_recursive_v1",
                "chunk_size": 1100,
                "chunk_overlap": 180,
            },
            "reranker": {
                "enabled": True,
                "backend": "flashrank",
                "model_name": "ms-marco-MiniLM-L-12-v2",
                "top_n": 8,
            },
        },
    }


def load_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load configuration with defaults and user overrides.

    Returns a fresh copy each time to avoid state pollution.
    """
    config = deepcopy(_default_config())

    # Load from config file if exists
    config_path = Path("~/.config/literature-rag-mcp/config.json").expanduser()
    if config_path.exists():
        with open(config_path) as f:
            user_config = json.load(f)
            _deep_merge(config, user_config)

    # Apply runtime overrides
    if overrides:
        _deep_merge(config, overrides)

    return config


def _deep_merge(base: dict, override: dict) -> None:
    """Deep merge override into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
