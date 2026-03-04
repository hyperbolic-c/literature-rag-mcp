# literature-rag-mcp Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 创建一个独立的文献 RAG MCP 工具，从 Zotero 读取 PDF 并向量化，支持语义搜索和单篇问答。

**Architecture:** 新项目采用可插拔架构：数据源层（Zotero 本地/API）+ 解析层（MinerU/已有MD）+ Embedding层（本地/API）+ RAG核心。RAG核心从现有 zotero-mcp 迁移。

**Tech Stack:** Python, FastMCP, ChromaDB, sentence-transformers, FlashRank

**前置决策（已确认）：**
- Embedding 由 ChromaDB 内置函数负责（通过 embedding_model + embedding_config 参数传入）
- 首版只支持 prebuilt_md，不含 MinerU（延后到 v0.2）

---

## Phase 1: 项目基础结构搭建

### Task 1: 创建项目骨架和 pyproject.toml

**Files:**
- Create: `literature-rag-mcp/pyproject.toml`
- Create: `literature-rag-mcp/README.md`

**Step 1: 创建目录结构**

```bash
mkdir -p literature-rag-mcp/src/literature_rag_mcp/{sources,parsers,embeddings,rag}
mkdir -p literature-rag-mcp/tests
```

**Step 2: 编写 pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "literature-rag-mcp"
dynamic = ["version"]
authors = [{ name = "Liam", email = "liam@example.com" }]
description = "A Model Context Protocol server for literature RAG"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.2.0",
    "fastmcp>=2.14.0",
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
    "flashrank>=0.2.10",
    "langchain-text-splitters>=0.3",
    "langchain-community>=0.3",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0.0", "black>=23.0.0", "isort>=5.12.0"]

[project.scripts]
literature-rag = "literature_rag_mcp.cli:main"

[tool.hatch.version]
path = "src/literature_rag_mcp/_version.py"

[tool.hatch.build.targets.wheel]
packages = ["src/literature_rag_mcp"]
```

**Step 3: 创建基础 __init__.py 文件**

```bash
touch literature-rag-mcp/src/literature_rag_mcp/__init__.py
touch literature-rag-mcp/src/literature_rag_mcp/sources/__init__.py
touch literature-rag-mcp/src/literature_rag_mcp/parsers/__init__.py
touch literature-rag-mcp/src/literature_rag_mcp/embeddings/__init__.py
touch literature-rag-mcp/src/literature_rag_mcp/rag/__init__.py
```

**Step 4: 提交**

```bash
git add literature-rag-mcp/
git commit -m "feat: create literature-rag-mcp project skeleton"
```

---

### Task 2: 配置加载模块

**Files:**
- Create: `literature-rag-mcp/src/literature_rag_mcp/config.py`

**Step 1: 编写测试**

```python
# tests/test_config.py
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
```

**Step 2: 运行测试验证失败**

```bash
cd literature-rag-mcp && pytest tests/test_config.py -v
# Expected: FAIL - ModuleNotFoundError
```

**Step 3: 编写实现**

```python
# src/literature_rag_mcp/config.py
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
```

**Step 4: 运行测试验证通过**

```bash
cd literature-rag-mcp && pytest tests/test_config.py -v
# Expected: PASS
```

**Step 5: 提交**

```bash
git add literature-rag-mcp/src/literature_rag_mcp/config.py literature-rag-mcp/tests/test_config.py
git commit -m "feat: add configuration management with fresh copy"
```

---

## Phase 2: 数据源层实现

### Task 3: AbstractSource 接口和 Zotero 本地实现（修正版）

**Files:**
- Create: `literature-rag-mcp/src/literature_rag_mcp/sources/base.py`
- Create: `literature-rag-mcp/src/literature_rag_mcp/sources/zotero_local.py`

**Step 1: 编写测试**

```python
# tests/sources/test_zotero_local.py
import pytest
from dataclasses import dataclass


def test_abstract_source_interface():
    from literature_rag_mcp.sources.base import AbstractSource
    assert hasattr(AbstractSource, 'get_items')
    assert hasattr(AbstractSource, 'get_item_by_key')
    assert hasattr(AbstractSource, 'get_attachments')


def test_zotero_local_implements_interface():
    from literature_rag_mcp.sources.zotero_local import ZoteroLocalSource
    from literature_rag_mcp.sources.base import AbstractSource
    assert issubclass(ZoteroLocalSource, AbstractSource)
```

**Step 2: 运行测试验证失败**

```bash
cd literature-rag-mcp && pytest tests/sources/test_zotero_local.py -v
# Expected: FAIL
```

**Step 3: 编写 AbstractSource 接口**

```python
# src/literature_rag_mcp/sources/base.py
"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SourceItem:
    """Represents an item from a data source."""
    key: str
    title: str
    item_type: str
    creators: str = ""
    abstract: str = ""
    doi: str = ""
    date: str = ""


@dataclass
class SourceAttachment:
    """Represents an attachment (PDF) for an item."""
    key: str
    parent_key: str
    filename: str
    content_type: str
    path: str = ""


class AbstractSource(ABC):
    """Abstract interface for data sources."""

    @abstractmethod
    def get_items(self, limit: int | None = None) -> list[SourceItem]:
        """Get items from the source."""
        pass

    @abstractmethod
    def get_item_by_key(self, key: str) -> SourceItem | None:
        """Get a single item by key."""
        pass

    @abstractmethod
    def get_attachments(self, item_key: str) -> list[SourceAttachment]:
        """Get attachments for an item."""
        pass
```

**Step 4: 编写 ZoteroLocalSource 实现（完全复用 local_db.py 的 SQL 模式）**

```python
# src/literature_rag_mcp/sources/zotero_local.py
"""Zotero local SQLite source implementation.

完全复用 zotero-mcp/local_db.py 的 SQL 查询模式，确保 schema 兼容性。
Schema: itemData + itemDataValues + fields 组合（fieldID=1=title, 2=abstract, 16=extra, DOI via fields table）
"""

import platform
import sqlite3
from pathlib import Path

from .base import AbstractSource, SourceItem, SourceAttachment


class ZoteroLocalSource(AbstractSource):
    """Read items from local Zotero SQLite database."""

    def __init__(self, db_path: str = "", storage_path: str = ""):
        self.db_path = db_path or self._find_zotero_db()
        self.storage_path = storage_path or self._get_storage_dir()
        self._connection: sqlite3.Connection | None = None

    def _find_zotero_db(self) -> str:
        """Auto-detect Zotero database location."""
        system = platform.system()
        if system == "Darwin":
            db_path = Path.home() / "Zotero" / "zotero.sqlite"
        elif system == "Windows":
            db_path = Path.home() / "Zotero" / "zotero.sqlite"
        else:
            db_path = Path.home() / "Zotero" / "zotero.sqlite"

        if not db_path.exists():
            raise FileNotFoundError(f"Zotero database not found at {db_path}")
        return str(db_path)

    def _get_storage_dir(self) -> Path:
        """Infer storage directory from database path."""
        return Path(self.db_path).parent / "storage"

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._connection is None:
            uri = f"file:{self.db_path}?immutable=1"
            self._connection = sqlite3.connect(uri, uri=True)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def get_items(self, limit: int | None = None) -> list[SourceItem]:
        """Get items from local database.

        完全复用 local_db.py:get_items_with_text() 的 SQL 模式。
        """
        conn = self._get_connection()

        # Query参考 local_db.py:398-450 - 使用 itemData + itemDataValues + fields 组合
        query = """
            SELECT i.key,
                   i.itemID,
                   it.typeName as item_type,
                   i.dateAdded,
                   i.dateModified,
                   title_val.value as title,
                   abstract_val.value as abstract,
                   extra_val.value as extra,
                   doi_val.value as doi,
                   GROUP_CONCAT(
                       CASE
                           WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL
                           THEN c.lastName || ', ' || c.firstName
                           WHEN c.lastName IS NOT NULL
                           THEN c.lastName
                           ELSE NULL
                       END, '; '
                   ) as creators
            FROM items i
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID

            -- Get title (fieldID = 1)
            LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
            LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID

            -- Get abstract (fieldID = 2)
            LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
            LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID

            -- Get extra field (fieldID = 16)
            LEFT JOIN itemData extra_data ON i.itemID = extra_data.itemID AND extra_data.fieldID = 16
            LEFT JOIN itemDataValues extra_val ON extra_data.valueID = extra_val.valueID

            -- Get DOI via fields table
            LEFT JOIN fields doi_f ON doi_f.fieldName = 'DOI'
            LEFT JOIN itemData doi_data ON i.itemID = doi_data.itemID AND doi_data.fieldID = doi_f.fieldID
            LEFT JOIN itemDataValues doi_val ON doi_data.valueID = doi_val.valueID

            -- Get creators
            LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
            LEFT JOIN creators c ON ic.creatorID = c.creatorID

            WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')

            GROUP BY i.itemID, i.key, i.itemTypeID, it.typeName, i.dateAdded, i.dateModified,
                     title_val.value, abstract_val.value, extra_val.value

            ORDER BY i.dateModified DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        items = []
        for row in conn.execute(query):
            creators = row["creators"] or ""
            # 去重：GROUP_CONCAT 可能产生重复
            unique_creators = "; ".join(dict.fromkeys(creators.split("; ")))
            items.append(SourceItem(
                key=row["key"],
                title=row["title"] or "",
                item_type=row["item_type"],
                creators=unique_creators,
                abstract=row["abstract"] or "",
                doi=row["doi"] or "",
                date=row["dateAdded"] or "",
            ))
        return items

    def get_item_by_key(self, key: str) -> SourceItem | None:
        """Get a single item by key."""
        conn = self._get_connection()

        query = """
            SELECT i.key,
                   it.typeName as item_type,
                   title_val.value as title,
                   abstract_val.value as abstract,
                   extra_val.value as extra,
                   doi_val.value as doi,
                   i.dateAdded,
                   GROUP_CONCAT(
                       CASE
                           WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL
                           THEN c.lastName || ', ' || c.firstName
                           WHEN c.lastName IS NOT NULL
                           THEN c.lastName
                           ELSE NULL
                       END, '; '
                   ) as creators
            FROM items i
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID

            LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
            LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID

            LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
            LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID

            LEFT JOIN itemData extra_data ON i.itemID = extra_data.itemID AND extra_data.fieldID = 16
            LEFT JOIN itemDataValues extra_val ON extra_data.valueID = extra_val.valueID

            LEFT JOIN fields doi_f ON doi_f.fieldName = 'DOI'
            LEFT JOIN itemData doi_data ON i.itemID = doi_data.itemID AND doi_data.fieldID = doi_f.fieldID
            LEFT JOIN itemDataValues doi_val ON doi_data.valueID = doi_val.valueID

            LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
            LEFT JOIN creators c ON ic.creatorID = c.creatorID

            WHERE i.key = ?

            GROUP BY i.itemID, i.key, it.typeName, title_val.value, abstract_val.value, extra_val.value
        """

        for row in conn.execute(query, (key,)):
            creators = row["creators"] or ""
            unique_creators = "; ".join(dict.fromkeys(creators.split("; ")))
            return SourceItem(
                key=row["key"],
                title=row["title"] or "",
                item_type=row["item_type"],
                creators=unique_creators,
                abstract=row["abstract"] or "",
                doi=row["doi"] or "",
                date=row["dateAdded"] or "",
            )
        return None

    def get_attachments(self, item_key: str) -> list[SourceAttachment]:
        """Get attachments for an item.

        完全复用 local_db.py:_iter_parent_attachments() 的 SQL 模式。
        """
        conn = self._get_connection()

        # First get the item's itemID from key
        item_id_query = "SELECT itemID FROM items WHERE key = ?"
        row = conn.execute(item_id_query, (item_key,)).fetchone()
        if not row:
            return []
        parent_item_id = row["itemID"]

        # Query attachments (参考 local_db.py:147)
        query = """
            SELECT ia.itemID as attachmentItemID,
                   ia.parentItemID as parentItemID,
                   ia.path as path,
                   ia.contentType as contentType,
                   att.key as attachmentKey
            FROM itemAttachments ia
            JOIN items att ON att.itemID = ia.itemID
            WHERE ia.parentItemID = ?
        """

        attachments = []
        for row in conn.execute(query, (parent_item_id,)):
            attachments.append(SourceAttachment(
                key=row["attachmentKey"],
                parent_key=item_key,
                filename=row["path"] or "",
                content_type=row["contentType"] or "",
            ))
        return attachments
```

**Step 5: 运行测试验证通过**

```bash
cd literature-rag-mcp && pytest tests/sources/test_zotero_local.py -v
# Expected: PASS
```

**Step 6: 提交**

```bash
git add literature-rag-mcp/src/literature_rag_mcp/sources/
git commit -m "feat: add Zotero local source implementation with correct schema"
```

---

## Phase 3: 解析器层实现

### Task 4: AbstractParser 接口和 PrebuiltMD 实现

**Files:**
- Create: `literature-rag-mcp/src/literature_rag_mcp/parsers/base.py`
- Create: `literature-rag-mcp/src/literature_rag_mcp/parsers/prebuilt_md.py`

**Step 1: 编写测试**

```python
# tests/parsers/test_prebuilt_md.py
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
```

**Step 2: 运行测试验证失败**

```bash
cd literature-rag-mcp && pytest tests/parsers/test_prebuilt_md.py -v
# Expected: FAIL
```

**Step 3: 编写 AbstractParser 接口**

```python
# src/literature_rag_mcp/parsers/base.py
"""Abstract base class for document parsers."""

from abc import ABC, abstractmethod


class AbstractParser(ABC):
    """Abstract interface for document parsers."""

    @abstractmethod
    def get_markdown(self, attachment_key: str) -> str:
        """Get markdown content for an attachment."""
        pass

    @abstractmethod
    def parse(self, pdf_path: str) -> str:
        """Parse a PDF file and return markdown."""
        pass
```

**Step 4: 编写 PrebuiltMdParser 实现**

```python
# src/literature_rag_mcp/parsers/prebuilt_md.py
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
```

**Step 5: 运行测试验证通过**

```bash
cd literature-rag-mcp && pytest tests/parsers/test_prebuilt_md.py -v
# Expected: PASS
```

**Step 6: 提交**

```bash
git add literature-rag-mcp/src/literature_rag_mcp/parsers/
git commit -m "feat: add prebuilt MD parser"
```

---

## Phase 4: RAG 核心迁移

### Task 5: 迁移 chroma_client

**Files:**
- Copy: `zotero-mcp/src/zotero_mcp/chroma_client.py` → `literature-rag-mcp/src/literature_rag_mcp/chroma_client.py`

**Step 1: 复制文件**

```bash
cp zotero-mcp/src/zotero_mcp/chroma_client.py literature-rag-mcp/src/literature_rag_mcp/
```

**Step 2: 验证导入**

```bash
cd literature-rag-mcp && python -c "from literature_rag_mcp.chroma_client import ChromaClient; print('OK')"
```

**Step 3: 提交**

```bash
git add literature-rag-mcp/src/literature_rag_mcp/chroma_client.py
git commit -m "feat: migrate chroma_client from zotero-mcp"
```

---

### Task 6: 迁移 chunkers, reference_parser, reranker

**Files:**
- Copy: `zotero-mcp/src/zotero_mcp/rag/chunkers.py` → `literature-rag-mcp/src/literature_rag_mcp/rag/chunkers.py`
- Copy: `zotero-mcp/src/zotero_mcp/rag/chunker_types.py` → `literature-rag-mcp/src/literature_rag_mcp/rag/chunker_types.py`
- Copy: `zotero-mcp/src/zotero_mcp/rag/reference_parser.py` → `literature-rag-mcp/src/literature_rag_mcp/rag/reference_parser.py`
- Copy: `zotero-mcp/src/zotero_mcp/rag/reranker.py` → `literature-rag-mcp/src/literature_rag_mcp/rag/reranker.py`

**Step 1: 复制文件**

```bash
cp zotero-mcp/src/zotero_mcp/rag/chunkers.py literature-rag-mcp/src/literature_rag_mcp/rag/
cp zotero-mcp/src/zotero_mcp/rag/chunker_types.py literature-rag-mcp/src/literature_rag_mcp/rag/
cp zotero-mcp/src/zotero_mcp/rag/reference_parser.py literature-rag-mcp/src/literature_rag_mcp/rag/
cp zotero-mcp/src/zotero_mcp/rag/reranker.py literature-rag-mcp/src/literature_rag_mcp/rag/
```

**Step 2: 验证导入**

```bash
cd literature-rag-mcp && python -c "from literature_rag_mcp.rag.chunkers import get_chunking_backend; print('OK')"
```

**Step 3: 提交**

```bash
git add literature-rag-mcp/src/literature_rag_mcp/rag/chunkers.py literature-rag-mcp/src/literature_rag_mcp/rag/chunker_types.py literature-rag-mcp/src/literature_rag_mcp/rag/reference_parser.py literature-rag-mcp/src/literature_rag_mcp/rag/reranker.py
git commit -m "feat: migrate chunkers, reference_parser, reranker from zotero-mcp"
```

---

## Phase 5: 实现 RAG Facade (LiteratureRAGRetriever)

### Task 7: LiteratureRAGRetriever 实现（修正版）

**Files:**
- Create: `literature-rag-mcp/src/literature_rag_mcp/rag/retriever.py`

**Step 1: 编写测试（修正版：按真实配置层级构造）**

```python
# tests/rag/test_retriever.py
import pytest
import tempfile


def test_retriever_initialization():
    """Test retriever can be initialized with required components.

    配置层级必须与 server/cli 传入的实际配置一致：
    - config["rag"] 包含 retrieve/chunk/reranker
    - config["embeddings"] 在顶层
    """
    from literature_rag_mcp.rag.retriever import LiteratureRAGRetriever

    # 真实配置层级（server/cli 传入的形状）
    config = {
        "embeddings": {"model": "all-MiniLM-L6-v2"},
        "rag": {
            "chunk": {"backend": "langchain", "strategy": "markdown_recursive_v1"},
            "reranker": {"enabled": False},
            "retrieve": {"candidate_k": 30},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        retriever = LiteratureRAGRetriever(
            chroma_path=tmpdir,
            config=config,
        )
        assert retriever is not None
        # Verify ChromaClient was initialized with correct embedding_model parameter
        assert retriever.chroma_client.embedding_model == "all-MiniLM-L6-v2"
        # Verify config is read from rag subtree
        assert retriever.config.get("rag", {}).get("retrieve", {}).get("candidate_k") == 30


def test_retriever_ingest_and_search_integration():
    """Integration test: ingest documents then search.

    Verifies the full pipeline:
    1. Ingest creates chunks in ChromaDB
    2. Search retrieves chunks with correct candidate_k config
    """
    from literature_rag_mcp.rag.retriever import LiteratureRAGRetriever

    # Mock source and parser
    class MockSource:
        def get_items(self, limit=None):
            from literature_rag_mcp.sources.base import SourceItem
            return [
                SourceItem(
                    key="ITEM1",
                    title="Test Paper",
                    item_type="journalArticle",
                    creators="John Doe",
                    abstract="Test abstract",
                    doi="",
                    date="2024-01-01",
                )
            ]

        def get_item_by_key(self, key):
            from literature_rag_mcp.sources.base import SourceItem
            return SourceItem(
                key=key,
                title="Test Paper",
                item_type="journalArticle",
                creators="John Doe",
                abstract="Test abstract",
                doi="",
                date="2024-01-01",
            )

        def get_attachments(self, item_key):
            from literature_rag_mcp.sources.base import SourceAttachment
            return [SourceAttachment(key="ATT1", parent_key=item_key, filename="test.pdf", content_type="application/pdf")]

    class MockParser:
        def get_markdown(self, attachment_key):
            # Return markdown content that will be chunked
            return "# Introduction\n\nThis is the introduction. # Methods\n\nThis describes methods. # Results\n\nThese are results."

    config = {
        "embeddings": {"type": "sentence_transformers", "model": "all-MiniLM-L6-v2"},
        "rag": {
            "chunk": {"backend": "langchain", "strategy": "markdown_recursive_v1", "chunk_size": 100, "chunk_overlap": 20},
            "reranker": {"enabled": False},
            "retrieve": {"candidate_k": 5},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        retriever = LiteratureRAGRetriever(
            chroma_path=tmpdir,
            config=config,
            source=MockSource(),
            parser=MockParser(),
        )

        # Ingest
        result = retriever.ingest(force_rebuild=True)
        assert result["status"] == "success"
        assert result["chunks_indexed"] > 0

        # Search - verify candidate_k from rag.retrieve is used
        search_result = retriever.search(query="introduction methods", limit=3)
        assert search_result["status"] == "success"
        assert len(search_result["results"]) <= 3
```

**Step 2: 运行测试验证失败**

```bash
cd literature-rag-mcp && pytest tests/rag/test_retriever.py -v
# Expected: FAIL
```

**Step 3: 编写实现（修正版：使用 embedding_model 参数）**

```python
# src/literature_rag_mcp/rag/retriever.py
"""Literature RAG Retriever - Facade combining all RAG components."""

import logging
from pathlib import Path
from typing import Any, Optional

from literature_rag_mcp.chroma_client import ChromaClient
from literature_rag_mcp.rag.chunkers import get_chunking_backend
from literature_rag_mcp.rag.reranker import Reranker

logger = logging.getLogger(__name__)


class LiteratureRAGRetriever:
    """Facade combining sources, parsers, embedding, and RAG components."""

    def __init__(
        self,
        chroma_path: str,
        config: dict[str, Any],
        source=None,
        parser=None,
    ):
        self.chroma_path = Path(chroma_path).expanduser()
        self.config = config
        self.source = source
        self.parser = parser

        # Initialize ChromaDB client with embedding configuration
        # 使用共享函数解析 embedding 配置（避免 CLI 和 Retriever 重复代码）
        from literature_rag_mcp.embedding_utils import resolve_embedding_config
        embeddings_cfg = config.get("embeddings", {})
        embedding_model, embedding_config = resolve_embedding_config(embeddings_cfg)

        # 读取 rag 子树配置（server/cli 传入完整 config）
        rag_cfg = config.get("rag", {})

        self.chroma_client = ChromaClient(
            collection_name="literature_chunks_v1",
            persist_directory=str(self.chroma_path),
            embedding_model=embedding_model,
            embedding_config=embedding_config,  # 透传 embedding 配置
        )

        # Initialize components - 从 rag 子树读取
        self.reranker = Reranker(rag_cfg.get("reranker", {}))
        self.chunking_backend = get_chunking_backend(rag_cfg.get("chunk", {}))

    def ingest(
        self,
        force_rebuild: bool = False,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Ingest documents from source into vector store."""
        if force_rebuild:
            # Use reset_collection() instead of delete_collection()
            self.chroma_client.reset_collection()

        if not self.source or not self.parser:
            return {"status": "error", "message": "Source and parser required"}

        items = self.source.get_items(limit=limit)
        total_chunks = 0

        for item in items:
            attachments = self.source.get_attachments(item.key)
            for attachment in attachments:
                # Get markdown content
                md_content = self.parser.get_markdown(attachment.key)
                if not md_content:
                    continue

                # Chunk the content
                records = self.chunking_backend.chunk(md_content)

                # Prepare for ChromaDB
                docs = [r.text for r in records]
                metas = [
                    {
                        "item_key": item.key,
                        "attachment_key": attachment.key,
                        "title": item.title,
                        "creators": item.creators,
                        "chunk_index": r.chunk_index,
                    }
                    for r in records
                ]
                ids = [f"{item.key}:{attachment.key}:{r.chunk_index}" for r in records]

                if docs:
                    # Use upsert_documents() instead of upsert()
                    self.chroma_client.upsert_documents(ids=ids, documents=docs, metadatas=metas)
                    total_chunks += len(docs)

        return {
            "status": "success",
            "items_processed": len(items),
            "chunks_indexed": total_chunks,
        }

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search documents by semantic similarity."""
        # Read config from rag subtree: rag.retrieve.candidate_k
        rag_cfg = self.config.get("rag", {})
        retrieve_cfg = rag_cfg.get("retrieve", {})
        candidate_k = retrieve_cfg.get("candidate_k", 30)

        # Search ChromaDB (embedding handled internally via embedding_model)
        results = self.chroma_client.search(
            query_texts=[query],
            n_results=max(candidate_k, limit),
            where=filters,
        )

        # Format results
        items = []
        for i, doc_id in enumerate(results.get("ids", [[]])[0]):
            metadata = results.get("metadatas", [[{}]])[0][i]
            distance = results.get("distances", [[1.0]])[0][i]
            # Convert distance to similarity score (assumes cosine/l2, normalized 0-1)
            similarity = 1.0 / (1.0 + distance) if distance else 1.0

            items.append({
                "item_key": metadata.get("item_key", ""),
                "attachment_key": metadata.get("attachment_key", ""),
                "title": metadata.get("title", ""),
                "text": results.get("documents", [[]])[0][i],
                "score": similarity,
            })

        # Rerank if enabled - use correct config key: rag.reranker.enabled
        if rag_cfg.get("reranker", {}).get("enabled", True):
            from literature_rag_mcp.rag.reranker import CandidateChunk
            candidates = [
                CandidateChunk(
                    item_key=item["item_key"],
                    attachment_key=item["attachment_key"],
                    text=item["text"],
                    metadata=item,
                    similarity_score=item["score"],
                    rank_score=item["score"],
                )
                for item in items
            ]
            reranked = self.reranker.rerank(query, candidates)
            items = [
                {
                    "item_key": c.item_key,
                    "attachment_key": c.attachment_key,
                    "text": c.text,
                    "score": c.rank_score,
                }
                for c in reranked[:limit]
            ]

        return {
            "status": "success",
            "query": query,
            "results": items[:limit],
        }

    def get_item_fulltext(self, item_key: str) -> dict[str, Any]:
        """Get full text for a specific item."""
        if not self.source or not self.parser:
            return {"status": "error", "message": "Source and parser required"}

        item = self.source.get_item_by_key(item_key)
        if not item:
            return {"status": "error", "message": "Item not found"}

        attachments = self.source.get_attachments(item_key)

        full_text_parts = []
        for attachment in attachments:
            md_content = self.parser.get_markdown(attachment.key)
            if md_content:
                full_text_parts.append(md_content)

        full_text = "\n\n---\n\n".join(full_text_parts)

        return {
            "status": "success",
            "item_key": item_key,
            "title": item.title,
            "creators": item.creators,
            "abstract": item.abstract,
            "full_text": full_text,
        }
```

**Step 4: 运行测试验证通过**

```bash
cd literature-rag-mcp && pytest tests/rag/test_retriever.py -v
# Expected: PASS
```

**Step 5: 提交**

```bash
git add literature-rag-mcp/src/literature_rag_mcp/rag/retriever.py
git commit -m "feat: implement LiteratureRAGRetriever facade"
```

---

### Task 7b: 共享的 Embedding 配置解析函数

**Files:**
- Create: `literature-rag-mcp/src/literature_rag_mcp/embedding_utils.py`

**Step 1: 编写共享函数**

```python
# src/literature_rag_mcp/embedding_utils.py
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
```

**Step 1.5: 编写测试**

```python
# tests/test_embedding_utils.py
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
```

**Step 1.6: 运行测试验证通过**

```bash
cd literature-rag-mcp && pytest tests/test_embedding_utils.py -v
# Expected: PASS
```

**Step 2: 提交**

```bash
git add literature-rag-mcp/src/literature_rag_mcp/embedding_utils.py literature-rag-mcp/tests/test_embedding_utils.py
git commit -m "feat: add shared embedding config resolution with tests"
```

---

## Phase 6: MCP Server 实现

### Task 8: FastMCP Server 和两个工具

**Files:**
- Create: `literature-rag-mcp/src/literature_rag_mcp/server.py`

**Step 1: 编写实现**

```python
# src/literature_rag_mcp/server.py
"""FastMCP server for literature RAG."""

import logging
from typing import Any

from fastmcp import FastMCP

from literature_rag_mcp.config import load_config
from literature_rag_mcp.sources.zotero_local import ZoteroLocalSource
from literature_rag_mcp.parsers.prebuilt_md import PrebuiltMdParser
from literature_rag_mcp.rag.retriever import LiteratureRAGRetriever

logger = logging.getLogger(__name__)

mcp = FastMCP("literature-rag-mcp")

# Global retriever instance
_retriever: LiteratureRAGRetriever | None = None


def get_retriever() -> LiteratureRAGRetriever:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        config = load_config()

        # Initialize source
        source = ZoteroLocalSource(
            db_path=config["source"].get("zotero_db_path", ""),
            storage_path=config["source"].get("storage_path", ""),
        )

        # Initialize parser
        parser = PrebuiltMdParser(
            md_root=config["parser"].get("md_root", ""),
        )

        # 构建 retriever 配置（rag 子树 + embeddings）
        # embeddings 配置在顶层，用于 ChromaClient
        # rag 子树用于 retriever 内部配置
        retriever_config = {
            "embeddings": config.get("embeddings", {}),
            "rag": config.get("rag", {}),
        }

        # Initialize retriever (embedding handled internally by ChromaDB)
        _retriever = LiteratureRAGRetriever(
            chroma_path=config["chroma_db_path"],
            config=retriever_config,
            source=source,
            parser=parser,
        )

    return _retriever


@mcp.tool()
def literature_search(
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Semantic search across literature.

    Args:
        query: Search query string
        limit: Maximum number of results (default 10)
        filters: Optional metadata filters (e.g., {"year": 2024})

    Returns:
        List of relevant document chunks with metadata
    """
    retriever = get_retriever()
    return retriever.search(query=query, limit=limit, filters=filters)


@mcp.tool()
def literature_qa(
    item_key: str,
    question: str = "",
) -> dict[str, Any]:
    """Get full text and relevant chunks for a specific item.

    Args:
        item_key: Zotero item key
        question: Optional question to find relevant passages

    Returns:
        Full text of the item and relevant chunks
    """
    retriever = get_retriever()
    result = retriever.get_item_fulltext(item_key)

    # If question provided, also search within this item
    if question and result.get("status") == "success":
        search_result = retriever.search(
            query=question,
            limit=5,
            filters={"item_key": item_key},
        )
        result["relevant_chunks"] = search_result.get("results", [])

    return result


if __name__ == "__main__":
    mcp.run()
```

**Step 2: 更新 __init__.py**

```python
# src/literature_rag_mcp/__init__.py
"""literature-rag-mcp - A Model Context Protocol server for literature RAG."""

from . import server

__all__ = ["server"]
```

**Step 3: 提交**

```bash
git add literature-rag-mcp/src/literature_rag_mcp/server.py literature-rag-mcp/src/literature_rag_mcp/__init__.py
git commit -m "feat: add FastMCP server with literature_search and literature_qa tools"
```

---

## Phase 7: CLI 工具

### Task 9: CLI 命令 (ingest, status)

**Files:**
- Create: `literature-rag-mcp/src/literature_rag_mcp/cli.py`

**Step 1: 编写实现（修正版：status 使用相同 embedding 配置）**

```python
# src/literature_rag_mcp/cli.py
"""CLI for literature-rag-mcp."""

import argparse
import sys

from literature_rag_mcp.config import load_config
from literature_rag_mcp.sources.zotero_local import ZoteroLocalSource
from literature_rag_mcp.parsers.prebuilt_md import PrebuiltMdParser
from literature_rag_mcp.rag.retriever import LiteratureRAGRetriever


def ingest_command(args):
    """Ingest documents into the vector database."""
    config = load_config()

    source = ZoteroLocalSource(
        db_path=config["source"].get("zotero_db_path", ""),
        storage_path=config["source"].get("storage_path", ""),
    )

    parser = PrebuiltMdParser(
        md_root=config["parser"].get("md_root", ""),
    )

    # 构建 retriever 配置（与 server.py 一致）
    retriever_config = {
        "embeddings": config.get("embeddings", {}),
        "rag": config.get("rag", {}),
    }

    retriever = LiteratureRAGRetriever(
        chroma_path=config["chroma_db_path"],
        config=retriever_config,
        source=source,
        parser=parser,
    )

    result = retriever.ingest(force_rebuild=args.rebuild, limit=args.limit)

    print(f"Indexed {result.get('chunks_indexed', 0)} chunks from {result.get('items_processed', 0)} items")
    return 0


def status_command(args):
    """Show database status.

    使用与 ingest 完全相同的配置，避免 embedding 冲突导致重建 collection。
    """
    config = load_config()

    # 使用共享函数解析 embedding 配置
    from literature_rag_mcp.chroma_client import ChromaClient
    from literature_rag_mcp.embedding_utils import resolve_embedding_config

    embeddings_cfg = config.get("embeddings", {})
    embedding_model, embedding_config = resolve_embedding_config(embeddings_cfg)

    client = ChromaClient(
        collection_name="literature_chunks_v1",
        persist_directory=config["chroma_db_path"],
        embedding_model=embedding_model,
        embedding_config=embedding_config,  # 与 ingest 使用相同配置
    )

    count = client.collection.count()
    print(f"Total chunks indexed: {count}")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(prog="literature-rag")
    subparsers = parser.add_subparsers(dest="command")

    # ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    ingest_parser.add_argument("--limit", type=int, default=None, help="Limit number of items")
    ingest_parser.set_defaults(func=ingest_command)

    # status subcommand
    status_parser = subparsers.add_parser("status", help="Show database status")
    status_parser.set_defaults(func=status_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: 提交**

```bash
git add literature-rag-mcp/src/literature_rag_mcp/cli.py
git commit -m "feat: add CLI with ingest and status commands"
```

---

## Phase 8: 集成测试

### Task 10: 端到端测试

**Step 1: 安装开发版本**

```bash
cd literature-rag-mcp
pip install -e .
```

**Step 2: 运行 CLI**

```bash
literature-rag status
# Expected: Should show "Total chunks indexed: 0" (empty DB)
```

**Step 3: 测试 MCP 工具导入**

```bash
python -c "from literature_rag_mcp.server import mcp; print('MCP server loaded OK')"
```

**Step 4: 提交**

```bash
git add .
git commit -m "feat: complete literature-rag-mcp v0.1"
```

---

## 实现完成

所有任务完成后，项目结构如下：

```
literature-rag-mcp/
├── pyproject.toml
├── README.md
├── src/literature_rag_mcp/
│   ├── __init__.py
│   ├── _version.py
│   ├── config.py
│   ├── cli.py
│   ├── server.py
│   ├── chroma_client.py
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── zotero_local.py
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── prebuilt_md.py
│   └── rag/
│       ├── __init__.py
│       ├── chunkers.py
│       ├── chunker_types.py
│       ├── reference_parser.py
│       ├── reranker.py
│       └── retriever.py
└── tests/
```
