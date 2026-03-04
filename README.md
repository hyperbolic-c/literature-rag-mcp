# literature-rag-mcp

A Model Context Protocol (MCP) server for literature RAG. Reads PDFs from Zotero and provides semantic search and Q&A capabilities.

## Features

- **Zotero Integration**: Read directly from local Zotero SQLite database
- **Multiple Parser Support**: Prebuilt MD files (MinerU support coming in v0.2)
- **Vector Search**: ChromaDB with sentence-transformers embeddings
- **RRank for improved searcheranking**: Flash results
- **MCP Tools**: `literature_search` and `literature_qa` tools for AI assistants
- **CLI**: Command-line interface for ingestion and status

## Installation

```bash
pip install literature-rag-mcp
```

Or install from source:

```bash
cd literature-rag-mcp
pip install -e .
```

## Quick Start

### 1. Configuration

Create `~/.config/literature-rag-mcp/config.json`:

```json
{
  "source": {
    "type": "zotero_local",
    "zotero_db_path": "",
    "storage_path": ""
  },
  "parser": {
    "type": "prebuilt_md",
    "md_root": "/path/to/your/md/files"
  },
  "embeddings": {
    "type": "sentence_transformers",
    "model": "all-MiniLM-L6-v2"
  },
  "chroma_db_path": "~/.config/literature-rag-mcp/chroma_db",
  "rag": {
    "retrieve": {
      "candidate_k": 30,
      "meta_weight": 0.7
    },
    "chunk": {
      "backend": "langchain",
      "strategy": "markdown_recursive_v1",
      "chunk_size": 1100,
      "chunk_overlap": 180
    },
    "reranker": {
      "enabled": true,
      "backend": "flashrank",
      "model_name": "ms-marco-MiniLM-L-12-v2",
      "top_n": 8
    }
  }
}
```

### 2. Ingest Documents

```bash
# Full ingestion
literature-rag ingest

# Force rebuild (clear existing index)
literature-rag ingest --rebuild

# Limit number of items
literature-rag ingest --limit 100
```

### 3. Check Status

```bash
literature-rag status
```

## MCP Tools

### literature_search

Semantic search across your literature library.

```python
# MCP tool call
literature_search(query: str, limit: int = 10, filters: dict = None)
```

### literature_qa

Get full text and relevant chunks for a specific item.

```python
# MCP tool call
literature_qa(item_key: str, question: str = "")
```

## Configuration Options

| Section | Option | Default | Description |
|---------|--------|---------|-------------|
| `source.type` | string | `zotero_local` | Data source type |
| `source.zotero_db_path` | string | auto-detect | Path to Zotero SQLite database |
| `source.storage_path` | string | auto-detect | Path to Zotero storage directory |
| `parser.type` | string | `prebuilt_md` | Parser type |
| `parser.md_root` | string | - | Root directory for MD files |
| `embeddings.type` | string | `sentence_transformers` | Embedding provider |
| `embeddings.model` | string | `all-MiniLM-L6-v2` | Embedding model name |
| `chroma_db_path` | string | `~/.config/literature-rag-mcp/chroma_db` | ChromaDB persistence path |
| `rag.retrieve.candidate_k` | int | 30 | Number of candidates to retrieve |
| `rag.retrieve.meta_weight` | float | 0.7 | Metadata weight in scoring |
| `rag.chunk.backend` | string | `langchain` | Chunking backend |
| `rag.chunk.strategy` | string | `markdown_recursive_v1` | Chunking strategy |
| `rag.chunk.chunk_size` | int | 1100 | Maximum chunk size |
| `rag.chunk.chunk_overlap` | int | 180 | Chunk overlap |
| `rag.reranker.enabled` | bool | true | Enable reranking |
| `rag.reranker.backend` | string | `flashrank` | Reranker backend |
| `rag.reranker.model_name` | string | `ms-marco-MiniLM-L-12-v2` | Reranker model |
| `rag.reranker.top_n` | int | 8 | Number of results after reranking |

## Architecture

```
literature-rag-mcp/
├── sources/           # Data source implementations
│   ├── base.py       # AbstractSource interface
│   └── zotero_local.py
├── parsers/          # Document parsers
│   ├── base.py       # AbstractParser interface
│   └── prebuilt_md.py
├── rag/              # RAG core components
│   ├── chunkers.py
│   ├── reranker.py
│   └── retriever.py
├── config.py         # Configuration management
├── cli.py            # CLI commands
└── server.py         # FastMCP server
```

## Supported Embedding Providers

- **sentence-transformers** (default): Local embeddings using HuggingFace models
- **openai**: OpenAI text-embedding-3-small
- **gemini**: Google Gemini embeddings

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
isort .
```

## License

MIT License
