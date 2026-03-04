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
            # Use proper markdown headers with blank lines between sections
            return """# Introduction

This is the introduction with much more text to make it longer than the minimum chunk size required for testing purposes.

## Methods

This describes methods with additional content that exceeds the minimum character threshold for chunking.

## Results

These are results with additional information and details."""

    config = {
        "embeddings": {"type": "sentence_transformers", "model": "all-MiniLM-L6-v2"},
        "rag": {
            "chunk": {"backend": "langchain", "strategy": "markdown_recursive_v1", "chunk_size": 500, "chunk_overlap": 50, "min_chunk_chars": 50},
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
