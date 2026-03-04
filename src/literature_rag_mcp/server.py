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
