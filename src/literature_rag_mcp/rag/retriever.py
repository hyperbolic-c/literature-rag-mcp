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
