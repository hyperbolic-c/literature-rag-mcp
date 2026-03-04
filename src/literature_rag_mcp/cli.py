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
