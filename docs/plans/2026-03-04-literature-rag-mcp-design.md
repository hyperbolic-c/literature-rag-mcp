# literature-rag-mcp 设计文档

**日期**：2026-03-04
**状态**：待实现

## 背景

`zotero-mcp` 当前集成了完整的 RAG 功能（`src/zotero_mcp/rag/`），但该功能与其他工具（BetterBibTeX、annotations、epub 处理等）耦合在同一个项目中。随着 RAG 成为核心使用路径，有必要将其独立为专注的轻量工具。

## 目标

将文献 RAG 核心流程独立为 `literature-rag-mcp`：

1. 连接 Zotero 读取文献条目及附件
2. 将 PDF 解析为 Markdown（可插拔解析器）
3. 基于 Markdown 进行向量化存储
4. 作为 MCP 工具暴露搜索和单篇问答接口

## 项目结构

```
literature-rag-mcp/
├── src/literature_rag_mcp/
│   ├── server.py              # FastMCP 入口，暴露 2 个 MCP 工具
│   ├── cli.py                 # ingest / status 命令
│   ├── config.py              # 配置加载
│   │
│   ├── sources/               # 数据源层（Zotero 连接）
│   │   ├── base.py            # AbstractSource 接口
│   │   ├── zotero_local.py    # 本地 SQLite 实现
│   │   └── zotero_api.py      # Web API 实现
│   │
│   ├── parsers/               # PDF 解析层
│   │   ├── base.py            # AbstractParser 接口
│   │   ├── mineru.py          # MinerU 调用实现
│   │   ├── pymupdf4llm.py     # pymupdf4llm 实现
│   │   └── prebuilt_md.py     # 直接读取已有 MD 文件目录
│   │
│   ├── embeddings/            # Embedding 层
│   │   ├── base.py            # AbstractEmbedder 接口
│   │   ├── sentence_transformers.py  # 默认本地模型
│   │   └── openai.py          # OpenAI API 实现
│   │
│   └── rag/                   # RAG 核心（从 zotero-mcp 迁移）
│       ├── chunkers.py
│       ├── reference_parser.py
│       ├── ingestor.py
│       ├── searcher.py
│       └── reranker.py
├── pyproject.toml
└── README.md
```

## MCP 工具接口

### 工具 1：`literature_search`

语义搜索，返回相关文献片段。

```
输入：
  - query: str           # 搜索查询
  - limit: int = 10      # 返回结果数
  - filters: dict        # 可选过滤条件（年份、作者等）

输出（每条结果）：
  - item_key: str
  - title: str
  - authors: list[str]
  - year: int
  - matched_chunks: list[{text, score}]
  - resolved_citations: list[str]
```

### 工具 2：`literature_qa`

单篇文献全文获取及问答支持。

```
输入：
  - item_key: str        # 文献标识符
  - question: str        # 针对该文献的问题

输出：
  - full_text: str       # 完整 Markdown 全文
  - relevant_chunks: list[{text, score}]  # 与问题最相关的片段
  - metadata: dict       # 标题、作者、DOI 等元数据
```

**设计说明**：`literature_qa` 返回全文 + 相关片段，由 Claude 在上下文中完成实际问答，MCP 工具只负责数据检索，不在工具层做 LLM 推理。

## 关键设计决策

### 数据源层（Sources）

- 本地 SQLite 优先，可降级到 Zotero Web API
- `AbstractSource` 接口统一两种模式，未来可扩展到普通 PDF 文件夹

### 解析层（Parsers）

| 实现 | 说明 |
|------|------|
| `prebuilt_md` | 直接读取已由 MinerU 等工具处理好的 MD 文件目录 |
| `mineru` | 调用 MinerU CLI 或 Python API 进行 PDF→MD 转换 |
| `pymupdf4llm` | 使用 pymupdf4llm 进行轻量 PDF→MD 转换 |

默认优先使用 `prebuilt_md`（性能最好），若指定路径不存在则自动降级到 `mineru`。

### Embedding 层

- 默认：本地 `sentence-transformers`（离线可用）
- 可配置切换到 OpenAI / Cohere 等 API 模型
- 向量存储：ChromaDB（与现有 `zotero-mcp` 一致）

### RAG 核心迁移策略

直接从 `zotero-mcp/src/zotero_mcp/rag/` 迁移以下模块，去除对 `zotero-mcp` 内部依赖：

- `chunkers.py`（无需修改）
- `reference_parser.py`（无需修改）
- `reranker.py`（无需修改）
- `ingestor.py`（替换数据源依赖为 `sources/` 层）
- `searcher.py`（替换数据源依赖为 `sources/` 层）

## 与现有 `zotero-mcp` 的关系

**完全独立部署**（推荐）：

- `zotero-mcp`：保留现有全部功能（annotations、BetterBibTeX、epub 等）
- `literature-rag-mcp`：专注 RAG 查询
- Claude Code 同时挂载两个 MCP 服务器，各司其职
- 两个项目共享同一个 ChromaDB 数据目录（可配置）

未来 `zotero-mcp` 中的 RAG 相关代码可选择性废弃，但不强制。

## 配置示例

```json
{
  "source": {
    "type": "zotero_local",
    "zotero_db_path": "~/Zotero/zotero.sqlite",
    "storage_path": "~/Zotero/storage"
  },
  "parser": {
    "type": "prebuilt_md",
    "md_root": "/path/to/mineru/output"
  },
  "embeddings": {
    "type": "sentence_transformers",
    "model": "all-MiniLM-L6-v2"
  },
  "chroma_db_path": "~/.config/literature-rag-mcp/chroma_db",
  "rag": {
    "chunk_size": 1100,
    "chunk_overlap": 180,
    "reranker_enabled": true,
    "candidate_k": 30
  }
}
```

## 实现优先级

1. **P0**：`sources/zotero_local.py` + `parsers/prebuilt_md.py` + RAG 核心迁移 + `literature_search` 工具
2. **P1**：`literature_qa` 工具 + `parsers/mineru.py` + CLI（ingest/status）
3. **P2**：`sources/zotero_api.py` + `parsers/pymupdf4llm.py` + `embeddings/openai.py`
