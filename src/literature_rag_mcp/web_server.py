"""FastAPI web server to serve the frontend and provide API endpoints."""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from literature_rag_mcp.server import get_retriever

app = FastAPI(title="Literature RAG Web Interface")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None

class QARequest(BaseModel):
    item_key: str
    question: str = ""

@app.post("/api/search")
async def search_endpoint(request: SearchRequest):
    """Semantic search API for frontend."""
    try:
        retriever = get_retriever()
        result = retriever.search(
            query=request.query, limit=request.limit, filters=request.filters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/qa")
async def qa_endpoint(request: QARequest):
    """Retrieve item content and answer QA."""
    try:
        retriever = get_retriever()
        result = retriever.get_item_fulltext(request.item_key)

        # If question provided, also search within this item
        if request.question and result.get("status") == "success":
            search_result = retriever.search(
                query=request.question,
                limit=5,
                filters={"item_key": request.item_key},
            )
            result["relevant_chunks"] = search_result.get("results", [])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Calculate absolute path for web/dist to mount static files
# In standard package layout this might be one directory above the current working directory, 
# or passed as an argument. Assume we are running from project root or similar.
# Since CLI runs from `literature-rag`, the CWD is usually where `web/` exists or can be detected.
# We will check if `./web/dist` exists, if so, mount it.

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Entry point to start the web server."""
    # Mount static files if they exist
    static_dir = os.path.join(os.getcwd(), "web", "dist")
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    
    uvicorn.run("literature_rag_mcp.web_server:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    start_server(reload=True)
