#!/usr/bin/env python3
"""
Unified Indexer REST API
========================

FastAPI-based REST API for the unified indexer, providing endpoints for:
- Index management (create, update, stats)
- Search (with domain filtering, query expansion)
- Domain management

Requirements:
    pip install fastapi uvicorn python-multipart

Run:
    uvicorn api_server:app --reload --port 8080
    
    # Or directly
    python api_server.py

API Docs:
    http://localhost:8080/docs      (Swagger UI)
    http://localhost:8080/redoc     (ReDoc)
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_indexer.pipeline import IndexingPipeline
from unified_indexer.models import SourceType


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_INDEX_PATH = os.environ.get("INDEX_PATH", "./index")
DEFAULT_VOCAB_PATH = os.environ.get("VOCAB_PATH", "./keywords.json")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB


# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class SearchRequest(BaseModel):
    """Search request body"""
    query: str = Field(..., description="Search query text", min_length=1)
    top_k: int = Field(10, description="Number of results", ge=1, le=100)
    domains: Optional[List[str]] = Field(None, description="Filter by domains")
    source_type: Optional[str] = Field(None, description="Filter: code, document, log, all")
    expand_query: bool = Field(False, description="Expand query using knowledge graph")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "OFAC sanctions screening",
                "top_k": 10,
                "domains": ["compliance"],
                "source_type": "code",
                "expand_query": True
            }
        }


class SearchResult(BaseModel):
    """Single search result"""
    rank: int
    score: float
    file_path: str
    domain: str
    source_type: str
    procedure_name: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    content_preview: str
    concepts: List[str] = []


class SearchResponse(BaseModel):
    """Search response"""
    query: str
    total_results: int
    search_time_ms: float
    domains_searched: Optional[List[str]] = None
    results: List[SearchResult]


class IndexFileRequest(BaseModel):
    """Request to index a file"""
    file_path: str = Field(..., description="Path to file to index")
    domain: str = Field("default", description="Domain tag for the file")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "/path/to/code.tal",
                "domain": "payments"
            }
        }


class IndexDirectoryRequest(BaseModel):
    """Request to index a directory"""
    directory: str = Field(..., description="Path to directory")
    domain: str = Field("default", description="Domain tag for all files")
    recursive: bool = Field(True, description="Recurse into subdirectories")
    
    class Config:
        json_schema_extra = {
            "example": {
                "directory": "/path/to/code",
                "domain": "payments",
                "recursive": True
            }
        }


class IndexContentRequest(BaseModel):
    """Request to index content directly"""
    content: str = Field(..., description="Content to index")
    filename: str = Field(..., description="Virtual filename for the content")
    domain: str = Field("default", description="Domain tag")
    source_type: str = Field("code", description="Source type: code, document")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "PROC validate_transfer;\nBEGIN\n    CALL check_amount;\nEND;",
                "filename": "validate.tal",
                "domain": "payments",
                "source_type": "code"
            }
        }


class IndexStats(BaseModel):
    """Index statistics"""
    total_chunks: int
    total_files: int
    domains: Dict[str, int]
    source_types: Dict[str, int]
    index_path: str
    last_updated: Optional[str] = None


class DomainInfo(BaseModel):
    """Domain information"""
    name: str
    chunk_count: int
    percentage: float


class DomainsResponse(BaseModel):
    """List of domains"""
    total_domains: int
    total_chunks: int
    domains: List[DomainInfo]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    index_loaded: bool
    total_chunks: int
    version: str


class IndexingResponse(BaseModel):
    """Response from indexing operation"""
    success: bool
    message: str
    files_processed: int = 0
    chunks_created: int = 0
    errors: List[str] = []


# =============================================================================
# APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="Unified Indexer API",
    description="""
REST API for semantic search through TAL/COBOL code, documents, and logs.

## Features
- **Semantic Search**: Vector + BM25 + concept matching
- **Domain Filtering**: Organize and search by business domain
- **Query Expansion**: Expand queries using knowledge graph
- **Multiple Source Types**: Code, documents, logs

## Quick Start
1. Index some files: `POST /index/directory`
2. Search: `POST /search`
3. Check domains: `GET /domains`
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[IndexingPipeline] = None


# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    
    # Load vocabulary if available
    vocab_data = []
    if os.path.exists(DEFAULT_VOCAB_PATH):
        try:
            with open(DEFAULT_VOCAB_PATH, 'r') as f:
                vocab_data = json.load(f)
                if isinstance(vocab_data, dict):
                    vocab_data = vocab_data.get('terms', [])
        except Exception as e:
            print(f"Warning: Could not load vocabulary: {e}")
    
    # Create pipeline
    pipeline = IndexingPipeline(
        vocabulary_data=vocab_data,
        embedder_type='hash'
    )
    
    # Load existing index if available
    if os.path.exists(DEFAULT_INDEX_PATH):
        try:
            pipeline.load(DEFAULT_INDEX_PATH)
            stats = pipeline.get_statistics()
            chunks = stats.get('pipeline', {}).get('total_chunks', 0)
            print(f"Loaded existing index: {chunks} chunks")
        except Exception as e:
            print(f"Warning: Could not load index: {e}")
    else:
        print("No existing index found. Starting fresh.")


# =============================================================================
# HEALTH & INFO ENDPOINTS
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API root - redirect to docs"""
    return {
        "message": "Unified Indexer API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint"""
    global pipeline
    
    if pipeline is None:
        return HealthResponse(
            status="unhealthy",
            index_loaded=False,
            total_chunks=0,
            version="1.0.0"
        )
    
    stats = pipeline.get_statistics()
    chunks = stats.get('pipeline', {}).get('total_chunks', 0)
    
    return HealthResponse(
        status="healthy",
        index_loaded=chunks > 0,
        total_chunks=chunks,
        version="1.0.0"
    )


@app.get("/stats", response_model=IndexStats, tags=["Info"])
async def get_stats():
    """Get index statistics"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    stats = pipeline.get_statistics()
    pipeline_stats = stats.get('pipeline', {})
    
    return IndexStats(
        total_chunks=pipeline_stats.get('total_chunks', 0),
        total_files=pipeline_stats.get('files_processed', 0),
        domains=pipeline.get_domains(),
        source_types=pipeline_stats.get('by_source_type', {}),
        index_path=DEFAULT_INDEX_PATH,
        last_updated=datetime.now().isoformat()
    )


# =============================================================================
# SEARCH ENDPOINTS
# =============================================================================

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Perform semantic search.
    
    Combines vector similarity, BM25 lexical matching, and concept matching
    using Reciprocal Rank Fusion (RRF).
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    import time
    start_time = time.time()
    
    # Parse source type filter
    source_types = None
    if request.source_type and request.source_type != "all":
        type_map = {
            "code": [SourceType.CODE],
            "document": [SourceType.DOCUMENT],
            "log": [SourceType.LOG]
        }
        source_types = type_map.get(request.source_type)
    
    # Perform search
    try:
        results = pipeline.search(
            query=request.query,
            top_k=request.top_k,
            source_types=source_types,
            domains=request.domains
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    search_time = (time.time() - start_time) * 1000
    
    # Format results
    formatted_results = []
    for i, r in enumerate(results, 1):
        chunk = r.chunk
        source_ref = chunk.source_ref
        
        # Get concepts from metadata
        concepts = []
        if hasattr(chunk, 'metadata') and chunk.metadata:
            concepts = chunk.metadata.get('concepts', [])
        
        formatted_results.append(SearchResult(
            rank=i,
            score=round(r.combined_score, 4),
            file_path=source_ref.file_path,
            domain=getattr(source_ref, 'domain', 'default'),
            source_type=chunk.source_type.value if chunk.source_type else 'unknown',
            procedure_name=source_ref.procedure_name,
            line_start=source_ref.line_start,
            line_end=source_ref.line_end,
            content_preview=chunk.text[:500] if chunk.text else "",
            concepts=concepts[:5]  # Limit concepts shown
        ))
    
    return SearchResponse(
        query=request.query,
        total_results=len(formatted_results),
        search_time_ms=round(search_time, 2),
        domains_searched=request.domains,
        results=formatted_results
    )


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_get(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results"),
    domain: Optional[str] = Query(None, description="Comma-separated domains"),
    type: Optional[str] = Query(None, description="Source type filter")
):
    """
    Perform semantic search (GET method for simple queries).
    
    Example: `/search?q=OFAC+screening&top_k=5&domain=compliance`
    """
    domains = domain.split(',') if domain else None
    
    return await search(SearchRequest(
        query=q,
        top_k=top_k,
        domains=domains,
        source_type=type
    ))


# =============================================================================
# DOMAIN ENDPOINTS
# =============================================================================

@app.get("/domains", response_model=DomainsResponse, tags=["Domains"])
async def list_domains():
    """List all available domains with chunk counts"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    domain_stats = pipeline.get_domains()
    total_chunks = sum(domain_stats.values())
    
    domains = []
    for name, count in sorted(domain_stats.items()):
        percentage = (count / total_chunks * 100) if total_chunks > 0 else 0
        domains.append(DomainInfo(
            name=name,
            chunk_count=count,
            percentage=round(percentage, 1)
        ))
    
    return DomainsResponse(
        total_domains=len(domains),
        total_chunks=total_chunks,
        domains=domains
    )


@app.get("/domains/{domain_name}", tags=["Domains"])
async def get_domain_info(domain_name: str):
    """Get detailed information about a specific domain"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    domain_stats = pipeline.get_domains()
    
    if domain_name not in domain_stats:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_name}' not found")
    
    count = domain_stats[domain_name]
    total = sum(domain_stats.values())
    
    return {
        "name": domain_name,
        "chunk_count": count,
        "percentage": round(count / total * 100, 1) if total > 0 else 0,
        "sample_search": f"/search?q=*&domain={domain_name}&top_k=5"
    }


# =============================================================================
# INDEXING ENDPOINTS
# =============================================================================

@app.post("/index/content", response_model=IndexingResponse, tags=["Indexing"])
async def index_content(request: IndexContentRequest):
    """
    Index content directly (without file upload).
    
    Useful for indexing code snippets or content from other sources.
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    try:
        # Determine source type
        source_type = SourceType.CODE if request.source_type == "code" else SourceType.DOCUMENT
        
        # Index content
        content_bytes = request.content.encode('utf-8')
        chunks = pipeline.index_content(
            content_bytes,
            request.filename,
            source_type,
            domain=request.domain
        )
        
        return IndexingResponse(
            success=True,
            message=f"Indexed {len(chunks)} chunks from content",
            files_processed=1,
            chunks_created=len(chunks)
        )
    
    except Exception as e:
        return IndexingResponse(
            success=False,
            message=f"Indexing failed: {str(e)}",
            errors=[str(e)]
        )


@app.post("/index/file", response_model=IndexingResponse, tags=["Indexing"])
async def index_file(request: IndexFileRequest):
    """
    Index a file from the server filesystem.
    
    The file path must be accessible from the server.
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    
    try:
        result = pipeline.index_file(request.file_path, domain=request.domain)
        
        return IndexingResponse(
            success=result.success,
            message=f"Indexed {result.chunks_created} chunks" if result.success else result.error,
            files_processed=1 if result.success else 0,
            chunks_created=result.chunks_created,
            errors=[result.error] if result.error else []
        )
    
    except Exception as e:
        return IndexingResponse(
            success=False,
            message=f"Indexing failed: {str(e)}",
            errors=[str(e)]
        )


@app.post("/index/directory", response_model=IndexingResponse, tags=["Indexing"])
async def index_directory(request: IndexDirectoryRequest):
    """
    Index all supported files in a directory.
    
    The directory path must be accessible from the server.
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    if not os.path.isdir(request.directory):
        raise HTTPException(status_code=404, detail=f"Directory not found: {request.directory}")
    
    try:
        stats = pipeline.index_directory(
            request.directory,
            recursive=request.recursive,
            domain=request.domain
        )
        
        return IndexingResponse(
            success=True,
            message=f"Indexed {stats.files_processed} files, {stats.total_chunks} chunks",
            files_processed=stats.files_processed,
            chunks_created=stats.total_chunks,
            errors=stats.errors
        )
    
    except Exception as e:
        return IndexingResponse(
            success=False,
            message=f"Indexing failed: {str(e)}",
            errors=[str(e)]
        )


@app.post("/index/upload", response_model=IndexingResponse, tags=["Indexing"])
async def upload_and_index(
    file: UploadFile = File(...),
    domain: str = Query("default", description="Domain tag for the file")
):
    """
    Upload and index a file.
    
    Supports TAL, COBOL, Python, Java, C, PDF, Markdown files.
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    # Check file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_SIZE // 1024 // 1024}MB)")
    
    try:
        # Determine source type from filename
        filename = file.filename or "uploaded_file"
        ext = Path(filename).suffix.lower()
        
        if ext in ['.pdf']:
            source_type = SourceType.DOCUMENT
        elif ext in ['.md', '.txt', '.html']:
            source_type = SourceType.DOCUMENT
        else:
            source_type = SourceType.CODE
        
        # Index content
        chunks = pipeline.index_content(
            content,
            filename,
            source_type,
            domain=domain
        )
        
        return IndexingResponse(
            success=True,
            message=f"Indexed {len(chunks)} chunks from {filename}",
            files_processed=1,
            chunks_created=len(chunks)
        )
    
    except Exception as e:
        return IndexingResponse(
            success=False,
            message=f"Indexing failed: {str(e)}",
            errors=[str(e)]
        )


@app.post("/index/save", tags=["Indexing"])
async def save_index(path: Optional[str] = None):
    """Save the current index to disk"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    save_path = path or DEFAULT_INDEX_PATH
    
    try:
        os.makedirs(save_path, exist_ok=True)
        pipeline.save(save_path)
        return {"success": True, "message": f"Index saved to {save_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save index: {str(e)}")


@app.post("/index/load", tags=["Indexing"])
async def load_index(path: Optional[str] = None):
    """Load an index from disk"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    load_path = path or DEFAULT_INDEX_PATH
    
    if not os.path.exists(load_path):
        raise HTTPException(status_code=404, detail=f"Index not found at {load_path}")
    
    try:
        pipeline.load(load_path)
        stats = pipeline.get_statistics()
        chunks = stats.get('pipeline', {}).get('total_chunks', 0)
        return {"success": True, "message": f"Loaded index with {chunks} chunks"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load index: {str(e)}")


@app.delete("/index", tags=["Indexing"])
async def clear_index():
    """Clear the current index (start fresh)"""
    global pipeline
    
    # Reinitialize pipeline
    vocab_data = []
    if os.path.exists(DEFAULT_VOCAB_PATH):
        try:
            with open(DEFAULT_VOCAB_PATH, 'r') as f:
                vocab_data = json.load(f)
                if isinstance(vocab_data, dict):
                    vocab_data = vocab_data.get('terms', [])
        except:
            pass
    
    pipeline = IndexingPipeline(
        vocabulary_data=vocab_data,
        embedder_type='hash'
    )
    
    return {"success": True, "message": "Index cleared"}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Unified Indexer REST API                        ║
╠══════════════════════════════════════════════════════════════╣
║  Server:    http://{host}:{port}                              
║  Docs:      http://{host}:{port}/docs                         
║  Health:    http://{host}:{port}/health                       
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=host, port=port)
