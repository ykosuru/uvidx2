#!/usr/bin/env python3
"""
Unified Indexer MCP Server
==========================

Model Context Protocol (MCP) server for semantic code/document search.
Integrates with VS Code, GitHub Copilot, and other MCP-compatible clients.

Features:
- Semantic search across code and documents
- Domain filtering (emts, gmts, etc.)
- Context retrieval for LLM code generation
- File content fetching

Installation:
    pip install mcp fastmcp

Run standalone:
    python mcp_server.py

Configure in VS Code (settings.json):
    {
        "mcp.servers": {
            "unified-indexer": {
                "command": "python",
                "args": ["/path/to/mcp_server.py"],
                "env": {
                    "INDEX_PATH": "/path/to/index"
                }
            }
        }
    }

Or with Claude Desktop (claude_desktop_config.json):
    {
        "mcpServers": {
            "unified-indexer": {
                "command": "python",
                "args": ["/path/to/mcp_server.py"],
                "env": {
                    "INDEX_PATH": "/path/to/index"
                }
            }
        }
    }
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        Resource,
        ResourceTemplate,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not installed. Run: pip install mcp", file=sys.stderr)

from unified_indexer.pipeline import IndexingPipeline
from unified_indexer.models import SourceType


# =============================================================================
# CONFIGURATION
# =============================================================================

INDEX_PATH = os.environ.get("INDEX_PATH", "./idx")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "./keywords.json")
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "10"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "8000"))


# =============================================================================
# GLOBAL STATE
# =============================================================================

pipeline: Optional[IndexingPipeline] = None


def get_pipeline() -> IndexingPipeline:
    """Get or initialize the pipeline"""
    global pipeline
    
    if pipeline is None:
        # Load vocabulary
        vocab_data = []
        if os.path.exists(VOCAB_PATH):
            try:
                with open(VOCAB_PATH, 'r') as f:
                    vocab_data = json.load(f)
                    if isinstance(vocab_data, dict):
                        vocab_data = vocab_data.get('terms', [])
            except Exception:
                pass
        
        pipeline = IndexingPipeline(
            vocabulary_data=vocab_data,
            embedder_type='hash'
        )
        
        if os.path.exists(INDEX_PATH):
            pipeline.load(INDEX_PATH)
    
    return pipeline


# =============================================================================
# MCP SERVER
# =============================================================================

if MCP_AVAILABLE:
    server = Server("unified-indexer")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools"""
        return [
            Tool(
                name="search_code",
                description="""Search for code and documentation in the indexed codebase.
                
Use this to find:
- Procedures/functions related to a concept
- Code examples for specific functionality
- Documentation about features
- Implementation patterns

Returns relevant code snippets with file paths and line numbers.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query (e.g., 'OFAC sanctions screening', 'wire transfer validation')"
                        },
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by domains (e.g., ['emts', 'gmts']). Omit to search all."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results (default: 10, max: 20)",
                            "default": 10
                        },
                        "source_type": {
                            "type": "string",
                            "enum": ["all", "code", "document"],
                            "description": "Filter by source type",
                            "default": "all"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_context_for_task",
                description="""Get relevant code context for a development task.
                
Use this when you need to:
- Understand existing patterns before writing new code
- Find similar implementations to reference
- Get documentation context for a feature

Returns comprehensive context formatted for LLM code generation.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Description of the development task (e.g., 'implement OFAC screening for incoming wires')"
                        },
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Relevant domains to search"
                        }
                    },
                    "required": ["task"]
                }
            ),
            Tool(
                name="list_domains",
                description="List all available domains in the index with chunk counts",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="get_file_content",
                description="Get the full content of a specific file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file"
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "Maximum lines to return (default: 500)",
                            "default": 500
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            Tool(
                name="get_procedure",
                description="Search for a specific procedure/function by name",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "procedure_name": {
                            "type": "string",
                            "description": "Name of the procedure to find"
                        },
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Domains to search"
                        }
                    },
                    "required": ["procedure_name"]
                }
            ),
            Tool(
                name="find_related_code",
                description="""Find code related to a given procedure or concept.
                
Use to discover:
- Functions that call or are called by a procedure
- Related implementations
- Similar patterns""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "reference": {
                            "type": "string",
                            "description": "Procedure name or concept to find related code for"
                        },
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Domains to search"
                        }
                    },
                    "required": ["reference"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls"""
        
        try:
            if name == "search_code":
                return await handle_search(arguments)
            elif name == "get_context_for_task":
                return await handle_get_context(arguments)
            elif name == "list_domains":
                return await handle_list_domains()
            elif name == "get_file_content":
                return await handle_get_file(arguments)
            elif name == "get_procedure":
                return await handle_get_procedure(arguments)
            elif name == "find_related_code":
                return await handle_find_related(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def handle_search(args: Dict[str, Any]) -> List[TextContent]:
        """Handle search_code tool"""
        pipe = get_pipeline()
        
        query = args.get("query", "")
        domains = args.get("domains")
        top_k = min(args.get("top_k", 10), 20)
        source_type = args.get("source_type", "all")
        
        # Parse source type
        source_types = None
        if source_type == "code":
            source_types = [SourceType.CODE]
        elif source_type == "document":
            source_types = [SourceType.DOCUMENT]
        
        results = pipe.search(
            query=query,
            top_k=top_k,
            source_types=source_types,
            domains=domains
        )
        
        if not results:
            return [TextContent(type="text", text=f"No results found for: {query}")]
        
        # Format results
        output = [f"## Search Results for: {query}\n"]
        output.append(f"Found {len(results)} results\n")
        
        for i, r in enumerate(results, 1):
            chunk = r.chunk
            ref = chunk.source_ref
            
            output.append(f"\n### Result {i} (score: {r.combined_score:.3f})")
            output.append(f"**File:** `{ref.file_path}`")
            
            if ref.procedure_name:
                output.append(f"**Procedure:** `{ref.procedure_name}`")
            
            if ref.line_start:
                line_info = f"Lines {ref.line_start}"
                if ref.line_end and ref.line_end != ref.line_start:
                    line_info += f"-{ref.line_end}"
                output.append(f"**Location:** {line_info}")
            
            if hasattr(ref, 'domains') and ref.domains:
                output.append(f"**Domains:** {', '.join(ref.domains)}")
            
            output.append(f"\n```\n{chunk.text[:1000]}{'...' if len(chunk.text) > 1000 else ''}\n```")
        
        return [TextContent(type="text", text="\n".join(output))]

    async def handle_get_context(args: Dict[str, Any]) -> List[TextContent]:
        """Handle get_context_for_task - returns comprehensive context for code generation"""
        pipe = get_pipeline()
        
        task = args.get("task", "")
        domains = args.get("domains")
        
        # Search for relevant code
        code_results = pipe.search(
            query=task,
            top_k=10,
            source_types=[SourceType.CODE],
            domains=domains
        )
        
        # Search for relevant documentation
        doc_results = pipe.search(
            query=task,
            top_k=5,
            source_types=[SourceType.DOCUMENT],
            domains=domains
        )
        
        output = [f"# Context for: {task}\n"]
        
        # Add documentation context
        if doc_results:
            output.append("## Relevant Documentation\n")
            for r in doc_results[:3]:
                chunk = r.chunk
                output.append(f"### From: {chunk.source_ref.file_path}")
                output.append(f"{chunk.text[:1500]}\n")
        
        # Add code context
        if code_results:
            output.append("## Relevant Code Examples\n")
            
            total_chars = 0
            for r in code_results:
                if total_chars > MAX_CONTEXT_CHARS:
                    output.append("\n*[Additional results truncated for context length]*")
                    break
                
                chunk = r.chunk
                ref = chunk.source_ref
                
                output.append(f"### {ref.procedure_name or 'Code'} from `{ref.file_path}`")
                
                if ref.line_start:
                    output.append(f"Lines {ref.line_start}-{ref.line_end or ref.line_start}")
                
                code_text = chunk.text[:2000]
                output.append(f"```\n{code_text}\n```\n")
                total_chars += len(code_text)
        
        if not code_results and not doc_results:
            output.append("No relevant context found. Consider:")
            output.append("- Using different search terms")
            output.append("- Specifying relevant domains")
        
        return [TextContent(type="text", text="\n".join(output))]

    async def handle_list_domains() -> List[TextContent]:
        """Handle list_domains tool"""
        pipe = get_pipeline()
        domains = pipe.get_domains()
        
        if not domains:
            return [TextContent(type="text", text="No domains found in index")]
        
        total = sum(domains.values())
        output = ["## Available Domains\n"]
        output.append(f"Total chunks: {total}\n")
        
        for name, count in sorted(domains.items()):
            pct = (count / total * 100) if total > 0 else 0
            output.append(f"- **{name}**: {count} chunks ({pct:.1f}%)")
        
        return [TextContent(type="text", text="\n".join(output))]

    async def handle_get_file(args: Dict[str, Any]) -> List[TextContent]:
        """Handle get_file_content tool"""
        file_path = args.get("file_path", "")
        max_lines = args.get("max_lines", 500)
        
        if not os.path.exists(file_path):
            return [TextContent(type="text", text=f"File not found: {file_path}")]
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:max_lines]
            
            content = "".join(lines)
            if len(lines) == max_lines:
                content += f"\n... (truncated at {max_lines} lines)"
            
            output = f"## File: {file_path}\n\n```\n{content}\n```"
            return [TextContent(type="text", text=output)]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error reading file: {e}")]

    async def handle_get_procedure(args: Dict[str, Any]) -> List[TextContent]:
        """Handle get_procedure tool"""
        pipe = get_pipeline()
        
        proc_name = args.get("procedure_name", "")
        domains = args.get("domains")
        
        # Search for the procedure
        results = pipe.search(
            query=proc_name,
            top_k=5,
            source_types=[SourceType.CODE],
            domains=domains
        )
        
        # Filter to exact procedure matches
        exact_matches = [
            r for r in results 
            if r.chunk.source_ref.procedure_name and 
               proc_name.lower() in r.chunk.source_ref.procedure_name.lower()
        ]
        
        if not exact_matches:
            # Fall back to all results
            exact_matches = results[:3]
        
        if not exact_matches:
            return [TextContent(type="text", text=f"Procedure not found: {proc_name}")]
        
        output = [f"## Procedure: {proc_name}\n"]
        
        for r in exact_matches[:3]:
            chunk = r.chunk
            ref = chunk.source_ref
            
            output.append(f"### `{ref.procedure_name}` in `{ref.file_path}`")
            if ref.line_start:
                output.append(f"Lines {ref.line_start}-{ref.line_end or ref.line_start}")
            output.append(f"```\n{chunk.text}\n```\n")
        
        return [TextContent(type="text", text="\n".join(output))]

    async def handle_find_related(args: Dict[str, Any]) -> List[TextContent]:
        """Handle find_related_code tool"""
        pipe = get_pipeline()
        
        reference = args.get("reference", "")
        domains = args.get("domains")
        
        # Search for related code
        results = pipe.search(
            query=reference,
            top_k=10,
            source_types=[SourceType.CODE],
            domains=domains
        )
        
        if not results:
            return [TextContent(type="text", text=f"No related code found for: {reference}")]
        
        output = [f"## Code Related to: {reference}\n"]
        
        seen_procs = set()
        for r in results:
            chunk = r.chunk
            ref = chunk.source_ref
            proc_name = ref.procedure_name or "unknown"
            
            # Skip if we've already shown this procedure
            if proc_name in seen_procs:
                continue
            seen_procs.add(proc_name)
            
            output.append(f"### `{proc_name}` (score: {r.combined_score:.3f})")
            output.append(f"**File:** `{ref.file_path}`")
            output.append(f"```\n{chunk.text[:800]}{'...' if len(chunk.text) > 800 else ''}\n```\n")
            
            if len(seen_procs) >= 5:
                break
        
        return [TextContent(type="text", text="\n".join(output))]

    @server.list_resources()
    async def list_resources() -> List[Resource]:
        """List available resources"""
        pipe = get_pipeline()
        domains = pipe.get_domains()
        
        resources = [
            Resource(
                uri="indexer://stats",
                name="Index Statistics",
                description="Current index statistics and configuration",
                mimeType="application/json"
            )
        ]
        
        # Add domain resources
        for domain in domains:
            resources.append(Resource(
                uri=f"indexer://domain/{domain}",
                name=f"Domain: {domain}",
                description=f"Browse {domain} domain content",
                mimeType="text/plain"
            ))
        
        return resources

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a resource"""
        pipe = get_pipeline()
        
        if uri == "indexer://stats":
            stats = pipe.get_statistics()
            domains = pipe.get_domains()
            return json.dumps({
                "index_path": INDEX_PATH,
                "total_chunks": stats.get('pipeline', {}).get('total_chunks', 0),
                "domains": domains,
                "source_types": stats.get('pipeline', {}).get('by_source_type', {})
            }, indent=2)
        
        elif uri.startswith("indexer://domain/"):
            domain = uri.replace("indexer://domain/", "")
            results = pipe.search("*", top_k=20, domains=[domain])
            
            output = [f"# Domain: {domain}\n"]
            files = set()
            for r in results:
                files.add(r.chunk.source_ref.file_path)
            
            output.append("## Files:\n")
            for f in sorted(files):
                output.append(f"- {f}")
            
            return "\n".join(output)
        
        return f"Unknown resource: {uri}"


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run the MCP server"""
    if not MCP_AVAILABLE:
        print("Error: MCP not installed. Run: pip install mcp", file=sys.stderr)
        sys.exit(1)
    
    # Verify index exists
    if not os.path.exists(INDEX_PATH):
        print(f"Warning: Index not found at {INDEX_PATH}", file=sys.stderr)
        print("Set INDEX_PATH environment variable to your index location", file=sys.stderr)
    
    print(f"Starting Unified Indexer MCP Server...", file=sys.stderr)
    print(f"Index: {INDEX_PATH}", file=sys.stderr)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
