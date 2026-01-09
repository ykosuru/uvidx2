#!/usr/bin/env python3
"""
Unified Indexer Python Client
==============================

Python client library for the Unified Indexer REST API.

Usage:
    from api_client import UnifiedIndexerClient
    
    client = UnifiedIndexerClient("http://localhost:8080")
    
    # Search
    results = client.search("OFAC screening", domains=["compliance"])
    
    # Index content
    client.index_content("PROC test; BEGIN END;", "test.tal", domain="payments")
    
    # List domains
    domains = client.list_domains()

Requirements:
    pip install requests
"""

import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Search result from the API"""
    rank: int
    score: float
    file_path: str
    domain: str
    source_type: str
    procedure_name: Optional[str]
    line_start: Optional[int]
    line_end: Optional[int]
    content_preview: str
    concepts: List[str]


@dataclass 
class SearchResponse:
    """Search response from the API"""
    query: str
    total_results: int
    search_time_ms: float
    domains_searched: Optional[List[str]]
    results: List[SearchResult]


class UnifiedIndexerClient:
    """
    Python client for Unified Indexer REST API.
    
    Example:
        client = UnifiedIndexerClient("http://localhost:8080")
        results = client.search("wire transfer validation")
    """
    
    def __init__(self, base_url: str = "http://localhost:8080", timeout: int = 30):
        """
        Initialize client.
        
        Args:
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make HTTP request"""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault('timeout', self.timeout)
        
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # HEALTH & INFO
    # =========================================================================
    
    def health(self) -> dict:
        """Check API health"""
        return self._request("GET", "/health")
    
    def stats(self) -> dict:
        """Get index statistics"""
        return self._request("GET", "/stats")
    
    def is_healthy(self) -> bool:
        """Check if API is healthy"""
        try:
            health = self.health()
            return health.get("status") == "healthy"
        except:
            return False
    
    # =========================================================================
    # SEARCH
    # =========================================================================
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        domains: Optional[List[str]] = None,
        source_type: Optional[str] = None,
        expand_query: bool = False
    ) -> SearchResponse:
        """
        Perform semantic search.
        
        Args:
            query: Search query text
            top_k: Number of results (1-100)
            domains: Filter by domains (e.g., ["payments", "compliance"])
            source_type: Filter by type ("code", "document", "log", "all")
            expand_query: Expand query using knowledge graph
            
        Returns:
            SearchResponse with results
            
        Example:
            results = client.search("OFAC screening", domains=["compliance"])
            for r in results.results:
                print(f"{r.rank}. {r.file_path} (score: {r.score})")
        """
        payload = {
            "query": query,
            "top_k": top_k,
            "expand_query": expand_query
        }
        
        if domains:
            payload["domains"] = domains
        if source_type:
            payload["source_type"] = source_type
        
        data = self._request("POST", "/search", json=payload)
        
        # Parse results
        results = [
            SearchResult(
                rank=r["rank"],
                score=r["score"],
                file_path=r["file_path"],
                domain=r["domain"],
                source_type=r["source_type"],
                procedure_name=r.get("procedure_name"),
                line_start=r.get("line_start"),
                line_end=r.get("line_end"),
                content_preview=r["content_preview"],
                concepts=r.get("concepts", [])
            )
            for r in data.get("results", [])
        ]
        
        return SearchResponse(
            query=data["query"],
            total_results=data["total_results"],
            search_time_ms=data["search_time_ms"],
            domains_searched=data.get("domains_searched"),
            results=results
        )
    
    def search_simple(self, query: str, top_k: int = 10, domain: Optional[str] = None) -> dict:
        """
        Simple search using GET method.
        
        Args:
            query: Search query
            top_k: Number of results
            domain: Single domain filter (comma-separated for multiple)
            
        Returns:
            Raw API response dict
        """
        params = {"q": query, "top_k": top_k}
        if domain:
            params["domain"] = domain
        
        return self._request("GET", "/search", params=params)
    
    # =========================================================================
    # DOMAINS
    # =========================================================================
    
    def list_domains(self) -> Dict[str, int]:
        """
        List all domains with chunk counts.
        
        Returns:
            Dict mapping domain name to chunk count
            
        Example:
            domains = client.list_domains()
            # {'payments': 1250, 'compliance': 890, 'settlement': 650}
        """
        data = self._request("GET", "/domains")
        return {d["name"]: d["chunk_count"] for d in data.get("domains", [])}
    
    def get_domain(self, domain_name: str) -> dict:
        """Get detailed info about a domain"""
        return self._request("GET", f"/domains/{domain_name}")
    
    # =========================================================================
    # INDEXING
    # =========================================================================
    
    def index_content(
        self,
        content: str,
        filename: str,
        domain: str = "default",
        source_type: str = "code"
    ) -> dict:
        """
        Index content directly.
        
        Args:
            content: Content to index
            filename: Virtual filename
            domain: Domain tag
            source_type: "code" or "document"
            
        Returns:
            Indexing result dict
            
        Example:
            result = client.index_content(
                "PROC validate; BEGIN END;",
                "validate.tal",
                domain="payments"
            )
        """
        payload = {
            "content": content,
            "filename": filename,
            "domain": domain,
            "source_type": source_type
        }
        return self._request("POST", "/index/content", json=payload)
    
    def index_file(self, file_path: str, domain: str = "default") -> dict:
        """
        Index a file from server filesystem.
        
        Args:
            file_path: Path to file (must be accessible from server)
            domain: Domain tag
            
        Returns:
            Indexing result dict
        """
        payload = {"file_path": file_path, "domain": domain}
        return self._request("POST", "/index/file", json=payload)
    
    def index_directory(
        self,
        directory: str,
        domain: str = "default",
        recursive: bool = True
    ) -> dict:
        """
        Index a directory from server filesystem.
        
        Args:
            directory: Path to directory
            domain: Domain tag for all files
            recursive: Recurse into subdirectories
            
        Returns:
            Indexing result dict
        """
        payload = {
            "directory": directory,
            "domain": domain,
            "recursive": recursive
        }
        return self._request("POST", "/index/directory", json=payload)
    
    def upload_file(self, file_path: str, domain: str = "default") -> dict:
        """
        Upload and index a local file.
        
        Args:
            file_path: Local path to file
            domain: Domain tag
            
        Returns:
            Indexing result dict
            
        Example:
            result = client.upload_file("./code/payments.tal", domain="payments")
        """
        import os
        filename = os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f)}
            params = {'domain': domain}
            
            response = self.session.post(
                f"{self.base_url}/index/upload",
                files=files,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
    
    def save_index(self, path: Optional[str] = None) -> dict:
        """Save index to disk"""
        params = {"path": path} if path else {}
        return self._request("POST", "/index/save", params=params)
    
    def load_index(self, path: Optional[str] = None) -> dict:
        """Load index from disk"""
        params = {"path": path} if path else {}
        return self._request("POST", "/index/load", params=params)
    
    def clear_index(self) -> dict:
        """Clear the index"""
        return self._request("DELETE", "/index")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_basic_usage():
    """Basic usage example"""
    client = UnifiedIndexerClient("http://localhost:8080")
    
    # Check health
    print("Checking API health...")
    if not client.is_healthy():
        print("API is not healthy!")
        return
    
    print("API is healthy!")
    
    # Index some content
    print("\nIndexing content...")
    result = client.index_content(
        content="""
        PROC validate_wire_transfer;
        BEGIN
            CALL check_amount;
            CALL verify_account;
            CALL screen_ofac;
        END;
        """,
        filename="wire_validation.tal",
        domain="payments"
    )
    print(f"  Indexed: {result}")
    
    # Search
    print("\nSearching for 'wire transfer'...")
    response = client.search("wire transfer", top_k=5)
    
    print(f"Found {response.total_results} results in {response.search_time_ms}ms")
    for r in response.results:
        print(f"  {r.rank}. {r.file_path} [{r.domain}] (score: {r.score:.3f})")
        print(f"     Preview: {r.content_preview[:100]}...")
    
    # List domains
    print("\nDomains:")
    domains = client.list_domains()
    for name, count in domains.items():
        print(f"  {name}: {count} chunks")


def example_domain_workflow():
    """Example workflow with domains"""
    client = UnifiedIndexerClient("http://localhost:8080")
    
    # Clear index
    client.clear_index()
    
    # Index content for different domains
    domains_content = {
        "payments": """
            PROC process_wire_transfer;
            BEGIN
                CALL validate_amount;
                CALL debit_account;
                CALL credit_account;
            END;
        """,
        "compliance": """
            PROC check_ofac_sanctions;
            BEGIN
                CALL lookup_sdn_list;
                CALL verify_customer;
            END;
        """,
        "settlement": """
            PROC settle_transaction;
            BEGIN
                CALL create_settlement_record;
                CALL send_swift_mt103;
            END;
        """
    }
    
    print("Indexing content by domain...")
    for domain, content in domains_content.items():
        result = client.index_content(content, f"{domain}.tal", domain=domain)
        print(f"  {domain}: {result['chunks_created']} chunks")
    
    # List domains
    print("\nDomains in index:")
    for name, count in client.list_domains().items():
        print(f"  {name}: {count}")
    
    # Search all domains
    print("\nSearch all domains for 'transfer':")
    results = client.search("transfer", top_k=5)
    for r in results.results:
        print(f"  [{r.domain}] {r.file_path}: {r.score:.3f}")
    
    # Search specific domain
    print("\nSearch 'payments' domain only:")
    results = client.search("transfer", domains=["payments"])
    for r in results.results:
        print(f"  [{r.domain}] {r.file_path}: {r.score:.3f}")
    
    # Search multiple domains
    print("\nSearch 'payments' + 'settlement' domains:")
    results = client.search("transfer", domains=["payments", "settlement"])
    for r in results.results:
        print(f"  [{r.domain}] {r.file_path}: {r.score:.3f}")


def example_batch_indexing():
    """Example of batch indexing multiple files"""
    import os
    
    client = UnifiedIndexerClient("http://localhost:8080")
    
    # Index a directory (if running on same machine as server)
    if os.path.exists("./code"):
        result = client.index_directory(
            directory="./code",
            domain="my_project",
            recursive=True
        )
        print(f"Indexed directory: {result}")
    
    # Or upload files individually
    files_to_upload = [
        ("./payments/wire.tal", "payments"),
        ("./compliance/ofac.tal", "compliance"),
        ("./docs/readme.md", "documentation"),
    ]
    
    for file_path, domain in files_to_upload:
        if os.path.exists(file_path):
            result = client.upload_file(file_path, domain=domain)
            print(f"Uploaded {file_path}: {result['chunks_created']} chunks")


if __name__ == "__main__":
    print("=" * 60)
    print("Unified Indexer Client Examples")
    print("=" * 60)
    
    print("\n--- Basic Usage ---")
    try:
        example_basic_usage()
    except requests.exceptions.ConnectionError:
        print("Could not connect to API. Is the server running?")
        print("Start with: python api_server.py")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n--- Domain Workflow ---")
    try:
        example_domain_workflow()
    except requests.exceptions.ConnectionError:
        print("Could not connect to API.")
    except Exception as e:
        print(f"Error: {e}")
