# Unified Indexer REST API Examples

## Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Index statistics |
| `/search` | POST | Semantic search |
| `/search?q=...` | GET | Simple search |
| `/domains` | GET | List domains |
| `/domains/{name}` | GET | Domain details |
| `/index/content` | POST | Index content directly |
| `/index/file` | POST | Index server file |
| `/index/directory` | POST | Index server directory |
| `/index/upload` | POST | Upload and index file |
| `/index/save` | POST | Save index to disk |
| `/index/load` | POST | Load index from disk |
| `/index` | DELETE | Clear index |

---

## cURL Examples

### Health Check

```bash
# Check API health
curl http://localhost:8080/health

# Response:
# {"status":"healthy","index_loaded":true,"total_chunks":1250,"version":"1.0.0"}
```

### Get Statistics

```bash
curl http://localhost:8080/stats

# Response:
# {
#   "total_chunks": 1250,
#   "total_files": 45,
#   "domains": {"payments": 500, "compliance": 400, "settlement": 350},
#   "source_types": {"code": 1100, "document": 150},
#   "index_path": "./index"
# }
```

### Search

```bash
# Simple GET search
curl "http://localhost:8080/search?q=OFAC+screening&top_k=5"

# POST search with more options
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "wire transfer validation",
    "top_k": 10,
    "domains": ["payments", "compliance"],
    "source_type": "code"
  }'

# Search single domain
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OFAC sanctions",
    "top_k": 5,
    "domains": ["compliance"]
  }'

# Search with query expansion
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OFAC",
    "top_k": 10,
    "expand_query": true
  }'
```

### List Domains

```bash
# List all domains
curl http://localhost:8080/domains

# Response:
# {
#   "total_domains": 3,
#   "total_chunks": 1250,
#   "domains": [
#     {"name": "compliance", "chunk_count": 400, "percentage": 32.0},
#     {"name": "payments", "chunk_count": 500, "percentage": 40.0},
#     {"name": "settlement", "chunk_count": 350, "percentage": 28.0}
#   ]
# }

# Get specific domain info
curl http://localhost:8080/domains/payments
```

### Index Content

```bash
# Index code content
curl -X POST http://localhost:8080/index/content \
  -H "Content-Type: application/json" \
  -d '{
    "content": "PROC validate_transfer;\nBEGIN\n    CALL check_amount;\nEND;",
    "filename": "validate.tal",
    "domain": "payments",
    "source_type": "code"
  }'

# Index document content
curl -X POST http://localhost:8080/index/content \
  -H "Content-Type: application/json" \
  -d '{
    "content": "# Wire Transfer Process\n\nThis document describes...",
    "filename": "wire_transfer.md",
    "domain": "documentation",
    "source_type": "document"
  }'
```

### Index File/Directory (Server-Side)

```bash
# Index a file from server filesystem
curl -X POST http://localhost:8080/index/file \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/code.tal",
    "domain": "payments"
  }'

# Index a directory
curl -X POST http://localhost:8080/index/directory \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "/path/to/code",
    "domain": "payments",
    "recursive": true
  }'
```

### Upload File

```bash
# Upload and index a file
curl -X POST http://localhost:8080/index/upload \
  -F "file=@./code/payments.tal" \
  -F "domain=payments"

# Upload multiple files
for f in ./code/*.tal; do
  curl -X POST http://localhost:8080/index/upload \
    -F "file=@$f" \
    -F "domain=code"
done
```

### Save/Load Index

```bash
# Save index to disk
curl -X POST "http://localhost:8080/index/save?path=./my_index"

# Load index from disk
curl -X POST "http://localhost:8080/index/load?path=./my_index"

# Clear index
curl -X DELETE http://localhost:8080/index
```

---

## JavaScript/TypeScript Examples

### Using Fetch API

```javascript
const API_BASE = 'http://localhost:8080';

// Health check
async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return response.json();
}

// Search
async function search(query, options = {}) {
  const response = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      top_k: options.topK || 10,
      domains: options.domains,
      source_type: options.sourceType,
      expand_query: options.expandQuery || false
    })
  });
  return response.json();
}

// List domains
async function listDomains() {
  const response = await fetch(`${API_BASE}/domains`);
  return response.json();
}

// Index content
async function indexContent(content, filename, domain = 'default') {
  const response = await fetch(`${API_BASE}/index/content`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      content,
      filename,
      domain,
      source_type: 'code'
    })
  });
  return response.json();
}

// Usage example
async function main() {
  // Check health
  const health = await checkHealth();
  console.log('API Status:', health.status);
  
  // Index some content
  await indexContent(
    'PROC validate; BEGIN END;',
    'test.tal',
    'payments'
  );
  
  // Search
  const results = await search('validate', { 
    domains: ['payments'],
    topK: 5 
  });
  
  console.log(`Found ${results.total_results} results`);
  results.results.forEach(r => {
    console.log(`${r.rank}. ${r.file_path} (${r.score.toFixed(3)})`);
  });
  
  // List domains
  const domains = await listDomains();
  console.log('Domains:', domains.domains.map(d => d.name));
}

main().catch(console.error);
```

### TypeScript Client Class

```typescript
interface SearchOptions {
  topK?: number;
  domains?: string[];
  sourceType?: 'code' | 'document' | 'log' | 'all';
  expandQuery?: boolean;
}

interface SearchResult {
  rank: number;
  score: number;
  file_path: string;
  domain: string;
  source_type: string;
  procedure_name?: string;
  content_preview: string;
  concepts: string[];
}

interface SearchResponse {
  query: string;
  total_results: number;
  search_time_ms: number;
  domains_searched?: string[];
  results: SearchResult[];
}

interface DomainInfo {
  name: string;
  chunk_count: number;
  percentage: number;
}

class UnifiedIndexerClient {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8080') {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    method: string,
    endpoint: string,
    body?: object
  ): Promise<T> {
    const options: RequestInit = {
      method,
      headers: { 'Content-Type': 'application/json' }
    };
    
    if (body) {
      options.body = JSON.stringify(body);
    }
    
    const response = await fetch(`${this.baseUrl}${endpoint}`, options);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
  }

  async health(): Promise<{ status: string; index_loaded: boolean; total_chunks: number }> {
    return this.request('GET', '/health');
  }

  async search(query: string, options: SearchOptions = {}): Promise<SearchResponse> {
    return this.request('POST', '/search', {
      query,
      top_k: options.topK || 10,
      domains: options.domains,
      source_type: options.sourceType,
      expand_query: options.expandQuery || false
    });
  }

  async listDomains(): Promise<{ total_domains: number; domains: DomainInfo[] }> {
    return this.request('GET', '/domains');
  }

  async indexContent(
    content: string,
    filename: string,
    domain: string = 'default',
    sourceType: string = 'code'
  ): Promise<{ success: boolean; chunks_created: number }> {
    return this.request('POST', '/index/content', {
      content,
      filename,
      domain,
      source_type: sourceType
    });
  }

  async clearIndex(): Promise<{ success: boolean }> {
    return this.request('DELETE', '/index');
  }
}

// Usage
const client = new UnifiedIndexerClient('http://localhost:8080');

async function example() {
  // Search with domain filter
  const results = await client.search('OFAC screening', {
    domains: ['compliance'],
    topK: 5
  });
  
  console.log(`Found ${results.total_results} results in ${results.search_time_ms}ms`);
  
  for (const result of results.results) {
    console.log(`${result.rank}. [${result.domain}] ${result.file_path}`);
    console.log(`   Score: ${result.score.toFixed(4)}`);
    console.log(`   Preview: ${result.content_preview.substring(0, 100)}...`);
  }
}
```

### React Hook Example

```typescript
import { useState, useCallback } from 'react';

interface UseSearchOptions {
  baseUrl?: string;
}

export function useSearch(options: UseSearchOptions = {}) {
  const baseUrl = options.baseUrl || 'http://localhost:8080';
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResponse | null>(null);

  const search = useCallback(async (
    query: string,
    searchOptions: SearchOptions = {}
  ) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${baseUrl}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          top_k: searchOptions.topK || 10,
          domains: searchOptions.domains,
          source_type: searchOptions.sourceType
        })
      });
      
      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      setResults(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Search failed';
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [baseUrl]);

  return { search, loading, error, results };
}

// Usage in component
function SearchComponent() {
  const { search, loading, error, results } = useSearch();
  const [query, setQuery] = useState('');
  const [domain, setDomain] = useState('all');

  const handleSearch = async () => {
    const domains = domain === 'all' ? undefined : [domain];
    await search(query, { domains, topK: 10 });
  };

  return (
    <div>
      <input 
        value={query} 
        onChange={e => setQuery(e.target.value)}
        placeholder="Search..."
      />
      <select value={domain} onChange={e => setDomain(e.target.value)}>
        <option value="all">All Domains</option>
        <option value="payments">Payments</option>
        <option value="compliance">Compliance</option>
      </select>
      <button onClick={handleSearch} disabled={loading}>
        {loading ? 'Searching...' : 'Search'}
      </button>
      
      {error && <div className="error">{error}</div>}
      
      {results && (
        <div>
          <p>Found {results.total_results} results in {results.search_time_ms}ms</p>
          {results.results.map(r => (
            <div key={r.rank}>
              <strong>{r.rank}. {r.file_path}</strong>
              <span>[{r.domain}]</span>
              <p>{r.content_preview.substring(0, 200)}...</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

---

## Node.js Examples

### Using Axios

```javascript
const axios = require('axios');

const client = axios.create({
  baseURL: 'http://localhost:8080',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' }
});

// Search
async function search(query, options = {}) {
  const { data } = await client.post('/search', {
    query,
    top_k: options.topK || 10,
    domains: options.domains,
    source_type: options.sourceType
  });
  return data;
}

// Index content
async function indexContent(content, filename, domain = 'default') {
  const { data } = await client.post('/index/content', {
    content,
    filename,
    domain,
    source_type: 'code'
  });
  return data;
}

// Upload file
const FormData = require('form-data');
const fs = require('fs');

async function uploadFile(filePath, domain = 'default') {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));
  
  const { data } = await client.post('/index/upload', form, {
    params: { domain },
    headers: form.getHeaders()
  });
  return data;
}

// Example usage
async function main() {
  // Index some TAL code
  await indexContent(`
    PROC process_payment;
    BEGIN
      CALL validate_amount;
      CALL debit_account;
    END;
  `, 'payment.tal', 'payments');
  
  // Search
  const results = await search('payment validation', {
    domains: ['payments'],
    topK: 5
  });
  
  console.log(`Found ${results.total_results} results`);
  results.results.forEach(r => {
    console.log(`  ${r.rank}. ${r.file_path} - ${r.score.toFixed(3)}`);
  });
}

main().catch(console.error);
```

---

## Integration Patterns

### Webhook on Index Update

```python
# In your indexing script
import requests

def notify_index_updated(domain, chunks_count):
    """Notify external system when index is updated"""
    webhook_url = "https://your-webhook-endpoint.com/index-updated"
    
    requests.post(webhook_url, json={
        "event": "index_updated",
        "domain": domain,
        "chunks_count": chunks_count,
        "timestamp": datetime.now().isoformat()
    })

# After indexing
result = client.index_directory("./code", domain="payments")
notify_index_updated("payments", result["chunks_created"])
```

### Scheduled Re-indexing

```python
# schedule_indexing.py
import schedule
import time

def reindex_all():
    """Re-index all directories"""
    client = UnifiedIndexerClient()
    
    directories = [
        ("./code/payments", "payments"),
        ("./code/compliance", "compliance"),
        ("./code/settlement", "settlement"),
    ]
    
    client.clear_index()
    
    for directory, domain in directories:
        result = client.index_directory(directory, domain=domain)
        print(f"Indexed {domain}: {result['chunks_created']} chunks")
    
    client.save_index()
    print("Index saved!")

# Schedule daily at 2 AM
schedule.every().day.at("02:00").do(reindex_all)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Search with Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_search(query_hash: str, domains_hash: str, top_k: int):
    """Cache search results for repeated queries"""
    # Reconstruct from hashes (simplified)
    return client.search(query_hash, top_k=top_k)

def search_with_cache(query: str, domains: list = None, top_k: int = 10):
    query_hash = hashlib.md5(query.encode()).hexdigest()
    domains_hash = hashlib.md5(str(sorted(domains or [])).encode()).hexdigest()
    return cached_search(query_hash, domains_hash, top_k)
```
