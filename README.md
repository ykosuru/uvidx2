# Unified Indexer for Payment Systems

A comprehensive semantic search and indexing system for legacy TAL/COBOL code, documentation, and logs with domain-specific vocabulary support for payment processing.

## Features

- **Multi-format parsing**: TAL, COBOL, Python, Java, C, Markdown, HTML, logs
- **Semantic search**: Vector embeddings + BM25 + concept matching with RRF fusion
- **Domain vocabulary**: Payment/wire transfer terminology with relationship expansion
- **TAL Enhanced Parser**: Full program parsing with call graphs, complexity metrics
- **LLM Integration**: Configurable provider for code analysis and explanation
- **Knowledge Graph**: Term co-occurrence and procedure relationships

## Quick Start

### 1. Install

```bash
unzip unified_indexer.zip
cd unified_indexer
pip install -r unified_indexer/requirements.txt
```

### 2. Build Index

```bash
# Index TAL code
python build_index.py \
    --tal-dir /path/to/TAL \
    --output ./my_index \
    --vocab keywords.json \
    --verbose

# Index multiple directories
python build_index.py \
    --tal-dir /path/to/TAL \
    --doc-dir /path/to/docs \
    --log-dir /path/to/logs \
    --output ./my_index
```

### 3. Search

```bash
# Basic search
python search_index.py --index ./my_index --query "OFAC sanctions screening"

# With query expansion
python search_index.py --index ./my_index --query "wire transfer" --expand-query

# Interactive mode
python search_index.py --index ./my_index --interactive
```

### 4. Run Tests

```bash
# Full test suite (84 tests)
python test_full_suite.py

# TAL parser tests (63 tests)
python test_tal_parsers.py

# If having issues
python diagnose_parser.py
```

## Directory Structure

```
.
├── build_index.py           # Index builder CLI
├── search_index.py          # Search CLI with LLM integration
├── llm_provider.py          # LLM provider base class
├── llm_config_loader.py     # Config file loader for LLM
├── llm_config.json          # LLM configuration profiles
├── knowledge_extractor.py   # Domain term extraction
├── keywords.json            # Payment domain vocabulary
│
├── tal_proc_parser.py       # Basic TAL procedure parser
├── tal_enhanced_parser.py   # Full TAL program parser
│
├── unified_indexer/         # Core package
│   ├── __init__.py
│   ├── pipeline.py          # Main indexing pipeline
│   ├── index.py             # Hybrid search index
│   ├── embeddings.py        # Embedding providers
│   ├── vocabulary.py        # Domain vocabulary
│   ├── models.py            # Data models
│   ├── call_graph.py        # Call graph analysis
│   └── parsers/
│       ├── base.py          # Parser base class
│       ├── code_parser.py   # Generic code parser
│       ├── tal_parser.py    # TAL integration
│       ├── document_parser.py
│       └── log_parser.py
│
├── test_full_suite.py       # Comprehensive tests (84)
├── test_tal_parsers.py      # TAL parser tests (63)
├── test_comprehensive.py    # Integration tests (55)
├── diagnose_parser.py       # Parser diagnostic tool
│
└── ARCHITECTURE.md          # Detailed architecture docs
```

## Configuration

### LLM Configuration (`llm_config.json`)

```json
{
    "default_profile": "internal",
    "profiles": {
        "internal": {
            "model": "gpt-4",
            "base_url": "http://localhost:8000",
            "api_key_env": "INTERNAL_API_KEY"
        },
        "openai": {
            "model": "gpt-4-turbo",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY"
        },
        "local": {
            "model": "llama-3-8b",
            "base_url": "http://localhost:11434/v1",
            "api_key": null
        }
    }
}
```

### Using LLM Provider

```python
from llm_config_loader import load_provider

# Load default profile
provider = load_provider()

# Load specific profile
provider = load_provider("openai")

# Override settings
provider = load_provider(model="gpt-4-turbo", base_url="http://custom:8000")
```

### Environment Variables

```bash
# LLM settings (override config file)
export LLM_MODEL=gpt-4
export LLM_BASE_URL=http://localhost:8000
export LLM_API_KEY=your-key

# Or profile-specific
export INTERNAL_API_KEY=your-internal-key
export OPENAI_API_KEY=sk-...
```

## Search Features

### Query Types

```bash
# Simple search
--query "OFAC screening"

# With query expansion (adds related terms)
--query "OFAC screening" --expand-query

# Search specific content type
--query "error handling" --source-type code

# Analyze with LLM
--query "wire transfer flow" --analyze
```

### Interactive Commands

```
:help              Show commands
:q, :quit          Exit
:code <query>      Search only code
:doc <query>       Search only documents
:analyze <query>   Search and analyze with LLM
:expand            Toggle query expansion
:verbose           Toggle verbose output
```

## TAL Parser Features

### Basic Parser (`tal_proc_parser.py`)

- Procedure declarations
- Parameter extraction
- Symbol table management
- MAIN/FORWARD/EXTERNAL detection

### Enhanced Parser (`tal_enhanced_parser.py`)

- Full program parsing
- DEFINE/LITERAL/STRUCT extraction
- Call graph generation
- Cyclomatic complexity metrics
- SUBPROC support
- Procedure body analysis

```python
from tal_enhanced_parser import parse_tal_string, get_call_graph

result = parse_tal_string(code)
print(f"Procedures: {len(result.procedures)}")
print(f"Defines: {len(result.defines)}")
print(f"Structs: {len(result.structs)}")
print(f"Call graph: {result.call_graph}")
```

## API Usage

### Programmatic Indexing

```python
from unified_indexer.pipeline import IndexingPipeline

# Create pipeline
pipeline = IndexingPipeline(
    vocabulary_data=[
        {"term": "wire transfer", "category": "payment"},
        {"term": "OFAC", "category": "compliance"},
    ],
    embedder_type="hash"
)

# Index directory
stats = pipeline.index_directory("/path/to/code")
print(f"Indexed {stats.files_processed} files, {stats.total_chunks} chunks")

# Search
results = pipeline.search("wire transfer validation", top_k=10)
for r in results:
    print(f"{r.chunk.source_ref.file_path}: {r.combined_score:.3f}")

# Save index
pipeline.save("./my_index")
```

### Loading Existing Index

```python
pipeline = IndexingPipeline.load("./my_index")
results = pipeline.search("OFAC screening")
```

## Test Summary

| Test Suite | Tests | Description |
|------------|-------|-------------|
| test_full_suite.py | 84 | All components + long queries |
| test_tal_parsers.py | 63 | TAL parser specific |
| test_comprehensive.py | 55 | Integration tests |

### Long Query Support

Tested with queries up to 1000+ words:

| Query Size | Response Time |
|------------|---------------|
| 50 words | ~1ms |
| 100 words | ~1ms |
| 500 words | ~2ms |
| 1000 words | ~3.5ms |

## Troubleshooting

### Parser Issues

```bash
# Run diagnostic
python diagnose_parser.py
```

### Common Issues

1. **"ParseResult has no attribute get"**
   - Replace `unified_indexer/parsers/tal_parser.py` with latest version

2. **"Enhanced parser failed, falling back"**
   - Check `tal_enhanced_parser.py` is in project root
   - Check `tal_proc_parser.py` is in project root

3. **"No module named unified_indexer"**
   - Run from project root directory
   - Ensure `unified_indexer/` folder exists

## Performance

- Index 100 files: ~0.17s
- Search (avg): ~1.4ms
- Parse 500 TAL procedures: ~0.2s

## License

Internal use only.

## Version

- Package version: 1.0.0
- Test suite: 84 tests passing
- Last updated: January 2025
