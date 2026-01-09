# Unified Indexer - CLI Reference Guide

Complete command-line reference for all tools in the Unified Indexer package.

---

## Table of Contents

1. [build_index.py](#build_indexpy) - Build search index
2. [search_index.py](#search_indexpy) - Search indexed content
3. [knowledge_extractor.py](#knowledge_extractorpy) - Extract domain vocabulary
4. [llm_config_loader.py](#llm_config_loaderpy) - LLM configuration
5. [diagnose_parser.py](#diagnose_parserpy) - Parser diagnostics
6. [Test Suites](#test-suites) - Running tests

---

## build_index.py

Build a search index from TAL code, documents, and other source files.

### Basic Usage

```bash
# Index TAL code directory
python build_index.py --tal-dir ./code --output ./my_index

# Index with domain tag
python build_index.py --tal-dir ./code --output ./my_index --domain payments

# Index multiple directories with different domains
python build_index.py \
    --tal-dir ./payments --domain payments --output ./my_index
python build_index.py \
    --tal-dir ./compliance --domain compliance --output ./my_index --incremental

# Infer domains from directory names
python build_index.py \
    --tal-dir ./code/payments \
    --pdf-dir ./docs/compliance \
    --output ./my_index \
    --infer-domains
```

### All Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--tal-dir DIR` | | Directory containing TAL code files | |
| `--pdf-dir DIR` | | Directory containing PDF documents | |
| `--code-dir DIR` | | Directory containing other code (C, Java, Python) | |
| `--output DIR` | `-o` | **Required.** Output directory for index | |
| `--add-file FILE` | | Add single file to existing index (repeatable) | |
| `--incremental` | `-i` | Only index new/modified files | False |
| `--domain NAME` | `-D` | Domain tag for all indexed files | `default` |
| `--domain-map FILE` | | JSON file mapping domains to directories | |
| `--infer-domains` | | Infer domain from parent directory name | False |
| `--vocab FILE` | `-v` | Vocabulary JSON file | `keywords.json` |
| `--recursive` | `-r` | Search directories recursively | True |
| `--no-recursive` | | Don't search recursively | |
| `--embedder TYPE` | `-e` | Embedder type (see below) | `hash` |
| `--dims N` | `-d` | Embedding dimensions | 1024 |
| `--domain-weight N` | | Weight for domain concepts (0.0-1.0) | 0.6 |
| `--knowledge-graph FILE` | `-kg` | Knowledge graph JSON for boosting | |
| `--tfidf-weight` | | Apply TF-IDF weights from KG | Auto |
| `--no-tfidf-weight` | | Disable TF-IDF weighting | |
| `--verbose` | `-V` | Show detailed progress | False |

### Domain Mapping File (domains.json)

```json
{
    "payments": {
        "directories": ["./code/payments", "./docs/payments"]
    },
    "compliance": {
        "directories": ["./code/compliance", "./docs/compliance"]
    },
    "settlement": {
        "directories": ["./code/settlement"]
    }
}
```

```bash
# Use domain mapping
python build_index.py --domain-map domains.json --output ./my_index
```

### Embedder Types

| Type | Description |
|------|-------------|
| `hash` | Fast locality-sensitive hashing (default) |
| `hybrid` | Hash + domain vocabulary vectors |
| `tfidf` | TF-IDF weighted embeddings |
| `domain` | Domain-concept focused |
| `bm25` | BM25 scoring |
| `payment` | Payment domain optimized |
| `payment_hybrid` | Payment + hybrid |

### Examples

```bash
# Basic index with verbose output
python build_index.py \
    --tal-dir ./tal_code \
    --output ./payment_index \
    --verbose

# Index with knowledge graph (better term weighting)
python build_index.py \
    --tal-dir ./code \
    --pdf-dir ./docs \
    --output ./my_index \
    --knowledge-graph ./knowledge_graph.json

# Incremental update (add new files only)
python build_index.py \
    --tal-dir ./code \
    --output ./my_index \
    --incremental

# Add single file to existing index
python build_index.py \
    --add-file ./new_procedure.tal \
    --output ./my_index

# Using hybrid embedder with custom dimensions
python build_index.py \
    --tal-dir ./code \
    --output ./my_index \
    --embedder hybrid \
    --dims 512 \
    --domain-weight 0.7
```

---

## search_index.py

Search through indexed content with semantic search, query expansion, and LLM analysis.

### Basic Usage

```bash
# Simple search
python search_index.py --index ./my_index --query "OFAC screening"

# Interactive mode
python search_index.py --index ./my_index --interactive

# Search with LLM analysis
python search_index.py --index ./my_index --query "wire transfer" --analyze
```

### All Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--index DIR` | `-i` | **Required.** Index directory | |
| `--query TEXT` | `-q` | Search query | |
| `--top N` | `-n` | Number of results | 5 |
| `--type TYPE` | `-t` | Filter: `code`, `document`, `log`, `all` | `all` |
| `--domain NAMES` | `-D` | Filter by domain(s), comma-separated | all |
| `--list-domains` | | List available domains and exit | |
| `--interactive` | `-I` | Interactive search mode | False |
| `--capability CAP` | `-c` | Search by business capability | |
| `--verbose` | `-v` | Show detailed results | False |
| `--vocab FILE` | | Vocabulary JSON file | `keywords.json` |

#### LLM Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--analyze` | `-a` | Analyze results with LLM | False |
| `--provider NAME` | `-p` | LLM provider | `tachyon` |
| `--model NAME` | `-m` | LLM model name | Provider default |
| `--min-score N` | | Minimum score for LLM analysis | 0.0 |
| `--full-file` | `-f` | Send full file to LLM (not chunks) | False |

#### Knowledge Graph Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--knowledge-graph FILE` | `-kg` | Knowledge graph JSON | |
| `--expand-query` | `-e` | Expand query with related terms | False |
| `--tfidf-boost` | `-b` | Boost by term distinctiveness | False |
| `--decompose` | `-d` | Break long queries into sub-queries | False |
| `--no-related` | | Don't show related terms | False |
| `--no-kg` | | Don't auto-load embedded KG | False |

### LLM Providers

| Provider | Description |
|----------|-------------|
| `tachyon` | Internal API (default) |
| `internal` | Internal API (alias) |
| `anthropic` | Anthropic Claude |
| `openai` | OpenAI GPT |
| `ollama` | Local Ollama |
| `stub` | Test stub (no API calls) |

### Examples

```bash
# List available domains
python search_index.py --index ./my_index --list-domains

# Search single domain
python search_index.py \
    --index ./my_index \
    --query "wire transfer" \
    --domain payments

# Search multiple domains
python search_index.py \
    --index ./my_index \
    --query "validation" \
    --domain payments,compliance

# Search code only
python search_index.py \
    --index ./my_index \
    --query "validate transfer" \
    --type code

# Search with query expansion
python search_index.py \
    --index ./my_index \
    --query "OFAC" \
    --knowledge-graph ./knowledge_graph.json \
    --expand-query

# Full knowledge graph features
python search_index.py \
    --index ./my_index \
    --query "sanctions screening" \
    --knowledge-graph ./knowledge_graph.json \
    --expand-query \
    --tfidf-boost

# LLM analysis with specific provider
python search_index.py \
    --index ./my_index \
    --query "implement wire transfer validation" \
    --analyze \
    --provider openai \
    --model gpt-4-turbo

# Decompose long queries
python search_index.py \
    --index ./my_index \
    --query "How does the system validate wire transfers, check OFAC sanctions, and handle errors?" \
    --decompose

# Verbose output with more results
python search_index.py \
    --index ./my_index \
    --query "error handling" \
    --top 20 \
    --verbose
```

### Interactive Mode Commands

Start with: `python search_index.py --index ./my_index --interactive`

| Command | Description |
|---------|-------------|
| `<query>` | Search for query |
| `:q`, `:quit` | Exit |
| `:help` | Show help |
| `:code <query>` | Search code only |
| `:doc <query>` | Search documents only |
| `:log <query>` | Search logs only |
| `:domain <name>` | Set domain filter (e.g., `:domain payments`) |
| `:domain <n1>,<n2>` | Set multiple domains |
| `:domain all` | Clear domain filter |
| `:domains` | List available domains |
| `:analyze <query>` | Search and analyze with LLM |
| `:expand` | Toggle query expansion |
| `:boost` | Toggle TF-IDF boosting |
| `:verbose` | Toggle verbose output |
| `:top N` | Set number of results |
| `:lookup <term>` | Look up term in knowledge graph |
| `:graph` | Show knowledge graph statistics |
| `:index` | Show index statistics |

---

## knowledge_extractor.py

Extract domain vocabulary and build knowledge graph from source documents and code.

### Basic Usage

```bash
# Extract from documents and code
python knowledge_extractor.py \
    --docs ./documents \
    --code ./tal_code \
    --output ./vocabulary.json

# Extract without LLM (heuristics only)
python knowledge_extractor.py \
    --docs ./documents \
    --code ./tal_code \
    --output ./vocabulary.json \
    --no-llm
```

### All Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--docs DIR` | `-d` | Document directory (PDFs, text) | |
| `--code DIR` | `-c` | Code directory (TAL, COBOL, etc.) | |
| `--output FILE` | `-o` | Output vocabulary JSON | |
| `--graph FILE` | `-g` | Output knowledge graph JSON | `knowledge_graph.json` |
| `--stats FILE` | `-s` | Output TF-IDF statistics JSON | `term_statistics.json` |
| `--existing FILE` | `-e` | Existing vocabulary to augment | |
| `--provider NAME` | `-p` | LLM provider for extraction | |
| `--model NAME` | `-m` | LLM model name | |
| `--no-llm` | | Heuristic extraction only | False |
| `--verbose` | `-v` | Verbose output | False |
| `--append` | `-a` | Append to existing extraction | False |

### Examples

```bash
# Full extraction with LLM
python knowledge_extractor.py \
    --docs ./pdfs \
    --code ./tal \
    --output ./vocab.json \
    --graph ./kg.json \
    --provider openai \
    --model gpt-4

# Heuristic only (no API calls)
python knowledge_extractor.py \
    --docs ./pdfs \
    --code ./tal \
    --output ./vocab.json \
    --no-llm \
    --verbose

# Append to existing extraction
python knowledge_extractor.py \
    --code ./new_code \
    --graph ./kg.json \
    --append

# Augment existing vocabulary
python knowledge_extractor.py \
    --docs ./new_docs \
    --existing ./old_vocab.json \
    --output ./merged_vocab.json
```

### Output Files

| File | Description |
|------|-------------|
| `vocabulary.json` | Domain terms with categories and relationships |
| `knowledge_graph.json` | Term co-occurrence and relationship graph |
| `term_statistics.json` | TF-IDF scores for term distinctiveness |

---

## llm_config_loader.py

Configure LLM provider settings via config file or environment variables.

### Config File (`llm_config.json`)

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

### Config File Locations (searched in order)

1. `./llm_config.json`
2. `./config/llm_config.json`
3. `~/.config/llm_config.json`
4. `$LLM_CONFIG_PATH` environment variable

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_MODEL` | Model name (overrides config) |
| `LLM_BASE_URL` | API base URL (overrides config) |
| `LLM_API_KEY` | API key (overrides config) |
| `LLM_CONFIG_PATH` | Path to config file |
| `INTERNAL_API_KEY` | API key for internal profile |
| `OPENAI_API_KEY` | API key for OpenAI profile |
| `ANTHROPIC_API_KEY` | API key for Anthropic profile |

### Python Usage

```python
from llm_config_loader import load_provider, create_provider

# Load default profile
provider = load_provider()

# Load specific profile
provider = load_provider("openai")

# Override settings
provider = load_provider(model="gpt-4-turbo", base_url="http://custom:8000")

# Backwards compatible
provider = create_provider("internal", "gpt-4")
```

### Testing Config

```bash
python llm_config_loader.py
```

---

## diagnose_parser.py

Diagnostic tool for troubleshooting TAL parser issues.

### Usage

```bash
python diagnose_parser.py
```

No command-line arguments. Runs automatically and tests:

1. `tal_enhanced_parser.py` - Direct import and parsing
2. `tal_proc_parser.py` - Procedure detection
3. `unified_indexer/parsers/tal_parser.py` - Integration layer
4. Full parsing through `TalCodeParser`
5. Source code checks for deprecated patterns

### Expected Output

```
============================================================
TEST 1: tal_enhanced_parser.py directly
============================================================
  ✓ PASSED

============================================================
TEST 2: tal_proc_parser.py
============================================================
  ✓ PASSED

============================================================
TEST 3: unified_indexer/parsers/tal_parser.py
============================================================
  Has tal_enhanced_parser: True
  Has tal_proc_parser: True

============================================================
TEST 4: Parse TAL code through TalCodeParser
============================================================
  Got 1 chunks
  ✓ PASSED

============================================================
TEST 5: Search for .get() patterns in source
============================================================
  ✓ No deprecated patterns found

============================================================
DIAGNOSIS COMPLETE
============================================================
```

---

## Test Suites

### test_full_suite.py (84 tests)

Comprehensive test suite covering all components.

```bash
# Run all tests
python test_full_suite.py

# Expected output
RESULTS: 84/84 passed, 0 failed, 0 skipped
```

**Test Categories:**
- TAL_PROC_PARSER (8 tests)
- TAL_ENHANCED_PARSER (28 tests)
- VOCABULARY (4 tests)
- EMBEDDING (5 tests)
- PARSER (4 tests)
- INDEX (5 tests)
- PIPELINE (3 tests)
- SEARCH_QUALITY (2 tests)
- ERROR_HANDLING (6 tests)
- LONG_QUERIES (11 tests)
- PERFORMANCE (3 tests)

### test_tal_parsers.py (63 tests)

TAL parser specific tests.

```bash
python test_tal_parsers.py

# Expected output
RESULTS: 63/63 passed
```

### test_comprehensive.py (55 tests)

Integration and end-to-end tests.

```bash
python test_comprehensive.py

# Expected output
RESULTS: 55/55 passed
```

---

## Common Workflows

### 1. Initial Setup

```bash
# Extract package
unzip unified_indexer_full.zip
cd unified_indexer_full

# Install dependencies
pip install -r unified_indexer/requirements.txt

# Run tests to verify
python test_full_suite.py
```

### 2. Build Index from Scratch

```bash
# Step 1: Extract vocabulary (optional but recommended)
python knowledge_extractor.py \
    --docs ./documents \
    --code ./tal_code \
    --output ./vocab.json \
    --no-llm

# Step 2: Build index
python build_index.py \
    --tal-dir ./tal_code \
    --pdf-dir ./documents \
    --output ./my_index \
    --vocab ./vocab.json \
    --knowledge-graph ./knowledge_graph.json \
    --verbose
```

### 3. Search Workflow

```bash
# Basic search
python search_index.py -i ./my_index -q "wire transfer"

# Enhanced search with all features
python search_index.py \
    --index ./my_index \
    --query "OFAC sanctions" \
    --knowledge-graph ./knowledge_graph.json \
    --expand-query \
    --tfidf-boost \
    --verbose

# Interactive exploration
python search_index.py --index ./my_index --interactive
```

### 4. Update Existing Index

```bash
# Add new files incrementally
python build_index.py \
    --tal-dir ./new_code \
    --output ./my_index \
    --incremental

# Add single file
python build_index.py \
    --add-file ./hotfix.tal \
    --output ./my_index
```

### 5. Troubleshooting

```bash
# Check parser health
python diagnose_parser.py

# Run full test suite
python test_full_suite.py

# Verbose search to see scoring
python search_index.py \
    --index ./my_index \
    --query "test" \
    --verbose
```

---

## Environment Setup

### Required Environment Variables

```bash
# For LLM features (optional)
export INTERNAL_API_KEY=your-key
# or
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=...

# Override LLM settings
export LLM_MODEL=gpt-4
export LLM_BASE_URL=http://localhost:8000
```

### Dependencies

```bash
pip install numpy           # Required
pip install PyMuPDF        # Optional: better PDF parsing
pip install requests       # Optional: LLM API calls
```

---

## Quick Reference Card

```bash
# BUILD INDEX
python build_index.py -o INDEX --tal-dir CODE [--pdf-dir DOCS] [-V]

# SEARCH
python search_index.py -i INDEX -q "QUERY" [-n 10] [-v] [--analyze]

# INTERACTIVE SEARCH
python search_index.py -i INDEX -I

# EXTRACT VOCABULARY
python knowledge_extractor.py -d DOCS -c CODE -o vocab.json [--no-llm]

# RUN TESTS
python test_full_suite.py

# DIAGNOSE ISSUES
python diagnose_parser.py
```
