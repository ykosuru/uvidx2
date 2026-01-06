# Unified Indexer Architecture

A hybrid search system for legacy code and documentation that combines vector similarity, BM25 lexical search, and domain vocabulary matching with Reciprocal Rank Fusion.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           UNIFIED INDEXER                                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     OFFLINE: INDEX BUILDING                          │    │
│  │                                                                      │    │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐                        │    │
│  │   │ TAL/COBOL│   │   PDFs   │   │   Logs   │   Source Files         │    │
│  │   │   Code   │   │   Docs   │   │  (JSON)  │                        │    │
│  │   └────┬─────┘   └────┬─────┘   └────┬─────┘                        │    │
│  │        │              │              │                               │    │
│  │        └──────────────┼──────────────┘                               │    │
│  │                       ▼                                              │    │
│  │              ┌─────────────────┐                                     │    │
│  │              │  Content Parser │  Language-specific parsing          │    │
│  │              │  + Vocabulary   │  + domain concept extraction        │    │
│  │              └────────┬────────┘                                     │    │
│  │                       ▼                                              │    │
│  │              ┌─────────────────┐                                     │    │
│  │              │ IndexableChunks │  Procedures, sections, log entries  │    │
│  │              └────────┬────────┘                                     │    │
│  │                       │                                              │    │
│  │        ┌──────────────┼──────────────┐                               │    │
│  │        ▼              ▼              ▼                               │    │
│  │   ┌─────────┐   ┌──────────┐   ┌──────────┐                         │    │
│  │   │ Vector  │   │  BM25    │   │ Concept  │   Three indexes         │    │
│  │   │  Store  │   │  Index   │   │  Index   │   built in parallel     │    │
│  │   └─────────┘   └──────────┘   └──────────┘                         │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      ONLINE: SEARCH                                  │    │
│  │                                                                      │    │
│  │   Query: "OFAC sanctions screening"                                  │    │
│  │        │                                                             │    │
│  │        ▼                                                             │    │
│  │   ┌─────────────────────────────────────────────────────────┐       │    │
│  │   │            KNOWLEDGE GRAPH EXPANSION                     │       │    │
│  │   │  "OFAC" → "OFAC sanctions SDN screening compliance"      │       │    │
│  │   └────────────────────────┬────────────────────────────────┘       │    │
│  │                            ▼                                         │    │
│  │        ┌───────────────────┼───────────────────┐                    │    │
│  │        ▼                   ▼                   ▼                    │    │
│  │   ┌─────────┐        ┌──────────┐        ┌──────────┐              │    │
│  │   │ Vector  │        │  BM25    │        │ Concept  │              │    │
│  │   │ Search  │        │  Search  │        │  Search  │              │    │
│  │   │(semantic)│       │(lexical) │        │(vocabulary)│             │    │
│  │   └────┬────┘        └────┬─────┘        └────┬─────┘              │    │
│  │        │                  │                   │                     │    │
│  │        └──────────────────┼───────────────────┘                     │    │
│  │                           ▼                                         │    │
│  │              ┌────────────────────────┐                             │    │
│  │              │ Reciprocal Rank Fusion │                             │    │
│  │              │    RRF(d) = Σ 1/(k+r)  │                             │    │
│  │              └───────────┬────────────┘                             │    │
│  │                          ▼                                          │    │
│  │                   Ranked Results                                    │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Extract knowledge graph from corpus (one-time)
python knowledge_extractor.py --input ./code ./docs --output ./kg_output

# 2. Build index with knowledge graph
python build_index.py --tal-dir ./code --pdf-dir ./docs \
    --knowledge-graph ./kg_output/knowledge_graph.json \
    --output ./index

# 3. Search
python search_index.py --index ./index --query "OFAC sanctions screening"

# Or interactive mode
python search_index.py --index ./index --interactive
```

---

## Components

### 1. Knowledge Extractor (`knowledge_extractor.py`)

Analyzes your corpus to build a knowledge graph:

```
knowledge_graph.json
├── nodes: Terms with TF-IDF scores, document frequency, source files
├── edges: Relationships (co_occurs_with, implements, contains)
└── statistics: Corpus-level metrics

vocabulary.json
└── Domain terms extracted from corpus, organized by category

term_statistics.json
├── Document frequency for each term
├── TF-IDF scores
└── Co-occurrence counts
```

**Why it matters:** The knowledge graph captures term importance and relationships that enhance search quality - no training required.

---

### 2. Index Builder (`build_index.py`)

Creates three parallel indexes:

```
./index/
├── index.pkl              # Vector store + BM25 index + concept index
├── index_meta.json        # Build configuration and statistics
├── knowledge_graph.json   # Embedded for search-time use
└── expansion_map.json     # Term relationships for query expansion
```

**What gets indexed:**

| Source Type | Parser | Chunking Strategy |
|-------------|--------|-------------------|
| TAL/COBOL | TalCodeParser | One chunk per procedure |
| PDF/DOCX | DocumentParser | Section-aware splitting |
| Logs | LogParser | Transaction/trace grouping |
| C/Java/Python | CodeParser | Function/class extraction |

---

### 3. Search Engine (`search_index.py`)

Hybrid search combining three signals via RRF:

```
Query: "OFAC screening"
         │
         ▼
   Query Expansion (via Knowledge Graph)
   "OFAC" → "OFAC sanctions SDN compliance"
         │
         ├─────────────────┬─────────────────┐
         ▼                 ▼                 ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │  VECTOR   │    │   BM25    │    │  CONCEPT  │
   │  SEARCH   │    │  SEARCH   │    │  SEARCH   │
   ├───────────┤    ├───────────┤    ├───────────┤
   │ Semantic  │    │ Lexical   │    │ Vocabulary│
   │ similarity│    │ + IDF     │    │ structure │
   └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
         │                │                │
         │ Ranks: A,B,C   │ Ranks: B,D,A   │ Ranks: A,C
         │                │                │
         └────────────────┼────────────────┘
                          ▼
                ┌─────────────────┐
                │  RRF FUSION     │
                │                 │
                │ A: 1/61 + 1/63  │
                │    + 1/61 = .049│  ← Winner
                │ B: 1/62 + 1/61  │
                │         = .033  │
                └────────┬────────┘
                         ▼
                  Ranked Results
```

---

## Retrieval Algorithms

### Vector Search (Hash Embeddings)

Captures **semantic similarity** using feature hashing:

```
text = "OFAC sanctions screening"
                ↓
features = ["OF", "FA", "AC", "OFAC", "sanctions", "screening", ...]
                ↓
hash(feature) % n_dimensions → sparse vector
                ↓
cosine_similarity(query_vec, doc_vec)
```

| Strengths | Weaknesses |
|-----------|------------|
| Finds semantically related content | May miss exact technical terms |
| No vocabulary limitations | No term importance weighting |
| Fast, no fitting required | |

---

### BM25 Search (Lexical)

Captures **exact term matching** with IDF weighting:

```
BM25(D, Q) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1×(1-b+b×|D|/avgdl))

where:
  f(qi, D)  = term frequency in document
  |D|       = document length
  avgdl     = average document length
  k1 = 1.5  = term frequency saturation
  b  = 0.75 = length normalization
```

| Strengths | Weaknesses |
|-----------|------------|
| Precise term matching (UETR, OFAC) | No semantic understanding |
| Rare terms score higher (IDF) | Vocabulary mismatch issues |
| Interpretable scoring | |

---

### Concept Search (Vocabulary)

Captures **domain structure** via controlled vocabulary:

```python
vocabulary = {
    "OFAC": {
        "keywords": ["ofac", "sanctions", "sdn"],
        "capabilities": ["Compliance", "Screening"]
    }
}

query = "OFAC screening"
      ↓
matched_concepts = ["OFAC"]
      ↓
chunks = concept_index.search("OFAC")  # All chunks tagged with OFAC
```

| Strengths | Weaknesses |
|-----------|------------|
| Domain-aware matching | Limited to known vocabulary |
| Consistent terminology | No semantic expansion |
| Capability-based filtering | |

---

### Reciprocal Rank Fusion (RRF)

Combines results using **ranks** instead of scores:

```
RRF_score(d) = Σ 1/(k + rank(d))

where k = 60 (standard constant)
```

**Why RRF?**

| Problem | RRF Solution |
|---------|--------------|
| BM25 scores are unbounded (0-∞) | Uses ranks (1, 2, 3...) |
| Vector scores are 0-1 | Uses ranks |
| Different score distributions | Ranks normalize naturally |

**Example:**
```
Vector: A=1, B=2, C=3    BM25: B=1, D=2, A=3    Concept: A=1, C=2

RRF scores:
  A: 1/61 + 1/63 + 1/61 = 0.049  ← Found by all 3, wins
  B: 1/62 + 1/61        = 0.033
  C: 1/63 + 1/62        = 0.032
  D: 1/62               = 0.016
```

---

## Knowledge Graph Integration

### At Index Time

```python
# Load knowledge graph
kg = load_knowledge_graph("knowledge_graph.json")

# Apply TF-IDF weights to vocabulary (high=1.5, low=0.5)
vocab = apply_tfidf_weights(vocab, kg)

# Embed in index directory for search-time use
save(kg, index_dir + "/knowledge_graph.json")
```

### At Search Time

```python
# Query expansion: "OFAC" → "OFAC sanctions SDN screening"
expanded = kg.expand_query(query)

# TF-IDF boosting: boost results matching distinctive terms
results = apply_tfidf_boost(results, kg, query)

# Related terms: show "Related: SDN_LIST, VALIDATE_BIC"
related = kg.get_related_terms(concept)
```

---

## Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: KNOWLEDGE EXTRACTION (one-time)                                  │
│                                                                          │
│   ./code/*.tal ─┐                                                        │
│   ./docs/*.pdf ─┼──▶ knowledge_extractor.py ──▶ knowledge_graph.json    │
│   ./logs/*.json─┘                               vocabulary.json          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: INDEX BUILDING                                                   │
│                                                                          │
│   Source files ─────▶ build_index.py ──▶ Vector + BM25 + Concept Index  │
│   knowledge_graph.json ─┘                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: SEARCH (repeated)                                                │
│                                                                          │
│   Query ──▶ search_index.py ──▶ Expand → Search → RRF → Results         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### build_index.py

```bash
python build_index.py \
    --tal-dir ./code              # TAL/COBOL source
    --pdf-dir ./docs              # PDF documentation  
    --code-dir ./other            # Other code (C, Java, Python)
    --knowledge-graph ./kg.json   # Knowledge graph
    --output ./index              # Output directory
    --embedder hash               # hash (default), hybrid, tfidf, bm25
    --dims 1024                   # Embedding dimensions
```

### Incremental Indexing

Add files to an existing index without rebuilding:

```bash
# Add a single file
python build_index.py --add-file ./new_document.pdf --output ./my_index

# Add multiple files
python build_index.py --add-file ./doc1.pdf --add-file ./doc2.pdf --output ./my_index

# Incremental directory scan (only new/modified files)
python build_index.py --pdf-dir ./docs --output ./my_index --incremental

# Re-scan all directories, adding only changes
python build_index.py --pdf-dir ./docs --tal-dir ./code \
    --output ./my_index --incremental
```

**How it works:**
1. Loads existing index from `--output` directory
2. Checks file manifest (`file_manifest.json`) for already-indexed files
3. Compares modification time and size to detect changes
4. Only indexes new or modified files
5. Updates manifest with newly indexed files

**File manifest tracks:**
- File path (absolute)
- Modification time
- File size

Files are re-indexed if mtime or size changes.

### Concurrent Access Safety

The index uses **generation-based versioning** for lock-free concurrent access:

```
┌─────────────────────────────────────────────────────────────────┐
│              GENERATION-BASED VERSIONING                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ./my_index/                                                     │
│  ├── generation.txt  ──► "2"  (atomic pointer)                  │
│  ├── gen_1/          ──► old data (will be cleaned up)          │
│  │   ├── chunks.json                                            │
│  │   └── embeddings.npy                                         │
│  └── gen_2/          ──► current data                           │
│      ├── chunks.json                                            │
│      └── embeddings.npy                                         │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  WRITE (build_index --incremental):                             │
│  1. Read generation.txt → 2                                      │
│  2. Write all files to gen_3/                                   │
│  3. Atomic: generation.txt → "3"                                │
│  4. Cleanup old gen_1/                                          │
│                                                                  │
│  READ (search_index):                                           │
│  1. Read generation.txt → 2                                      │
│  2. Load from gen_2/                                            │
│  3. Search in memory (never touches disk again)                 │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CONCURRENT SCENARIO:                                           │
│                                                                  │
│  Search A ─── reads gen 2 ───────────────────► continues        │
│                    │                              │              │
│  Build ──────────────┼── writes gen 3 ──► atomic update         │
│                      │                              │            │
│  Search B ───────────┴───────── reads gen 3 ──────►             │
│                                                                  │
│  ✓ No blocking                                                  │
│  ✓ Always consistent (all old OR all new)                       │
│  ✓ Searches never see partial writes                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Guarantees:**
- **Non-blocking**: Builds never block searches
- **Consistent snapshots**: Readers see complete generation (never mixed)
- **Atomic pointer**: Single file write switches generations
- **Auto-cleanup**: Old generations removed (keeps last 2)
- **Legacy compatible**: Reads old format without generation subdirs

**Verbose logging** (`--verbose` / `-V`):
```
============================================================
GENERATION UPDATE: 2 → 3
============================================================
  📂 Current generation: 2
  📂 Writing new generation: 3
  📁 Target directory: /tmp/test_index/gen_3

  Writing gen_3/...
    • chunks.json (6 chunks)
    • embeddings.npy (6 vectors, 1024 dims)
    • bm25_index.json (57 terms)
    • metadata.json

  ⚡ ATOMIC POINTER SWAP
    Writing generation.txt.tmp → '3'
    Renaming generation.txt.tmp → generation.txt
    ✓ Generation pointer updated: 2 → 3

  🔄 Searches now see generation 3

  🧹 Cleanup: removing 1 old generation(s)
    Removing gen_1/
    Keeping generations: [3, 2]
```

### search_index.py

```bash
python search_index.py \
    --index ./index               # Index directory
    --query "OFAC screening"      # Search query
    --top 10                      # Number of results
    --expand-query                # Enable query expansion
    --tfidf-boost                 # Enable TF-IDF boosting
    --interactive                 # Interactive mode
    --verbose                     # Show component scores
```

### Interactive Commands

```
<query>           Search for text
:code <query>     Search only in code
:doc <query>      Search only in documents
:expand           Toggle query expansion
:boost            Toggle TF-IDF boosting
:decompose        Toggle query decomposition for long queries
:lookup <term>    Look up term in knowledge graph
:stats            Show index statistics
:quit             Exit
```

---

## Query Decomposition

Long queries with multiple concepts often perform poorly because different concepts compete for attention. Query decomposition splits them into focused sub-queries while **preserving the original query for context**:

```
Original: "How does OFAC sanctions screening work for wire transfers with BIC validation?"
           │
           ▼
    ┌──────────────────────────────────────────────┐
    │         KEYWORD EXTRACTION                    │
    │  vocabulary + knowledge graph → keywords      │
    │                                               │
    │  Found: OFAC, sanctions screening,            │
    │         wire transfer, BIC, validation        │
    └──────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────┐
    │         SUB-QUERIES GENERATED                 │
    │                                               │
    │  1. Original query (PRESERVED for context)   │
    │  2. "sanctions screening" (focused)           │
    │  3. "wire transfer" (focused)                 │
    │  4. "BIC validation" (focused)                │
    └──────────────────────────────────────────────┘
           │
           ├──────────────┬──────────────┬──────────────┐
           ▼              ▼              ▼              ▼
      Search #1      Search #2      Search #3      Search #4
      (context)      (focused)      (focused)      (focused)
           │              │              │              │
           └──────────────┴──────────────┴──────────────┘
                                   │
                                   ▼
                     ┌─────────────────────────┐
                     │      RRF FUSION         │
                     │  OR                     │
                     │      LLM SYNTHESIS      │
                     │  (results by sub-query) │
                     └────────────┬────────────┘
                                  ▼
                           Final Results
```

### Key Improvement: Original Query First

The original query is **always included first** to preserve relationships:

```
"OFAC screening FOR wire transfers" 
       ↓
Without original: searches "OFAC" and "wire transfers" separately (loses relationship)
With original:    searches full query (context) + focused terms (precision)
```

### LLM-Aware Decomposition

When using `--analyze --decompose`, results are organized **by sub-query** for the LLM:

```
=== Perspective 1: Full Context Search ===
Query: "How does OFAC screening work for wire transfers?"
Results: [code matching the full query with relationships intact]

=== Perspective 2: Focused on 'sanctions screening' ===
Results: [all OFAC/sanctions related code]

=== Perspective 3: Focused on 'wire transfer' ===
Results: [all wire transfer code]
```

The LLM can then synthesize across all perspectives, understanding relationships that simple RRF fusion cannot.

### Usage

```bash
# Basic decomposition (RRF fusion)
python search_index.py --index ./index --decompose \
    --query "How does OFAC screening work for wire transfers with BIC validation?"

# With LLM analysis (results grouped by sub-query)
python search_index.py --index ./index --decompose --analyze \
    --query "How does OFAC screening work for wire transfers with BIC validation?"

# Interactive mode
:decompose        # Toggle query decomposition on/off
:analyze <query>  # Uses decomposition if enabled
```

---

## Why This Architecture?

### Three Orthogonal Signals

| Signal | What It Captures | Example |
|--------|------------------|---------|
| Vector | Semantic similarity | "payment" finds "transaction" |
| BM25 | Exact terms + rarity | "UETR" finds exact matches |
| Concept | Domain structure | "OFAC" finds compliance code |

Each finds things the others miss. RRF combines them robustly.

### Pre-computed Knowledge Graph

| Benefit | Description |
|---------|-------------|
| No training step | Just load and apply at index time |
| Inspectable | Examine term relationships in JSON |
| Updateable | Re-run extractor when corpus changes |
| Reusable | Same KG works for multiple indexes |

### Hash Embeddings (Default)

| Benefit | Description |
|---------|-------------|
| No fitting | Works immediately on any vocabulary |
| Deterministic | Same input = same output |
| Fast | Simple hashing operations |
| Orthogonal | Different signal from BM25/Concept |

---

## File Reference

| File | Purpose |
|------|---------|
| `knowledge_extractor.py` | Build knowledge graph from corpus |
| `build_index.py` | Create search index |
| `search_index.py` | Search the index |
| `keywords.json` | Domain vocabulary |
| `llm_provider.py` | Optional LLM integration |
| `unified_indexer/` | Core library |

---

## Extending

### Add Neural Embeddings

For a fourth orthogonal signal:

```python
from sentence_transformers import SentenceTransformer

class NeuralEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
```

### Add New Parsers

```python
class MyParser(ContentParser):
    def parse(self, content: bytes, source_path: str) -> List[IndexableChunk]:
        # Parse content into chunks
        pass
```

Register in `pipeline.py`:
```python
self.parsers[SourceType.MY_TYPE] = MyParser(self.vocabulary)
```
