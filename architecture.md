# Unified Indexer Architecture Document

**Version:** 1.0  
**Date:** January 2025  
**Status:** Production Ready

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture Principles](#3-architecture-principles)
4. [Component Architecture](#4-component-architecture)
5. [Data Models](#5-data-models)
6. [Processing Pipeline](#6-processing-pipeline)
7. [Embedding Strategies](#7-embedding-strategies)
8. [Search and Retrieval](#8-search-and-retrieval)
9. [LLM Integration](#9-llm-integration)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Extension Points](#11-extension-points)
12. [Security Considerations](#12-security-considerations)
13. [Performance Characteristics](#13-performance-characteristics)
14. [Appendices](#14-appendices)

---

## 1. Executive Summary

### 1.1 Purpose

The Unified Indexer is a domain-aware search and retrieval system designed specifically for payment systems modernization. It enables cross-referencing between legacy TAL/COBOL code, PDF documentation, and transaction logs using hybrid vector + concept matching.

### 1.2 Key Capabilities

- **Multi-Source Indexing**: Index TAL code, PDF documents, and transaction logs
- **Domain-Aware Search**: Payment systems vocabulary with 50+ domain terms
- **Hybrid Retrieval**: Combines vector similarity with exact concept matching
- **Local Embeddings**: Five embedding approaches with no external API dependencies
- **LLM Analysis**: Send search results to LLM for implementation extraction
- **Cross-Reference Search**: Find code that implements documented requirements

### 1.3 Design Goals

| Goal | Approach |
|------|----------|
| No external dependencies | Local embeddings (numpy only) |
| Domain relevance | Aho-Corasick vocabulary matching |
| Legacy code support | TAL/COBOL-aware parsing |
| Offline operation | No API calls required for core functionality |
| Extensibility | Strategy pattern for parsers and embedders |

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  build_index.py          search_index.py           Interactive CLI          │
│  (Index Builder)         (Search Client)           (:analyze, :cap, etc)    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INDEXING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Content   │    │   Domain    │    │  Embedding  │    │   Hybrid    │  │
│  │   Parsers   │───▶│  Vocabulary │───▶│   Engine    │───▶│    Index    │  │
│  │             │    │  (Aho-Cor)  │    │             │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                   │                  │                  │          │
│        ▼                   ▼                  ▼                  ▼          │
│   TAL Parser         keywords.json      HashEmbedder       VectorStore     │
│   PDF Parser         50+ terms          HybridEmbedder     ConceptIndex    │
│   Log Parser         Capabilities       TFIDFEmbedder      SourceIndex     │
│                                         DomainEmbedder                      │
│                                         BM25Embedder                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SEARCH & RETRIEVAL                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Query Parser   │───▶│  Hybrid Search  │───▶│  Result Ranker  │         │
│  │                 │    │                 │    │                 │         │
│  │ • Tokenization  │    │ • Vector Search │    │ • Score Fusion  │         │
│  │ • Concept Match │    │ • Concept Match │    │ • Filtering     │         │
│  │ • Expansion     │    │ • Capability    │    │ • Deduplication │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LLM INTEGRATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ Result Formatter│───▶│  LLM Provider   │───▶│ Analysis Output │         │
│  │                 │    │                 │    │                 │         │
│  │ • Score Filter  │    │ • Anthropic     │    │ • Impl Details  │         │
│  │ • Text Extract  │    │ • OpenAI        │    │ • Business Rules│         │
│  │ • Image Encode  │    │ • Internal API  │    │ • Data Flow     │         │
│  │ • Context Build │    │ • Ollama        │    │ • Compliance    │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interaction Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Source  │     │  Parser  │     │ Embedder │     │  Index   │
│  Files   │     │          │     │          │     │          │
└────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │                │
     │  Read File     │                │                │
     │───────────────▶│                │                │
     │                │                │                │
     │                │  Parse & Chunk │                │
     │                │───────────────▶│                │
     │                │                │                │
     │                │  IndexableChunk│                │
     │                │◀───────────────│                │
     │                │                │                │
     │                │                │  Get Embedding │
     │                │                │───────────────▶│
     │                │                │                │
     │                │                │  Vector        │
     │                │                │◀───────────────│
     │                │                │                │
     │                │                │  Store Chunk   │
     │                │                │───────────────▶│
     │                │                │                │
```

---

## 3. Architecture Principles

### 3.1 Design Patterns

| Pattern | Application | Benefit |
|---------|-------------|---------|
| **Strategy** | Parsers, Embedders | Swappable implementations |
| **Factory** | `create_provider()`, `create_embedder()` | Simplified instantiation |
| **Composite** | HybridIndex (Vector + Concept) | Combined functionality |
| **Template Method** | LLMProvider abstract class | Consistent interface |
| **Builder** | ContentItem, LLMRequest | Complex object construction |

### 3.2 Separation of Concerns

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                        │
│  build_index.py, search_index.py, CLI commands                  │
├─────────────────────────────────────────────────────────────────┤
│                        APPLICATION LAYER                         │
│  IndexingPipeline, analyze_search_results()                     │
├─────────────────────────────────────────────────────────────────┤
│                          DOMAIN LAYER                            │
│  DomainVocabulary, VocabularyEntry, DomainMatch                 │
├─────────────────────────────────────────────────────────────────┤
│                       INFRASTRUCTURE LAYER                       │
│  Parsers, Embedders, VectorStore, ConceptIndex, LLMProvider     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Key Design Decisions

1. **Local-First Embeddings**: No OpenAI/external API required for core functionality
2. **Domain Vocabulary as First-Class Citizen**: Aho-Corasick automaton for O(n) matching
3. **Hybrid Search**: Combine semantic (vector) with exact (concept) matching
4. **Chunk-Based Architecture**: All content normalized to IndexableChunk
5. **Pluggable LLM Integration**: Abstract provider pattern for multiple backends

---

## 4. Component Architecture

### 4.1 Package Structure

```
unified_indexer/
├── __init__.py              # Public API exports
├── models.py                # Data classes (IndexableChunk, SearchResult, etc.)
├── vocabulary.py            # DomainVocabulary, Aho-Corasick automaton
├── embeddings.py            # 5 embedding strategies
├── index.py                 # VectorStore, ConceptIndex, HybridIndex
├── pipeline.py              # IndexingPipeline orchestrator
├── parsers/
│   ├── __init__.py
│   ├── base.py              # ContentParser abstract base
│   ├── tal_parser.py        # TAL/COBOL code parser
│   ├── document_parser.py   # PDF/DOCX/TXT parser
│   └── log_parser.py        # JSON/structured log parser
├── examples/
│   └── usage_example.py
└── tests/
    └── test_unified_indexer.py

# External modules
llm_provider.py              # LLM integration (abstract + implementations)
build_index.py               # CLI for building indexes
search_index.py              # CLI for searching
keywords.json                # Domain vocabulary definitions
```

### 4.2 Core Components

#### 4.2.1 IndexingPipeline

**Responsibility**: Orchestrate parsing, embedding, and indexing operations.

```python
class IndexingPipeline:
    """Central orchestrator for the indexing system"""
    
    def __init__(self,
                 vocabulary_path: Optional[str] = None,
                 vocabulary_data: Optional[List[Dict]] = None,
                 embedder_type: Optional[str] = "hash",
                 embedding_fn: Optional[Callable] = None):
        
        self.vocabulary = DomainVocabulary()
        self.index = HybridIndex(self.vocabulary)
        self.embedder = create_embedder(embedder_type, self.vocabulary)
        
        # Register parsers
        self.parsers = {
            SourceType.CODE: TalCodeParser(self.vocabulary),
            SourceType.DOCUMENT: DocumentParser(self.vocabulary),
            SourceType.LOG: LogParser(self.vocabulary)
        }
```

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `index_content(data, filename, source_type)` | Index raw content |
| `index_file(file_path)` | Index a single file |
| `index_directory(path, recursive)` | Batch index directory |
| `search(query, top_k, source_types)` | Hybrid search |
| `search_cross_reference(query, from_type, to_types)` | Cross-reference search |
| `save(directory)` / `load(directory)` | Persistence |

#### 4.2.2 DomainVocabulary

**Responsibility**: Manage domain terms and perform efficient multi-pattern matching.

```python
class DomainVocabulary:
    """Domain vocabulary with Aho-Corasick matching"""
    
    def __init__(self):
        self.entries: List[VocabularyEntry] = []
        self.automaton = AhoCorasickAutomaton()
        
        # Fast lookup indexes
        self.by_canonical: Dict[str, VocabularyEntry] = {}
        self.by_capability: Dict[str, List[VocabularyEntry]] = {}
        self.by_category: Dict[str, List[VocabularyEntry]] = {}
        self.term_to_entry: Dict[str, VocabularyEntry] = {}
```

**Aho-Corasick Automaton**:

```
                    [root]
                   /  |  \
                  /   |   \
                [o]  [w]  [m]
                /     |     \
              [f]   [i]    [t]
              /       |       \
            [a]     [r]      [-]
            /         |         \
          [c]       [e]        [1]
                              /   \
        Match: "ofac"    [0]     [0]
                          |       |
                         [3]     [2]
                          
        Match: "wire"  Match: "mt-103"  Match: "mt-202"
```

**Complexity**: O(n + m) where n = text length, m = total matches

#### 4.2.3 VectorStore

**Responsibility**: Store and search embeddings using cosine similarity.

```python
class VectorStore:
    """Simple in-memory vector store for embeddings"""
    
    def __init__(self):
        self.embeddings: Dict[str, np.ndarray] = {}  # chunk_id -> embedding
        self.chunks: Dict[str, IndexableChunk] = {}   # chunk_id -> chunk
    
    def add(self, chunk_id: str, embedding: List[float], chunk: IndexableChunk):
        """Add a chunk with its embedding vector"""
        self.embeddings[chunk_id] = np.array(embedding)
        self.chunks[chunk_id] = chunk
    
    def search(self, query_embedding: List[float], top_k: int = 10, 
               filter_fn: Optional[Callable] = None) -> List[Tuple[str, float]]:
        """Search using cosine similarity"""
        query_vec = np.array(query_embedding)
        
        results = []
        for chunk_id, embedding in self.embeddings.items():
            if filter_fn and not filter_fn(self.chunks[chunk_id]):
                continue
            
            # Cosine similarity
            similarity = np.dot(query_vec, embedding) / (
                np.linalg.norm(query_vec) * np.linalg.norm(embedding) + 1e-8
            )
            results.append((chunk_id, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
```

**Storage Format** (embeddings.json):
```json
{
  "chunk-uuid-1": [0.123, -0.456, 0.789, ...],
  "chunk-uuid-2": [0.234, -0.567, 0.890, ...],
  ...
}
```

**Instantiation Flow**:
```
IndexingPipeline.__init__()
    │
    └──▶ HybridIndex.__init__()
              │
              ├──▶ self.vector_store = VectorStore()  ◀── Created here
              │
              └──▶ self.concept_index = ConceptIndex()
```

**Usage in Indexing**:
```python
# In HybridIndex.index_chunk()
def index_chunk(self, chunk: IndexableChunk):
    # Add to concept index (always)
    self.concept_index.add(chunk)
    
    # Add to vector store (if embedding function available)
    if self.embedding_fn:
        embedding = self.embedding_fn(chunk.embedding_text)
        self.vector_store.add(chunk.chunk_id, embedding, chunk)
```

**Production Alternatives**:

For large-scale deployments, replace the in-memory VectorStore with:

| Store | Use Case | Integration |
|-------|----------|-------------|
| **FAISS** | Large scale, CPU/GPU | `faiss.IndexFlatIP(dim)` |
| **ChromaDB** | Persistent, easy setup | `chromadb.PersistentClient()` |
| **Qdrant** | Production, filtering | REST API or Python client |
| **Pinecone** | Managed cloud | `pinecone.Index()` |
| **Milvus** | Enterprise, distributed | `pymilvus.Collection()` |

See [Section 11.5 Custom Vector Store](#115-custom-vector-store) for integration examples.

#### 4.2.4 HybridIndex

**Responsibility**: Combine vector and concept-based retrieval.

```python
class HybridIndex:
    """Hybrid retrieval combining vectors and concepts"""
    
    def __init__(self, vocabulary: DomainVocabulary):
        self.vocabulary = vocabulary
        self.vector_store = VectorStore()
        self.concept_index = ConceptIndex()
        self.source_index = SourceIndex()
        self.embedding_fn: Optional[Callable] = None
```

**Search Algorithm**:

```
Input: query, top_k, source_types, capabilities

1. CONCEPT EXTRACTION
   concepts = vocabulary.match_text(query)
   expanded_terms = vocabulary.expand_query(query)

2. VECTOR SEARCH (if embedding_fn available)
   query_embedding = embedding_fn(query)
   vector_results = vector_store.search(query_embedding, top_k * 3)

3. CONCEPT SEARCH
   For each concept in concepts:
       chunk_ids = concept_index.search_concept(concept)
       Add to concept_results with score

4. SCORE FUSION
   For each chunk_id in (vector_results ∪ concept_results):
       combined_score = (vector_weight * vector_score) + 
                        (concept_weight * concept_score)

5. FILTERING & RANKING
   Apply source_type filter
   Apply capability filter
   Sort by combined_score DESC
   Return top_k results
```

#### 4.2.4 LLMProvider

**Responsibility**: Abstract interface for LLM integration.

```python
class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def invoke_llm(self,
                   system_prompt: str,
                   user_prompt: str,
                   content_type: ContentType = ContentType.TEXT,
                   content_items: Optional[List[ContentItem]] = None,
                   **kwargs) -> LLMResponse:
        """Send prompts to LLM and get response"""
        pass
```

**Provider Hierarchy**:

```
                    LLMProvider (ABC)
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
   StubLLMProvider  InternalAPIProvider  CloudProviders
   (Testing)        (Your wrapper)           │
                                    ┌────────┼────────┐
                                    │        │        │
                                    ▼        ▼        ▼
                              Anthropic   OpenAI   Ollama
```

---

## 5. Data Models

### 5.1 Core Entities

#### IndexableChunk

The fundamental unit of indexed content.

```python
@dataclass
class IndexableChunk:
    chunk_id: str                    # Unique identifier (UUID)
    text: str                        # Raw text content
    embedding_text: str              # Preprocessed text for embedding
    source_type: SourceType          # CODE, DOCUMENT, LOG
    semantic_type: SemanticType      # PROCEDURE, SECTION, LOG_ENTRY, etc.
    source_ref: SourceReference      # File, line numbers, etc.
    domain_matches: List[DomainMatch] # Matched vocabulary terms
    context_before: Optional[str]    # Surrounding context
    context_after: Optional[str]
    metadata: Dict[str, Any]         # Additional attributes
    
    @property
    def capability_set(self) -> Set[str]:
        """Business capabilities from domain matches"""
```

#### VocabularyEntry

Domain vocabulary term definition.

```python
@dataclass
class VocabularyEntry:
    canonical_term: str              # Primary term (e.g., "wire transfer")
    keywords: List[str]              # All matching keywords
    related_keywords: List[str]      # Related terms for expansion
    description: str                 # Human-readable description
    metadata_category: str           # Category (e.g., "payment-systems")
    business_capabilities: List[str] # Business capabilities
```

#### SearchResult

Search result with scoring details.

```python
@dataclass
class SearchResult:
    chunk: IndexableChunk
    vector_score: float              # Similarity score (0-1)
    concept_score: float             # Concept match score
    combined_score: float            # Weighted combination
    matched_concepts: List[str]      # Which concepts matched
```

### 5.2 Entity Relationships

```
┌─────────────────────┐       ┌─────────────────────┐
│   VocabularyEntry   │       │    DomainMatch      │
├─────────────────────┤       ├─────────────────────┤
│ canonical_term      │──┐    │ matched_term        │
│ keywords[]          │  │    │ canonical_term      │──┐
│ related_keywords[]  │  │    │ capabilities[]      │  │
│ business_caps[]     │  │    │ category            │  │
│ metadata_category   │  │    │ confidence          │  │
└─────────────────────┘  │    │ position            │  │
                         │    └─────────────────────┘  │
                         │              │              │
                         └──────────────┼──────────────┘
                                        │
                                        ▼
┌─────────────────────┐       ┌─────────────────────┐
│   IndexableChunk    │◀──────│    SearchResult     │
├─────────────────────┤       ├─────────────────────┤
│ chunk_id            │       │ chunk               │
│ text                │       │ vector_score        │
│ embedding_text      │       │ concept_score       │
│ source_type         │       │ combined_score      │
│ semantic_type       │       │ matched_concepts[]  │
│ source_ref          │       └─────────────────────┘
│ domain_matches[]────┼───────────────────▲
│ capability_set      │                   │
└─────────────────────┘                   │
         │                                │
         ▼                                │
┌─────────────────────┐                   │
│   SourceReference   │                   │
├─────────────────────┤                   │
│ file_path           │       ┌───────────────────────┐
│ line_start/end      │       │      HybridIndex      │
│ procedure_name      │       ├───────────────────────┤
│ page_number         │       │ vector_store          │
│ section_title       │       │ concept_index         │
└─────────────────────┘       │ source_index          │
                              │ search() ─────────────┼──▶
                              └───────────────────────┘
```

### 5.3 Vocabulary Schema (keywords.json)

```json
{
  "version": "1.0",
  "description": "Payment Systems Domain Vocabulary",
  "entries": [
    {
      "keywords": "wire transfer,funds transfer,electronic transfer",
      "metadata": "payment-systems",
      "description": "Electronic transfer of funds between institutions",
      "related_keywords": "domestic wire,international wire",
      "business_capability": ["Payment Processing", "Wire Transfer"]
    }
  ]
}
```

---

## 6. Processing Pipeline

### 6.1 Indexing Flow

```
┌─────────────┐
│ Source File │
│ (PDF/TAL/   │
│  LOG)       │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                      PARSER SELECTION                         │
│                                                               │
│  .tal/.txt  ────▶  TalCodeParser                             │
│  .pdf       ────▶  DocumentParser (pdfplumber)               │
│  .docx      ────▶  DocumentParser (python-docx)              │
│  .log/.json ────▶  LogParser                                 │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                         PARSING                               │
│                                                               │
│  1. Read content (handle encoding)                           │
│  2. Split into chunks (procedures, sections, entries)        │
│  3. Extract metadata (line numbers, page, procedure name)    │
│  4. Match domain vocabulary (Aho-Corasick)                   │
│  5. Build embedding text (clean, normalize)                  │
│  6. Create IndexableChunk objects                            │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                       EMBEDDING                               │
│                                                               │
│  For each chunk:                                             │
│    embedding_vector = embedder.get_embedding(chunk.text)     │
│                                                               │
│  Embedder options:                                           │
│    • HashEmbedder (1024 dims, no fitting)                    │
│    • HybridEmbedder (domain + hash, 634 dims)                │
│    • TFIDFEmbedder (requires fitting)                        │
│    • DomainConceptEmbedder (interpretable)                   │
│    • BM25Embedder (retrieval-optimized)                      │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                        INDEXING                               │
│                                                               │
│  VectorStore:                                                │
│    embeddings[chunk_id] = embedding_vector                   │
│    chunks[chunk_id] = chunk                                  │
│                                                               │
│  ConceptIndex:                                               │
│    For each domain_match in chunk:                           │
│      concept_to_chunks[canonical_term].add(chunk_id)         │
│      capability_to_chunks[capability].add(chunk_id)          │
│                                                               │
│  SourceIndex:                                                │
│    source_type_to_chunks[source_type].add(chunk_id)          │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 TAL Code Parsing

```
Input: TAL source file

┌────────────────────────────────────────────────────┐
│ ! Wire Transfer Module                             │
│ ! Handles MT-103 processing                        │
│                                                    │
│ INT PROC PROCESS^WIRE(rec);                        │
│     INT .rec;                                      │
│ BEGIN                                              │
│     IF OFAC^CHECK(rec) THEN ...                    │
│ END;                                               │
│                                                    │
│ INT PROC VALIDATE^AMOUNT(amt);                     │
│ ...                                                │
└────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────┐
│              PROCEDURE EXTRACTION                   │
│                                                    │
│  Pattern: (INT|PROC|STRING|FIXED)?\s*PROC\s+       │
│           ([A-Z][A-Z0-9^_]*)\s*\(                  │
│                                                    │
│  Find: BEGIN ... END blocks                        │
│  Track: Line numbers, nesting level                │
└────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────┐
│                   OUTPUT                            │
│                                                    │
│  Chunk 1:                                          │
│    text: "INT PROC PROCESS^WIRE(rec)..."           │
│    procedure_name: "PROCESS^WIRE"                  │
│    line_start: 4, line_end: 8                      │
│    domain_matches: ["wire transfer", "OFAC"]       │
│    capabilities: ["Wire Transfer", "OFAC Screen"]  │
│                                                    │
│  Chunk 2:                                          │
│    text: "INT PROC VALIDATE^AMOUNT(amt)..."        │
│    ...                                             │
└────────────────────────────────────────────────────┘
```

### 6.3 Document Parsing

```
Input: PDF Document

┌────────────────────────────────────────────────────┐
│  Wire Transfer Processing Guide                    │
│  ═══════════════════════════                       │
│                                                    │
│  1. Overview                                       │
│  This document describes...                        │
│                                                    │
│  2. OFAC Screening                                 │
│  All wire transfers must undergo...                │
│                                                    │
│  [DIAGRAM: Process Flow]                           │
│                                                    │
└────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────┐
│              PDF EXTRACTION                         │
│                                                    │
│  Using: pdfplumber                                 │
│  Extract: Text per page                            │
│  Extract: Tables (if present)                      │
│  Extract: Images (base64 encode for LLM)           │
└────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────┐
│              SECTION DETECTION                      │
│                                                    │
│  Pattern: Numbered headers, title case             │
│  Split: By section or by page                      │
│  Maintain: Section hierarchy                       │
└────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────┐
│                   OUTPUT                            │
│                                                    │
│  Chunk 1:                                          │
│    text: "1. Overview\nThis document..."           │
│    page_number: 1                                  │
│    section_title: "Overview"                       │
│    domain_matches: ["wire transfer"]               │
│                                                    │
│  Chunk 2:                                          │
│    text: "2. OFAC Screening\nAll wire..."          │
│    page_number: 1                                  │
│    section_title: "OFAC Screening"                 │
│    domain_matches: ["OFAC", "wire transfer"]       │
│    metadata: {images: [base64_data]}               │
└────────────────────────────────────────────────────┘
```

---

## 7. Embedding Strategies

### 7.1 Embedder Comparison

| Embedder | Dimensions | Fitting | Speed | Accuracy | Interpretable |
|----------|------------|---------|-------|----------|---------------|
| Hash | 1024 | No | ★★★★★ | ★★★ | ★★ |
| Hybrid | 634 | No | ★★★★ | ★★★★★ | ★★★ |
| TF-IDF | Variable | Yes | ★★★ | ★★★★ | ★★★ |
| Domain | Variable | No | ★★★★★ | ★★★ | ★★★★★ |
| BM25 | Variable | Yes | ★★★ | ★★★★ | ★★★ |

### 7.2 HashEmbedder

**Algorithm**: Feature hashing with signed random projections

```python
def get_embedding(text: str) -> np.ndarray:
    tokens = tokenize(text)
    vector = np.zeros(n_features)  # Default: 1024
    
    for token in tokens:
        # Dual hashing for index and sign
        h1 = md5(token) % n_features  # Index
        h2 = sha1(token) % 2          # Sign (+1 or -1)
        sign = 1 if h2 == 0 else -1
        
        # TF weighting with domain boost
        weight = 1 + log(count)
        if token in domain_terms:
            weight *= domain_boost  # Default: 2.0
        
        vector[h1] += sign * weight
    
    return normalize(vector)  # L2 norm
```

**Advantages**:
- No vocabulary building
- Fixed dimension output
- Handles OOV terms
- Domain term boosting

### 7.3 HybridEmbedder

**Algorithm**: Concatenate domain concepts + text features

```python
def get_embedding(text: str) -> np.ndarray:
    # Domain component (122 dimensions for default vocab)
    domain_vec = domain_embedder.transform(text) * domain_weight  # 0.6
    
    # Text component (512 dimensions)
    text_vec = hash_embedder.transform(text) * text_weight  # 0.4
    
    # Concatenate: 122 + 512 = 634 dimensions
    return np.concatenate([domain_vec, text_vec])
```

**Advantages**:
- Best of both worlds
- Strong domain term matching
- General text similarity

### 7.4 DomainConceptEmbedder

**Algorithm**: One dimension per domain concept

```python
def get_embedding(text: str) -> np.ndarray:
    vector = np.zeros(n_concepts)  # ~122 for default vocab
    
    matches = vocabulary.match_text(text)
    for match in matches:
        idx = concept_to_idx[match.canonical_term]
        vector[idx] = match.confidence
        
        # Capabilities at lower weight
        for cap in match.capabilities:
            cap_idx = concept_to_idx[cap]
            vector[cap_idx] = 0.5 * match.confidence
    
    return normalize(vector)
```

**Advantages**:
- Fully interpretable
- `explain_embedding()` shows activated concepts
- No training required

### 7.5 PaymentDomainEmbedder

For payment systems specifically, we provide structured semantic dimensions instead of one dimension per keyword.

**80 Semantic Dimensions in 8 Categories:**

| Category | Dimensions | Examples |
|----------|------------|----------|
| Message Types | 10 | MSG_MT_CUSTOMER, MSG_MT_INSTITUTION, MSG_ISO20022 |
| Networks | 10 | NET_SWIFT, NET_FEDWIRE, NET_CHIPS, NET_ACH |
| Compliance | 10 | COMP_OFAC, COMP_AML, COMP_KYC, COMP_SANCTIONS |
| Processing Stages | 10 | STAGE_SCREENING, STAGE_SETTLEMENT, STAGE_ROUTING |
| Parties | 10 | PARTY_ORIGINATOR, PARTY_BENEFICIARY, PARTY_INTERMEDIARY |
| Data Elements | 10 | DATA_AMOUNT, DATA_BIC, DATA_IBAN, DATA_REFERENCE |
| Errors | 10 | ERR_VALIDATION, ERR_COMPLIANCE, ERR_RETURN, ERR_REPAIR |
| Capabilities | 10 | CAP_HIGH_VALUE, CAP_URGENT, CAP_INQUIRY |

**Embedder Options:**

| Embedder | Dimensions | Use Case |
|----------|------------|----------|
| `payment` | 80 | Pure payment semantic matching |
| `payment_hybrid` | 80 + text_dim | Payment semantics + general text |

```bash
# Payment domain only (80 dimensions)
python build_index.py -o ./index --embedder payment

# Payment hybrid (80 + 512 = 592)
python build_index.py -o ./index --embedder payment_hybrid

# Payment hybrid with larger text dim
python build_index.py -o ./index --embedder payment_hybrid --dims 1024
```

**Dimension Activation Example:**

```
Text: "MT-103 wire transfer OFAC screening"

Activated Dimensions:
  COMP_OFAC: 0.392        (Compliance - OFAC)
  MSG_MT_CUSTOMER: 0.392  (Message - Customer Transfer)
  STAGE_SCREENING: 0.392  (Processing - Screening)
  CAP_HIGH_VALUE: 0.353   (Capability - High Value)
  COMP_SANCTIONS: 0.353   (Compliance - Sanctions)
```

**Advantages over Generic Embeddings:**

1. **Interpretable**: Each dimension has clear payment meaning
2. **Domain-optimized**: Captures payment semantics precisely  
3. **Cross-reference friendly**: Similar concepts activate same dimensions
4. **Compact**: 80 dims vs 1024+ for hash embeddings

### 7.6 LearnedDomainEmbedder

For any domain, dimensions can be learned automatically from your corpus:

**Learning Process:**

```
Documents → Term Extraction → Co-occurrence Analysis → Clustering → Dimensions
```

1. **Term Extraction**: TF-IDF ranking of significant terms and n-grams
2. **Co-occurrence**: Build term co-occurrence matrix within windows
3. **Similarity**: Compute PMI (Pointwise Mutual Information) between terms
4. **Clustering**: Hierarchical clustering groups related terms
5. **Dimensions**: Each cluster becomes a semantic dimension

**Usage:**

```bash
# Step 1: Learn dimensions from corpus
python learn_dimensions.py --input ./docs --output dimensions.json --dims 80

# Step 2: Build index with learned dimensions
python build_index.py --pdf-dir ./docs -o ./index --embedder learned --dimensions dimensions.json

# Or learn and index in one step:
python build_index.py --pdf-dir ./docs --tal-dir ./code -o ./index --learn-dims --dims 80
```

**Example Learned Dimensions:**

```
From payment processing documents:

Dimension 1: WIRE_TRANSFER
  Terms: wire, wire transfer, ofac, mt-103, screening, funds transfer
  
Dimension 2: SETTLEMENT
  Terms: settlement, fedwire, rtgs, value date, finality

Dimension 3: COMPLIANCE
  Terms: compliance, aml, kyc, sanctions, watchlist, screening
```

**Advantages:**

| Feature | Hardcoded Dims | Learned Dims |
|---------|----------------|--------------|
| Domain adaptation | ❌ Manual | ✅ Automatic |
| Term coverage | Fixed vocabulary | Corpus-specific |
| Maintenance | Update code | Re-learn from docs |
| Multi-domain | Separate files | Learn per domain |

**Configuration Options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_dimensions` | 80 | Target number of dimensions |
| `min_term_frequency` | 3 | Minimum term occurrences |
| `cooccurrence_window` | 50 | Character window for co-occurrence |
| `extract_bigrams` | True | Extract 2-word phrases |
| `extract_trigrams` | True | Extract 3-word phrases |

## 8. Search and Retrieval

### 8.1 Search Algorithm

```
hybrid_search(query, top_k=10, vector_weight=0.5, concept_weight=0.5):

    ┌─────────────────────────────────────────────────────────────┐
    │                    QUERY PROCESSING                          │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  query = "OFAC sanctions screening for wire transfers"       │
    │                                                              │
    │  1. Concept Extraction (Aho-Corasick)                       │
    │     matches = ["OFAC", "sanctions screening", "wire transfer"]│
    │                                                              │
    │  2. Query Expansion                                          │
    │     expanded = ["OFAC", "sanctions", "blocked persons",      │
    │                 "wire transfer", "funds transfer", ...]      │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    VECTOR SEARCH                             │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  query_embedding = embedder.get_embedding(query)             │
    │                                                              │
    │  For each chunk in vector_store:                            │
    │      similarity = cosine(query_embedding, chunk_embedding)   │
    │                                                              │
    │  vector_results = top_k * 3 by similarity                   │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   CONCEPT SEARCH                             │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  For each concept in matches:                               │
    │      chunk_ids = concept_index.search_concept(concept)       │
    │      Add 1.0 to concept_score[chunk_id]                     │
    │                                                              │
    │  For each term in expanded:                                 │
    │      chunk_ids = concept_index.search_concept(term)          │
    │      Add 0.5 to concept_score[chunk_id]  (lower weight)     │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    SCORE FUSION                              │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  all_chunk_ids = vector_results ∪ concept_results           │
    │                                                              │
    │  For each chunk_id:                                         │
    │      v_score = vector_results.get(chunk_id, 0)              │
    │      c_score = concept_results.get(chunk_id, 0)             │
    │                                                              │
    │      # Normalize concept score                               │
    │      c_score_norm = c_score / max_concept_score             │
    │                                                              │
    │      combined = (vector_weight * v_score) +                 │
    │                 (concept_weight * c_score_norm)              │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 FILTERING & RANKING                          │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Apply filters:                                             │
    │    - source_types (CODE, DOCUMENT, LOG)                     │
    │    - capabilities (e.g., "OFAC Screening")                  │
    │                                                              │
    │  Sort by combined_score DESC                                │
    │                                                              │
    │  Return top_k SearchResult objects                          │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
```

### 8.2 Cross-Reference Search

Find code that implements documented requirements:

```python
def search_cross_reference(query, from_type, to_types, top_k):
    """
    Example: Find code that handles an error from logs
    
    search_cross_reference(
        query="OFAC screening failed",
        from_type=SourceType.LOG,
        to_types=[SourceType.CODE, SourceType.DOCUMENT],
        top_k=5
    )
    """
    
    # 1. Search primary source type
    primary_results = search(query, source_types=[from_type])
    
    # 2. Extract concepts from primary results
    all_concepts = set()
    for result in primary_results:
        all_concepts.update(result.matched_concepts)
    
    # 3. Search other types using extracted concepts
    cross_results = {}
    for target_type in to_types:
        expanded_query = query + " " + " ".join(all_concepts)
        cross_results[target_type] = search(
            expanded_query,
            source_types=[target_type],
            top_k=top_k
        )
    
    return cross_results
```

---

## 9. LLM Integration

### 9.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SEARCH RESULTS                              │
│                                                                  │
│  Results with score >= min_score (default 0.70)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RESULT FORMATTER                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  format_search_results_for_llm(results, min_score):             │
│                                                                  │
│  1. Filter by min_score                                         │
│  2. Separate CODE, DOCUMENT, LOG chunks                         │
│  3. Extract images from document metadata                       │
│  4. Sanitize text (encoding, control chars)                     │
│  5. Format with source attribution                              │
│                                                                  │
│  Output:                                                        │
│    - context_text: Formatted string                             │
│    - image_items: List[ContentItem] for base64 images           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM REQUEST                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  System Prompt: WIRE_PAYMENTS_SYSTEM_PROMPT                     │
│    - Expert in SWIFT, ISO 20022, Fedwire, CHIPS                │
│    - Knows TAL/COBOL legacy systems                            │
│    - Extracts implementation details                            │
│    - Identifies business rules                                  │
│    - Notes compliance requirements                              │
│                                                                  │
│  User Prompt:                                                   │
│    - Original query                                             │
│    - Formatted context (code + docs + images)                   │
│    - Instructions for analysis                                  │
│                                                                  │
│  Content Type:                                                  │
│    - TEXT: Text only                                            │
│    - CODE: Code with syntax context                             │
│    - IMAGE: Base64 encoded images                               │
│    - MIXED: Text + images                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM PROVIDER                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │ InternalAPI     │  POST /v1/chat/completions                 │
│  │ Provider        │  Content-Type: application/json            │
│  │                 │  Authorization: Bearer {api_key}           │
│  └─────────────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  {                                                              │
│    "model": "gpt-4",                                            │
│    "messages": [                                                │
│      {"role": "system", "content": "..."},                      │
│      {"role": "user", "content": [...]}                         │
│    ],                                                           │
│    "temperature": 0.3,                                          │
│    "max_tokens": 4096                                           │
│  }                                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM RESPONSE                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Analysis includes:                                             │
│    1. Direct answer to query                                    │
│    2. Implementation details from code                          │
│    3. Business rules from documentation                         │
│    4. Data flow and processing sequence                         │
│    5. Error conditions and exception handling                   │
│    6. Compliance and regulatory considerations                  │
│    7. Recommendations                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Provider Configuration

| Provider | Environment Variable | Default Model |
|----------|---------------------|---------------|
| `anthropic` | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 |
| `openai` | `OPENAI_API_KEY` | gpt-4 |
| `internal` | `INTERNAL_API_KEY`, `INTERNAL_API_URL` | gpt-4 |
| `ollama` | (none - local) | llama3 |
| `stub` | (none - testing) | stub-model |

### 9.3 Image Handling

```python
class ContentItem:
    content_type: ContentType
    text: Optional[str]
    image_data: Optional[bytes]       # Raw bytes
    image_base64: Optional[str]       # Base64 encoded
    image_media_type: str             # "image/png", "image/jpeg"
    
    def to_base64(self) -> str:
        """Lazy encode image to base64"""
        if self.image_base64:
            return self.image_base64
        if self.image_data:
            self.image_base64 = base64.b64encode(self.image_data).decode('utf-8')
            return self.image_base64
        return None
```

**API Format for Images**:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Analyze this code..."},
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,iVBORw0KGgo..."
      }
    }
  ]
}
```

---

## 10. Deployment Architecture

### 10.1 File Structure (Deployed)

```
deployment/
├── unified_indexer/          # Core package
├── keywords.json             # Domain vocabulary
├── llm_provider.py           # LLM integration
├── build_index.py            # Index builder CLI
├── search_index.py           # Search CLI
├── setup.py                  # Package installation
├── requirements.txt          # Dependencies
└── indexes/                  # Saved indexes
    ├── payment_system/
    │   ├── chunks.json
    │   ├── embeddings.npy
    │   ├── vocabulary.json
    │   └── pipeline_stats.json
    └── another_project/
        └── ...
```

### 10.2 Dependencies

**Required**:
```
numpy>=1.21.0
```

**Optional**:
```
pdfplumber>=0.7.0          # PDF parsing
python-docx>=0.8.11        # DOCX parsing
beautifulsoup4>=4.11.0     # HTML parsing
requests>=2.28.0           # HTTP for LLM providers
anthropic>=0.18.0          # Anthropic API
openai>=1.0.0              # OpenAI API
```

### 10.3 Installation

```bash
# Basic (index and search TAL/text only)
pip install numpy

# With PDF support
pip install numpy pdfplumber

# With LLM support
pip install numpy requests anthropic

# Full installation
pip install -e ".[all]"
```

### 10.4 Configuration

**Environment Variables**:

```bash
# LLM Providers
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export INTERNAL_API_KEY="your-key"
export INTERNAL_API_URL="https://your-api.internal.com"

# Optional
export UNIFIED_INDEXER_VOCAB="./custom_vocab.json"
```

---

## 11. Extension Points

### 11.1 Custom Parser

```python
from unified_indexer.parsers.base import ContentParser
from unified_indexer.models import IndexableChunk, SourceType, SemanticType

class CobolParser(ContentParser):
    """Custom parser for COBOL code"""
    
    def parse(self, content: bytes, file_path: str) -> List[IndexableChunk]:
        text = content.decode('utf-8', errors='replace')
        chunks = []
        
        # Find COBOL paragraphs/sections
        for match in re.finditer(r'(\d{6}\s+)?(\w+)\s+SECTION\.', text):
            section_name = match.group(2)
            section_text = self._extract_section(text, match.start())
            
            chunk = IndexableChunk(
                chunk_id=str(uuid.uuid4()),
                text=section_text,
                embedding_text=self._clean_for_embedding(section_text),
                source_type=SourceType.CODE,
                semantic_type=SemanticType.PROCEDURE,
                source_ref=SourceReference(
                    file_path=file_path,
                    procedure_name=section_name
                ),
                domain_matches=self.vocabulary.match_text(section_text)
            )
            chunks.append(chunk)
        
        return chunks

# Register with pipeline
pipeline.register_parser('.cbl', CobolParser(vocabulary))
pipeline.register_parser('.cob', CobolParser(vocabulary))
```

### 11.2 Custom Embedder

```python
from unified_indexer.embeddings import create_embedder
from unified_indexer.vocabulary import DomainVocabulary

class SentenceTransformerEmbedder:
    """Custom embedder using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def transform(self, text: str) -> np.ndarray:
        return self.model.encode(text)
    
    def get_embedding(self, text: str) -> List[float]:
        return self.transform(text).tolist()
    
    @property
    def n_features(self) -> int:
        return self.model.get_sentence_embedding_dimension()

# Use with pipeline
pipeline = IndexingPipeline(vocabulary_data=vocab)
pipeline.embedder = SentenceTransformerEmbedder()
pipeline.index.set_embedding_function(pipeline.embedder.get_embedding)
```

### 11.3 Custom LLM Provider

```python
from llm_provider import LLMProvider, LLMResponse, ContentType, ContentItem

class AzureOpenAIProvider(LLMProvider):
    """Custom provider for Azure OpenAI"""
    
    def __init__(self, 
                 deployment_name: str,
                 api_key: str,
                 endpoint: str,
                 api_version: str = "2024-02-15-preview"):
        super().__init__(model=deployment_name, api_key=api_key, base_url=endpoint)
        self.api_version = api_version
    
    def invoke_llm(self,
                   system_prompt: str,
                   user_prompt: str,
                   content_type: ContentType = ContentType.TEXT,
                   content_items: Optional[List[ContentItem]] = None,
                   **kwargs) -> LLMResponse:
        
        import requests
        
        url = f"{self.base_url}/openai/deployments/{self.model}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        params = {"api-version": self.api_version}
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        response = requests.post(url, headers=headers, params=params, json=payload)
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=self.model,
            provider="azure",
            success=True
        )
```

### 11.4 Custom Vocabulary

```json
{
  "version": "1.0",
  "description": "Insurance Claims Domain Vocabulary",
  "entries": [
    {
      "keywords": "claim,insurance claim,loss claim",
      "metadata": "claims-processing",
      "description": "Insurance claim submission",
      "related_keywords": "first notice of loss,fnol",
      "business_capability": ["Claim Intake", "FNOL Processing"]
    },
    {
      "keywords": "adjuster,claims adjuster,loss adjuster",
      "metadata": "claims-processing",
      "description": "Claims adjustment and evaluation",
      "related_keywords": "field adjuster,desk adjuster",
      "business_capability": ["Claim Adjustment", "Loss Evaluation"]
    }
  ]
}
```

### 11.5 Custom Vector Store

Replace the in-memory VectorStore with a production-grade solution:

#### FAISS Integration

```python
import faiss
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
from unified_indexer.models import IndexableChunk

class FAISSVectorStore:
    """FAISS-backed vector store for large-scale similarity search"""
    
    def __init__(self, dimension: int = 634, use_gpu: bool = False):
        self.dimension = dimension
        self.use_gpu = use_gpu
        
        # Create index (Inner Product for cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Maintain mapping from FAISS index position to chunk_id
        self.id_map: List[str] = []
        self.chunks: Dict[str, IndexableChunk] = {}
    
    def add(self, chunk_id: str, embedding: List[float], chunk: IndexableChunk):
        """Add a chunk with its embedding"""
        vector = np.array([embedding], dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(vector)
        
        self.index.add(vector)
        self.id_map.append(chunk_id)
        self.chunks[chunk_id] = chunk
    
    def add_batch(self, items: List[Tuple[str, List[float], IndexableChunk]]):
        """Batch add for efficiency"""
        vectors = np.array([item[1] for item in items], dtype=np.float32)
        faiss.normalize_L2(vectors)
        
        self.index.add(vectors)
        
        for chunk_id, _, chunk in items:
            self.id_map.append(chunk_id)
            self.chunks[chunk_id] = chunk
    
    def search(self, 
               query_embedding: List[float], 
               top_k: int = 10,
               filter_fn: Optional[Callable[[IndexableChunk], bool]] = None
              ) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        
        # Search more if filtering
        search_k = top_k * 3 if filter_fn else top_k
        
        scores, indices = self.index.search(query, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing
                continue
            
            chunk_id = self.id_map[idx]
            chunk = self.chunks[chunk_id]
            
            if filter_fn and not filter_fn(chunk):
                continue
            
            results.append((chunk_id, float(score)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, path: str):
        """Save index to disk"""
        faiss.write_index(self.index, f"{path}/faiss.index")
        # Also save id_map and chunks separately
    
    def load(self, path: str):
        """Load index from disk"""
        self.index = faiss.read_index(f"{path}/faiss.index")

# Usage with pipeline
from unified_indexer import IndexingPipeline

pipeline = IndexingPipeline(vocabulary_data=vocab, embedder_type='hybrid')

# Replace default vector store
pipeline.index.vector_store = FAISSVectorStore(dimension=634)
```

#### ChromaDB Integration

```python
import chromadb
from chromadb.config import Settings

class ChromaVectorStore:
    """ChromaDB-backed vector store with persistence"""
    
    def __init__(self, 
                 collection_name: str = "unified_indexer",
                 persist_directory: str = "./chroma_db"):
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.chunks: Dict[str, IndexableChunk] = {}
    
    def add(self, chunk_id: str, embedding: List[float], chunk: IndexableChunk):
        """Add a chunk with its embedding"""
        self.collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            metadatas=[{
                "source_type": chunk.source_type.value,
                "file_path": chunk.source_ref.file_path or "",
                "capabilities": ",".join(chunk.capability_set)
            }],
            documents=[chunk.text[:1000]]  # Store truncated text
        )
        self.chunks[chunk_id] = chunk
    
    def search(self,
               query_embedding: List[float],
               top_k: int = 10,
               filter_fn: Optional[Callable[[IndexableChunk], bool]] = None
              ) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 3 if filter_fn else top_k
        )
        
        output = []
        for chunk_id, distance in zip(results['ids'][0], results['distances'][0]):
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue
            
            if filter_fn and not filter_fn(chunk):
                continue
            
            # Convert distance to similarity
            similarity = 1 - distance
            output.append((chunk_id, similarity))
            
            if len(output) >= top_k:
                break
        
        return output

# Usage
pipeline.index.vector_store = ChromaVectorStore(
    collection_name="payment_system",
    persist_directory="./my_index/chroma"
)
```

#### Qdrant Integration

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantVectorStore:
    """Qdrant-backed vector store"""
    
    def __init__(self,
                 collection_name: str = "unified_indexer",
                 url: str = "localhost",
                 port: int = 6333,
                 dimension: int = 634):
        
        self.client = QdrantClient(host=url, port=port)
        self.collection_name = collection_name
        
        # Create collection if not exists
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
        except:
            pass  # Collection exists
        
        self.chunks: Dict[str, IndexableChunk] = {}
        self._id_counter = 0
        self._id_to_chunk: Dict[int, str] = {}
    
    def add(self, chunk_id: str, embedding: List[float], chunk: IndexableChunk):
        """Add a chunk with its embedding"""
        point_id = self._id_counter
        self._id_counter += 1
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "chunk_id": chunk_id,
                    "source_type": chunk.source_type.value,
                    "capabilities": list(chunk.capability_set)
                }
            )]
        )
        
        self._id_to_chunk[point_id] = chunk_id
        self.chunks[chunk_id] = chunk
    
    def search(self,
               query_embedding: List[float],
               top_k: int = 10,
               filter_fn: Optional[Callable] = None
              ) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k * 3 if filter_fn else top_k
        )
        
        output = []
        for hit in results:
            chunk_id = hit.payload.get("chunk_id")
            chunk = self.chunks.get(chunk_id)
            
            if not chunk:
                continue
            
            if filter_fn and not filter_fn(chunk):
                continue
            
            output.append((chunk_id, hit.score))
            
            if len(output) >= top_k:
                break
        
        return output
```

---

## 12. Security Considerations

### 12.1 Data Protection

| Concern | Mitigation |
|---------|------------|
| Sensitive code exposure | Local embeddings, no external API for indexing |
| API key leakage | Environment variables, not in code |
| Index file access | File system permissions |
| LLM data transmission | HTTPS, API provider security |

### 12.2 Input Sanitization

```python
def sanitize_text_for_llm(text: str) -> str:
    """Sanitize text before sending to LLM"""
    
    # Handle encoding issues
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    # Remove null bytes and problematic control chars
    sanitized = ''.join(
        char if char.isprintable() or char in '\n\t\r' else ' '
        for char in text
    )
    
    # Collapse multiple blank lines
    sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
    
    return sanitized.strip()
```

### 12.3 API Key Management

```python
# Recommended: Use environment variables
api_key = os.environ.get("ANTHROPIC_API_KEY")

# Alternative: Secrets manager integration
from your_secrets_manager import get_secret
api_key = get_secret("anthropic-api-key")

# Never: Hardcode in source
api_key = "sk-ant-..."  # DON'T DO THIS
```

---

## 13. Performance Characteristics

### 13.1 Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Vocabulary match (Aho-Corasick) | O(n + m) | n = text length, m = matches |
| Hash embedding | O(t) | t = tokens in text |
| Vector search (brute force) | O(n × d) | n = chunks, d = dimensions |
| Concept index lookup | O(1) | Hash map lookup |
| Save/Load index | O(n) | n = total chunks |

### 13.2 Space Complexity

| Component | Size | Notes |
|-----------|------|-------|
| VectorStore | n × d × 4 bytes | Float32 embeddings |
| ConceptIndex | O(c × k) | c = concepts, k = avg chunks per concept |
| Chunk storage | Variable | Depends on text length |

### 13.3 Benchmarks (Approximate)

| Operation | 1K chunks | 10K chunks | 100K chunks |
|-----------|-----------|------------|-------------|
| Index build | ~5 sec | ~30 sec | ~5 min |
| Search (top 10) | ~10 ms | ~50 ms | ~500 ms |
| Save index | ~1 sec | ~5 sec | ~30 sec |
| Load index | ~1 sec | ~5 sec | ~30 sec |

### 13.4 Optimization Recommendations

For large indexes (>100K chunks):

1. **Use FAISS for vector search**:
```python
# Replace VectorStore with FAISS
import faiss
index = faiss.IndexFlatIP(dimension)  # Inner product
index.add(embeddings)
```

2. **Batch processing**:
```python
pipeline.index_directory(path, batch_size=100)
```

3. **Selective embedding**:
```python
# Skip embedding for exact-match-only use cases
pipeline = IndexingPipeline(embedder_type=None)
```

---

## 14. Appendices

### 14.1 CLI Reference

#### build_index.py

```
Usage: build_index.py [OPTIONS]

Options:
  --pdf-dir PATH       Directory containing PDF documents
  --tal-dir PATH       Directory containing TAL code files
  --output, -o PATH    Output directory for index (required)
  --vocab, -v PATH     Vocabulary file (default: keywords.json)
  --embedder, -e TYPE  Embedder type: hash, hybrid, tfidf, domain, bm25
  --recursive, -r      Search directories recursively
  --no-recursive       Don't search recursively

Examples:
  python build_index.py --pdf-dir ./docs --tal-dir ./code -o ./index
  python build_index.py --tal-dir ./code -o ./index --embedder hybrid
```

#### search_index.py

```
Usage: search_index.py [OPTIONS]

Options:
  --index, -i PATH     Index directory (required)
  --query, -q TEXT     Search query
  --top, -n INT        Number of results (default: 5)
  --type, -t TYPE      Filter: code, document, log, all
  --capability, -c     Search by business capability
  --interactive, -I    Interactive mode
  --analyze, -a        Enable LLM analysis
  --provider, -p TYPE  LLM provider: anthropic, openai, internal, ollama, stub
  --model, -m NAME     LLM model name
  --api-url URL        Base URL for internal API
  --min-score FLOAT    Minimum score for LLM (default: 0.70)
  --verbose, -v        Verbose output

Examples:
  python search_index.py -i ./index -q "OFAC screening"
  python search_index.py -i ./index -q "wire transfer" --analyze
  python search_index.py -i ./index --interactive
```

### 14.2 Vocabulary Categories

| Category | Description | Example Terms |
|----------|-------------|---------------|
| `payment-systems` | Core payment concepts | wire transfer, funds transfer |
| `swift-mt-messages` | SWIFT MT message types | MT-103, MT-202, MT-940 |
| `iso20022-pacs-messages` | ISO 20022 pacs.* | pacs.008, pacs.009, pacs.004 |
| `iso20022-pain-messages` | ISO 20022 pain.* | pain.001 |
| `iso20022-camt-messages` | ISO 20022 camt.* | camt.053 |
| `compliance-fraud` | Compliance/AML | OFAC, AML, KYC, BSA |
| `payment-networks` | Payment rails | Fedwire, CHIPS, ACH, SWIFT |
| `payment-parties` | Transaction parties | beneficiary, originator |
| `exception-handling` | Exceptions | payment return, repair |
| `processing` | Processing steps | validation, routing, settlement |
| `identifiers` | IDs and codes | BIC, IBAN, ABA, LEI |

### 14.3 Glossary

| Term | Definition |
|------|------------|
| **Chunk** | Atomic unit of indexed content (procedure, section, log entry) |
| **Domain Match** | Vocabulary term found in content |
| **Capability** | Business capability mapped from domain terms |
| **Hybrid Search** | Combines vector similarity + concept matching |
| **Aho-Corasick** | Algorithm for efficient multi-pattern string matching |
| **Embedding** | Dense vector representation of text |
| **TAL** | Transaction Application Language (HP NonStop) |

### 14.4 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2025 | Initial release |

---

**Document End**
