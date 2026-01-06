#!/usr/bin/env python3
"""
Index Search - Semantic search through indexed code and documents

================================================================================
OVERVIEW
================================================================================

Enhanced semantic search with knowledge graph support for query expansion
and TF-IDF weighted scoring. Leverages extracted domain knowledge to:
- Expand queries using related terms (synonyms, co-occurring concepts)
- Boost scores for distinctive terms (high TF-IDF)
- Show related concepts in search results

================================================================================
ARCHITECTURE
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                            SEARCH PIPELINE                                   │
│                                                                             │
│  USER QUERY              PROCESSING STAGES                    RESULTS       │
│  ──────────              ─────────────────                    ───────       │
│                                                                             │
│  "OFAC screening"                                                           │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: Query Expansion (--expand-query)                           │   │
│  │                                                                      │   │
│  │  Knowledge Graph lookup:                                             │   │
│  │    "OFAC" → co_occurs_with: [sanctions, SDN, screening]             │   │
│  │           → implements: [screen_ofac, validate_ofac]                │   │
│  │                                                                      │   │
│  │  Expanded: "OFAC screening sanctions SDN screen_ofac"               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: Multi-Signal Search                                        │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ Vector       │  │ BM25         │  │ Concept      │              │   │
│  │  │ Similarity   │  │ Lexical      │  │ Matching     │              │   │
│  │  │ (semantic)   │  │ (exact term) │  │ (vocabulary) │              │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │   │
│  │         │                 │                 │                       │   │
│  │         └────────────┬────┴─────────────────┘                       │   │
│  │                      ▼                                              │   │
│  │        Reciprocal Rank Fusion (RRF)                                 │   │
│  │        RRF_score = Σ 1/(k + rank)                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: TF-IDF Boosting (--tfidf-boost, optional)                  │   │
│  │                                                                      │   │
│  │  For each result:                                                    │   │
│  │    - Check if matched terms have high TF-IDF scores                 │   │
│  │    - Boost score by up to 0.3 for distinctive term matches          │   │
│  │    - Re-rank results by boosted scores                              │   │
│  │                                                                      │   │
│  │  Example: "UETR" (TF-IDF=3.0) match → +0.2 boost                    │   │
│  │           "wire" (TF-IDF=0.5) match → +0.05 boost                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: Result Display                                             │   │
│  │                                                                      │   │
│  │  Result #1  |  Score: 0.892  |  Type: CODE                          │   │
│  │  ────────────────────────────────────────                           │   │
│  │  📁 File: /src/compliance/ofac_check.tal                            │   │
│  │  🔧 Procedure: SCREEN_OFAC                                          │   │
│  │  🏷️  Concepts: OFAC, sanctions, compliance                          │   │
│  │  🔗 Related: SDN_LIST, VALIDATE_BIC, CHECK_SANCTIONS  ← from KG     │   │
│  │                                                                      │   │
│  │  📝 Content:                                                         │   │
│  │     PROC SCREEN_OFAC(customer_name, result);                        │   │
│  │     ...                                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
BM25 LEXICAL SEARCH
================================================================================

BM25 (Best Matching 25) provides lexical retrieval to complement vector search:

    BM25 score(D, Q) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1×(1-b+b×|D|/avgdl))

    where:
    - f(qi, D) = frequency of term qi in document D
    - |D| = document length
    - avgdl = average document length
    - k1 = term frequency saturation (default: 1.5)
    - b = length normalization (default: 0.75)

BM25 excels at:
- Exact term matching (acronyms like UETR, OFAC, BIC)
- Technical terms that vector embeddings may miss
- Complementing semantic search with lexical precision

================================================================================
RECIPROCAL RANK FUSION (RRF)
================================================================================

RRF combines results from multiple retrievers using ranks, not scores:

    RRF_score(d) = Σ 1/(k + rank(d))

    where k = 60 (standard constant)

Example:
    Vector:  [A: rank 1, B: rank 2, C: rank 3]
    BM25:    [B: rank 1, D: rank 2, A: rank 3]

    RRF scores:
      A: 1/61 + 1/63 = 0.0323
      B: 1/62 + 1/61 = 0.0325  ← Winner (found by both)
      C: 1/63        = 0.0159
      D: 1/62        = 0.0161

Benefits:
- Robust to score distribution differences
- No normalization needed
- Proven in production (Elasticsearch, Pinecone)

================================================================================
KNOWLEDGE GRAPH CLASS
================================================================================

The KnowledgeGraph class provides fast lookups for search enhancement:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  class KnowledgeGraph:                                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Data Structures                                                      │   │
│  │                                                                      │   │
│  │  nodes: Dict[str, Dict]                                             │   │
│  │    └─ "ofac" → {id, label, type, tf_idf_score, co_occurs_with}     │   │
│  │                                                                      │   │
│  │  _outgoing_edges: Dict[str, List[Dict]]                             │   │
│  │    └─ "ofac" → [{source: ofac, target: sanctions, type: co_occurs}] │   │
│  │                                                                      │   │
│  │  _incoming_edges: Dict[str, List[Dict]]                             │   │
│  │    └─ "sanctions" → [{source: ofac, target: sanctions, ...}]        │   │
│  │                                                                      │   │
│  │  _label_to_id: Dict[str, str]                                       │   │
│  │    └─ "OFAC" → "ofac" (case-insensitive lookup)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Key Methods                                                          │   │
│  │                                                                      │   │
│  │  get_node(term) → Dict                                              │   │
│  │    Returns node data for a term (by ID or label)                    │   │
│  │                                                                      │   │
│  │  get_related_terms(term, max_terms=10) → List[(term, type, weight)] │   │
│  │    BFS traversal to find related terms via edges                    │   │
│  │                                                                      │   │
│  │  get_tfidf_score(term) → float                                      │   │
│  │    Returns TF-IDF score for boosting (0.0 if not found)             │   │
│  │                                                                      │   │
│  │  expand_query(query) → str                                          │   │
│  │    Expands query by adding related terms                            │   │
│  │    "OFAC" → "ofac sanctions sdn screening"                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
TF-IDF BOOSTING ALGORITHM
================================================================================

The apply_tfidf_boost() function re-ranks results:

    Input: List of search results, Knowledge Graph, Original query
    
    For each result:
        boost = 0.0
        
        # Check matched concepts
        for concept in result.matched_concepts:
            if concept has high TF-IDF:
                boost += 0.2 × (tfidf / max_tfidf)
        
        # Check content text
        for query_term in original_query:
            if term in result.text and has high TF-IDF:
                boost += 0.1 × (tfidf / max_tfidf)
        
        # Check procedure name
        if query_term in procedure_name:
            boost += 0.15 × (tfidf / max_tfidf)
        
        # Cap total boost at 0.3
        result.score += min(boost, 0.3)
    
    Output: Results re-sorted by boosted scores

================================================================================
AUTO-LOADING KNOWLEDGE GRAPH
================================================================================

When index was built with --knowledge-graph, the KG is embedded:

    ./my_index/
    ├── index.pkl
    ├── index_meta.json      ← Contains: "knowledge_graph": true
    └── knowledge_graph.json ← Auto-loaded by search

Search auto-loading logic:
    1. If --knowledge-graph flag provided → use that path
    2. Else if ./index/knowledge_graph.json exists → auto-load
    3. Else → search without KG features
    4. If --no-kg flag → skip auto-loading

================================================================================
USAGE
================================================================================

    # Basic semantic search
    python search_index.py --index ./my_index --query "OFAC screening"
    
    # With knowledge graph query expansion (auto-loaded from index)
    python search_index.py --index ./my_index --query "OFAC screening" --expand-query
    
    # With TF-IDF boosting
    python search_index.py --index ./my_index --query "OFAC" --tfidf-boost
    
    # Both features
    python search_index.py --index ./my_index --query "OFAC" -e -b
    
    # Interactive mode
    python search_index.py --index ./my_index --interactive
    
    # With LLM analysis
    python search_index.py --index ./my_index --query "wire transfer" --analyze

================================================================================
ARGUMENTS
================================================================================

    --index           Directory containing the saved index
    --query           Search query string (natural language)
    --top             Number of results to return (default: 5)
    --type            Filter by source type: code, document, log, or all
    --interactive     Start interactive search mode
    --capability      Search by business capability instead of text
    --verbose         Show more details in results
    
    Knowledge Graph Options:
    --knowledge-graph Path to knowledge_graph.json (overrides auto-load)
    --expand-query    Expand query using related terms from knowledge graph
    --tfidf-boost     Boost scores based on TF-IDF (distinctive terms)
    --no-related      Don't show related terms in results
    --no-kg           Don't auto-load embedded knowledge graph
    
    LLM Options:
    --analyze         Send results to LLM for analysis
    --provider        LLM provider (default: tachyon)
    --model           LLM model name
    --min-score       Minimum score for LLM analysis (default: 0.10)

================================================================================
INTERACTIVE COMMANDS
================================================================================

    <query>           Search for text
    :analyze <query>  Search and analyze with LLM
    :cap <capability> Search by business capability
    :code <query>     Search only in code
    :doc <query>      Search only in documents
    :top <n>          Set number of results
    :verbose          Toggle verbose output
    
    Knowledge Graph (when loaded):
    :expand           Toggle query expansion ON/OFF
    :boost            Toggle TF-IDF boosting ON/OFF
    :related          Toggle related terms display
    :lookup <term>    Show term details (TF-IDF, relationships)
    :graph            Show knowledge graph statistics
"""

import sys
import os
import re
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_indexer import IndexingPipeline, SourceType

# Import LLM provider (optional dependency)
try:
    from llm_provider import (
        LLMProvider,
        create_provider,
        analyze_search_results
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Default keywords file location
DEFAULT_KEYWORDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keywords.json")


# =============================================================================
# KNOWLEDGE GRAPH SUPPORT
#
# The KnowledgeGraph class loads and indexes the knowledge_graph.json file
# produced by knowledge_extractor.py. It provides fast lookups for:
#   - Query expansion: finding related terms to add to searches
#   - TF-IDF boosting: getting distinctiveness scores for terms
#   - Result enrichment: showing related concepts in search results
# =============================================================================

class KnowledgeGraph:
    """
    Knowledge graph for query expansion and TF-IDF boosting.
    
    Loads the knowledge_graph.json produced by knowledge_extractor.py and
    provides methods for:
    - Finding related terms for query expansion
    - Looking up TF-IDF scores for result boosting
    - Getting context about terms
    
    Attributes:
        nodes: Dict mapping normalized term to node data
        edges: List of relationship edges
        statistics: Summary statistics from the graph
        
    Example node:
        {
            "id": "ofac",
            "label": "OFAC",
            "type": "acronym",
            "tf_idf_score": 2.5,
            "term_frequency": 45,
            "document_frequency": 3,
            "co_occurs_with": ["sanctions", "sdn", "screening"]
        }
    
    Example edge:
        {
            "source": "ofac",
            "target": "sanctions",
            "type": "co_occurs_with",
            "evidence": "Co-occurred in 5 documents"
        }
    """
    
    def __init__(self, graph_path: str):
        """
        Load knowledge graph from JSON file.
        
        Args:
            graph_path: Path to knowledge_graph.json
        """
        # Primary data structures
        self.nodes: Dict[str, Dict] = {}      # node_id -> node data
        self.edges: List[Dict] = []            # list of all edges
        self.statistics: Dict = {}             # summary stats from extractor
        
        # Index structures for O(1) lookups
        # These are built during _load() for fast graph traversal
        self._outgoing_edges: Dict[str, List[Dict]] = defaultdict(list)  # source -> [edges]
        self._incoming_edges: Dict[str, List[Dict]] = defaultdict(list)  # target -> [edges]
        self._label_to_id: Dict[str, str] = {}  # "OFAC" -> "ofac" (case-insensitive)
        
        self._load(graph_path)
    
    def _load(self, graph_path: str):
        """
        Load and index the knowledge graph from JSON.
        
        Builds index structures for fast lookups:
        - nodes dict for direct ID access
        - _label_to_id for case-insensitive label lookup
        - _outgoing_edges for forward traversal (A→B)
        - _incoming_edges for reverse traversal (B←A)
        """
        with open(graph_path, 'r') as f:
            data = json.load(f)
        
        # Index nodes by their normalized ID
        # Also create label→id mapping for flexible lookups
        for node in data.get('nodes', []):
            node_id = node.get('id')
            if not node_id:
                continue  # Skip nodes without ID
            self.nodes[node_id] = node
            # Allow lookup by original label (e.g., "OFAC" → "ofac")
            if 'label' in node:
                self._label_to_id[node['label'].lower()] = node_id
        
        # Index edges for fast graph traversal
        # Build both forward and reverse indexes
        self.edges = data.get('edges', [])
        for edge in self.edges:
            source = edge.get('source')
            target = edge.get('target')
            if not source or not target:
                continue  # Skip malformed edges
            # Forward: source → [all edges from source]
            self._outgoing_edges[source].append(edge)
            # Reverse: target → [all edges to target]
            self._incoming_edges[target].append(edge)
        
        self.statistics = data.get('statistics', {})
    
    def normalize(self, term: str) -> str:
        """
        Normalize a term to match node IDs.
        
        Transforms: "Wire-Transfer" → "wire_transfer"
        - Lowercase
        - Replace spaces/hyphens with underscores
        - Remove special characters
        """
        normalized = term.lower()
        normalized = re.sub(r'[-\s]+', '_', normalized)  # spaces/hyphens → underscore
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)  # remove special chars
        return normalized
    
    def get_node(self, term: str) -> Optional[Dict]:
        """
        Get node data for a term.
        
        Tries multiple lookup strategies:
        1. Normalized form (wire_transfer)
        2. Label lookup (WIRE_TRANSFER → wire_transfer)
        
        Args:
            term: Term to look up (original or normalized form)
            
        Returns:
            Node dict with id, label, type, tf_idf_score, etc. or None
        """
        # Strategy 1: Try normalized form directly
        normalized = self.normalize(term)
        if normalized in self.nodes:
            return self.nodes[normalized]
        
        # Strategy 2: Try label lookup (case-insensitive)
        if term.lower() in self._label_to_id:
            node_id = self._label_to_id[term.lower()]
            return self.nodes.get(node_id)
        
        return None
    
    def get_related_terms(self, term: str, 
                          relationship_types: Optional[List[str]] = None,
                          max_depth: int = 1,
                          max_terms: int = 10) -> List[Tuple[str, str, float]]:
        """
        Get terms related to the given term via graph relationships.
        
        Uses BFS (Breadth-First Search) to traverse the graph and find
        connected terms. Can follow relationships in both directions.
        
        Args:
            term: Term to find relationships for
            relationship_types: Filter by relationship type (None = all types)
                              Options: co_occurs_with, implements, contains, related_to
            max_depth: How many hops to follow (1 = direct relationships only)
            max_terms: Maximum number of related terms to return
            
        Returns:
            List of (related_term, relationship_type, weight) tuples
            Weight is based on co-occurrence count or 1.0 for structural relationships
            
        Example:
            get_related_terms("OFAC") → [
                ("sanctions", "co_occurs_with", 5.0),
                ("SDN", "co_occurs_with", 3.0),
                ("SCREEN_OFAC", "rev_implements", 1.0)
            ]
        """
        normalized = self.normalize(term)
        related = []
        seen = {normalized}  # Track visited nodes to avoid cycles
        
        # BFS traversal - process nodes level by level
        current_level = [normalized]
        
        for depth in range(max_depth):
            next_level = []
            
            for current in current_level:
                # === Check OUTGOING edges (current → target) ===
                for edge in self._outgoing_edges.get(current, []):
                    # Skip if relationship type doesn't match filter
                    if relationship_types and edge['type'] not in relationship_types:
                        continue
                    
                    target = edge['target']
                    if target not in seen:
                        seen.add(target)
                        
                        # Extract weight from evidence (e.g., "Co-occurred in 5 documents")
                        weight = 1.0
                        if 'Co-occurred in' in edge.get('evidence', ''):
                            try:
                                count = int(edge['evidence'].split()[2])
                                weight = count
                            except:
                                pass
                        
                        # Get display label from node, fallback to ID
                        node = self.nodes.get(target, {})
                        label = node.get('label', target)
                        related.append((label, edge['type'], weight))
                        next_level.append(target)
                
                # === Check INCOMING edges (source → current) ===
                # These are "reverse" relationships
                for edge in self._incoming_edges.get(current, []):
                    if relationship_types and edge['type'] not in relationship_types:
                        continue
                    
                    source = edge['source']
                    if source not in seen:
                        seen.add(source)
                        
                        weight = 1.0
                        if 'Co-occurred in' in edge.get('evidence', ''):
                            try:
                                count = int(edge['evidence'].split()[2])
                                weight = count
                            except:
                                pass
                        
                        node = self.nodes.get(source, {})
                        label = node.get('label', source)
                        # Mark as reverse relationship
                        related.append((label, f"rev_{edge['type']}", weight))
                        next_level.append(source)
            
            current_level = next_level
        
        # Sort by weight (higher = more related) and limit
        related.sort(key=lambda x: -x[2])
        return related[:max_terms]
    
    def get_tfidf_score(self, term: str) -> float:
        """
        Get TF-IDF score for a term.
        
        Higher scores indicate more distinctive terms.
        
        Args:
            term: Term to look up
            
        Returns:
            TF-IDF score or 0.0 if not found
        """
        node = self.get_node(term)
        if node:
            return node.get('tf_idf_score', 0.0)
        return 0.0
    
    def get_co_occurring_terms(self, term: str, min_count: int = 2) -> List[Tuple[str, int]]:
        """
        Get terms that frequently co-occur with this term.
        
        Args:
            term: Term to find co-occurrences for
            min_count: Minimum co-occurrence count
            
        Returns:
            List of (term, count) tuples sorted by count
        """
        node = self.get_node(term)
        if not node:
            return []
        
        co_occurs = node.get('co_occurs_with', [])
        # co_occurs is just a list of term names from the extractor
        # Return with count=1 if we don't have counts
        return [(t, 1) for t in co_occurs]
    
    def expand_query(self, query: str, max_expansion: int = 5) -> str:
        """
        Expand a query using related terms from the knowledge graph.
        
        This adds synonyms and related concepts to improve recall.
        
        Args:
            query: Original query string
            max_expansion: Maximum terms to add per query word
            
        Returns:
            Expanded query string
            
        Example:
            expand_query("OFAC screening")
            -> "OFAC screening sanctions SDN compliance screen_ofac"
        """
        # Tokenize query
        words = query.lower().split()
        expanded_terms = set(words)
        
        for word in words:
            # Get related terms
            related = self.get_related_terms(
                word, 
                relationship_types=['co_occurs_with', 'implements', 'related_to'],
                max_terms=max_expansion
            )
            
            for term, rel_type, weight in related:
                # Add the related term (normalized)
                expanded_terms.add(term.lower().replace('_', ' '))
            
            # Also check co-occurring terms directly from node
            co_terms = self.get_co_occurring_terms(word)
            for term, count in co_terms[:3]:
                expanded_terms.add(term.lower().replace('_', ' '))
        
        return ' '.join(expanded_terms)
    
    def get_term_context(self, term: str) -> Dict:
        """
        Get full context about a term including relationships.
        
        Returns:
            Dict with node data and relationship info
        """
        node = self.get_node(term)
        if not node:
            return {}
        
        context = dict(node)
        context['outgoing_relationships'] = self._outgoing_edges.get(node['id'], [])
        context['incoming_relationships'] = self._incoming_edges.get(node['id'], [])
        
        return context


def load_knowledge_graph(graph_path: str) -> Optional[KnowledgeGraph]:
    """
    Load knowledge graph if path is provided and file exists.
    
    Args:
        graph_path: Path to knowledge_graph.json
        
    Returns:
        KnowledgeGraph instance or None
    """
    if not graph_path:
        return None
    
    if not os.path.exists(graph_path):
        print(f"⚠️  Knowledge graph not found: {graph_path}")
        return None
    
    try:
        kg = KnowledgeGraph(graph_path)
        print(f"📊 Knowledge graph loaded: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
        return kg
    except Exception as e:
        print(f"⚠️  Error loading knowledge graph: {e}")
        return None


# =============================================================================
# VOCABULARY LOADING
#
# Loads the vocabulary (keywords.json or vocabulary.json) that defines
# domain-specific terms and their metadata for concept matching.
# =============================================================================


def load_vocabulary(vocab_path: str) -> list:
    """
    Load vocabulary from JSON file.
    
    Supports two formats:
    1. List format: [{keywords, metadata, ...}, ...]
    2. Dict format: {entries: [{...}, ...]}
    
    Args:
        vocab_path: Path to vocabulary JSON file
        
    Returns:
        List of vocabulary entries
    """
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found: {vocab_path}")
        print(f"Please ensure 'keywords.json' exists in the same directory as this script,")
        print(f"or specify a custom vocabulary file with --vocab")
        sys.exit(1)
    
    with open(vocab_path, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: direct list or dict with 'entries' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get('entries', [data])
    else:
        print(f"Error: Invalid vocabulary format in {vocab_path}")
        sys.exit(1)


# =============================================================================
# RESULT DISPLAY
#
# Functions for formatting and printing search results with optional
# knowledge graph context (related terms, TF-IDF scores).
# =============================================================================


def print_result(result, index: int, verbose: bool = False, 
                 knowledge_graph: Optional[KnowledgeGraph] = None,
                 show_related: bool = True):
    """
    Print a single search result with optional knowledge graph context.
    
    Displays:
    - Score and source type
    - File path, line numbers, procedure name
    - Matched concepts and business capabilities
    - Related terms from knowledge graph (if loaded)
    - Content preview
    
    Args:
        result: Search result object from pipeline.search()
        index: Result index for numbering (0-based)
        verbose: Show detailed scores and metadata
        knowledge_graph: Optional KG for showing related terms
        show_related: Whether to show related terms from KG
    """
    chunk = result.chunk
    
    # === Header: Score and type ===
    print(f"\n{'─' * 60}")
    score_info = f"Score: {result.combined_score:.3f}"
    if verbose:
        # Show component scores: vector, bm25, concept
        score_info += f" (v:{result.vector_score:.3f} b:{result.bm25_score:.3f} c:{result.concept_score:.3f})"
    print(f"Result #{index + 1}  |  {score_info}  |  Type: {chunk.source_type.value.upper()}")
    if verbose:
        print(f"Method: {result.retrieval_method}")
    print(f"{'─' * 60}")
    
    # === Source location ===
    source_ref = chunk.source_ref
    if source_ref.file_path:
        print(f"📁 File: {source_ref.file_path}")
    
    if source_ref.line_start:
        line_info = f"Lines {source_ref.line_start}"
        if source_ref.line_end and source_ref.line_end != source_ref.line_start:
            line_info += f"-{source_ref.line_end}"
        print(f"📍 {line_info}")
    
    if source_ref.procedure_name:
        print(f"🔧 Procedure: {source_ref.procedure_name}")
    
    if source_ref.page_number:
        print(f"📄 Page: {source_ref.page_number}")
    
    if result.matched_concepts:
        concepts = result.matched_concepts[:5]
        print(f"🏷️  Concepts: {', '.join(concepts)}")
    
    capabilities = list(chunk.capability_set)[:3]
    if capabilities:
        print(f"💼 Capabilities: {', '.join(capabilities)}")
    
    # Show calls if available
    calls = chunk.metadata.get('calls', [])
    if calls:
        print(f"📞 Calls: {', '.join(calls[:10])}")
        if len(calls) > 10:
            print(f"   ... and {len(calls) - 10} more")
    
    # Show related terms from knowledge graph
    if knowledge_graph and show_related:
        # Get related terms for matched concepts or procedure name
        related_shown = set()
        terms_to_check = []
        
        if source_ref.procedure_name:
            terms_to_check.append(source_ref.procedure_name)
        if result.matched_concepts:
            terms_to_check.extend(result.matched_concepts[:3])
        
        all_related = []
        for term in terms_to_check:
            node = knowledge_graph.get_node(term)
            if node:
                # Get TF-IDF info
                tfidf = node.get('tf_idf_score', 0)
                if tfidf > 0 and verbose:
                    print(f"📈 TF-IDF({term}): {tfidf:.3f}")
                
                # Get related terms
                related = knowledge_graph.get_related_terms(term, max_terms=5)
                for rel_term, rel_type, weight in related:
                    if rel_term.lower() not in related_shown:
                        related_shown.add(rel_term.lower())
                        all_related.append((rel_term, rel_type))
        
        if all_related:
            rel_display = [f"{t}" for t, rt in all_related[:5]]
            print(f"🔗 Related: {', '.join(rel_display)}")
    
    print(f"\n📝 Content:")
    text = chunk.text.strip()
    
    max_len = 500 if verbose else 200
    if len(text) > max_len:
        text = text[:max_len] + "..."
    
    for line in text.split('\n')[:10]:
        print(f"   {line}")
    
    if verbose and chunk.metadata:
        print(f"\n🔍 Metadata: {chunk.metadata}")


def print_results(results, verbose: bool = False,
                  knowledge_graph: Optional[KnowledgeGraph] = None,
                  show_related: bool = True):
    """Print all search results with optional knowledge graph context."""
    if not results:
        print("\n⚠️  No results found.")
        return
    
    print(f"\n{'═' * 60}")
    print(f"Found {len(results)} result(s)")
    print(f"{'═' * 60}")
    
    for i, result in enumerate(results):
        print_result(result, i, verbose, knowledge_graph, show_related)
    
    print(f"\n{'═' * 60}")


def print_llm_analysis(response, verbose: bool = False):
    """Print LLM analysis response"""
    print(f"\n{'═' * 60}")
    print("🤖 LLM ANALYSIS")
    print(f"{'═' * 60}")
    
    if not response.success:
        print(f"\n❌ Error: {response.error}")
        return
    
    print(f"\nProvider: {response.provider} | Model: {response.model}")
    if response.tokens_used:
        print(f"Tokens used: {response.tokens_used}")
    
    print(f"\n{'─' * 60}")
    print(response.content)
    print(f"{'─' * 60}")


def search_once(pipeline: IndexingPipeline, 
                query: str, 
                top_k: int = 5,
                source_type: str = "all",
                verbose: bool = False,
                knowledge_graph: Optional[KnowledgeGraph] = None,
                expand_query: bool = False,
                tfidf_boost: bool = False,
                show_related: bool = True):
    """
    Perform a semantic search with optional knowledge graph enhancement.
    
    This is the main search function that orchestrates the search pipeline:
    
    1. QUERY EXPANSION (if enabled):
       - Look up query terms in knowledge graph
       - Add related terms (co-occurring, implementing procedures)
       - "OFAC" → "OFAC sanctions SDN screening"
    
    2. MULTI-SIGNAL SEARCH:
       - Vector similarity (semantic meaning)
       - Concept matching (domain vocabulary)
       - Keyword matching (exact terms)
       - Combined into single score
    
    3. TF-IDF BOOSTING (if enabled):
       - Look up TF-IDF scores for query terms
       - Boost results matching distinctive terms
       - Re-rank by boosted scores
    
    4. RESULT DISPLAY:
       - Show file, procedure, concepts
       - Show related terms from knowledge graph
       - Show content preview
    
    Args:
        pipeline: The indexing pipeline with loaded index
        query: Search query (natural language)
        top_k: Number of results to return
        source_type: Filter by type (all, code, document, log)
        verbose: Show detailed scores and metadata
        knowledge_graph: Optional KnowledgeGraph for expansion/boosting
        expand_query: If True, expand query using related terms from KG
        tfidf_boost: If True, boost scores based on TF-IDF scores
        show_related: If True, show related terms in results
        
    Returns:
        List of search results
    """
    
    # === Step 1: Parse source type filter ===
    source_types = None
    if source_type == "code":
        source_types = [SourceType.CODE]
    elif source_type == "document":
        source_types = [SourceType.DOCUMENT]
    elif source_type == "log":
        source_types = [SourceType.LOG]
    # "all" leaves source_types as None (no filter)
    
    # === Step 2: Query expansion using knowledge graph ===
    search_query = query
    if knowledge_graph and expand_query:
        # expand_query() looks up each term and adds related terms
        expanded = knowledge_graph.expand_query(query)
        if expanded != query.lower():
            search_query = expanded
            print(f"\n🔎 Original query: \"{query}\"")
            print(f"📖 Expanded query: \"{search_query}\"")
        else:
            # No expansion happened (terms not in KG)
            print(f"\n🔎 Searching for: \"{query}\"")
    else:
        print(f"\n🔎 Searching for: \"{query}\"")
    
    if source_types:
        print(f"   Filtered to: {source_type}")
    
    # === Step 3: Execute the search ===
    # If TF-IDF boosting is enabled, fetch extra results for re-ranking
    # This ensures we have enough candidates after re-ranking
    fetch_k = top_k * 3 if tfidf_boost and knowledge_graph else top_k * 2  # Fetch more for dedup
    
    # pipeline.search() does the heavy lifting:
    # - Embeds the query
    # - Finds similar vectors (semantic search)
    # - Matches concepts (domain vocabulary)
    # - Combines scores from multiple signals
    results = pipeline.search(search_query, top_k=fetch_k, source_types=source_types)
    
    if not results:
        print("\n⚠️  No results found.")
        return results
    
    # === Step 4: TF-IDF boosting (re-rank results) ===
    # This step boosts results that match distinctive terms
    if knowledge_graph and tfidf_boost and results:
        results = apply_tfidf_boost(results, knowledge_graph, query)
    
    # === Step 5: Deduplicate by file ===
    # Keep only the highest-scoring chunk per file
    seen_files = set()
    deduped_results = []
    for r in results:
        file_path = r.chunk.source_ref.file_path or r.chunk.chunk_id
        if file_path not in seen_files:
            seen_files.add(file_path)
            deduped_results.append(r)
    results = deduped_results[:top_k]
    
    # === Step 6: Display results ===
    print(f"\n{'═' * 60}")
    print(f"Found {len(results)} result(s)")
    if knowledge_graph and (expand_query or tfidf_boost):
        features = []
        if expand_query:
            features.append("query expansion")
        if tfidf_boost:
            features.append("TF-IDF boost")
        print(f"   Knowledge graph: {', '.join(features)}")
    print(f"{'═' * 60}")
    
    # print_result() shows file, procedure, concepts, related terms, content
    for i, result in enumerate(results):
        print_result(result, i, verbose, knowledge_graph, show_related)
    
    print(f"\n{'═' * 60}")
    
    return results


def apply_tfidf_boost(results, knowledge_graph: KnowledgeGraph, 
                      original_query: str) -> list:
    """
    Re-rank results by boosting scores for matches on high TF-IDF terms.
    
    Terms with high TF-IDF (distinctive terms) get a score boost.
    This helps surface results that match rare/specific terminology.
    
    Algorithm:
        1. Extract query terms and look up their TF-IDF scores
        2. For each result, calculate boost based on:
           - Matched concepts (+0.2 max per term)
           - Content text matches (+0.1 max per term)
           - Procedure name matches (+0.15 max per term)
        3. Normalize boosts by max TF-IDF to keep scale consistent
        4. Cap total boost at 0.3 to prevent overwhelming base score
        5. Re-sort results by boosted score
    
    Args:
        results: List of search results from pipeline.search()
        knowledge_graph: KnowledgeGraph with TF-IDF scores
        original_query: Original query for term extraction
        
    Returns:
        Re-ranked list of results with boosted scores
        
    Example:
        Query: "UETR validation"
        UETR has TF-IDF=3.0 (distinctive), validation has TF-IDF=0.5 (common)
        
        Result matching "UETR" in procedure name:
            boost = 0.15 * (3.0/3.0) = 0.15
        
        Result matching "validation" in text:
            boost = 0.1 * (0.5/3.0) = 0.017
    """
    # Step 1: Extract query terms and get their TF-IDF scores from knowledge graph
    query_terms = original_query.lower().split()
    
    term_scores = {}  # term -> TF-IDF score
    for term in query_terms:
        score = knowledge_graph.get_tfidf_score(term)
        if score > 0:
            term_scores[term] = score
    
    # If no query terms have TF-IDF data, return results unchanged
    if not term_scores:
        return results
    
    # Step 2: Calculate boost for each result
    boosted_results = []
    max_tfidf = max(term_scores.values()) if term_scores else 1.0  # For normalization
    
    for result in results:
        boost = 0.0
        chunk = result.chunk
        
        # --- Boost for concept matches ---
        # Concepts are domain terms matched from vocabulary
        # These are high-value matches, so boost is larger (0.2 max)
        for concept in (result.matched_concepts or []):
            concept_lower = concept.lower()
            for term, tfidf in term_scores.items():
                if term in concept_lower:
                    # Normalize: distinctive terms (high TF-IDF) get full boost
                    boost += 0.2 * (tfidf / max_tfidf)
        
        # --- Boost for content text matches ---
        # Raw text matches are less precise, smaller boost (0.1 max)
        text_lower = chunk.text.lower()
        for term, tfidf in term_scores.items():
            if term in text_lower:
                boost += 0.1 * (tfidf / max_tfidf)
        
        # --- Boost for procedure name matches ---
        # Procedure names are meaningful identifiers, medium boost (0.15 max)
        if chunk.source_ref.procedure_name:
            proc_lower = chunk.source_ref.procedure_name.lower()
            for term, tfidf in term_scores.items():
                if term in proc_lower:
                    boost += 0.15 * (tfidf / max_tfidf)
        
        # Cap total boost at 0.3 to prevent runaway scores
        boost = min(boost, 0.3)
        
        # Apply boost to combined score
        result.combined_score = result.combined_score + boost
        boosted_results.append(result)
    
    # Step 3: Re-sort by boosted scores (highest first)
    boosted_results.sort(key=lambda r: -r.combined_score)
    
    return boosted_results


# =============================================================================
# QUERY DECOMPOSITION
#
# Long queries with multiple concepts often perform poorly because:
# - Different concepts compete for attention in the embedding
# - A single query mixes unrelated terms
#
# Query decomposition splits long queries into focused sub-queries:
# "How does OFAC screening work for wire transfers with BIC validation?"
#   → ["OFAC screening", "wire transfers", "BIC validation"]
#
# Each sub-query is searched separately, then results are fused via RRF.
# =============================================================================


def extract_domain_keywords(query: str, 
                            vocabulary: list,
                            knowledge_graph: Optional[KnowledgeGraph] = None) -> List[Tuple[str, str]]:
    """
    Extract domain-specific keywords from a query.
    
    Matches query terms against vocabulary entries and knowledge graph nodes
    to identify domain concepts.
    
    Args:
        query: The user's search query
        vocabulary: List of vocabulary entries from keywords.json (or dict with 'entries' key)
        knowledge_graph: Optional KnowledgeGraph for additional terms
        
    Returns:
        List of (keyword, source) tuples where source is 'vocab' or 'kg'
        
    Example:
        query = "How does OFAC sanctions screening work for wire transfers?"
        returns = [("OFAC", "vocab"), ("sanctions", "kg"), ("wire transfer", "vocab")]
    """
    query_lower = query.lower()
    keywords = []
    seen = set()
    
    # Handle vocabulary as dict with 'entries' key or list
    if isinstance(vocabulary, dict):
        vocab_entries = vocabulary.get('entries', [])
    else:
        vocab_entries = vocabulary if vocabulary else []
    
    # Build vocabulary term lookup
    vocab_terms = {}  # normalized_term -> original_form
    for entry in vocab_entries:
        if isinstance(entry, str):
            # Simple string entry
            term_lower = entry.lower().strip()
            if term_lower and len(term_lower) >= 2:
                vocab_terms[term_lower] = entry
            continue
            
        kw_field = entry.get('keywords', '')
        if isinstance(kw_field, str):
            terms = [t.strip() for t in kw_field.split(',')]
        else:
            terms = [str(t) for t in kw_field]
        
        for term in terms:
            term_lower = term.lower().strip()
            if term_lower and len(term_lower) >= 2:
                vocab_terms[term_lower] = term
                # Also index without underscores/hyphens
                normalized = term_lower.replace('_', ' ').replace('-', ' ')
                if normalized != term_lower:
                    vocab_terms[normalized] = term
    
    # Match vocabulary terms (prefer longer matches)
    for term_lower, term_orig in sorted(vocab_terms.items(), key=lambda x: -len(x[0])):
        if term_lower in query_lower and term_lower not in seen:
            keywords.append((term_orig, 'vocab'))
            seen.add(term_lower)
            # Also mark component words as seen to avoid duplicates
            for word in term_lower.split():
                seen.add(word)
    
    # Match knowledge graph nodes
    if knowledge_graph:
        for node_id, node in knowledge_graph.nodes.items():
            label = node.get('label', node_id)
            label_lower = label.lower()
            
            # Skip if already matched
            if label_lower in seen or node_id in seen:
                continue
            
            # Check if term appears in query
            if label_lower in query_lower or node_id.replace('_', ' ') in query_lower:
                keywords.append((label, 'kg'))
                seen.add(label_lower)
                seen.add(node_id)
    
    return keywords


def decompose_query(query: str,
                    vocabulary: list,
                    knowledge_graph: Optional[KnowledgeGraph] = None,
                    min_keywords: int = 2,
                    max_subqueries: int = 4) -> List[str]:
    """
    Decompose a long query into focused sub-queries based on domain keywords.
    
    Strategy:
    1. Extract domain keywords from the query
    2. Group related keywords (using knowledge graph co-occurrence)
    3. Create sub-queries for each group
    4. Fall back to original query if not enough keywords found
    
    Args:
        query: Original search query
        vocabulary: Vocabulary entries for keyword extraction
        knowledge_graph: Optional KG for grouping related terms
        min_keywords: Minimum keywords needed to decompose (default: 2)
        max_subqueries: Maximum number of sub-queries to generate (default: 4)
        
    Returns:
        List of sub-query strings. Returns [query] if decomposition not warranted.
        
    Example:
        query = "How does OFAC screening work and how are wire transfers validated with BIC?"
        returns = ["OFAC screening sanctions", "wire transfers BIC validation"]
    """
    # Extract domain keywords
    keywords = extract_domain_keywords(query, vocabulary, knowledge_graph)
    
    # Not enough keywords to decompose
    if len(keywords) < min_keywords:
        return [query]
    
    # Short query doesn't need decomposition
    word_count = len(query.split())
    if word_count <= 5:
        return [query]
    
    # Group related keywords using knowledge graph
    keyword_groups = []
    used_keywords = set()
    
    if knowledge_graph:
        # Find clusters of related keywords
        for kw, source in keywords:
            if kw.lower() in used_keywords:
                continue
            
            # Start a new group with this keyword
            group = [kw]
            used_keywords.add(kw.lower())
            
            # Find related keywords from the same query
            kw_node = knowledge_graph.get_node(kw)
            if kw_node:
                co_occurs = kw_node.get('co_occurs_with', [])
                related_terms = set(t.lower() for t in co_occurs)
                
                # Add related keywords to this group
                for other_kw, _ in keywords:
                    if other_kw.lower() in used_keywords:
                        continue
                    if other_kw.lower() in related_terms:
                        group.append(other_kw)
                        used_keywords.add(other_kw.lower())
            
            keyword_groups.append(group)
    else:
        # Without KG, each keyword is its own group
        for kw, source in keywords:
            if kw.lower() not in used_keywords:
                keyword_groups.append([kw])
                used_keywords.add(kw.lower())
    
    # Limit number of groups
    keyword_groups = keyword_groups[:max_subqueries]
    
    # Create sub-queries from groups
    sub_queries = []
    for group in keyword_groups:
        if len(group) == 1:
            # Single keyword - just use the keyword (cleaner)
            sub_queries.append(group[0])
        else:
            # Multiple related keywords - combine them
            sub_queries.append(' '.join(group))
    
    # If we only got one sub-query, return original only
    if len(sub_queries) <= 1:
        return [query]
    
    # CRITICAL: Include original query FIRST to preserve context/relationships
    # Then add focused sub-queries for concept boosting
    # RRF will combine: original (context) + focused (precision)
    return [query] + sub_queries


def search_decomposed(pipeline: IndexingPipeline,
                      query: str,
                      vocabulary: list,
                      top_k: int = 5,
                      source_type: str = "all",
                      verbose: bool = False,
                      knowledge_graph: Optional[KnowledgeGraph] = None,
                      expand_query: bool = False,
                      tfidf_boost: bool = False,
                      show_related: bool = True) -> list:
    """
    Search using query decomposition for long queries.
    
    Decomposes the query into focused sub-queries, runs each separately,
    then combines results using Reciprocal Rank Fusion (RRF).
    
    Benefits:
    - Each sub-query focuses on one concept
    - Results from multiple perspectives are combined
    - Documents matching multiple concepts rank higher
    
    Args:
        pipeline: The indexing pipeline
        query: Original search query
        vocabulary: Vocabulary for keyword extraction
        top_k: Number of final results
        source_type: Filter by source type
        verbose: Show detailed output
        knowledge_graph: Optional KG for expansion/boosting
        expand_query: Enable query expansion
        tfidf_boost: Enable TF-IDF boosting
        show_related: Show related terms in results
        
    Returns:
        List of search results
    """
    from unified_indexer.index import reciprocal_rank_fusion
    
    # Decompose query into sub-queries
    sub_queries = decompose_query(query, vocabulary, knowledge_graph)
    
    # If no decomposition, use regular search
    if len(sub_queries) == 1:
        return search_once(pipeline, query, top_k, source_type, verbose,
                          knowledge_graph, expand_query, tfidf_boost, show_related)
    
    # Parse source type filter
    source_types = None
    if source_type == "code":
        source_types = [SourceType.CODE]
    elif source_type == "document":
        source_types = [SourceType.DOCUMENT]
    elif source_type == "log":
        source_types = [SourceType.LOG]
    
    print(f"\n🔎 Query decomposition:")
    print(f"   Original: \"{query}\"")
    print(f"   Sub-queries ({len(sub_queries)}):")
    for i, sq in enumerate(sub_queries, 1):
        print(f"      {i}. \"{sq}\"")
    
    # Run each sub-query
    all_results = []  # List of (chunk_id, score) lists for RRF
    chunk_lookup = {}  # chunk_id -> SearchResult (keep best)
    
    for sq in sub_queries:
        # Expand sub-query if enabled
        search_sq = sq
        if knowledge_graph and expand_query:
            search_sq = knowledge_graph.expand_query(sq)
        
        # Search
        fetch_k = top_k * 2  # Fetch extra for fusion
        results = pipeline.search(search_sq, top_k=fetch_k, source_types=source_types)
        
        # Apply TF-IDF boost if enabled
        if knowledge_graph and tfidf_boost and results:
            results = apply_tfidf_boost(results, knowledge_graph, sq)
        
        # Collect for RRF
        result_tuples = []
        for r in results:
            chunk_id = r.chunk.chunk_id
            result_tuples.append((chunk_id, r.combined_score))
            
            # Keep the result with highest score for each chunk
            if chunk_id not in chunk_lookup or r.combined_score > chunk_lookup[chunk_id].combined_score:
                chunk_lookup[chunk_id] = r
        
        all_results.append(result_tuples)
        
        if verbose:
            print(f"\n   Sub-query \"{sq}\": {len(results)} results")
    
    # Fuse results using RRF
    if not all_results or not any(all_results):
        print("\n⚠️  No results found.")
        return []
    
    fused_scores = reciprocal_rank_fusion(all_results, k=60)
    
    # Normalize RRF scores to 0-1 range
    # Raw RRF scores are tiny (e.g., 0.01-0.05), need normalization for thresholds
    if fused_scores:
        max_rrf = max(fused_scores.values())
        min_rrf = min(fused_scores.values())
        rrf_range = max_rrf - min_rrf
        if rrf_range > 0:
            fused_scores = {
                cid: (score - min_rrf) / rrf_range 
                for cid, score in fused_scores.items()
            }
        else:
            # Single result or all same score - give full score
            fused_scores = {cid: 1.0 for cid in fused_scores}
    
    # Build final result list
    final_results = []
    for chunk_id, rrf_score in sorted(fused_scores.items(), key=lambda x: -x[1]):
        if chunk_id in chunk_lookup:
            result = chunk_lookup[chunk_id]
            # Update combined_score to normalized RRF score
            result.combined_score = rrf_score
            final_results.append(result)
    
    # Deduplicate by file - keep only best chunk per file
    seen_files = {}
    deduped_results = []
    for result in final_results:
        file_path = result.chunk.source_ref.file_path or result.chunk.chunk_id
        if file_path not in seen_files:
            seen_files[file_path] = result
            deduped_results.append(result)
        # else: skip duplicate file, keep higher-scored one (results already sorted)
    
    final_results = deduped_results[:top_k]
    
    # Assign ranks
    for i, result in enumerate(final_results):
        result.rank = i + 1
    
    # Display results
    print(f"\n{'═' * 60}")
    print(f"Found {len(final_results)} result(s) (fused from {len(sub_queries)} sub-queries)")
    print(f"{'═' * 60}")
    
    for i, result in enumerate(final_results):
        print_result(result, i, verbose, knowledge_graph, show_related)
    
    print(f"\n{'═' * 60}")
    
    return final_results


def search_and_analyze(pipeline: IndexingPipeline,
                       query: str,
                       provider: 'LLMProvider',
                       top_k: int = 20,
                       source_type: str = "all",
                       min_score: float = 0.10,
                       verbose: bool = False,
                       knowledge_graph: Optional[KnowledgeGraph] = None,
                       expand_query: bool = False,
                       tfidf_boost: bool = False,
                       full_file: bool = False):
    """
    Search and analyze with LLM, with optional knowledge graph enhancement.
    
    The LLM receives relevant code/documentation for analysis.
    Knowledge graph can expand the query and boost distinctive terms.
    
    Args:
        full_file: If True, send full file content to LLM instead of just chunks
    """
    
    source_types = None
    if source_type == "code":
        source_types = [SourceType.CODE]
    elif source_type == "document":
        source_types = [SourceType.DOCUMENT]
    elif source_type == "log":
        source_types = [SourceType.LOG]
    
    # Query expansion
    search_query = query
    if knowledge_graph and expand_query:
        expanded = knowledge_graph.expand_query(query)
        if expanded != query.lower():
            search_query = expanded
            print(f"\n🔎 Original query: \"{query}\"")
            print(f"📖 Expanded query: \"{search_query}\"")
        else:
            print(f"\n🔎 Searching for: \"{query}\"")
    else:
        print(f"\n🔎 Searching for: \"{query}\"")
    
    if source_types:
        print(f"   Filtered to: {source_type}")
    print(f"   Min score for analysis: {min_score}")
    
    # Get results for analysis
    results = pipeline.search(search_query, top_k=top_k, source_types=source_types)
    
    # Apply TF-IDF boosting
    if knowledge_graph and tfidf_boost and results:
        results = apply_tfidf_boost(results, knowledge_graph, query)
    
    # Show results summary
    high_score_count = len([r for r in results if r.combined_score >= min_score])
    print(f"\n📊 Found {len(results)} results, {high_score_count} with score >= {min_score}")
    
    if verbose:
        for i, result in enumerate(results[:10]):
            print_result(result, i, verbose, knowledge_graph)
    else:
        # Show brief summary
        print("\nTop results:")
        for i, r in enumerate(results[:5]):
            chunk = r.chunk
            source = chunk.source_ref.file_path or "unknown"
            proc = chunk.metadata.get('procedure_name') or chunk.metadata.get('function_name') or ''
            if proc:
                print(f"  {i+1}. [{r.combined_score:.3f}] {proc} ({Path(source).name})")
            else:
                print(f"  {i+1}. [{r.combined_score:.3f}] {chunk.source_type.value}: {Path(source).name}")
    
    # Send to LLM for analysis
    print(f"\n🤖 Sending to LLM for analysis...")
    if full_file:
        print(f"   📁 Full file mode: enabled")
    
    response = analyze_search_results(
        query=query,
        results=results,
        provider=provider,
        min_score=min_score,
        max_chunks=20,
        verbose=verbose,
        full_file=full_file
    )
    
    print_llm_analysis(response, verbose)
    
    return results, response


def search_and_analyze_decomposed(pipeline: IndexingPipeline,
                                   query: str,
                                   vocabulary: list,
                                   provider: 'LLMProvider',
                                   top_k: int = 20,
                                   source_type: str = "all",
                                   min_score: float = 0.10,
                                   verbose: bool = False,
                                   knowledge_graph: Optional[KnowledgeGraph] = None,
                                   expand_query: bool = False,
                                   tfidf_boost: bool = False,
                                   full_file: bool = False):
    """
    Search with query decomposition and analyze with LLM.
    
    Unlike regular search_and_analyze, this:
    1. Decomposes the query into sub-queries
    2. Searches each sub-query separately
    3. Sends ALL results to LLM organized by sub-query
    4. LLM can see results from each perspective and synthesize
    
    This is better for complex multi-concept queries because the LLM
    can understand relationships between concepts that RRF fusion cannot.
    
    Args:
        pipeline: The indexing pipeline
        query: Original search query
        vocabulary: Vocabulary for keyword extraction
        provider: LLM provider for analysis
        top_k: Results per sub-query
        source_type: Filter by source type
        min_score: Minimum score threshold
        verbose: Show detailed output
        knowledge_graph: Optional KG for expansion
        expand_query: Enable query expansion
        tfidf_boost: Enable TF-IDF boosting
    """
    # Parse source type
    source_types = None
    if source_type == "code":
        source_types = [SourceType.CODE]
    elif source_type == "document":
        source_types = [SourceType.DOCUMENT]
    elif source_type == "log":
        source_types = [SourceType.LOG]
    
    # Decompose query
    sub_queries = decompose_query(query, vocabulary, knowledge_graph)
    
    # If no decomposition, fall back to regular search_and_analyze
    if len(sub_queries) == 1:
        return search_and_analyze(pipeline, query, provider, top_k, source_type,
                                  min_score, verbose, knowledge_graph, 
                                  expand_query, tfidf_boost, full_file)
    
    print(f"\n🔎 Query decomposition for LLM analysis:")
    print(f"   Original: \"{query}\"")
    print(f"   Sub-queries ({len(sub_queries)}):")
    for i, sq in enumerate(sub_queries, 1):
        label = "(full context)" if i == 1 else "(focused)"
        print(f"      {i}. \"{sq}\" {label}")
    
    # Search each sub-query and collect results
    results_by_subquery = {}  # sub_query -> list of results
    all_chunks_seen = set()   # Track unique chunks
    all_files_seen = set()    # Track unique files
    
    for sq in sub_queries:
        # Expand if enabled
        search_sq = sq
        if knowledge_graph and expand_query:
            search_sq = knowledge_graph.expand_query(sq)
        
        # Search
        results = pipeline.search(search_sq, top_k=top_k, source_types=source_types)
        
        # Apply TF-IDF boost
        if knowledge_graph and tfidf_boost and results:
            results = apply_tfidf_boost(results, knowledge_graph, sq)
        
        # Filter by score and dedupe by chunk and file
        filtered = []
        for r in results:
            if r.combined_score >= min_score:
                chunk_id = r.chunk.chunk_id
                file_path = r.chunk.source_ref.file_path or chunk_id
                # Dedupe by both chunk_id and file_path
                if chunk_id not in all_chunks_seen and file_path not in all_files_seen:
                    filtered.append(r)
                    all_chunks_seen.add(chunk_id)
                    all_files_seen.add(file_path)
        
        results_by_subquery[sq] = filtered
        
        if verbose:
            print(f"\n   Sub-query \"{sq}\": {len(filtered)} results (score >= {min_score})")
    
    # Show summary
    total_results = sum(len(r) for r in results_by_subquery.values())
    print(f"\n📊 Total unique results: {total_results} across {len(sub_queries)} sub-queries")
    
    if verbose:
        for sq, results in results_by_subquery.items():
            print(f"\n   [{sq}]")
            for i, r in enumerate(results[:3]):
                chunk = r.chunk
                proc = chunk.metadata.get('procedure_name') or chunk.metadata.get('function_name') or ''
                source = Path(chunk.source_ref.file_path or 'unknown').name
                print(f"      {i+1}. [{r.combined_score:.3f}] {proc or source}")
    
    # Build LLM prompt with results organized by sub-query
    print(f"\n🤖 Sending to LLM for analysis...")
    if full_file:
        print(f"   📁 Full file mode: enabled")
    
    context_parts = []
    context_parts.append(f"Original Question: {query}\n")
    context_parts.append(f"The query was decomposed into {len(sub_queries)} search perspectives:\n")
    
    # Track files already included (for full_file deduplication)
    files_included_in_context = set()
    
    for i, (sq, results) in enumerate(results_by_subquery.items(), 1):
        if i == 1:
            context_parts.append(f"\n=== Perspective {i}: Full Context Search ===")
            context_parts.append(f"Query: \"{sq}\"\n")
        else:
            context_parts.append(f"\n=== Perspective {i}: Focused on '{sq}' ===")
        
        if not results:
            context_parts.append("No relevant results found.\n")
            continue
        
        for j, r in enumerate(results[:5], 1):  # Top 5 per sub-query
            chunk = r.chunk
            proc = chunk.metadata.get('procedure_name') or chunk.metadata.get('function_name')
            source = chunk.source_ref.file_path or 'unknown'
            
            # Skip if we already included this file in full_file mode
            if full_file and source in files_included_in_context:
                continue
            
            context_parts.append(f"\n--- Result {i}.{j} [{r.combined_score:.3f}] ---")
            context_parts.append(f"Source: {source}")
            if proc:
                context_parts.append(f"Procedure: {proc}")
            
            # Use full file content if enabled
            if full_file and source != 'unknown':
                try:
                    with open(source, 'r', encoding='utf-8', errors='ignore') as f:
                        file_content = f.read(50000)  # Max 50KB
                        if len(file_content) == 50000:
                            file_content += "\n... (truncated at 50KB)"
                    context_parts.append(f"Full File Content:\n{file_content}")
                    files_included_in_context.add(source)
                except:
                    context_parts.append(f"Content:\n{chunk.text[:1500]}")
                    if len(chunk.text) > 1500:
                        context_parts.append("... (truncated)")
            else:
                context_parts.append(f"Content:\n{chunk.text[:1500]}")
                if len(chunk.text) > 1500:
                    context_parts.append("... (truncated)")
    
    context = "\n".join(context_parts)
    
    # Create analysis prompt
    prompt = f"""Analyze the following search results to answer the user's question.

The results are organized by search perspective - the first perspective searched the full query
for context, while subsequent perspectives focused on specific concepts mentioned in the query.

Use information from ALL perspectives to provide a comprehensive answer.
Cite specific procedures, files, or sections when referencing the code/documentation.

{context}

---

Based on these results from multiple search perspectives, please provide:
1. A direct answer to the question: "{query}"
2. Key relevant procedures/functions found
3. Any important relationships between the concepts searched
4. Suggestions for further exploration if the answer is incomplete
"""
    
    # Call LLM
    response = provider.complete(prompt)
    
    print_llm_analysis(response, verbose)
    
    # Return flattened results for compatibility
    all_results = []
    for results in results_by_subquery.values():
        all_results.extend(results)
    
    return all_results, response



def search_by_capability(pipeline: IndexingPipeline,
                         capability: str,
                         top_k: int = 5,
                         verbose: bool = False):
    """Search by business capability"""
    print(f"\n🔎 Searching by capability: \"{capability}\"")
    
    results = pipeline.get_by_capability(capability, top_k=top_k)
    print_results(results, verbose)
    
    return results


def list_capabilities(pipeline: IndexingPipeline):
    """List all available business capabilities"""
    stats = pipeline.index.get_statistics()
    
    if 'concept_index' in stats and 'capabilities' in stats['concept_index']:
        capabilities = stats['concept_index']['capabilities']
        print("\n📋 Available Business Capabilities:")
        for cap in sorted(capabilities):
            print(f"   • {cap}")
    else:
        print("\n⚠️  No capabilities indexed yet.")


def interactive_mode(pipeline: IndexingPipeline, 
                     provider: 'LLMProvider' = None,
                     min_score: float = 0.10,
                     verbose: bool = False,
                     knowledge_graph: Optional[KnowledgeGraph] = None,
                     vocabulary: list = None,
                     full_file: bool = False):
    """
    Run interactive search mode with optional knowledge graph support.
    
    Args:
        pipeline: The indexing pipeline
        provider: Optional LLM provider for analysis
        min_score: Minimum score for LLM analysis
        verbose: Verbose output mode
        knowledge_graph: Optional knowledge graph for query expansion/boosting
        vocabulary: Vocabulary list for query decomposition
        full_file: Send full file content to LLM instead of chunks
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE SEARCH MODE")
    print("=" * 60)
    
    llm_status = "✓ enabled" if provider else "✗ disabled"
    kg_status = "✓ enabled" if knowledge_graph else "✗ disabled"
    
    # Knowledge graph feature toggles
    expand_query = True if knowledge_graph else False
    tfidf_boost = True if knowledge_graph else False
    show_related = True if knowledge_graph else False
    decompose_queries = True if (vocabulary and knowledge_graph) else False
    
    print(f"""
Commands:
  <query>           Search for text
  :analyze <query>  Search and analyze with LLM ({llm_status})
  :cap <capability> Search by business capability
  :caps             List all capabilities
  :code <query>     Search only in code
  :doc <query>      Search only in documents
  :top <n>          Set number of results (default: 5)
  :minscore <n>     Set min score for LLM (default: {min_score})
  :verbose          Toggle verbose output

Knowledge Graph ({kg_status}):
  :expand           Toggle query expansion (current: {'ON' if expand_query else 'OFF'})
  :boost            Toggle TF-IDF boosting (current: {'ON' if tfidf_boost else 'OFF'})
  :related          Toggle related terms display (current: {'ON' if show_related else 'OFF'})
  :decompose        Toggle query decomposition for long queries (current: {'ON' if decompose_queries else 'OFF'})
  :fullfile         Toggle full file mode for LLM (current: {'ON' if full_file else 'OFF'})
  :lookup <term>    Look up term in knowledge graph
  :graph            Show knowledge graph stats
  
  :stats            Show index statistics
  :help             Show this help
  :quit             Exit

Examples:
  OFAC sanctions
  :analyze wire transfer processing
  :cap Payment Processing
  :code wire transfer
  :lookup UETR
""")
    
    top_k = 5
    
    while True:
        try:
            query = input("\n🔍 Search> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() in [":quit", ":exit", ":q"]:
            print("Goodbye!")
            break
        
        elif query.lower() == ":help":
            kg_info = ""
            if knowledge_graph:
                kg_info = f"""
Knowledge Graph Commands:
  :expand           Toggle query expansion (current: {'ON' if expand_query else 'OFF'})
  :boost            Toggle TF-IDF boosting (current: {'ON' if tfidf_boost else 'OFF'})
  :related          Toggle related terms display (current: {'ON' if show_related else 'OFF'})
  :lookup <term>    Look up term in knowledge graph
  :graph            Show knowledge graph stats
"""
            print(f"""
Commands:
  <query>           Search for text
  :analyze <query>  Search and analyze with LLM
  :cap <capability> Search by business capability
  :caps             List all capabilities  
  :code <query>     Search only in code
  :doc <query>      Search only in documents
  :top <n>          Set number of results
  :minscore <n>     Set min score for LLM analysis (current: {min_score})
  :verbose          Toggle verbose output
  :stats            Show index statistics
  :quit             Exit
{kg_info}""")
        
        elif query.lower() == ":verbose":
            verbose = not verbose
            print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
        
        elif query.lower() == ":expand":
            if knowledge_graph:
                expand_query = not expand_query
                print(f"Query expansion: {'ON' if expand_query else 'OFF'}")
            else:
                print("❌ Knowledge graph not loaded. Use --knowledge-graph option.")
        
        elif query.lower() == ":boost":
            if knowledge_graph:
                tfidf_boost = not tfidf_boost
                print(f"TF-IDF boosting: {'ON' if tfidf_boost else 'OFF'}")
            else:
                print("❌ Knowledge graph not loaded. Use --knowledge-graph option.")
        
        elif query.lower() == ":related":
            if knowledge_graph:
                show_related = not show_related
                print(f"Related terms display: {'ON' if show_related else 'OFF'}")
            else:
                print("❌ Knowledge graph not loaded. Use --knowledge-graph option.")
        
        elif query.lower() == ":decompose":
            if knowledge_graph and vocabulary:
                decompose_queries = not decompose_queries
                print(f"Query decomposition: {'ON' if decompose_queries else 'OFF'}")
                if decompose_queries:
                    print("   Long queries will be split into focused sub-queries")
            else:
                print("❌ Requires knowledge graph and vocabulary. Use --knowledge-graph option.")
        
        elif query.lower() == ":fullfile":
            full_file = not full_file
            print(f"Full file mode: {'ON' if full_file else 'OFF'}")
            if full_file:
                print("   LLM will receive full file content (up to 50KB per file)")
            else:
                print("   LLM will receive only relevant chunks")
        
        elif query.lower().startswith(":lookup "):
            term = query[8:].strip()
            if knowledge_graph:
                context = knowledge_graph.get_term_context(term)
                if context:
                    print(f"\n📖 Term: {context.get('label', term)}")
                    print(f"   Type: {context.get('type', 'unknown')}")
                    print(f"   Source: {context.get('source', 'unknown')}")
                    print(f"   TF-IDF: {context.get('tf_idf_score', 0):.3f}")
                    print(f"   Term Frequency: {context.get('term_frequency', 0)}")
                    print(f"   Document Frequency: {context.get('document_frequency', 0)}")
                    
                    co_occurs = context.get('co_occurs_with', [])
                    if co_occurs:
                        print(f"   Co-occurs with: {', '.join(co_occurs[:10])}")
                    
                    related = knowledge_graph.get_related_terms(term, max_terms=10)
                    if related:
                        print(f"\n   Related terms:")
                        for rt, rtype, weight in related:
                            print(f"      • {rt} ({rtype})")
                else:
                    print(f"❌ Term '{term}' not found in knowledge graph")
            else:
                print("❌ Knowledge graph not loaded. Use --knowledge-graph option.")
        
        elif query.lower() == ":graph":
            if knowledge_graph:
                print(f"\n📊 Knowledge Graph Statistics:")
                print(f"   Nodes: {len(knowledge_graph.nodes)}")
                print(f"   Edges: {len(knowledge_graph.edges)}")
                stats = knowledge_graph.statistics
                if stats:
                    print(f"   Cross-referenced: {stats.get('cross_referenced', 0)}")
                    print(f"   PDF only: {stats.get('pdf_only', 0)}")
                    print(f"   Code only: {stats.get('code_only', 0)}")
            else:
                print("❌ Knowledge graph not loaded. Use --knowledge-graph option.")
        
        elif query.lower() == ":caps":
            list_capabilities(pipeline)
        
        elif query.lower() == ":stats":
            stats = pipeline.get_statistics()
            print("\n📊 Index Statistics:")
            print(f"   Total chunks: {stats['pipeline']['total_chunks']}")
            print(f"   By type:")
            for t, count in stats['pipeline'].get('by_source_type', {}).items():
                print(f"      {t}: {count}")
            print(f"   Vocabulary entries: {stats['vocabulary'].get('total_entries', 0)}")
        
        elif query.lower().startswith(":top "):
            try:
                top_k = int(query[5:].strip())
                print(f"Results per query set to: {top_k}")
            except ValueError:
                print("Invalid number")
        
        elif query.lower().startswith(":minscore "):
            try:
                min_score = float(query[10:].strip())
                print(f"Min score for LLM analysis set to: {min_score}")
            except ValueError:
                print("Invalid number")
        
        elif query.lower().startswith(":analyze "):
            q = query[9:].strip()
            if not provider:
                print("❌ LLM analysis not available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
            else:
                if decompose_queries and vocabulary:
                    # Use decomposed search for LLM - sends results from each sub-query
                    search_and_analyze_decomposed(pipeline, q, vocabulary, provider, 
                                                  top_k=20, min_score=min_score, 
                                                  verbose=verbose,
                                                  knowledge_graph=knowledge_graph,
                                                  expand_query=expand_query,
                                                  tfidf_boost=tfidf_boost,
                                                  full_file=full_file)
                else:
                    search_and_analyze(pipeline, q, provider, top_k=20, 
                                       min_score=min_score, verbose=verbose,
                                       knowledge_graph=knowledge_graph,
                                       expand_query=expand_query,
                                       tfidf_boost=tfidf_boost,
                                       full_file=full_file)
        
        elif query.lower().startswith(":cap "):
            capability = query[5:].strip()
            search_by_capability(pipeline, capability, top_k, verbose)
        
        elif query.lower().startswith(":code "):
            q = query[6:].strip()
            if decompose_queries and vocabulary:
                search_decomposed(pipeline, q, vocabulary, top_k, "code", verbose,
                                 knowledge_graph=knowledge_graph,
                                 expand_query=expand_query,
                                 tfidf_boost=tfidf_boost,
                                 show_related=show_related)
            else:
                search_once(pipeline, q, top_k, "code", verbose,
                           knowledge_graph=knowledge_graph,
                           expand_query=expand_query,
                           tfidf_boost=tfidf_boost,
                           show_related=show_related)
        
        elif query.lower().startswith(":doc "):
            q = query[5:].strip()
            if decompose_queries and vocabulary:
                search_decomposed(pipeline, q, vocabulary, top_k, "document", verbose,
                                 knowledge_graph=knowledge_graph,
                                 expand_query=expand_query,
                                 tfidf_boost=tfidf_boost,
                                 show_related=show_related)
            else:
                search_once(pipeline, q, top_k, "document", verbose,
                           knowledge_graph=knowledge_graph,
                           expand_query=expand_query,
                           tfidf_boost=tfidf_boost,
                           show_related=show_related)
        
        elif query.startswith(":"):
            print(f"Unknown command: {query}. Type :help for available commands.")
        
        else:
            if decompose_queries and vocabulary:
                search_decomposed(pipeline, query, vocabulary, top_k, "all", verbose,
                                 knowledge_graph=knowledge_graph,
                                 expand_query=expand_query,
                                 tfidf_boost=tfidf_boost,
                                 show_related=show_related)
            else:
                search_once(pipeline, query, top_k, "all", verbose,
                           knowledge_graph=knowledge_graph,
                           expand_query=expand_query,
                           tfidf_boost=tfidf_boost,
                           show_related=show_related)


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search through indexed code and documents with knowledge graph support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic semantic search
  python search_index.py --index ./my_index --query "wire transfer validation"
  
  # With knowledge graph query expansion
  python search_index.py --index ./my_index --query "OFAC screening" \\
      --knowledge-graph ./knowledge_graph.json --expand-query
  
  # With TF-IDF boosting for distinctive terms
  python search_index.py --index ./my_index --query "UETR validation" \\
      --knowledge-graph ./knowledge_graph.json --tfidf-boost
  
  # Full knowledge graph support (expansion + boosting)
  python search_index.py --index ./my_index --query "sanctions" \\
      --knowledge-graph ./knowledge_graph.json --expand-query --tfidf-boost
  
  # With LLM analysis  
  python search_index.py --index ./my_index --query "implement OFAC screening" \\
      --analyze --knowledge-graph ./knowledge_graph.json
  
  # Interactive mode with knowledge graph
  python search_index.py --index ./my_index --interactive \\
      --knowledge-graph ./knowledge_graph.json

Knowledge Graph Features:
  --expand-query    Expands "OFAC" to include "sanctions", "SDN", "screening" etc.
  --tfidf-boost     Boosts results matching distinctive terms (high TF-IDF)
  
Interactive commands for knowledge graph:
  :expand           Toggle query expansion
  :boost            Toggle TF-IDF boosting
  :lookup <term>    Look up term details and relationships
  :graph            Show knowledge graph statistics
        """
    )
    
    parser.add_argument("--index", "-i", type=str, required=True, 
                        help="Directory containing the saved index")
    parser.add_argument("--query", "-q", type=str, 
                        help="Search query string")
    parser.add_argument("--top", "-n", type=int, default=5,
                        help="Number of results to return (default: 5)")
    parser.add_argument("--type", "-t", type=str, default="all",
                        choices=["code", "document", "log", "all"],
                        help="Filter by source type (default: all)")
    parser.add_argument("--interactive", "-I", action="store_true",
                        help="Start interactive search mode")
    parser.add_argument("--capability", "-c", type=str,
                        help="Search by business capability")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show more details in results")
    parser.add_argument("--vocab", type=str, default=DEFAULT_KEYWORDS_FILE,
                        help="Path to vocabulary JSON file (default: keywords.json)")
    
    # LLM options
    parser.add_argument("--analyze", "-a", action="store_true",
                        help="Send results to LLM for implementation guidance")
    parser.add_argument("--provider", "-p", type=str, default="tachyon",
                        choices=["tachyon", "anthropic", "openai", "ollama", "internal", "stub"],
                        help="LLM provider (default: tachyon)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="LLM model name (provider-specific)")
    parser.add_argument("--min-score", type=float, default=0.10,
                        help="Minimum score for LLM analysis (default: 0.10)")
    
    # Knowledge graph options
    parser.add_argument("--knowledge-graph", "-kg", type=str, default=None,
                        help="Path to knowledge_graph.json for query expansion and TF-IDF boosting")
    parser.add_argument("--expand-query", "-e", action="store_true",
                        help="Expand query using related terms from knowledge graph")
    parser.add_argument("--tfidf-boost", "-b", action="store_true",
                        help="Boost scores based on TF-IDF (distinctive terms)")
    parser.add_argument("--decompose", "-d", action="store_true",
                        help="Decompose long queries into focused sub-queries")
    parser.add_argument("--no-related", action="store_true",
                        help="Don't show related terms in results")
    parser.add_argument("--no-kg", action="store_true",
                        help="Don't auto-load embedded knowledge graph from index")
    parser.add_argument("--full-file", "-f", action="store_true",
                        help="Send full file content to LLM (instead of just chunks)")
    
    args = parser.parse_args()
    
    # Validate
    if not args.interactive and not args.query and not args.capability:
        print("Error: Either --query, --capability, or --interactive is required")
        sys.exit(1)
    
    if not os.path.exists(args.index):
        print(f"Error: Index directory not found: {args.index}")
        sys.exit(1)
    
    print("=" * 60)
    print("UNIFIED INDEXER - SEARCH")
    print("=" * 60)
    
    # Load vocabulary
    print(f"\nLoading vocabulary from: {args.vocab}")
    vocab_data = load_vocabulary(args.vocab)
    print(f"Vocabulary entries: {len(vocab_data)}")
    
    # Create pipeline and load index
    print(f"Loading index from: {args.index}")
    pipeline = IndexingPipeline(
        vocabulary_data=vocab_data,
        embedder_type=None  # Will be restored from saved index
    )
    pipeline.load(args.index)
    
    stats = pipeline.get_statistics()
    total_chunks = stats['pipeline']['total_chunks']
    print(f"Index loaded: {total_chunks} chunks")
    
    # Load knowledge graph
    # Priority: 1) explicit --knowledge-graph flag, 2) embedded in index directory
    # Skip if --no-kg flag is set
    knowledge_graph = None
    kg_source = None
    
    if not args.no_kg:
        if args.knowledge_graph:
            # Explicit path provided
            knowledge_graph = load_knowledge_graph(args.knowledge_graph)
            kg_source = args.knowledge_graph
        else:
            # Check for embedded knowledge graph in index directory
            embedded_kg_path = os.path.join(args.index, "knowledge_graph.json")
            if os.path.exists(embedded_kg_path):
                knowledge_graph = load_knowledge_graph(embedded_kg_path)
                kg_source = "embedded in index"
        
        if knowledge_graph:
            if kg_source == "embedded in index":
                print(f"📊 Knowledge graph: auto-loaded from index")
            
            if args.expand_query or args.tfidf_boost:
                features = []
                if args.expand_query:
                    features.append("query expansion")
                if args.tfidf_boost:
                    features.append("TF-IDF boost")
                print(f"   Features enabled: {', '.join(features)}")
    
    # Determine show_related setting
    show_related = not args.no_related
    
    # Setup LLM provider if needed
    llm_provider = None
    if args.analyze or args.interactive:
        if LLM_AVAILABLE:
            try:
                llm_provider = create_provider(args.provider, args.model)
                print(f"LLM Provider: {args.provider} ({llm_provider.model})")
            except Exception as e:
                print(f"⚠️  LLM setup failed: {e}")
                if args.analyze:
                    print("   Continuing without LLM analysis...")
        else:
            print("⚠️  LLM provider module not available")
    
    # Run search
    if args.interactive:
        interactive_mode(pipeline, llm_provider, args.min_score, args.verbose, 
                        knowledge_graph, vocabulary=vocab_data, full_file=args.full_file)
    elif args.analyze and args.query:
        if llm_provider:
            if args.decompose and knowledge_graph:
                # Use decomposed search - sends results from each sub-query to LLM
                search_and_analyze_decomposed(
                    pipeline, args.query, vocab_data, llm_provider,
                    top_k=20,
                    source_type=args.type,
                    min_score=args.min_score,
                    verbose=args.verbose,
                    knowledge_graph=knowledge_graph,
                    expand_query=args.expand_query,
                    tfidf_boost=args.tfidf_boost,
                    full_file=args.full_file
                )
            else:
                search_and_analyze(
                    pipeline, args.query, llm_provider,
                    top_k=20,
                    source_type=args.type,
                    min_score=args.min_score,
                    verbose=args.verbose,
                    knowledge_graph=knowledge_graph,
                    expand_query=args.expand_query,
                    tfidf_boost=args.tfidf_boost,
                    full_file=args.full_file
                )
        else:
            print("❌ LLM analysis requires a valid provider. Check API keys.")
            search_once(pipeline, args.query, args.top, args.type, args.verbose,
                       knowledge_graph=knowledge_graph,
                       expand_query=args.expand_query,
                       tfidf_boost=args.tfidf_boost,
                       show_related=show_related)
    elif args.capability:
        search_by_capability(pipeline, args.capability, args.top, args.verbose)
    else:
        # Use decomposition for long queries if enabled
        if args.decompose and knowledge_graph:
            search_decomposed(pipeline, args.query, vocab_data, args.top, args.type, 
                             args.verbose,
                             knowledge_graph=knowledge_graph,
                             expand_query=args.expand_query,
                             tfidf_boost=args.tfidf_boost,
                             show_related=show_related)
        else:
            search_once(pipeline, args.query, args.top, args.type, args.verbose,
                       knowledge_graph=knowledge_graph,
                       expand_query=args.expand_query,
                       tfidf_boost=args.tfidf_boost,
                       show_related=show_related)


if __name__ == "__main__":
    main()
