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
USAGE
================================================================================

    # Basic semantic search
    python search_index.py --index ./my_index --query "OFAC screening"
    
    # With knowledge graph query expansion
    python search_index.py --index ./my_index --query "OFAC screening" \\
        --knowledge-graph ./knowledge_graph.json --expand-query
    
    # Interactive mode with knowledge graph
    python search_index.py --index ./my_index --interactive \\
        --knowledge-graph ./knowledge_graph.json
    
    # With LLM analysis
    python search_index.py --index ./my_index --query "wire transfer" --analyze

================================================================================
KNOWLEDGE GRAPH FEATURES
================================================================================

Query Expansion (--expand-query):
    Original: "OFAC screening"
    Expanded: "OFAC screening sanctions SDN compliance screen_ofac"
    
    Uses relationships from knowledge graph:
    - co_occurs_with: Terms that appear together in documents
    - implements: Procedures that implement concepts
    - contains: Structure fields
    - related_to: Semantically related terms

TF-IDF Boosting (--tfidf-boost):
    Boosts scores for matches on distinctive terms (high TF-IDF score).
    Terms that appear in few documents are considered more important.

Related Terms Display:
    Shows related concepts in search results when knowledge graph is loaded.

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
    --knowledge-graph Path to knowledge_graph.json from knowledge_extractor
    --expand-query    Expand query using related terms from knowledge graph
    --tfidf-boost     Boost scores based on TF-IDF (distinctive terms)
    --show-related    Show related terms in results (default: on with KG)
    
    LLM Options:
    --analyze         Send results to LLM for analysis
    --provider        LLM provider (default: tachyon)
    --model           LLM model name
    --min-score       Minimum score for LLM analysis (default: 0.50)
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_indexer import IndexingPipeline, SourceType

# Import LLM provider
try:
    from llm_provider import (
        LLMProvider,
        create_provider,
        analyze_search_results,
        format_search_results_for_llm,
        WIRE_PAYMENTS_SYSTEM_PROMPT,
        ContentType
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Default keywords file location
DEFAULT_KEYWORDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keywords.json")


# =============================================================================
# KNOWLEDGE GRAPH SUPPORT
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
    """
    
    def __init__(self, graph_path: str):
        """
        Load knowledge graph from JSON file.
        
        Args:
            graph_path: Path to knowledge_graph.json
        """
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Dict] = []
        self.statistics: Dict = {}
        
        # Index structures for fast lookup
        self._outgoing_edges: Dict[str, List[Dict]] = defaultdict(list)  # term -> edges from term
        self._incoming_edges: Dict[str, List[Dict]] = defaultdict(list)  # term -> edges to term
        self._label_to_id: Dict[str, str] = {}  # original label -> normalized id
        
        self._load(graph_path)
    
    def _load(self, graph_path: str):
        """Load and index the knowledge graph."""
        with open(graph_path, 'r') as f:
            data = json.load(f)
        
        # Load nodes (keyed by normalized id)
        for node in data.get('nodes', []):
            node_id = node['id']
            self.nodes[node_id] = node
            # Also index by original label for flexible lookup
            if 'label' in node:
                self._label_to_id[node['label'].lower()] = node_id
        
        # Load and index edges
        self.edges = data.get('edges', [])
        for edge in self.edges:
            source = edge['source']
            target = edge['target']
            self._outgoing_edges[source].append(edge)
            self._incoming_edges[target].append(edge)
        
        self.statistics = data.get('statistics', {})
    
    def normalize(self, term: str) -> str:
        """Normalize a term to match node IDs."""
        import re
        normalized = term.lower()
        normalized = re.sub(r'[-\s]+', '_', normalized)
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        return normalized
    
    def get_node(self, term: str) -> Optional[Dict]:
        """
        Get node data for a term.
        
        Args:
            term: Term to look up (original or normalized form)
            
        Returns:
            Node dict with id, label, type, tf_idf_score, etc. or None
        """
        # Try normalized form
        normalized = self.normalize(term)
        if normalized in self.nodes:
            return self.nodes[normalized]
        
        # Try label lookup
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
        
        Args:
            term: Term to find relationships for
            relationship_types: Filter by relationship type (None = all types)
                              Options: co_occurs_with, implements, contains, related_to
            max_depth: How many hops to follow (1 = direct relationships only)
            max_terms: Maximum number of related terms to return
            
        Returns:
            List of (related_term, relationship_type, weight) tuples
            Weight is based on co-occurrence count or 1.0 for structural relationships
        """
        normalized = self.normalize(term)
        related = []
        seen = {normalized}
        
        # BFS to find related terms
        current_level = [normalized]
        
        for depth in range(max_depth):
            next_level = []
            
            for current in current_level:
                # Check outgoing edges
                for edge in self._outgoing_edges.get(current, []):
                    if relationship_types and edge['type'] not in relationship_types:
                        continue
                    
                    target = edge['target']
                    if target not in seen:
                        seen.add(target)
                        
                        # Calculate weight (co-occurrence count or 1.0)
                        weight = 1.0
                        if 'Co-occurred in' in edge.get('evidence', ''):
                            try:
                                count = int(edge['evidence'].split()[2])
                                weight = count
                            except:
                                pass
                        
                        node = self.nodes.get(target, {})
                        label = node.get('label', target)
                        related.append((label, edge['type'], weight))
                        next_level.append(target)
                
                # Check incoming edges (reverse relationships)
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
# =============================================================================


def load_vocabulary(vocab_path: str) -> list:
    """Load vocabulary from JSON file"""
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found: {vocab_path}")
        print(f"Please ensure 'keywords.json' exists in the same directory as this script,")
        print(f"or specify a custom vocabulary file with --vocab")
        sys.exit(1)
    
    with open(vocab_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get('entries', [data])
    else:
        print(f"Error: Invalid vocabulary format in {vocab_path}")
        sys.exit(1)


def print_result(result, index: int, verbose: bool = False, 
                 knowledge_graph: Optional[KnowledgeGraph] = None,
                 show_related: bool = True):
    """
    Print a single search result with optional knowledge graph context.
    
    Args:
        result: Search result object
        index: Result index for numbering
        verbose: Show detailed scores and metadata
        knowledge_graph: Optional KG for showing related terms
        show_related: Whether to show related terms from KG
    """
    chunk = result.chunk
    
    print(f"\n{'─' * 60}")
    score_info = f"Score: {result.combined_score:.3f}"
    if verbose:
        score_info += f" (v:{result.vector_score:.3f} c:{result.concept_score:.3f} k:{result.keyword_score:.3f})"
    print(f"Result #{index + 1}  |  {score_info}  |  Type: {chunk.source_type.value.upper()}")
    if verbose:
        print(f"Method: {result.retrieval_method}")
    print(f"{'─' * 60}")
    
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
    
    Args:
        pipeline: The indexing pipeline
        query: Search query
        top_k: Number of results
        source_type: Filter by type (all, code, document, log)
        verbose: Show full content
        knowledge_graph: Optional KnowledgeGraph for expansion/boosting
        expand_query: Expand query using related terms from KG
        tfidf_boost: Boost scores based on TF-IDF scores
        show_related: Show related terms in results
        
    Returns:
        List of search results
    """
    
    source_types = None
    if source_type == "code":
        source_types = [SourceType.CODE]
    elif source_type == "document":
        source_types = [SourceType.DOCUMENT]
    elif source_type == "log":
        source_types = [SourceType.LOG]
    
    # Query expansion using knowledge graph
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
    
    # Perform search (request more results if we're going to re-rank)
    fetch_k = top_k * 2 if tfidf_boost and knowledge_graph else top_k
    results = pipeline.search(search_query, top_k=fetch_k, source_types=source_types)
    
    if not results:
        print("\n⚠️  No results found.")
        return results
    
    # TF-IDF boosting: re-rank results based on TF-IDF scores
    if knowledge_graph and tfidf_boost and results:
        results = apply_tfidf_boost(results, knowledge_graph, query)
        results = results[:top_k]  # Trim to requested count
    
    # Print results
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
    
    Args:
        results: List of search results
        knowledge_graph: KnowledgeGraph with TF-IDF scores
        original_query: Original query for term extraction
        
    Returns:
        Re-ranked list of results
    """
    # Extract query terms
    query_terms = original_query.lower().split()
    
    # Get TF-IDF scores for query terms
    term_scores = {}
    for term in query_terms:
        score = knowledge_graph.get_tfidf_score(term)
        if score > 0:
            term_scores[term] = score
    
    if not term_scores:
        return results  # No TF-IDF data, return unchanged
    
    # Calculate boost for each result
    boosted_results = []
    max_tfidf = max(term_scores.values()) if term_scores else 1.0
    
    for result in results:
        boost = 0.0
        chunk = result.chunk
        
        # Check matched concepts
        for concept in (result.matched_concepts or []):
            concept_lower = concept.lower()
            for term, tfidf in term_scores.items():
                if term in concept_lower:
                    # Normalize boost to 0-0.2 range
                    boost += 0.2 * (tfidf / max_tfidf)
        
        # Check content text
        text_lower = chunk.text.lower()
        for term, tfidf in term_scores.items():
            if term in text_lower:
                # Smaller boost for text matches
                boost += 0.1 * (tfidf / max_tfidf)
        
        # Check procedure name
        if chunk.source_ref.procedure_name:
            proc_lower = chunk.source_ref.procedure_name.lower()
            for term, tfidf in term_scores.items():
                if term in proc_lower:
                    boost += 0.15 * (tfidf / max_tfidf)
        
        # Apply boost (cap at 0.3 total boost)
        boost = min(boost, 0.3)
        
        # Create new result with boosted score
        # We store the boost in a simple way by modifying combined_score
        result.combined_score = result.combined_score + boost
        boosted_results.append(result)
    
    # Re-sort by boosted score
    boosted_results.sort(key=lambda r: -r.combined_score)
    
    return boosted_results


def search_and_analyze(pipeline: IndexingPipeline,
                       query: str,
                       provider: 'LLMProvider',
                       top_k: int = 20,
                       source_type: str = "all",
                       min_score: float = 0.50,
                       verbose: bool = False,
                       knowledge_graph: Optional[KnowledgeGraph] = None,
                       expand_query: bool = False,
                       tfidf_boost: bool = False):
    """
    Search and analyze with LLM, with optional knowledge graph enhancement.
    
    The LLM receives relevant code/documentation for analysis.
    Knowledge graph can expand the query and boost distinctive terms.
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
    
    response = analyze_search_results(
        query=query,
        results=results,
        provider=provider,
        min_score=min_score,
        max_chunks=20,
        verbose=verbose
    )
    
    print_llm_analysis(response, verbose)
    
    return results, response



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
                     min_score: float = 0.50,
                     verbose: bool = False,
                     knowledge_graph: Optional[KnowledgeGraph] = None):
    """
    Run interactive search mode with optional knowledge graph support.
    
    Args:
        pipeline: The indexing pipeline
        provider: Optional LLM provider for analysis
        min_score: Minimum score for LLM analysis
        verbose: Verbose output mode
        knowledge_graph: Optional knowledge graph for query expansion/boosting
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
                search_and_analyze(pipeline, q, provider, top_k=20, 
                                   min_score=min_score, verbose=verbose,
                                   knowledge_graph=knowledge_graph,
                                   expand_query=expand_query,
                                   tfidf_boost=tfidf_boost)
        
        elif query.lower().startswith(":cap "):
            capability = query[5:].strip()
            search_by_capability(pipeline, capability, top_k, verbose)
        
        elif query.lower().startswith(":code "):
            q = query[6:].strip()
            search_once(pipeline, q, top_k, "code", verbose,
                       knowledge_graph=knowledge_graph,
                       expand_query=expand_query,
                       tfidf_boost=tfidf_boost,
                       show_related=show_related)
        
        elif query.lower().startswith(":doc "):
            q = query[5:].strip()
            search_once(pipeline, q, top_k, "document", verbose,
                       knowledge_graph=knowledge_graph,
                       expand_query=expand_query,
                       tfidf_boost=tfidf_boost,
                       show_related=show_related)
        
        elif query.startswith(":"):
            print(f"Unknown command: {query}. Type :help for available commands.")
        
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
    parser.add_argument("--min-score", type=float, default=0.50,
                        help="Minimum score for LLM analysis (default: 0.50)")
    
    # Knowledge graph options
    parser.add_argument("--knowledge-graph", "-kg", type=str, default=None,
                        help="Path to knowledge_graph.json for query expansion and TF-IDF boosting")
    parser.add_argument("--expand-query", "-e", action="store_true",
                        help="Expand query using related terms from knowledge graph")
    parser.add_argument("--tfidf-boost", "-b", action="store_true",
                        help="Boost scores based on TF-IDF (distinctive terms)")
    parser.add_argument("--no-related", action="store_true",
                        help="Don't show related terms in results")
    
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
    
    # Load knowledge graph if provided
    knowledge_graph = None
    if args.knowledge_graph:
        knowledge_graph = load_knowledge_graph(args.knowledge_graph)
        if knowledge_graph and (args.expand_query or args.tfidf_boost):
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
                        knowledge_graph)
    elif args.analyze and args.query:
        if llm_provider:
            search_and_analyze(
                pipeline, args.query, llm_provider,
                top_k=20,
                source_type=args.type,
                min_score=args.min_score,
                verbose=args.verbose,
                knowledge_graph=knowledge_graph,
                expand_query=args.expand_query,
                tfidf_boost=args.tfidf_boost
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
        search_once(pipeline, args.query, args.top, args.type, args.verbose,
                   knowledge_graph=knowledge_graph,
                   expand_query=args.expand_query,
                   tfidf_boost=args.tfidf_boost,
                   show_related=show_related)


if __name__ == "__main__":
    main()
