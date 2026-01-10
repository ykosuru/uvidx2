#!/usr/bin/env python3
"""
Enhanced API Server with optimal search for LLM context
"""

from flask import Flask, request, jsonify
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_indexer.pipeline import IndexingPipeline
from unified_indexer.models import SourceType
from search_index import (
    KnowledgeGraph, 
    load_knowledge_graph,
    apply_tfidf_boost,
    decompose_query,
    load_vocabulary
)
from unified_indexer.index import reciprocal_rank_fusion

app = Flask(__name__)

# Global state
pipeline: Optional[IndexingPipeline] = None
knowledge_graph: Optional[KnowledgeGraph] = None
vocabulary: Optional[list] = None

DEFAULT_KEYWORDS_FILE = os.path.join(os.path.dirname(__file__), "keywords.json")


def init_index():
    """Initialize index on startup"""
    global pipeline, knowledge_graph, vocabulary
    
    index_path = os.environ.get('INDEX_PATH', './index')
    vocab_path = os.environ.get('VOCAB_PATH', DEFAULT_KEYWORDS_FILE)
    
    print(f"Loading index from: {index_path}")
    print(f"Loading vocabulary from: {vocab_path}")
    
    # Load vocabulary
    vocabulary = load_vocabulary(vocab_path)
    
    # Create and load pipeline
    pipeline = IndexingPipeline(
        vocabulary_data=vocabulary,
        embedder_type=None
    )
    pipeline.load(index_path)
    
    # Auto-load knowledge graph from index
    kg_path = os.path.join(index_path, "knowledge_graph.json")
    if os.path.exists(kg_path):
        knowledge_graph = load_knowledge_graph(kg_path)
        print(f"Knowledge graph loaded: {len(knowledge_graph.nodes)} nodes")
    
    stats = pipeline.get_statistics()
    print(f"Index loaded: {stats['pipeline']['total_chunks']} chunks")


@dataclass
class SearchConfig:
    """Search configuration with sensible defaults for LLM context"""
    top_k: int = 10
    expand_query: bool = True      # Enable by default
    tfidf_boost: bool = True       # Enable by default
    decompose: bool = True         # Enable for long queries
    balance_sources: bool = True   # Balance docs and code
    min_score: float = 0.3         # Filter low-quality results
    doc_ratio: float = 0.4         # 40% docs, 60% code (adjustable)
    domains: Optional[List[str]] = None


def balanced_search(query: str, config: SearchConfig) -> List[Dict]:
    """
    Search with source type balancing for optimal LLM context.
    
    Returns a mix of documentation and code results.
    """
    results_by_type = {}
    
    # Determine how many of each type to fetch
    if config.balance_sources:
        doc_count = max(2, int(config.top_k * config.doc_ratio))
        code_count = config.top_k - doc_count
    else:
        doc_count = config.top_k
        code_count = config.top_k
    
    # Prepare search query (with expansion if enabled)
    search_query = query
    if knowledge_graph and config.expand_query:
        expanded = knowledge_graph.expand_query(query)
        if expanded != query.lower():
            search_query = expanded
    
    # Search documents
    doc_results = pipeline.search(
        search_query, 
        top_k=doc_count * 2,  # Fetch extra for filtering
        source_types=[SourceType.DOCUMENT],
        domains=config.domains
    )
    
    # Search code
    code_results = pipeline.search(
        search_query,
        top_k=code_count * 2,
        source_types=[SourceType.CODE],
        domains=config.domains
    )
    
    # Apply TF-IDF boosting
    if knowledge_graph and config.tfidf_boost:
        doc_results = apply_tfidf_boost(doc_results, knowledge_graph, query)
        code_results = apply_tfidf_boost(code_results, knowledge_graph, query)
    
    # Filter by minimum score and deduplicate by file
    def filter_and_dedup(results, max_count):
        seen_files = set()
        filtered = []
        for r in results:
            if r.combined_score < config.min_score:
                continue
            file_path = r.chunk.source_ref.file_path or r.chunk.chunk_id
            if file_path not in seen_files:
                seen_files.add(file_path)
                filtered.append(r)
                if len(filtered) >= max_count:
                    break
        return filtered
    
    doc_results = filter_and_dedup(doc_results, doc_count)
    code_results = filter_and_dedup(code_results, code_count)
    
    # Combine and format results
    all_results = []
    
    for r in doc_results:
        all_results.append(format_result(r, 'document'))
    
    for r in code_results:
        all_results.append(format_result(r, 'code'))
    
    # Sort by score
    all_results.sort(key=lambda x: -x['score'])
    
    # Assign ranks
    for i, r in enumerate(all_results):
        r['rank'] = i + 1
    
    return all_results[:config.top_k]


def decomposed_search(query: str, config: SearchConfig) -> List[Dict]:
    """
    Search with query decomposition for complex queries.
    
    Splits query into focused sub-queries, searches each, then fuses with RRF.
    """
    if not vocabulary or not knowledge_graph:
        return balanced_search(query, config)
    
    # Decompose query
    sub_queries = decompose_query(query, vocabulary, knowledge_graph)
    
    if len(sub_queries) <= 1:
        return balanced_search(query, config)
    
    # Search each sub-query
    all_results_lists = []
    chunk_lookup = {}
    
    for sq in sub_queries:
        # Expand sub-query
        search_sq = sq
        if config.expand_query:
            search_sq = knowledge_graph.expand_query(sq)
        
        # Search both types
        results = pipeline.search(search_sq, top_k=config.top_k * 2)
        
        if config.tfidf_boost:
            results = apply_tfidf_boost(results, knowledge_graph, sq)
        
        # Collect for RRF
        result_tuples = []
        for r in results:
            chunk_id = r.chunk.chunk_id
            result_tuples.append((chunk_id, r.combined_score))
            if chunk_id not in chunk_lookup or r.combined_score > chunk_lookup[chunk_id].combined_score:
                chunk_lookup[chunk_id] = r
        
        all_results_lists.append(result_tuples)
    
    # RRF fusion
    if not all_results_lists or not any(all_results_lists):
        return balanced_search(query, config)
    
    fused_scores = reciprocal_rank_fusion(all_results_lists, k=60)
    
    # Normalize scores
    if fused_scores:
        max_score = max(fused_scores.values())
        min_score = min(fused_scores.values())
        score_range = max_score - min_score
        if score_range > 0:
            fused_scores = {
                cid: (score - min_score) / score_range
                for cid, score in fused_scores.items()
            }
    
    # Build results with source balancing
    doc_results = []
    code_results = []
    seen_files = set()
    
    for chunk_id, score in sorted(fused_scores.items(), key=lambda x: -x[1]):
        if chunk_id not in chunk_lookup:
            continue
        
        r = chunk_lookup[chunk_id]
        file_path = r.chunk.source_ref.file_path or chunk_id
        
        if file_path in seen_files:
            continue
        seen_files.add(file_path)
        
        if score < config.min_score:
            continue
        
        r.combined_score = score
        
        if r.chunk.source_type == SourceType.DOCUMENT:
            doc_results.append(r)
        else:
            code_results.append(r)
    
    # Balance results
    if config.balance_sources:
        doc_count = max(2, int(config.top_k * config.doc_ratio))
        code_count = config.top_k - doc_count
        doc_results = doc_results[:doc_count]
        code_results = code_results[:code_count]
    
    # Combine and format
    all_results = []
    for r in doc_results:
        all_results.append(format_result(r, 'document'))
    for r in code_results:
        all_results.append(format_result(r, 'code'))
    
    all_results.sort(key=lambda x: -x['score'])
    for i, r in enumerate(all_results):
        r['rank'] = i + 1
    
    return all_results[:config.top_k]


def format_result(result, source_type: str) -> Dict:
    """Format a search result for API response"""
    chunk = result.chunk
    source_ref = chunk.source_ref
    
    return {
        'score': round(result.combined_score, 4),
        'file_path': source_ref.file_path,
        'source_type': source_type,
        'procedure_name': source_ref.procedure_name,
        'line_start': source_ref.line_start,
        'line_end': source_ref.line_end,
        'page_number': source_ref.page_number,
        'content_preview': chunk.text[:1000],
        'concepts': result.matched_concepts[:10] if result.matched_concepts else [],
        'domain': chunk.metadata.get('domain', 'default'),
        'calls': chunk.metadata.get('calls', [])[:10],
    }


@app.route('/health', methods=['GET'])
def health():
    if pipeline is None:
        return jsonify({'status': 'error', 'message': 'Index not loaded'}), 503
    return jsonify({'status': 'healthy'})


@app.route('/search', methods=['POST'])
def search():
    """
    Enhanced search endpoint with all optimizations enabled by default.
    
    Request body:
    {
        "query": "wire transfer validation",
        "top_k": 10,
        "expand_query": true,      // default: true
        "tfidf_boost": true,       // default: true
        "decompose": true,         // default: true for long queries
        "balance_sources": true,   // default: true
        "min_score": 0.3,          // default: 0.3
        "doc_ratio": 0.4,          // default: 0.4 (40% docs, 60% code)
        "domains": ["payments"]    // optional domain filter
    }
    """
    if pipeline is None:
        return jsonify({'error': 'Index not loaded'}), 503
    
    data = request.json or {}
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    # Build config from request (with smart defaults)
    config = SearchConfig(
        top_k=data.get('top_k', 10),
        expand_query=data.get('expand_query', True),
        tfidf_boost=data.get('tfidf_boost', True),
        decompose=data.get('decompose', True),
        balance_sources=data.get('balance_sources', True),
        min_score=data.get('min_score', 0.3),
        doc_ratio=data.get('doc_ratio', 0.4),
        domains=data.get('domains')
    )
    
    # Use decomposed search for longer queries
    word_count = len(query.split())
    if config.decompose and word_count > 5 and knowledge_graph:
        results = decomposed_search(query, config)
    else:
        results = balanced_search(query, config)
    
    return jsonify({
        'query': query,
        'total_results': len(results),
        'config': {
            'expand_query': config.expand_query,
            'tfidf_boost': config.tfidf_boost,
            'decompose': config.decompose,
            'balance_sources': config.balance_sources,
        },
        'results': results
    })


@app.route('/domains', methods=['GET'])
def list_domains():
    """List available domains"""
    if pipeline is None:
        return jsonify({'error': 'Index not loaded'}), 503
    
    domains = pipeline.get_domains()
    return jsonify({
        'domains': [
            {'name': name, 'chunk_count': count, 'percentage': round(count / sum(domains.values()) * 100, 1)}
            for name, count in sorted(domains.items())
        ]
    })


@app.route('/context', methods=['POST'])
def get_context():
    """
    Get comprehensive context for a task - optimized for LLM consumption.
    
    Returns structured context with:
    - Documentation (business rules, message formats)
    - Code (implementations, procedures)
    - Related concepts
    
    Request body:
    {
        "query": "implement OFAC screening",
        "max_docs": 3,
        "max_code": 5,
        "include_calls": true
    }
    """
    if pipeline is None:
        return jsonify({'error': 'Index not loaded'}), 503
    
    data = request.json or {}
    query = data.get('query', '')
    max_docs = data.get('max_docs', 3)
    max_code = data.get('max_code', 5)
    include_calls = data.get('include_calls', True)
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    # Search with expansion
    search_query = query
    expanded_terms = []
    if knowledge_graph:
        expanded = knowledge_graph.expand_query(query)
        if expanded != query.lower():
            search_query = expanded
            expanded_terms = expanded.split()
    
    # Get documents
    doc_results = pipeline.search(
        search_query, 
        top_k=max_docs * 2,
        source_types=[SourceType.DOCUMENT]
    )
    if knowledge_graph:
        doc_results = apply_tfidf_boost(doc_results, knowledge_graph, query)
    
    # Dedupe docs by file
    seen_files = set()
    docs = []
    for r in doc_results:
        fp = r.chunk.source_ref.file_path
        if fp not in seen_files:
            seen_files.add(fp)
            docs.append({
                'file': fp,
                'page': r.chunk.source_ref.page_number,
                'score': round(r.combined_score, 3),
                'content': r.chunk.text[:2000]
            })
            if len(docs) >= max_docs:
                break
    
    # Get code
    code_results = pipeline.search(
        search_query,
        top_k=max_code * 2,
        source_types=[SourceType.CODE]
    )
    if knowledge_graph:
        code_results = apply_tfidf_boost(code_results, knowledge_graph, query)
    
    # Dedupe code by file
    seen_files = set()
    code = []
    all_calls = set()
    for r in code_results:
        fp = r.chunk.source_ref.file_path
        if fp not in seen_files:
            seen_files.add(fp)
            calls = r.chunk.metadata.get('calls', [])
            all_calls.update(calls)
            code.append({
                'file': fp,
                'procedure': r.chunk.source_ref.procedure_name,
                'lines': f"{r.chunk.source_ref.line_start}-{r.chunk.source_ref.line_end}",
                'score': round(r.combined_score, 3),
                'calls': calls[:10],
                'content': r.chunk.text[:2000]
            })
            if len(code) >= max_code:
                break
    
    # Get related procedures via call graph
    related_procs = []
    if include_calls and all_calls:
        for call_name in list(all_calls)[:10]:
            call_results = pipeline.search(call_name, top_k=1, source_types=[SourceType.CODE])
            if call_results:
                r = call_results[0]
                if r.chunk.source_ref.procedure_name and r.chunk.source_ref.procedure_name not in [c['procedure'] for c in code]:
                    related_procs.append({
                        'procedure': r.chunk.source_ref.procedure_name,
                        'file': r.chunk.source_ref.file_path,
                        'called_by': [c['procedure'] for c in code if call_name in c.get('calls', [])]
                    })
    
    return jsonify({
        'query': query,
        'expanded_terms': expanded_terms,
        'documentation': docs,
        'code': code,
        'related_procedures': related_procs[:5],
        'concepts': list(set(
            concept 
            for r in (doc_results[:3] + code_results[:5])
            for concept in (r.matched_concepts or [])
        ))[:15]
    })


if __name__ == '__main__':
    init_index()
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
