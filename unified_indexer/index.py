"""
Hybrid Index - Combines vector similarity with concept-based retrieval

This index stores chunks in both a vector store (for semantic search)
and a concept index (for exact domain term matching). Search results
are fused using reciprocal rank fusion.

Concurrency Model:
- Uses generation-based versioning for lock-free reads
- Writes create new generation, atomic pointer update
- Readers always see consistent snapshot (old or new, never mixed)
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from collections import defaultdict
import numpy as np

from .models import (
    IndexableChunk,
    SearchResult,
    SourceType
)
from .vocabulary import DomainVocabulary


def get_current_generation(index_dir: Path) -> int:
    """Get the current generation number from index directory."""
    gen_file = index_dir / 'generation.txt'
    if gen_file.exists():
        try:
            return int(gen_file.read_text().strip())
        except (ValueError, IOError):
            pass
    return 0


def get_generation_path(index_dir: Path, generation: int) -> Path:
    """Get the path to a specific generation's data."""
    if generation == 0:
        # Legacy: no generation subdirectory
        return index_dir
    return index_dir / f'gen_{generation}'


class VectorStore:
    """
    Simple in-memory vector store for embeddings
    
    For production, replace with ChromaDB, Qdrant, Pinecone, etc.
    """
    
    def __init__(self):
        self.embeddings: Dict[str, np.ndarray] = {}
        self.chunks: Dict[str, IndexableChunk] = {}
    
    def add(self, chunk_id: str, embedding: List[float], chunk: IndexableChunk):
        """Add a chunk with its embedding"""
        self.embeddings[chunk_id] = np.array(embedding)
        self.chunks[chunk_id] = chunk
    
    def search(self, 
               query_embedding: List[float], 
               top_k: int = 10,
               filter_fn: Optional[Callable[[IndexableChunk], bool]] = None) -> List[Tuple[str, float]]:
        """
        Search for similar chunks
        
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        if not self.embeddings:
            return []
        
        query_vec = np.array(query_embedding)
        
        # Compute similarities
        results = []
        for chunk_id, embedding in self.embeddings.items():
            # Apply filter if provided
            if filter_fn and not filter_fn(self.chunks[chunk_id]):
                continue
            
            # Cosine similarity
            similarity = np.dot(query_vec, embedding) / (
                np.linalg.norm(query_vec) * np.linalg.norm(embedding) + 1e-8
            )
            results.append((chunk_id, float(similarity)))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_chunk(self, chunk_id: str) -> Optional[IndexableChunk]:
        """Get chunk by ID"""
        return self.chunks.get(chunk_id)
    
    def __len__(self):
        return len(self.chunks)


class ConceptIndex:
    """
    Index for exact domain concept matching
    
    Maps canonical terms and business capabilities to chunk IDs
    for fast exact-match retrieval.
    """
    
    def __init__(self):
        # concept (canonical term) -> set of chunk_ids
        self.concept_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        
        # capability -> set of chunk_ids  
        self.capability_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        
        # category -> set of chunk_ids
        self.category_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        
        # source_type -> set of chunk_ids
        self.source_type_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        
        # All chunks
        self.chunks: Dict[str, IndexableChunk] = {}
    
    def add(self, chunk: IndexableChunk):
        """Index a chunk by its domain concepts"""
        chunk_id = chunk.chunk_id
        self.chunks[chunk_id] = chunk
        
        # Index by source type
        self.source_type_to_chunks[chunk.source_type.value].add(chunk_id)
        
        # Index by domain matches
        for match in chunk.domain_matches:
            # By canonical term
            canonical_lower = match.canonical_term.lower()
            self.concept_to_chunks[canonical_lower].add(chunk_id)
            
            # By capability
            for capability in match.capabilities:
                self.capability_to_chunks[capability].add(chunk_id)
            
            # By category
            self.category_to_chunks[match.category].add(chunk_id)
    
    def search_concept(self, concept: str) -> Set[str]:
        """Find chunks containing a specific concept"""
        return self.concept_to_chunks.get(concept.lower(), set())
    
    def search_capability(self, capability: str) -> Set[str]:
        """Find chunks for a business capability"""
        return self.capability_to_chunks.get(capability, set())
    
    def search_category(self, category: str) -> Set[str]:
        """Find chunks in a metadata category"""
        return self.category_to_chunks.get(category, set())
    
    def search_source_type(self, source_type: str) -> Set[str]:
        """Find chunks by source type"""
        return self.source_type_to_chunks.get(source_type, set())
    
    def get_chunk(self, chunk_id: str) -> Optional[IndexableChunk]:
        """Get chunk by ID"""
        return self.chunks.get(chunk_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'total_chunks': len(self.chunks),
            'unique_concepts': len(self.concept_to_chunks),
            'unique_capabilities': len(self.capability_to_chunks),
            'unique_categories': len(self.category_to_chunks),
            'chunks_by_source': {
                st: len(chunks) 
                for st, chunks in self.source_type_to_chunks.items()
            }
        }
    
    def __len__(self):
        return len(self.chunks)


# =============================================================================
# BM25 INDEX
# =============================================================================

class BM25Index:
    """
    BM25 (Best Matching 25) index for lexical retrieval.
    
    BM25 is a bag-of-words retrieval function that ranks documents based on 
    query term frequencies. It's particularly effective for:
    - Exact term matching (acronyms, technical terms)
    - Queries where semantic similarity misses lexical matches
    - Complementing vector search in hybrid retrieval
    
    Algorithm:
        score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
        
        where:
        - f(qi, D) = frequency of term qi in document D
        - |D| = document length
        - avgdl = average document length
        - k1 = term frequency saturation parameter (default: 1.5)
        - b = length normalization parameter (default: 0.75)
        - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
    
    Attributes:
        k1: Term frequency saturation (higher = more weight to term frequency)
        b: Length normalization (0 = no normalization, 1 = full normalization)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        
        # Document storage
        self.chunk_ids: List[str] = []                    # Ordered list of chunk IDs
        self.chunks: Dict[str, IndexableChunk] = {}       # chunk_id -> chunk
        self.doc_tokens: List[List[str]] = []             # Tokenized documents
        self.doc_lengths: List[int] = []                  # Document lengths
        
        # Index structures (built on first search or explicit build)
        self.term_doc_freqs: Dict[str, int] = {}          # term -> document frequency
        self.inverted_index: Dict[str, Dict[int, int]] = {}  # term -> {doc_idx -> term_freq}
        self.avgdl: float = 0.0                           # Average document length
        self.N: int = 0                                   # Total number of documents
        
        # IDF cache
        self._idf_cache: Dict[str, float] = {}
        self._index_built = False
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Uses simple whitespace + punctuation splitting with lowercasing.
        More sophisticated tokenization (stemming, lemmatization) could improve results.
        """
        import re
        # Split on non-alphanumeric, keep underscores for technical terms
        tokens = re.split(r'[^\w]+', text.lower())
        # Filter empty and single-char tokens (except meaningful ones)
        return [t for t in tokens if len(t) >= 2 or t in ('a', 'i')]
    
    def add(self, chunk_id: str, chunk: IndexableChunk):
        """
        Add a chunk to the BM25 index.
        
        Note: Call build_index() after adding all chunks, or it will be
        built automatically on first search.
        
        Args:
            chunk_id: Unique identifier for the chunk
            chunk: The chunk to index
        """
        # Tokenize the chunk text
        tokens = self._tokenize(chunk.text)
        
        # Store document
        doc_idx = len(self.chunk_ids)
        self.chunk_ids.append(chunk_id)
        self.chunks[chunk_id] = chunk
        self.doc_tokens.append(tokens)
        self.doc_lengths.append(len(tokens))
        
        # Update inverted index incrementally
        term_freqs: Dict[str, int] = {}
        for token in tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1
        
        for term, freq in term_freqs.items():
            # Update document frequency
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
                self.term_doc_freqs[term] = 0
            
            if doc_idx not in self.inverted_index[term]:
                self.term_doc_freqs[term] += 1
            
            self.inverted_index[term][doc_idx] = freq
        
        # Mark index as needing rebuild for IDF
        self._index_built = False
    
    def build_index(self):
        """
        Build/rebuild the BM25 index statistics.
        
        Call this after adding all documents for optimal performance.
        Computes:
        - Average document length
        - IDF scores for all terms
        """
        self.N = len(self.chunk_ids)
        
        if self.N == 0:
            self.avgdl = 0.0
            return
        
        # Calculate average document length
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0.0
        
        # Pre-compute IDF for all terms
        self._idf_cache = {}
        for term, df in self.term_doc_freqs.items():
            # BM25 IDF formula (with +1 to avoid negative IDF for common terms)
            idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            self._idf_cache[term] = idf
        
        self._index_built = True
    
    def _get_idf(self, term: str) -> float:
        """Get IDF score for a term."""
        if term in self._idf_cache:
            return self._idf_cache[term]
        
        # Compute IDF for unknown term
        df = self.term_doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        
        idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        self._idf_cache[term] = idf
        return idf
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               filter_fn: Optional[Callable[[IndexableChunk], bool]] = None) -> List[Tuple[str, float]]:
        """
        Search the index using BM25 scoring.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filter_fn: Optional function to filter chunks (chunk -> bool)
            
        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        if not self._index_built:
            self.build_index()
        
        if self.N == 0:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Calculate BM25 scores for all matching documents
        scores: Dict[int, float] = {}
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            idf = self._get_idf(term)
            
            # Score each document containing this term
            for doc_idx, term_freq in self.inverted_index[term].items():
                doc_len = self.doc_lengths[doc_idx]
                
                # BM25 scoring formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                term_score = idf * (numerator / denominator)
                
                if doc_idx not in scores:
                    scores[doc_idx] = 0.0
                scores[doc_idx] += term_score
        
        # Apply filter if provided
        if filter_fn:
            filtered_scores = {}
            for doc_idx, score in scores.items():
                chunk_id = self.chunk_ids[doc_idx]
                chunk = self.chunks.get(chunk_id)
                if chunk and filter_fn(chunk):
                    filtered_scores[doc_idx] = score
            scores = filtered_scores
        
        # Sort by score and return top_k
        sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
        
        results = []
        for doc_idx, score in sorted_docs[:top_k]:
            chunk_id = self.chunk_ids[doc_idx]
            results.append((chunk_id, score))
        
        return results
    
    def get_chunk(self, chunk_id: str) -> Optional[IndexableChunk]:
        """Get a chunk by ID."""
        return self.chunks.get(chunk_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_documents': self.N,
            'unique_terms': len(self.term_doc_freqs),
            'average_doc_length': self.avgdl,
            'k1': self.k1,
            'b': self.b
        }
    
    def __len__(self):
        return len(self.chunk_ids)


# =============================================================================
# RECIPROCAL RANK FUSION
# =============================================================================

def reciprocal_rank_fusion(result_lists: List[List[Tuple[str, float]]], 
                           k: int = 60) -> Dict[str, float]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion (RRF).
    
    RRF is a simple but effective method for combining results from multiple
    retrieval systems. It uses ranks rather than scores, making it robust
    to score distribution differences between systems.
    
    Formula: RRF_score(d) = Σ 1/(k + rank(d))
    
    Args:
        result_lists: List of result lists, each containing (chunk_id, score) tuples
        k: Ranking constant (default: 60, as used by Elasticsearch)
           Higher k = more weight to lower-ranked results
           
    Returns:
        Dict mapping chunk_id -> fused RRF score
        
    Example:
        >>> vector_results = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
        >>> bm25_results = [("B", 12.5), ("D", 10.0), ("A", 8.0)]
        >>> fused = reciprocal_rank_fusion([vector_results, bm25_results])
        >>> # B appears rank 2 in vector (1/62) and rank 1 in BM25 (1/61)
        >>> # B gets highest combined score
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    
    for results in result_lists:
        if not results:
            continue
        
        # Sort by score descending to establish ranks
        sorted_results = sorted(results, key=lambda x: -x[1])
        
        for rank, (chunk_id, score) in enumerate(sorted_results):
            # RRF formula: 1/(k + rank), where rank is 0-indexed
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
    
    return dict(rrf_scores)


class HybridIndex:
    """
    Hybrid retrieval index combining vector, BM25, and concept search.
    
    This index uses multiple retrieval signals:
    - Vector similarity search (semantic matching)
    - BM25 lexical search (exact term matching)
    - Concept index (domain vocabulary matching)
    
    Results are combined using Reciprocal Rank Fusion (RRF), which:
    - Uses ranks instead of scores (robust to score distribution differences)
    - Naturally handles missing results from some retrievers
    - Proven effective in production search systems
    
    Search Strategy:
        1. If both vector and BM25 return results → RRF fusion
        2. If only one returns results → Use that retriever
        3. Fall back to concept matching if neither works
    
    Features:
    - Filtering by source type, capability, etc.
    - Automatic fallback when one retriever fails
    - Configurable weights and thresholds
    """
    
    def __init__(self, 
                 vocabulary: DomainVocabulary,
                 embedding_fn: Optional[Callable[[str], List[float]]] = None,
                 bm25_k1: float = 1.5,
                 bm25_b: float = 0.75):
        """
        Initialize hybrid index with vector, BM25, and concept components.
        
        Args:
            vocabulary: Domain vocabulary for query expansion
            embedding_fn: Function to generate embeddings (text -> vector)
            bm25_k1: BM25 term frequency saturation (default: 1.5)
            bm25_b: BM25 length normalization (default: 0.75)
        """
        self.vocabulary = vocabulary
        self.embedding_fn = embedding_fn
        
        # Initialize all three retrieval components
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index(k1=bm25_k1, b=bm25_b)
        self.concept_index = ConceptIndex()
        
        # Index metadata
        self.metadata: Dict[str, Any] = {
            'total_indexed': 0,
            'by_source_type': defaultdict(int),
            'bm25_enabled': True
        }
    
    def set_embedding_function(self, fn: Callable[[str], List[float]]):
        """Set the embedding function for vector search"""
        self.embedding_fn = fn
    
    def index_chunk(self, chunk: IndexableChunk):
        """
        Add a chunk to all indexes (vector, BM25, concept).
        
        Args:
            chunk: Chunk to index
        """
        # Add to concept index (always)
        self.concept_index.add(chunk)
        
        # Add to BM25 index (always - lexical search doesn't need embeddings)
        self.bm25_index.add(chunk.chunk_id, chunk)
        
        # Add to vector store if embedding function available
        if self.embedding_fn:
            try:
                embedding = self.embedding_fn(chunk.embedding_text)
                chunk.embedding = embedding
                self.vector_store.add(chunk.chunk_id, embedding, chunk)
            except Exception as e:
                print(f"Warning: Failed to embed chunk {chunk.chunk_id}: {e}")
        
        # Update metadata
        self.metadata['total_indexed'] += 1
        self.metadata['by_source_type'][chunk.source_type.value] += 1
    
    def index_chunks(self, chunks: List[IndexableChunk], batch_size: int = 100):
        """
        Index multiple chunks.
        
        Args:
            chunks: List of chunks to index
            batch_size: Batch size for embedding (if applicable)
        """
        for chunk in chunks:
            self.index_chunk(chunk)
        
        # Build BM25 index after all chunks are added (more efficient)
        self.bm25_index.build_index()
    
    def search(self,
               query: str,
               top_k: int = 10,
               source_types: Optional[List[SourceType]] = None,
               capabilities: Optional[List[str]] = None,
               use_rrf: bool = True,
               rrf_k: int = 60,
               vector_weight: float = 1.0,
               bm25_weight: float = 1.0,
               concept_weight: float = 0.5) -> List[SearchResult]:
        """
        Hybrid search combining vector, BM25, and concept matching.
        
        Uses Reciprocal Rank Fusion (RRF) by default to combine results from
        multiple retrieval systems. RRF is robust to score distribution
        differences between retrievers.
        
        Search Strategy:
            1. Run vector search (semantic similarity)
            2. Run BM25 search (lexical matching)
            3. Run concept search (domain vocabulary)
            4. Fuse results using RRF or weighted combination
        
        Args:
            query: Search query
            top_k: Number of results to return
            source_types: Filter by source types (None = all)
            capabilities: Filter by business capabilities (None = all)
            use_rrf: If True, use Reciprocal Rank Fusion (recommended)
                     If False, use weighted score combination
            rrf_k: RRF constant (default: 60, higher = more weight to lower ranks)
            vector_weight: Weight for vector results in RRF (default: 1.0)
            bm25_weight: Weight for BM25 results in RRF (default: 1.0)
            concept_weight: Weight for concept results in RRF (default: 0.5)
            
        Returns:
            List of SearchResult objects ranked by fused score
        """
        # Extract concepts from query for matching
        query_concepts = self.vocabulary.match_text(query, deduplicate=True)
        expanded_terms = self.vocabulary.expand_query(query)
        
        # Create filter function for source types and capabilities
        def filter_fn(chunk: IndexableChunk) -> bool:
            if source_types and chunk.source_type not in source_types:
                return False
            if capabilities:
                chunk_caps = chunk.capability_set
                if not any(cap in chunk_caps for cap in capabilities):
                    return False
            return True
        
        # ========== 1. Vector Search ==========
        vector_results: List[Tuple[str, float]] = []
        
        if self.embedding_fn and len(self.vector_store) > 0:
            query_embedding = self.embedding_fn(query)
            vector_results = self.vector_store.search(
                query_embedding, 
                top_k=top_k * 3,  # Get more for fusion
                filter_fn=filter_fn
            )
        
        # ========== 2. BM25 Search ==========
        bm25_results: List[Tuple[str, float]] = []
        
        if len(self.bm25_index) > 0:
            bm25_results = self.bm25_index.search(
                query,
                top_k=top_k * 3,
                filter_fn=filter_fn
            )
        
        # ========== 3. Concept Search ==========
        concept_scores: Dict[str, float] = {}
        
        # Match by extracted concepts
        for concept_match in query_concepts:
            chunk_ids = self.concept_index.search_concept(concept_match.canonical_term)
            for chunk_id in chunk_ids:
                if chunk_id not in concept_scores:
                    concept_scores[chunk_id] = 0.0
                concept_scores[chunk_id] += 1.0
        
        # Match by expanded terms
        for term in expanded_terms:
            entry = self.vocabulary.get_entry_by_term(term)
            if entry:
                chunk_ids = self.concept_index.search_concept(entry.canonical_term)
                for chunk_id in chunk_ids:
                    if chunk_id not in concept_scores:
                        concept_scores[chunk_id] = 0.0
                    concept_scores[chunk_id] += 0.5
        
        # Apply source type filter to concept results
        if source_types:
            allowed_chunks = set()
            for st in source_types:
                allowed_chunks.update(self.concept_index.search_source_type(st.value))
            concept_scores = {
                cid: score for cid, score in concept_scores.items()
                if cid in allowed_chunks
            }
        
        # Convert to list format for fusion
        concept_results = [(cid, score) for cid, score in concept_scores.items()]
        
        # ========== 4. Result Fusion ==========
        if use_rrf:
            # Build weighted result lists for RRF
            # Apply weights by repeating results (simple but effective)
            weighted_lists = []
            
            if vector_results and vector_weight > 0:
                weighted_lists.append(vector_results)
            
            if bm25_results and bm25_weight > 0:
                weighted_lists.append(bm25_results)
            
            if concept_results and concept_weight > 0:
                weighted_lists.append(concept_results)
            
            if not weighted_lists:
                return []
            
            # Apply RRF
            fused_scores = reciprocal_rank_fusion(weighted_lists, k=rrf_k)
            
            # Normalize RRF scores to 0-1 range
            # RRF scores are typically very small (e.g., 0.01-0.05)
            if fused_scores:
                max_rrf = max(fused_scores.values())
                min_rrf = min(fused_scores.values())
                rrf_range = max_rrf - min_rrf
                if rrf_range > 0:
                    # Normalize to 0-1 range
                    fused_scores = {
                        cid: (score - min_rrf) / rrf_range 
                        for cid, score in fused_scores.items()
                    }
                else:
                    # Single result or all same score - give full score
                    fused_scores = {cid: 1.0 for cid in fused_scores}
            
        else:
            # Fallback to weighted score combination
            # Normalize each result list's scores to 0-1
            def normalize_scores(results: List[Tuple[str, float]]) -> Dict[str, float]:
                if not results:
                    return {}
                scores = [s for _, s in results]
                min_s, max_s = min(scores), max(scores)
                range_s = max_s - min_s if max_s != min_s else 1.0
                return {cid: (s - min_s) / range_s for cid, s in results}
            
            v_norm = normalize_scores(vector_results)
            b_norm = normalize_scores(bm25_results)
            c_norm = normalize_scores(concept_results)
            
            all_ids = set(v_norm.keys()) | set(b_norm.keys()) | set(c_norm.keys())
            total_weight = vector_weight + bm25_weight + concept_weight
            
            fused_scores = {}
            for chunk_id in all_ids:
                score = (
                    vector_weight * v_norm.get(chunk_id, 0) +
                    bm25_weight * b_norm.get(chunk_id, 0) +
                    concept_weight * c_norm.get(chunk_id, 0)
                ) / total_weight if total_weight > 0 else 0
                fused_scores[chunk_id] = score
        
        # ========== 5. Build SearchResult objects ==========
        # Create lookup dicts for individual scores
        vector_dict = dict(vector_results)
        bm25_dict = dict(bm25_results)
        concept_dict = dict(concept_results)
        
        fused_results = []
        for chunk_id, combined_score in fused_scores.items():
            # Get chunk from any store
            chunk = (
                self.vector_store.get_chunk(chunk_id) or 
                self.bm25_index.get_chunk(chunk_id) or
                self.concept_index.get_chunk(chunk_id)
            )
            
            if not chunk:
                continue
            
            # Determine matched concepts
            matched_concepts = []
            matched_capabilities = []
            for qc in query_concepts:
                for cm in chunk.domain_matches:
                    if cm.canonical_term.lower() == qc.canonical_term.lower():
                        matched_concepts.append(cm.canonical_term)
                        matched_capabilities.extend(cm.capabilities)
            
            # Determine retrieval method
            methods = []
            if chunk_id in vector_dict:
                methods.append("vector")
            if chunk_id in bm25_dict:
                methods.append("bm25")
            if chunk_id in concept_dict:
                methods.append("concept")
            method = "+".join(methods) if methods else "unknown"
            
            result = SearchResult(
                chunk=chunk,
                vector_score=vector_dict.get(chunk_id, 0.0),
                bm25_score=bm25_dict.get(chunk_id, 0.0),
                concept_score=concept_dict.get(chunk_id, 0.0),
                keyword_score=0.0,  # Deprecated, kept for compatibility
                combined_score=combined_score,
                matched_concepts=list(set(matched_concepts)),
                matched_capabilities=list(set(matched_capabilities)),
                retrieval_method=method
            )
            fused_results.append(result)
        
        # Sort by combined score
        fused_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(fused_results[:top_k]):
            result.rank = i + 1
        
        return fused_results[:top_k]
    
    def _keyword_search(self,
                        query: str,
                        source_types: Optional[List[SourceType]] = None,
                        capabilities: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Perform keyword/grep search on chunk text content.
        
        This is a fallback for when semantic search doesn't find good matches,
        especially useful for exact technical terms, acronyms, or domain-specific jargon.
        
        Args:
            query: Search query
            source_types: Filter by source types
            capabilities: Filter by capabilities
            
        Returns:
            Dict mapping chunk_id -> keyword match score
        """
        import re
        results: Dict[str, float] = {}
        
        # Common abbreviation expansions for payment domain
        abbreviations = {
            'fedin': ['fedwire inbound', 'fed in', 'fedin', 'fedwire in'],
            'fedout': ['fedwire outbound', 'fed out', 'fedout', 'fedwire out'],
            'ltr': ['ltr', 'letter', 'advise ltr', 'advice ltr'],
            'mt103': ['mt103', 'mt-103', 'mt 103'],
            'mt202': ['mt202', 'mt-202', 'mt 202'],
            'pacs008': ['pacs008', 'pacs.008', 'pacs 008'],
            'pacs009': ['pacs009', 'pacs.009', 'pacs 009'],
            'ofac': ['ofac', 'sanctions screening', 'sanctions'],
            'aml': ['aml', 'anti-money laundering', 'anti money laundering'],
            'bic': ['bic', 'swift code', 'bank identifier'],
            'iban': ['iban', 'international bank account'],
            'cov': ['cov', 'cover', 'coverage'],
            'stp': ['stp', 'straight through processing'],
            'swift': ['swift', 'society for worldwide'],
        }
        
        def tokenize(text: str) -> List[str]:
            """Tokenize text on common delimiters"""
            # Split on whitespace, underscores, hyphens, dots, slashes, etc.
            tokens = re.split(r'[\s_\-./\\,;:()\[\]{}]+', text.lower())
            # Filter empty and single-char tokens
            return [t.strip() for t in tokens if len(t.strip()) >= 2]
        
        def get_expanded_terms(term: str) -> List[str]:
            """Get expanded forms of abbreviations"""
            term_lower = term.lower()
            expanded = [term_lower]
            if term_lower in abbreviations:
                expanded.extend(abbreviations[term_lower])
            return list(set(expanded))
        
        # Stop words to filter out
        stop_words = {'how', 'to', 'the', 'a', 'an', 'is', 'are', 'what', 'where', 'when', 'why', 
                      'can', 'do', 'does', 'for', 'in', 'on', 'at', 'by', 'with', 'from', 'implement',
                      'implementation', 'implementing', 'about', 'this', 'that', 'which'}
        
        # Tokenize query
        query_tokens = tokenize(query)
        query_terms = [t for t in query_tokens if t not in stop_words]
        
        if not query_terms:
            return results
        
        # Build expanded query terms (including abbreviation expansions)
        expanded_query_terms = set()
        for term in query_terms:
            for exp in get_expanded_terms(term):
                expanded_query_terms.add(exp)
                # Also add individual tokens from multi-word expansions
                for tok in tokenize(exp):
                    if tok not in stop_words:
                        expanded_query_terms.add(tok)
        
        # Search all chunks
        all_chunks = list(self.concept_index.chunks.values())
        
        # Also include chunks from vector store that might not be in concept index
        for chunk_id, chunk in self.vector_store.chunks.items():
            if chunk_id not in self.concept_index.chunks:
                all_chunks.append(chunk)
        
        for chunk in all_chunks:
            # Apply filters
            if source_types and chunk.source_type not in source_types:
                continue
            if capabilities:
                chunk_caps = chunk.capability_set
                if not any(cap in chunk_caps for cap in capabilities):
                    continue
            
            # Collect all searchable text sources
            text_parts = [chunk.text]
            if chunk.embedding_text:
                text_parts.append(chunk.embedding_text)
            
            # Get filename (important for matching)
            filename = ""
            if chunk.source_ref and hasattr(chunk.source_ref, 'file_path') and chunk.source_ref.file_path:
                filename = chunk.source_ref.file_path
                text_parts.append(filename)
            
            # Get procedure name
            if chunk.source_ref and hasattr(chunk.source_ref, 'procedure_name') and chunk.source_ref.procedure_name:
                text_parts.append(chunk.source_ref.procedure_name)
            
            # Create searchable representations
            full_text = " ".join(text_parts)
            full_text_lower = full_text.lower()
            full_text_tokens = set(tokenize(full_text))
            
            # Tokenize filename separately for boosted matching
            filename_tokens = set(tokenize(filename)) if filename else set()
            
            # Score based on matches
            score = 0.0
            matched_info = []
            
            # === FILENAME TOKEN MATCHING (highest priority) ===
            filename_matches = 0
            for term in query_terms:
                term_matched = False
                # Direct match
                if term in filename_tokens:
                    filename_matches += 1
                    term_matched = True
                    matched_info.append(f"filename:{term}")
                else:
                    # Check expanded forms
                    for exp in get_expanded_terms(term):
                        exp_toks = tokenize(exp)
                        if any(et in filename_tokens for et in exp_toks):
                            filename_matches += 1
                            term_matched = True
                            matched_info.append(f"filename:{term}→{exp}")
                            break
            
            if query_terms and filename_matches > 0:
                filename_coverage = filename_matches / len(query_terms)
                score += 2.0 * filename_coverage  # Strong boost for filename matches
                if filename_matches >= len(query_terms):
                    score += 1.0  # Big bonus for all query terms in filename
            
            # === CONTENT TOKEN MATCHING ===
            content_matches = 0
            for term in query_terms:
                # Direct token match
                if term in full_text_tokens:
                    content_matches += 1
                    matched_info.append(f"token:{term}")
                else:
                    # Check expanded forms in tokens
                    for exp in get_expanded_terms(term):
                        exp_toks = tokenize(exp)
                        if any(et in full_text_tokens for et in exp_toks):
                            content_matches += 1
                            matched_info.append(f"token:{term}→{exp}")
                            break
            
            if query_terms and content_matches > 0:
                content_coverage = content_matches / len(query_terms)
                score += 1.0 * content_coverage
                if content_matches >= len(query_terms):
                    score += 0.5  # Bonus for all terms present
            
            # === SUBSTRING/PHRASE MATCHING ===
            # Check for expanded phrases in the raw text
            for term in query_terms:
                for exp in get_expanded_terms(term):
                    if len(exp) >= 3 and exp in full_text_lower:
                        score += 0.5
                        matched_info.append(f"substr:{exp}")
                        break
            
            # === MULTI-WORD PHRASE MATCHING ===
            if len(query_terms) >= 2:
                # Try matching consecutive query terms
                query_phrase = ' '.join(query_terms)
                # Normalize both for comparison
                normalized_text = re.sub(r'[\s_\-./\\]+', ' ', full_text_lower)
                if query_phrase in normalized_text:
                    score += 1.5
                    matched_info.append(f"phrase:{query_phrase}")
            
            if score > 0:
                # Normalize score to 0-1 range (cap at 1.0)
                score = min(score / 4.0, 1.0)
                results[chunk.chunk_id] = score
        
        return results
    
    def search_by_capability(self, 
                             capability: str,
                             top_k: int = 10) -> List[SearchResult]:
        """
        Find all chunks related to a business capability
        
        Args:
            capability: Business capability name
            top_k: Maximum results
            
        Returns:
            List of SearchResult objects
        """
        chunk_ids = self.concept_index.search_capability(capability)
        
        results = []
        for chunk_id in list(chunk_ids)[:top_k]:
            chunk = self.concept_index.get_chunk(chunk_id)
            if chunk:
                results.append(SearchResult(
                    chunk=chunk,
                    concept_score=1.0,
                    combined_score=1.0,
                    matched_capabilities=[capability],
                    retrieval_method="concept"
                ))
        
        return results
    
    def search_cross_reference(self,
                               query: str,
                               source_type: SourceType,
                               reference_types: List[SourceType],
                               top_k: int = 5) -> Dict[str, List[SearchResult]]:
        """
        Find related content across different source types
        
        Example: Find code that handles errors from logs
        
        Args:
            query: Search query
            source_type: Primary source type to search
            reference_types: Related source types to cross-reference
            top_k: Results per source type
            
        Returns:
            Dict mapping source type to results
        """
        # Search primary source type
        primary_results = self.search(
            query, 
            top_k=top_k, 
            source_types=[source_type]
        )
        
        results = {source_type.value: primary_results}
        
        # Extract concepts from primary results
        all_concepts = set()
        all_capabilities = set()
        
        for result in primary_results:
            all_concepts.update(result.chunk.canonical_terms)
            all_capabilities.update(result.chunk.capability_set)
        
        # Search reference types using extracted concepts
        for ref_type in reference_types:
            ref_results = []
            
            # Search by each concept
            for concept in list(all_concepts)[:5]:  # Limit concepts
                concept_hits = self.concept_index.search_concept(concept)
                
                for chunk_id in concept_hits:
                    chunk = self.concept_index.get_chunk(chunk_id)
                    if chunk and chunk.source_type == ref_type:
                        ref_results.append(SearchResult(
                            chunk=chunk,
                            concept_score=1.0,
                            combined_score=1.0,
                            matched_concepts=[concept],
                            retrieval_method="cross_reference"
                        ))
            
            # Deduplicate and limit
            seen_ids = set()
            unique_results = []
            for r in ref_results:
                if r.chunk.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk.chunk_id)
                    unique_results.append(r)
            
            results[ref_type.value] = unique_results[:top_k]
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            **self.metadata,
            'vector_store_size': len(self.vector_store),
            'concept_index': self.concept_index.get_statistics()
        }
    
    def save(self, directory: str, use_numpy: bool = True, verbose: bool = True):
        """
        Save index to disk using generation-based versioning.
        
        NON-BLOCKING: Does not interfere with concurrent searches.
        
        Process:
        1. Write new data to gen_N+1/ subdirectory
        2. Atomic update of generation.txt pointer
        3. Old generation cleaned up after grace period
        
        Args:
            directory: Directory to save index files
            use_numpy: If True, save embeddings as .npy binary (faster, smaller)
                       If False, save as .json (human readable)
            verbose: If True, print detailed generation logging
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Get current generation and prepare next
        current_gen = get_current_generation(path)
        next_gen = current_gen + 1
        gen_path = path / f'gen_{next_gen}'
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"GENERATION UPDATE: {current_gen} → {next_gen}")
            print(f"{'='*60}")
            print(f"  📂 Current generation: {current_gen}")
            print(f"  📂 Writing new generation: {next_gen}")
            print(f"  📁 Target directory: {gen_path}")
        
        # Write to new generation directory
        gen_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save chunks
            chunks_data = [
                chunk.to_dict() 
                for chunk in self.concept_index.chunks.values()
            ]
            
            if verbose:
                print(f"\n  Writing gen_{next_gen}/...")
                print(f"    • chunks.json ({len(chunks_data)} chunks)")
            
            with open(gen_path / 'chunks.json', 'w') as f:
                json.dump(chunks_data, f, indent=2)
            
            # Save embeddings if available
            if self.vector_store.embeddings:
                if use_numpy:
                    # Binary numpy format - faster and smaller
                    chunk_ids = list(self.vector_store.embeddings.keys())
                    embeddings_array = np.array([
                        self.vector_store.embeddings[cid] for cid in chunk_ids
                    ])
                    
                    # Save as .npy file
                    np.save(gen_path / 'embeddings.npy', embeddings_array)
                    with open(gen_path / 'embedding_ids.json', 'w') as f:
                        json.dump(chunk_ids, f)
                    
                    if verbose:
                        print(f"    • embeddings.npy ({embeddings_array.shape[0]} vectors, {embeddings_array.shape[1]} dims)")
                    
                    # Save dimensions info
                    embed_meta = {
                        'format': 'numpy',
                        'shape': list(embeddings_array.shape),
                        'dtype': str(embeddings_array.dtype)
                    }
                    with open(gen_path / 'embeddings_meta.json', 'w') as f:
                        json.dump(embed_meta, f, indent=2)
                else:
                    # JSON format - human readable but larger
                    embeddings_data = {
                        chunk_id: embedding.tolist()
                        for chunk_id, embedding in self.vector_store.embeddings.items()
                    }
                    with open(gen_path / 'embeddings.json', 'w') as f:
                        json.dump(embeddings_data, f)
                    if verbose:
                        print(f"    • embeddings.json ({len(embeddings_data)} vectors)")
            
            # Save BM25 index
            bm25_data = {
                'N': self.bm25_index.N,
                'avgdl': self.bm25_index.avgdl,
                'doc_lengths': self.bm25_index.doc_lengths,
                'chunk_ids': self.bm25_index.chunk_ids,
                'term_doc_freqs': self.bm25_index.term_doc_freqs,
                'inverted_index': {
                    term: dict(postings)
                    for term, postings in self.bm25_index.inverted_index.items()
                },
                'idf_cache': self.bm25_index._idf_cache,
                'index_built': self.bm25_index._index_built,
                'k1': self.bm25_index.k1,
                'b': self.bm25_index.b
            }
            with open(gen_path / 'bm25_index.json', 'w') as f:
                json.dump(bm25_data, f, indent=2)
            
            if verbose:
                print(f"    • bm25_index.json ({len(bm25_data['term_doc_freqs'])} terms)")
            
            # Save metadata
            with open(gen_path / 'metadata.json', 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            if verbose:
                print(f"    • metadata.json")
            
            # ATOMIC: Update generation pointer
            # Write to temp file then rename for atomicity
            if verbose:
                print(f"\n  ⚡ ATOMIC POINTER SWAP")
                print(f"    Writing generation.txt.tmp → '{next_gen}'")
            
            gen_file = path / 'generation.txt'
            temp_gen_file = path / 'generation.txt.tmp'
            temp_gen_file.write_text(str(next_gen))
            
            if verbose:
                print(f"    Renaming generation.txt.tmp → generation.txt")
            
            temp_gen_file.rename(gen_file)
            
            if verbose:
                print(f"    ✓ Generation pointer updated: {current_gen} → {next_gen}")
                print(f"\n  🔄 Searches now see generation {next_gen}")
            
            print(f"\nIndex saved to {directory} (generation {next_gen})")
            
            # Clean up old generations (keep last 2 for safety)
            self._cleanup_old_generations(path, keep_last=2, verbose=verbose)
            
        except Exception as e:
            # Clean up failed generation
            if verbose:
                print(f"\n  ❌ Error during save: {e}")
                print(f"    Cleaning up failed generation {next_gen}...")
            if gen_path.exists():
                shutil.rmtree(gen_path, ignore_errors=True)
            raise
    
    def _cleanup_old_generations(self, index_dir: Path, keep_last: int = 2, verbose: bool = False):
        """Remove old generation directories, keeping the most recent ones."""
        current_gen = get_current_generation(index_dir)
        
        # Find all generation directories
        gen_dirs = []
        for item in index_dir.iterdir():
            if item.is_dir() and item.name.startswith('gen_'):
                try:
                    gen_num = int(item.name.split('_')[1])
                    gen_dirs.append((gen_num, item))
                except (ValueError, IndexError):
                    pass
        
        # Sort by generation number
        gen_dirs.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old generations (keep most recent ones)
        to_remove = gen_dirs[keep_last:]
        if to_remove and verbose:
            print(f"\n  🧹 Cleanup: removing {len(to_remove)} old generation(s)")
        
        for gen_num, gen_path in to_remove:
            try:
                if verbose:
                    print(f"    Removing gen_{gen_num}/")
                shutil.rmtree(gen_path)
            except Exception as e:
                if verbose:
                    print(f"    ⚠️  Could not remove gen_{gen_num}: {e}")
        
        if verbose and gen_dirs:
            kept = gen_dirs[:keep_last]
            kept_nums = [g[0] for g in kept]
            print(f"    Keeping generations: {kept_nums}")
    
    def load(self, directory: str, verbose: bool = False):
        """
        Load index from disk using generation-based versioning.
        
        NON-BLOCKING: Never waits for writes. Always gets consistent snapshot.
        
        Args:
            directory: Directory containing index files
            verbose: If True, print detailed generation logging
        """
        path = Path(directory)
        
        if not path.exists():
            raise FileNotFoundError(f"Index directory not found: {directory}")
        
        # Get current generation and its path
        current_gen = get_current_generation(path)
        gen_path = get_generation_path(path, current_gen)
        
        if verbose:
            print(f"\n  📖 Loading generation {current_gen} from {gen_path}")
        
        # Handle legacy format (no generation subdirectory)
        if current_gen == 0 and not (gen_path / 'chunks.json').exists():
            # Check if chunks.json exists in root (legacy)
            if (path / 'chunks.json').exists():
                gen_path = path
                if verbose:
                    print(f"    Using legacy format (no generation subdirs)")
            else:
                raise FileNotFoundError(f"No index data found in {directory}")
        
        # Load chunks
        chunks_file = gen_path / 'chunks.json'
        if not chunks_file.exists():
            raise FileNotFoundError(f"chunks.json not found in {gen_path}")
            
        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)
        
        for chunk_data in chunks_data:
            chunk = IndexableChunk.from_dict(chunk_data)
            self.concept_index.add(chunk)
        
        if verbose:
            print(f"    Loaded {len(chunks_data)} chunks")
        
        # Load embeddings - try numpy format first, then JSON
        numpy_path = gen_path / 'embeddings.npy'
        json_path = gen_path / 'embeddings.json'
        ids_path = gen_path / 'embedding_ids.json'
        
        if numpy_path.exists() and ids_path.exists():
            # Load numpy binary format
            embeddings_array = np.load(numpy_path)
            with open(ids_path, 'r') as f:
                chunk_ids = json.load(f)
            
            for i, chunk_id in enumerate(chunk_ids):
                chunk = self.concept_index.get_chunk(chunk_id)
                if chunk:
                    embedding = embeddings_array[i].tolist()
                    self.vector_store.add(chunk_id, embedding, chunk)
        
        elif json_path.exists():
            # Load JSON format (legacy)
            with open(json_path, 'r') as f:
                embeddings_data = json.load(f)
            
            for chunk_id, embedding in embeddings_data.items():
                chunk = self.concept_index.get_chunk(chunk_id)
                if chunk:
                    self.vector_store.add(chunk_id, embedding, chunk)
        
        # Load BM25 index if available
        bm25_path = gen_path / 'bm25_index.json'
        if bm25_path.exists():
            with open(bm25_path, 'r') as f:
                bm25_data = json.load(f)
            
            self.bm25_index.N = bm25_data.get('N', 0)
            self.bm25_index.avgdl = bm25_data.get('avgdl', 0)
            self.bm25_index.doc_lengths = bm25_data.get('doc_lengths', [])
            self.bm25_index.chunk_ids = bm25_data.get('chunk_ids', [])
            self.bm25_index.term_doc_freqs = bm25_data.get('term_doc_freqs', {})
            self.bm25_index.inverted_index = {
                term: {int(k): v for k, v in postings.items()}
                for term, postings in bm25_data.get('inverted_index', {}).items()
            }
            self.bm25_index._idf_cache = bm25_data.get('idf_cache', {})
            self.bm25_index._index_built = bm25_data.get('index_built', False)
            self.bm25_index.k1 = bm25_data.get('k1', 1.5)
            self.bm25_index.b = bm25_data.get('b', 0.75)
            
            # Rebuild chunks dict from concept_index
            for chunk_id in self.bm25_index.chunk_ids:
                chunk = self.concept_index.get_chunk(chunk_id)
                if chunk:
                    self.bm25_index.chunks[chunk_id] = chunk
        
        # Load metadata
        metadata_path = gen_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        gen_info = f" (generation {current_gen})" if current_gen > 0 else ""
        print(f"Index loaded from {directory}{gen_info}: {len(self.concept_index)} chunks")
