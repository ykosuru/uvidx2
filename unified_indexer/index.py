"""
Hybrid Index - Combines vector similarity with concept-based retrieval

This index stores chunks in both a vector store (for semantic search)
and a concept index (for exact domain term matching). Search results
are fused using reciprocal rank fusion.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

from .models import (
    IndexableChunk,
    SearchResult,
    SourceType,
    DomainMatch
)
from .vocabulary import DomainVocabulary


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


class HybridIndex:
    """
    Hybrid retrieval index combining vector and concept search
    
    Features:
    - Vector similarity search for semantic matching
    - Concept index for exact domain term matching
    - Reciprocal rank fusion for result combination
    - Filtering by source type, capability, etc.
    """
    
    def __init__(self, 
                 vocabulary: DomainVocabulary,
                 embedding_fn: Optional[Callable[[str], List[float]]] = None):
        """
        Initialize hybrid index
        
        Args:
            vocabulary: Domain vocabulary for query expansion
            embedding_fn: Function to generate embeddings (text -> vector)
        """
        self.vocabulary = vocabulary
        self.embedding_fn = embedding_fn
        
        self.vector_store = VectorStore()
        self.concept_index = ConceptIndex()
        
        # Index metadata
        self.metadata: Dict[str, Any] = {
            'total_indexed': 0,
            'by_source_type': defaultdict(int)
        }
    
    def set_embedding_function(self, fn: Callable[[str], List[float]]):
        """Set the embedding function for vector search"""
        self.embedding_fn = fn
    
    def index_chunk(self, chunk: IndexableChunk):
        """
        Add a chunk to both indexes
        
        Args:
            chunk: Chunk to index
        """
        # Add to concept index
        self.concept_index.add(chunk)
        
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
        Index multiple chunks
        
        Args:
            chunks: List of chunks to index
            batch_size: Batch size for embedding (if applicable)
        """
        for chunk in chunks:
            self.index_chunk(chunk)
    
    def search(self,
               query: str,
               top_k: int = 10,
               source_types: Optional[List[SourceType]] = None,
               capabilities: Optional[List[str]] = None,
               vector_weight: float = 0.5,
               concept_weight: float = 0.5,
               keyword_weight: float = 0.3,
               keyword_fallback_threshold: float = 0.3) -> List[SearchResult]:
        """
        Hybrid search combining vector, concept, and keyword matching
        
        Args:
            query: Search query
            top_k: Number of results to return
            source_types: Filter by source types (None = all)
            capabilities: Filter by business capabilities (None = all)
            vector_weight: Weight for vector search scores
            concept_weight: Weight for concept match scores
            keyword_weight: Weight for keyword/grep match scores
            keyword_fallback_threshold: If top vector score below this, boost keyword weight
            
        Returns:
            List of SearchResult objects ranked by combined score
        """
        # Extract concepts from query
        query_concepts = self.vocabulary.match_text(query, deduplicate=True)
        
        # Expand query with synonyms
        expanded_terms = self.vocabulary.expand_query(query)
        
        # ========== Vector Search ==========
        vector_results: Dict[str, float] = {}
        
        if self.embedding_fn and len(self.vector_store) > 0:
            # Create filter function
            def filter_fn(chunk: IndexableChunk) -> bool:
                if source_types and chunk.source_type not in source_types:
                    return False
                if capabilities:
                    chunk_caps = chunk.capability_set
                    if not any(cap in chunk_caps for cap in capabilities):
                        return False
                return True
            
            # Search vector store
            query_embedding = self.embedding_fn(query)
            vector_hits = self.vector_store.search(
                query_embedding, 
                top_k=top_k * 3,  # Get more for fusion
                filter_fn=filter_fn
            )
            
            for chunk_id, score in vector_hits:
                vector_results[chunk_id] = score
        
        # ========== Keyword/Grep Search ==========
        keyword_results: Dict[str, float] = {}
        keyword_results = self._keyword_search(query, source_types, capabilities)
        
        # Check if we need to boost keyword weight (low vector scores)
        effective_keyword_weight = keyword_weight
        top_vector_score = max(vector_results.values()) if vector_results else 0
        
        if top_vector_score < keyword_fallback_threshold:
            # Boost keyword weight when vector search isn't finding good matches
            # The lower the vector score, the more we rely on keywords
            if top_vector_score < 0.15:
                # Very low vector scores - keywords should dominate
                effective_keyword_weight = 0.8
            elif top_vector_score < 0.25:
                effective_keyword_weight = 0.6
            else:
                effective_keyword_weight = 0.5
        
        if not vector_results:
            # No vector results at all, rely entirely on keywords
            effective_keyword_weight = 0.9
        
        # ========== Concept Search ==========
        concept_results: Dict[str, float] = {}
        
        # Search by extracted concepts
        for concept_match in query_concepts:
            chunk_ids = self.concept_index.search_concept(concept_match.canonical_term)
            for chunk_id in chunk_ids:
                if chunk_id not in concept_results:
                    concept_results[chunk_id] = 0.0
                concept_results[chunk_id] += 1.0
        
        # Search by expanded terms
        for term in expanded_terms:
            entry = self.vocabulary.get_entry_by_term(term)
            if entry:
                chunk_ids = self.concept_index.search_concept(entry.canonical_term)
                for chunk_id in chunk_ids:
                    if chunk_id not in concept_results:
                        concept_results[chunk_id] = 0.0
                    concept_results[chunk_id] += 0.5  # Lower weight for expanded terms
        
        # Search by capability filter
        if capabilities:
            for capability in capabilities:
                chunk_ids = self.concept_index.search_capability(capability)
                for chunk_id in chunk_ids:
                    if chunk_id not in concept_results:
                        concept_results[chunk_id] = 0.0
                    concept_results[chunk_id] += 0.5
        
        # Apply source type filter to concept results
        if source_types:
            allowed_chunks = set()
            for st in source_types:
                allowed_chunks.update(
                    self.concept_index.search_source_type(st.value)
                )
            concept_results = {
                cid: score for cid, score in concept_results.items()
                if cid in allowed_chunks
            }
        
        # Normalize concept scores
        if concept_results:
            max_concept_score = max(concept_results.values())
            if max_concept_score > 0:
                concept_results = {
                    cid: score / max_concept_score 
                    for cid, score in concept_results.items()
                }
        
        # ========== Result Fusion ==========
        all_chunk_ids = set(vector_results.keys()) | set(concept_results.keys()) | set(keyword_results.keys())
        
        fused_results = []
        for chunk_id in all_chunk_ids:
            v_score = vector_results.get(chunk_id, 0.0)
            c_score = concept_results.get(chunk_id, 0.0)
            k_score = keyword_results.get(chunk_id, 0.0)
            
            # Weighted combination (normalize weights)
            total_weight = vector_weight + concept_weight + effective_keyword_weight
            combined_score = (
                (vector_weight * v_score) + 
                (concept_weight * c_score) + 
                (effective_keyword_weight * k_score)
            ) / total_weight if total_weight > 0 else 0
            
            # Boost score when we have strong concept AND keyword matches
            # This indicates a direct term match which should rank highly
            if c_score >= 0.5 and k_score >= 0.5:
                # Both concept and keyword matched - high confidence
                boost = min(c_score, k_score) * 0.3  # Up to 0.3 boost
                combined_score = min(combined_score + boost, 1.0)
            elif c_score >= 0.8 or k_score >= 0.8:
                # At least one very strong match
                boost = max(c_score, k_score) * 0.15
                combined_score = min(combined_score + boost, 1.0)
            
            # Get chunk from either store
            chunk = (
                self.vector_store.get_chunk(chunk_id) or 
                self.concept_index.get_chunk(chunk_id)
            )
            
            if chunk:
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
                if v_score > 0:
                    methods.append("vector")
                if c_score > 0:
                    methods.append("concept")
                if k_score > 0:
                    methods.append("keyword")
                method = "+".join(methods) if methods else "unknown"
                
                result = SearchResult(
                    chunk=chunk,
                    vector_score=v_score,
                    concept_score=c_score,
                    keyword_score=k_score,
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
    
    def save(self, directory: str, use_numpy: bool = True):
        """
        Save index to disk
        
        Args:
            directory: Directory to save index files
            use_numpy: If True, save embeddings as .npy binary (faster, smaller)
                       If False, save as .json (human readable)
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        chunks_data = [
            chunk.to_dict() 
            for chunk in self.concept_index.chunks.values()
        ]
        
        with open(path / 'chunks.json', 'w') as f:
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
                np.save(path / 'embeddings.npy', embeddings_array)
                with open(path / 'embedding_ids.json', 'w') as f:
                    json.dump(chunk_ids, f)
                
                # Save dimensions info
                embed_meta = {
                    'format': 'numpy',
                    'shape': list(embeddings_array.shape),
                    'dtype': str(embeddings_array.dtype)
                }
                with open(path / 'embeddings_meta.json', 'w') as f:
                    json.dump(embed_meta, f, indent=2)
            else:
                # JSON format - human readable but larger
                embeddings_data = {
                    chunk_id: embedding.tolist()
                    for chunk_id, embedding in self.vector_store.embeddings.items()
                }
                with open(path / 'embeddings.json', 'w') as f:
                    json.dump(embeddings_data, f)
        
        # Save metadata
        with open(path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Index saved to {directory}")
    
    def load(self, directory: str):
        """
        Load index from disk
        
        Args:
            directory: Directory containing index files
        """
        path = Path(directory)
        
        if not path.exists():
            raise FileNotFoundError(f"Index directory not found: {directory}")
        
        # Load chunks
        with open(path / 'chunks.json', 'r') as f:
            chunks_data = json.load(f)
        
        for chunk_data in chunks_data:
            chunk = IndexableChunk.from_dict(chunk_data)
            self.concept_index.add(chunk)
        
        # Load embeddings - try numpy format first, then JSON
        numpy_path = path / 'embeddings.npy'
        json_path = path / 'embeddings.json'
        ids_path = path / 'embedding_ids.json'
        
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
        
        # Load metadata
        metadata_path = path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        print(f"Index loaded from {directory}: {len(self.concept_index)} chunks")
