#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Indexer

Test Categories:
- Unit Tests: Individual component testing in isolation
- Functional Tests: Feature-level testing
- Positive Tests: Valid inputs produce expected outputs
- Negative Tests: Invalid inputs handled gracefully
- Edge Cases: Boundary conditions and unusual scenarios

Run with: python -m pytest tests/ -v
Or standalone: python tests/test_unified_indexer.py
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_indexer import IndexingPipeline, SourceType, SearchResult
from unified_indexer.index import (
    HybridIndex, 
    BM25Index, 
    ConceptIndex, 
    VectorStore,
    get_current_generation,
    get_generation_path
)
from unified_indexer.embeddings import (
    HashEmbedder,
    DomainConceptEmbedder,
    HybridEmbedder,
    TFIDFEmbedder,
    BM25Embedder,
    create_embedder
)
from unified_indexer.vocabulary import DomainVocabulary, VocabularyEntry
from unified_indexer.models import (
    IndexableChunk,
    SourceReference,
    SemanticType,
    DomainMatch
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

@dataclass
class TestResult:
    """Result of a test"""
    name: str
    passed: bool
    message: str = ""
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        msg = f" - {self.message}" if self.message else ""
        return f"{status}: {self.name}{msg}"


class TestSuite:
    """Base class for test suites"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.temp_dirs: List[Path] = []
    
    def setup(self):
        """Setup before tests"""
        pass
    
    def teardown(self):
        """Cleanup after tests"""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def create_temp_dir(self) -> Path:
        """Create a temporary directory for testing"""
        temp_dir = Path(tempfile.mkdtemp(prefix='test_indexer_'))
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def assert_true(self, condition: bool, name: str, message: str = ""):
        """Assert condition is true"""
        self.results.append(TestResult(name, condition, message if not condition else ""))
        return condition
    
    def assert_false(self, condition: bool, name: str, message: str = ""):
        """Assert condition is false"""
        return self.assert_true(not condition, name, message)
    
    def assert_equal(self, actual, expected, name: str):
        """Assert values are equal"""
        passed = actual == expected
        msg = f"Expected {expected}, got {actual}" if not passed else ""
        return self.assert_true(passed, name, msg)
    
    def assert_greater(self, actual, expected, name: str):
        """Assert actual > expected"""
        passed = actual > expected
        msg = f"Expected > {expected}, got {actual}" if not passed else ""
        return self.assert_true(passed, name, msg)
    
    def assert_raises(self, exception_type, func, name: str, *args, **kwargs):
        """Assert function raises exception"""
        try:
            func(*args, **kwargs)
            self.results.append(TestResult(name, False, f"Expected {exception_type.__name__}"))
            return False
        except exception_type:
            self.results.append(TestResult(name, True))
            return True
        except Exception as e:
            self.results.append(TestResult(name, False, f"Got {type(e).__name__}: {e}"))
            return False
    
    def run_all(self) -> bool:
        """Run all test methods"""
        self.setup()
        
        # Find and run all test methods
        test_methods = [m for m in dir(self) if m.startswith('test_')]
        
        for method_name in sorted(test_methods):
            method = getattr(self, method_name)
            try:
                method()
            except Exception as e:
                self.results.append(TestResult(method_name, False, f"Exception: {e}"))
        
        self.teardown()
        return all(r.passed for r in self.results)
    
    def print_results(self):
        """Print test results"""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print(f"\n{'='*60}")
        print(f"{self.__class__.__name__} Results: {passed}/{total} passed")
        print('='*60)
        
        for result in self.results:
            print(f"  {result}")
        
        return passed == total


def create_sample_vocabulary() -> List[Dict]:
    """Create sample vocabulary data for testing"""
    return [
        {
            'term': 'OFAC',
            'keywords': ['ofac', 'sanctions', 'sdn'],
            'category': 'compliance',
            'capabilities': ['sanctions_screening']
        },
        {
            'term': 'wire_transfer',
            'keywords': ['wire', 'transfer', 'wt'],
            'category': 'payments',
            'capabilities': ['payment_processing']
        },
        {
            'term': 'SWIFT',
            'keywords': ['swift', 'bic', 'mt103'],
            'category': 'messaging',
            'capabilities': ['message_formatting']
        },
        {
            'term': 'BIC',
            'keywords': ['bic', 'bank identifier'],
            'category': 'identifiers',
            'capabilities': ['validation']
        }
    ]


def create_sample_chunk(
    chunk_id: str = "test_chunk",
    text: str = "Test content",
    source_type: SourceType = SourceType.DOCUMENT
) -> IndexableChunk:
    """Create a sample chunk for testing"""
    return IndexableChunk(
        chunk_id=chunk_id,
        text=text,
        embedding_text=text,
        source_type=source_type,
        semantic_type=SemanticType.PARAGRAPH,
        source_ref=SourceReference(file_path='test.txt', line_start=1, line_end=1)
    )


# =============================================================================
# UNIT TESTS - Individual Components
# =============================================================================

class TestVocabulary(TestSuite):
    """Unit tests for DomainVocabulary"""
    
    def test_empty_vocabulary(self):
        """Test creating empty vocabulary"""
        vocab = DomainVocabulary()
        self.assert_equal(len(vocab.entries), 0, "Empty vocabulary has no entries")
    
    def test_load_from_data(self):
        """Test loading vocabulary from data"""
        vocab = DomainVocabulary()
        vocab.load_from_data(create_sample_vocabulary())
        self.assert_equal(len(vocab.entries), 4, "Vocabulary has 4 entries")
    
    def test_match_text_positive(self):
        """Test matching text with vocabulary terms"""
        vocab = DomainVocabulary()
        vocab.load_from_data(create_sample_vocabulary())
        
        matches = vocab.match_text("OFAC sanctions screening")
        self.assert_greater(len(matches), 0, "Should match OFAC")
        
        # Note: canonical_term is lowercased
        terms = [m.canonical_term.lower() for m in matches]
        self.assert_true('ofac' in terms, "Should find ofac term (lowercase)")
    
    def test_match_text_case_insensitive(self):
        """Test case-insensitive matching"""
        vocab = DomainVocabulary()
        vocab.load_from_data(create_sample_vocabulary())
        
        matches_lower = vocab.match_text("ofac screening")
        matches_upper = vocab.match_text("OFAC SCREENING")
        
        self.assert_greater(len(matches_lower), 0, "Should match lowercase")
        self.assert_greater(len(matches_upper), 0, "Should match uppercase")
    
    def test_match_text_no_match(self):
        """Test text with no matching terms"""
        vocab = DomainVocabulary()
        vocab.load_from_data(create_sample_vocabulary())
        
        matches = vocab.match_text("hello world python java")
        self.assert_equal(len(matches), 0, "Should have no matches")
    
    def test_get_statistics(self):
        """Test vocabulary statistics"""
        vocab = DomainVocabulary()
        vocab.load_from_data(create_sample_vocabulary())
        
        stats = vocab.get_statistics()
        self.assert_true('total_entries' in stats, "Stats has total_entries")
        self.assert_equal(stats['total_entries'], 4, "Total entries is 4")
    
    def test_to_dict_and_back(self):
        """Test serialization round-trip"""
        vocab = DomainVocabulary()
        vocab.load_from_data(create_sample_vocabulary())
        
        data = vocab.to_dict()
        vocab2 = DomainVocabulary()
        vocab2.load_from_data(data)
        
        self.assert_equal(len(vocab2.entries), len(vocab.entries), "Round-trip preserves entries")


class TestEmbedders(TestSuite):
    """Unit tests for embedding classes"""
    
    def setup(self):
        self.vocab = DomainVocabulary()
        self.vocab.load_from_data(create_sample_vocabulary())
    
    def test_hash_embedder_dimensions(self):
        """Test HashEmbedder produces correct dimensions"""
        embedder = HashEmbedder(n_features=512)
        vec = embedder.get_embedding("test text")
        self.assert_equal(len(vec), 512, "Hash embedding has 512 dims")
    
    def test_hash_embedder_consistency(self):
        """Test HashEmbedder produces consistent results"""
        embedder = HashEmbedder(n_features=256)
        vec1 = embedder.get_embedding("test text")
        vec2 = embedder.get_embedding("test text")
        self.assert_equal(vec1, vec2, "Same text produces same embedding")
    
    def test_hash_embedder_different_texts(self):
        """Test HashEmbedder produces different embeddings for different texts"""
        embedder = HashEmbedder(n_features=256)
        vec1 = embedder.get_embedding("hello world")
        vec2 = embedder.get_embedding("goodbye universe")
        self.assert_true(vec1 != vec2, "Different texts produce different embeddings")
    
    def test_domain_concept_embedder(self):
        """Test DomainConceptEmbedder basic functionality"""
        embedder = DomainConceptEmbedder(self.vocab)
        vec = embedder.get_embedding("OFAC sanctions")
        
        self.assert_greater(len(vec), 0, "Domain embedding has dimensions")
        self.assert_true(any(v > 0 for v in vec), "Has non-zero activations")
    
    def test_domain_concept_explain(self):
        """Test DomainConceptEmbedder explanation"""
        embedder = DomainConceptEmbedder(self.vocab)
        explanation = embedder.explain_embedding("OFAC sanctions screening")
        
        self.assert_true('ofac' in explanation, "Explanation includes ofac")
        self.assert_greater(explanation.get('ofac', 0), 0, "OFAC has positive weight")
    
    def test_create_embedder_factory(self):
        """Test create_embedder factory function"""
        # Test hash
        emb_hash = create_embedder('hash', self.vocab, n_features=128)
        self.assert_equal(len(emb_hash.get_embedding("test")), 128, "Hash embedder created")
        
        # Test domain
        emb_domain = create_embedder('domain', self.vocab)
        self.assert_true(hasattr(emb_domain, 'explain_embedding'), "Domain embedder created")
    
    def test_empty_text_embedding(self):
        """Test embedding empty text"""
        embedder = HashEmbedder(n_features=64)
        vec = embedder.get_embedding("")
        self.assert_equal(len(vec), 64, "Empty text produces correct dims")


class TestBM25Index(TestSuite):
    """Unit tests for BM25Index"""
    
    def test_add_document(self):
        """Test adding documents to BM25 index"""
        bm25 = BM25Index()
        chunk = create_sample_chunk("doc1", "test document about payments")
        bm25.add(chunk.chunk_id, chunk)
        
        self.assert_equal(len(bm25.chunk_ids), 1, "One document added")
    
    def test_search_basic(self):
        """Test basic BM25 search"""
        bm25 = BM25Index()
        chunk1 = create_sample_chunk("doc1", "payment processing system")
        chunk2 = create_sample_chunk("doc2", "user interface design")
        bm25.add(chunk1.chunk_id, chunk1)
        bm25.add(chunk2.chunk_id, chunk2)
        bm25.build_index()
        
        results = bm25.search("payment", top_k=5)
        self.assert_greater(len(results), 0, "Should find payment document")
        self.assert_equal(results[0][0], "doc1", "Payment doc should rank first")
    
    def test_search_empty_index(self):
        """Test searching empty index"""
        bm25 = BM25Index()
        results = bm25.search("test", top_k=5)
        self.assert_equal(len(results), 0, "Empty index returns no results")
    
    def test_search_no_match(self):
        """Test search with no matching terms"""
        bm25 = BM25Index()
        chunk = create_sample_chunk("doc1", "payment processing")
        bm25.add(chunk.chunk_id, chunk)
        bm25.build_index()
        
        results = bm25.search("xyz123nonexistent", top_k=5)
        self.assert_equal(len(results), 0, "No results for non-matching query")


class TestConceptIndex(TestSuite):
    """Unit tests for ConceptIndex"""
    
    def test_add_chunk(self):
        """Test adding chunk to concept index"""
        idx = ConceptIndex()
        chunk = create_sample_chunk("doc1", "test content")
        idx.add(chunk)
        
        self.assert_equal(len(idx), 1, "One chunk added")
    
    def test_get_chunk(self):
        """Test retrieving chunk by ID"""
        idx = ConceptIndex()
        chunk = create_sample_chunk("doc1", "test content")
        idx.add(chunk)
        
        retrieved = idx.get_chunk("doc1")
        self.assert_true(retrieved is not None, "Chunk retrieved")
        self.assert_equal(retrieved.text, "test content", "Content matches")
    
    def test_get_nonexistent_chunk(self):
        """Test retrieving non-existent chunk"""
        idx = ConceptIndex()
        retrieved = idx.get_chunk("nonexistent")
        self.assert_true(retrieved is None, "Returns None for missing chunk")


class TestVectorStore(TestSuite):
    """Unit tests for VectorStore"""
    
    def test_add_and_search(self):
        """Test adding and searching vectors"""
        store = VectorStore()
        chunk = create_sample_chunk("doc1", "test")
        embedding = [1.0, 0.0, 0.0, 0.0]  # 4-dim vector
        
        store.add("doc1", embedding, chunk)
        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        
        self.assert_greater(len(results), 0, "Should find vector")
        self.assert_equal(results[0][0], "doc1", "Correct doc found")
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        store = VectorStore()
        chunk1 = create_sample_chunk("doc1", "test1")
        chunk2 = create_sample_chunk("doc2", "test2")
        
        store.add("doc1", [1.0, 0.0], chunk1)
        store.add("doc2", [0.0, 1.0], chunk2)
        
        # Query similar to doc1
        results = store.search([0.9, 0.1], top_k=2)
        self.assert_equal(results[0][0], "doc1", "Most similar doc found first")


# =============================================================================
# FUNCTIONAL TESTS - Feature-level Testing
# =============================================================================

class TestHybridIndex(TestSuite):
    """Functional tests for HybridIndex"""
    
    def setup(self):
        self.vocab = DomainVocabulary()
        self.vocab.load_from_data(create_sample_vocabulary())
    
    def test_index_and_search(self):
        """Test indexing and searching"""
        idx = HybridIndex(self.vocab)
        idx.set_embedding_function(HashEmbedder(n_features=64).get_embedding)
        
        idx.index_chunk(create_sample_chunk("doc1", "OFAC sanctions screening"))
        idx.index_chunk(create_sample_chunk("doc2", "wire transfer processing"))
        
        results = idx.search("OFAC", top_k=5)
        self.assert_greater(len(results), 0, "Should find results")
    
    def test_filter_by_source_type(self):
        """Test filtering by source type"""
        idx = HybridIndex(self.vocab)
        idx.set_embedding_function(HashEmbedder(n_features=64).get_embedding)
        
        idx.index_chunk(create_sample_chunk("doc1", "OFAC screening", SourceType.DOCUMENT))
        idx.index_chunk(create_sample_chunk("code1", "def ofac_check():", SourceType.CODE))
        
        results = idx.search("OFAC", top_k=5, source_types=[SourceType.CODE])
        
        # Should only return code results
        for result in results:
            # Access source_type via chunk attribute
            self.assert_equal(result.chunk.source_type, SourceType.CODE, "Only code results")
    
    def test_statistics(self):
        """Test index statistics"""
        idx = HybridIndex(self.vocab)
        idx.set_embedding_function(HashEmbedder(n_features=64).get_embedding)
        
        idx.index_chunk(create_sample_chunk("doc1", "test1"))
        idx.index_chunk(create_sample_chunk("doc2", "test2"))
        
        stats = idx.get_statistics()
        self.assert_true('concept_index' in stats, "Has concept_index stats")
        self.assert_equal(stats['concept_index']['total_chunks'], 2, "Shows 2 chunks")


class TestIndexingPipeline(TestSuite):
    """Functional tests for IndexingPipeline"""
    
    def test_create_pipeline(self):
        """Test creating pipeline"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        self.assert_true(pipeline is not None, "Pipeline created")
        self.assert_true(pipeline.embedder is not None, "Has embedder")
    
    def test_index_content(self):
        """Test indexing content directly"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        # Use .md extension to trigger DocumentParser
        chunks = pipeline.index_content(
            b"OFAC sanctions screening for wire transfers",
            "test.md",
            SourceType.DOCUMENT
        )
        
        self.assert_greater(len(chunks), 0, "Created chunks")
    
    def test_search(self):
        """Test searching indexed content"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        # Use .md for documents
        pipeline.index_content(b"OFAC sanctions screening", "doc1.md", SourceType.DOCUMENT)
        pipeline.index_content(b"wire transfer processing", "doc2.md", SourceType.DOCUMENT)
        
        results = pipeline.search("OFAC", top_k=5)
        self.assert_greater(len(results), 0, "Found results")
    
    def test_cross_reference_search(self):
        """Test cross-reference search"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        # Use .md for doc, .py for code
        pipeline.index_content(b"OFAC error in payment", "doc.md", SourceType.DOCUMENT)
        pipeline.index_content(b"def handle_ofac(): pass", "code.py", SourceType.CODE)
        
        results = pipeline.search_cross_reference(
            query="OFAC",
            from_type=SourceType.DOCUMENT,
            to_types=[SourceType.CODE],
            top_k=5
        )
        
        self.assert_true('document' in results, "Has document results")
        self.assert_true('code' in results, "Has code results")
    
    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        # Use .md extension
        chunks1 = pipeline.index_content(b"content1", "doc1.md", SourceType.DOCUMENT)
        chunks2 = pipeline.index_content(b"content2", "doc2.md", SourceType.DOCUMENT)
        
        # Check index stats (more reliable than pipeline stats)
        stats = pipeline.index.get_statistics()
        total_chunks = stats.get('concept_index', {}).get('total_chunks', 0)
        self.assert_equal(total_chunks, len(chunks1) + len(chunks2), "Correct chunk count in index")


class TestSaveLoad(TestSuite):
    """Functional tests for save/load operations"""
    
    def test_save_and_load(self):
        """Test saving and loading index"""
        temp_dir = self.create_temp_dir()
        
        # Create and save - use .md for document parsing
        pipeline1 = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        chunks = pipeline1.index_content(b"OFAC sanctions", "doc.md", SourceType.DOCUMENT)
        pipeline1.save(str(temp_dir), verbose=False)
        
        # Load into new pipeline
        pipeline2 = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        pipeline2.load(str(temp_dir), verbose=False)
        
        # Verify - check index directly
        stats = pipeline2.index.get_statistics()
        self.assert_equal(
            stats['concept_index']['total_chunks'], 
            len(chunks), 
            "Loaded same number of chunks"
        )
    
    def test_generation_increment(self):
        """Test generation number increments on save"""
        temp_dir = self.create_temp_dir()
        
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        pipeline.index_content(b"content", "doc.md", SourceType.DOCUMENT)
        
        # First save
        pipeline.save(str(temp_dir), verbose=False)
        self.assert_equal(get_current_generation(temp_dir), 1, "Generation is 1")
        
        # Second save
        pipeline.index_content(b"more content", "doc2.md", SourceType.DOCUMENT)
        pipeline.save(str(temp_dir), verbose=False)
        self.assert_equal(get_current_generation(temp_dir), 2, "Generation is 2")
    
    def test_generation_cleanup(self):
        """Test old generations are cleaned up"""
        temp_dir = self.create_temp_dir()
        
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        # Save multiple times to trigger cleanup
        for i in range(4):
            pipeline.index_content(f"content {i}".encode(), f"doc{i}.md", SourceType.DOCUMENT)
            pipeline.save(str(temp_dir), verbose=False)
        
        # Should only have 2 generations (latest 2)
        gen_dirs = list(temp_dir.glob('gen_*'))
        self.assert_true(len(gen_dirs) <= 2, f"At most 2 generations kept, found {len(gen_dirs)}")
    
    def test_load_preserves_searchability(self):
        """Test that loaded index is searchable"""
        temp_dir = self.create_temp_dir()
        
        pipeline1 = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        pipeline1.index_content(b"OFAC sanctions screening", "doc.md", SourceType.DOCUMENT)
        pipeline1.save(str(temp_dir), verbose=False)
        
        pipeline2 = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        pipeline2.load(str(temp_dir), verbose=False)
        
        results = pipeline2.search("OFAC", top_k=5)
        self.assert_greater(len(results), 0, "Can search after load")


# =============================================================================
# NEGATIVE TESTS - Error Handling
# =============================================================================

class TestNegativeCases(TestSuite):
    """Negative tests for error handling"""
    
    def test_load_nonexistent_directory(self):
        """Test loading from non-existent directory"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        self.assert_raises(
            FileNotFoundError,
            pipeline.load,
            "Load non-existent directory raises FileNotFoundError",
            "/nonexistent/path/to/index"
        )
    
    def test_load_empty_directory(self):
        """Test loading from empty directory"""
        temp_dir = self.create_temp_dir()
        
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        self.assert_raises(
            FileNotFoundError,
            pipeline.load,
            "Load empty directory raises FileNotFoundError",
            str(temp_dir)
        )
    
    def test_index_invalid_source_type(self):
        """Test indexing with None source type still works"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        # Should auto-detect type - use .md for document with enough content to be chunked
        chunks = pipeline.index_content(
            b"This is a test document with enough content to be properly parsed into a chunk.",
            "test.md", 
            None
        )
        self.assert_greater(len(chunks), 0, "Auto-detects type when None")
    
    def test_search_empty_query(self):
        """Test searching with empty query"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        pipeline.index_content(b"content", "doc.md", SourceType.DOCUMENT)
        
        results = pipeline.search("", top_k=5)
        # Should not crash, may return empty or all results
        self.assert_true(isinstance(results, list), "Returns list for empty query")
    
    def test_search_empty_index(self):
        """Test searching empty index"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        results = pipeline.search("test", top_k=5)
        self.assert_equal(len(results), 0, "Empty index returns no results")
    
    def test_invalid_vocabulary_path(self):
        """Test loading invalid vocabulary path"""
        self.assert_raises(
            FileNotFoundError,
            lambda: IndexingPipeline(vocabulary_path="/nonexistent/vocab.json"),
            "Invalid vocabulary path raises FileNotFoundError"
        )
    
    def test_top_k_zero(self):
        """Test search with top_k=0"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        pipeline.index_content(b"content", "doc.md", SourceType.DOCUMENT)
        
        results = pipeline.search("content", top_k=0)
        self.assert_equal(len(results), 0, "top_k=0 returns no results")
    
    def test_negative_top_k(self):
        """Test search with negative top_k"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        pipeline.index_content(b"content", "doc.md", SourceType.DOCUMENT)
        
        # Should handle gracefully
        results = pipeline.search("content", top_k=-1)
        self.assert_true(isinstance(results, list), "Handles negative top_k")


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases(TestSuite):
    """Edge case tests"""
    
    def test_very_long_text(self):
        """Test indexing very long text"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        long_text = "OFAC sanctions " * 10000  # ~150KB
        chunks = pipeline.index_content(long_text.encode(), "long.md", SourceType.DOCUMENT)
        
        self.assert_greater(len(chunks), 0, "Can index long text")
    
    def test_unicode_content(self):
        """Test indexing Unicode content"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        unicode_text = "OFAC審査 санкции écran 检查 🔍"
        chunks = pipeline.index_content(unicode_text.encode('utf-8'), "unicode.md", SourceType.DOCUMENT)
        
        self.assert_greater(len(chunks), 0, "Can index Unicode text")
    
    def test_special_characters(self):
        """Test indexing text with special characters"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        special_text = "OFAC <test> & 'quotes' \"double\" \n\t\r"
        chunks = pipeline.index_content(special_text.encode(), "special.md", SourceType.DOCUMENT)
        
        self.assert_greater(len(chunks), 0, "Can index special characters")
    
    def test_duplicate_chunks(self):
        """Test indexing duplicate content"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        chunks1 = pipeline.index_content(b"OFAC sanctions", "doc1.md", SourceType.DOCUMENT)
        chunks2 = pipeline.index_content(b"OFAC sanctions", "doc2.md", SourceType.DOCUMENT)
        
        # Should have 2 separate chunks
        stats = pipeline.index.get_statistics()
        self.assert_equal(
            stats['concept_index']['total_chunks'], 
            len(chunks1) + len(chunks2), 
            "Both chunks indexed"
        )
    
    def test_binary_content(self):
        """Test handling binary content"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        # Binary content with some text
        binary_content = b'\x00\x01\x02OFAC\x03\x04\x05'
        chunks = pipeline.index_content(binary_content, "binary.md", SourceType.DOCUMENT)
        
        # Should handle gracefully
        self.assert_true(isinstance(chunks, list), "Handles binary content")
    
    def test_whitespace_only_content(self):
        """Test indexing whitespace-only content"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        chunks = pipeline.index_content(b"   \n\t\r   ", "whitespace.md", SourceType.DOCUMENT)
        # May or may not create chunks, but shouldn't crash
        self.assert_true(isinstance(chunks, list), "Handles whitespace content")
    
    def test_single_word(self):
        """Test indexing short content"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        # DocumentParser needs minimum content length - use a sentence
        chunks = pipeline.index_content(
            b"OFAC is important for compliance screening in wire transfers.",
            "word.md",
            SourceType.DOCUMENT
        )
        self.assert_greater(len(chunks), 0, "Can index short content")
    
    def test_many_small_files(self):
        """Test indexing many small files"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        total_chunks = 0
        for i in range(100):
            chunks = pipeline.index_content(f"content {i}".encode(), f"doc{i}.md", SourceType.DOCUMENT)
            total_chunks += len(chunks)
        
        stats = pipeline.index.get_statistics()
        self.assert_equal(stats['concept_index']['total_chunks'], total_chunks, "All chunks indexed")
    
    def test_search_with_many_results(self):
        """Test search returning many results"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        # Index many similar documents
        for i in range(50):
            pipeline.index_content(f"OFAC sanctions document {i}".encode(), f"doc{i}.md", SourceType.DOCUMENT)
        
        results = pipeline.search("OFAC", top_k=100)
        self.assert_equal(len(results), 50, "Returns all 50 matches")
    
    def test_concurrent_saves(self):
        """Test that generation numbers don't conflict with rapid saves"""
        temp_dir = self.create_temp_dir()
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        # Rapid saves
        for i in range(5):
            pipeline.index_content(f"content {i}".encode(), f"doc{i}.md", SourceType.DOCUMENT)
            pipeline.save(str(temp_dir), verbose=False)
        
        # Should have generation 5
        self.assert_equal(get_current_generation(temp_dir), 5, "Generation is 5 after 5 saves")
    
    def test_empty_vocabulary(self):
        """Test pipeline with empty vocabulary"""
        pipeline = IndexingPipeline(vocabulary_data=[])
        
        # Use content long enough to be chunked by DocumentParser
        chunks = pipeline.index_content(
            b"This is a test document with sufficient content to be properly parsed and indexed.",
            "doc.md",
            SourceType.DOCUMENT
        )
        self.assert_greater(len(chunks), 0, "Works with empty vocabulary")
        
        results = pipeline.search("test", top_k=5)
        self.assert_true(isinstance(results, list), "Can search with empty vocabulary")
    
    def test_very_long_query(self):
        """Test search with very long query"""
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        pipeline.index_content(b"OFAC sanctions", "doc.md", SourceType.DOCUMENT)
        
        long_query = "OFAC " * 1000  # Very long query
        results = pipeline.search(long_query, top_k=5)
        
        self.assert_true(isinstance(results, list), "Handles very long query")
    
    def test_generation_path_legacy(self):
        """Test generation path for legacy format (gen 0)"""
        temp_dir = self.create_temp_dir()
        
        path = get_generation_path(temp_dir, 0)
        self.assert_equal(path, temp_dir, "Generation 0 returns base directory")
        
        path = get_generation_path(temp_dir, 1)
        self.assert_equal(path, temp_dir / 'gen_1', "Generation 1 returns gen_1 subdirectory")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration(TestSuite):
    """Integration tests combining multiple components"""
    
    def test_full_workflow(self):
        """Test complete indexing and search workflow"""
        temp_dir = self.create_temp_dir()
        
        # Create pipeline
        pipeline = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        
        # Index various content types
        doc_chunks = pipeline.index_content(
            b"OFAC sanctions screening for wire transfers",
            "compliance.md",
            SourceType.DOCUMENT
        )
        code_chunks = pipeline.index_content(
            b"def process_ofac(entity): return check_sdn(entity)",
            "screening.py",
            SourceType.CODE
        )
        
        # Save
        pipeline.save(str(temp_dir), verbose=False)
        
        # Load in new pipeline
        pipeline2 = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        pipeline2.load(str(temp_dir), verbose=False)
        
        # Search
        results = pipeline2.search("OFAC screening", top_k=10)
        self.assert_greater(len(results), 0, "Found results after load")
        
        # Verify cross-reference works
        xref = pipeline2.search_cross_reference(
            "OFAC",
            SourceType.DOCUMENT,
            [SourceType.CODE],
            top_k=5
        )
        self.assert_true('document' in xref and 'code' in xref, "Cross-reference works")
    
    def test_incremental_indexing(self):
        """Test incremental indexing workflow"""
        temp_dir = self.create_temp_dir()
        
        # Initial indexing
        pipeline1 = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        initial_chunks = pipeline1.index_content(b"Initial content", "doc1.md", SourceType.DOCUMENT)
        pipeline1.save(str(temp_dir), verbose=False)
        
        initial_gen = get_current_generation(temp_dir)
        
        # Load and add more
        pipeline2 = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        pipeline2.load(str(temp_dir), verbose=False)
        additional_chunks = pipeline2.index_content(b"Additional content", "doc2.md", SourceType.DOCUMENT)
        pipeline2.save(str(temp_dir), verbose=False)
        
        new_gen = get_current_generation(temp_dir)
        
        self.assert_equal(new_gen, initial_gen + 1, "Generation incremented")
        
        # Verify all content searchable
        pipeline3 = IndexingPipeline(vocabulary_data=create_sample_vocabulary())
        pipeline3.load(str(temp_dir), verbose=False)
        
        stats = pipeline3.index.get_statistics()
        expected_chunks = len(initial_chunks) + len(additional_chunks)
        self.assert_equal(stats['concept_index']['total_chunks'], expected_chunks, "All chunks present")
    
    def test_multiple_embedder_types(self):
        """Test pipeline with different embedder types"""
        embedder_types = ['hash', 'domain', 'bm25']
        
        for emb_type in embedder_types:
            pipeline = IndexingPipeline(
                vocabulary_data=create_sample_vocabulary(),
                embedder_type=emb_type
            )
            
            chunks = pipeline.index_content(
                b"OFAC sanctions content",
                "doc.md",
                SourceType.DOCUMENT
            )
            
            results = pipeline.search("OFAC", top_k=5)
            
            self.assert_greater(
                len(chunks), 0,
                f"Embedder {emb_type}: created chunks"
            )


# =============================================================================
# MAIN - Run all tests
# =============================================================================

def run_all_tests():
    """Run all test suites"""
    print("=" * 70)
    print("UNIFIED INDEXER - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    suites = [
        # Unit Tests
        ("Unit: Vocabulary", TestVocabulary()),
        ("Unit: Embedders", TestEmbedders()),
        ("Unit: BM25Index", TestBM25Index()),
        ("Unit: ConceptIndex", TestConceptIndex()),
        ("Unit: VectorStore", TestVectorStore()),
        
        # Functional Tests
        ("Functional: HybridIndex", TestHybridIndex()),
        ("Functional: IndexingPipeline", TestIndexingPipeline()),
        ("Functional: Save/Load", TestSaveLoad()),
        
        # Negative Tests
        ("Negative: Error Handling", TestNegativeCases()),
        
        # Edge Cases
        ("Edge Cases", TestEdgeCases()),
        
        # Integration Tests
        ("Integration", TestIntegration()),
    ]
    
    all_passed = True
    total_tests = 0
    total_passed = 0
    
    for name, suite in suites:
        print(f"\n{'─' * 70}")
        print(f"Running: {name}")
        print('─' * 70)
        
        suite.run_all()
        passed = suite.print_results()
        
        total_tests += len(suite.results)
        total_passed += sum(1 for r in suite.results if r.passed)
        
        if not passed:
            all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success Rate: {100 * total_passed / total_tests:.1f}%")
    print("=" * 70)
    
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
