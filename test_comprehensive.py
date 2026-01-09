#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Indexer
=============================================

Tests cover:
- Positive cases: Normal expected behavior
- Negative cases: Error handling, invalid inputs
- Edge cases: Boundary conditions, unusual inputs

Run with: python -m pytest test_comprehensive.py -v
Or:       python test_comprehensive.py
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct imports (avoid circular dependency)
from unified_indexer.pipeline import IndexingPipeline
from unified_indexer.models import SourceType, IndexableChunk, SearchResult
from unified_indexer.vocabulary import DomainVocabulary
from unified_indexer.index import HybridIndex
from unified_indexer.embeddings import (
    create_embedder, HashEmbedder, TFIDFEmbedder, 
    BM25Embedder, DomainConceptEmbedder
)
from unified_indexer.parsers.tal_parser import TalCodeParser
from unified_indexer.parsers.document_parser import DocumentParser
from unified_indexer.parsers.code_parser import GenericCodeParser
from unified_indexer.parsers.log_parser import LogParser


# =============================================================================
# TEST FIXTURES
# =============================================================================

SAMPLE_VOCABULARY = [
    {"keywords": "wire,wire transfer,wires", "business_capability": "payment_transfer", "description": "Wire transfer processing"},
    {"keywords": "OFAC,sanctions,SDN", "business_capability": "compliance", "description": "OFAC sanctions screening"},
    {"keywords": "BIC,SWIFT,routing", "business_capability": "routing", "description": "Bank routing codes"},
    {"keywords": "validation,validate,validator", "business_capability": "validation", "description": "Input validation"},
    {"keywords": "error,exception,failure", "business_capability": "error_handling", "description": "Error handling"},
]

SAMPLE_TAL_CODE = """
-- Wire Transfer Validation Procedure
PROC validate_wire_transfer(wire_data);
BEGIN
    -- Check OFAC sanctions list
    IF check_ofac_screening(wire_data.beneficiary) THEN
        RETURN error_blocked;
    END;
    
    -- Validate BIC code
    IF NOT validate_bic(wire_data.bic_code) THEN
        RETURN error_invalid_bic;
    END;
    
    RETURN success;
END;

PROC check_ofac_screening(name);
BEGIN
    -- SDN list lookup
    CALL sdn_lookup(name);
END;
"""

SAMPLE_PYTHON_CODE = """
def process_payment(payment_data):
    \"\"\"Process wire transfer payment.\"\"\"
    # Validate input
    if not validate_payment(payment_data):
        raise ValidationError("Invalid payment data")
    
    # Check sanctions
    if check_ofac(payment_data.beneficiary):
        raise ComplianceError("OFAC blocked")
    
    # Route payment
    routing_code = get_swift_routing(payment_data.bic)
    return execute_transfer(routing_code, payment_data)
"""

SAMPLE_LOG_ENTRIES = """
{"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "message": "Processing wire transfer", "transaction_id": "TXN001"}
{"timestamp": "2024-01-15T10:30:01Z", "level": "ERROR", "message": "OFAC screening failed", "transaction_id": "TXN001", "error_code": "BLOCKED"}
{"timestamp": "2024-01-15T10:31:00Z", "level": "INFO", "message": "Wire transfer completed", "transaction_id": "TXN002"}
"""

SAMPLE_KNOWLEDGE_GRAPH = {
    "nodes": [
        {"id": "ofac", "label": "OFAC", "type": "concept", "tf_idf_score": 2.5},
        {"id": "wire_transfer", "label": "WIRE_TRANSFER", "type": "concept", "tf_idf_score": 1.8},
        {"id": "sanctions", "label": "SANCTIONS", "type": "concept", "tf_idf_score": 2.2},
        {"id": "bic", "label": "BIC", "type": "concept", "tf_idf_score": 1.5},
    ],
    "edges": [
        {"source": "ofac", "target": "sanctions", "type": "co_occurs_with"},
        {"source": "wire_transfer", "target": "bic", "type": "related_to"},
    ]
}


class TestResult:
    """Simple test result tracker."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def ok(self, name):
        self.passed += 1
        print(f"  ✓ {name}")
    
    def fail(self, name, reason):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  ✗ {name}: {reason}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailures:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason}")
        print(f"{'='*60}")
        return self.failed == 0


# =============================================================================
# POSITIVE TESTS - Normal Expected Behavior
# =============================================================================

def test_positive_cases(results: TestResult):
    """Test normal expected behavior."""
    print("\n" + "="*60)
    print("POSITIVE TESTS - Normal Expected Behavior")
    print("="*60)
    
    # --- Test: Pipeline Creation ---
    try:
        pipeline = IndexingPipeline(
            vocabulary_data=SAMPLE_VOCABULARY,
            embedder_type="hash"
        )
        assert pipeline is not None
        assert pipeline.vocabulary is not None
        results.ok("Pipeline creation with vocabulary")
    except Exception as e:
        results.fail("Pipeline creation with vocabulary", str(e))
    
    # --- Test: Index TAL Code ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        chunks = pipeline.index_content(
            SAMPLE_TAL_CODE.encode('utf-8'),
            "test.tal",
            SourceType.CODE
        )
        assert len(chunks) > 0
        results.ok(f"Index TAL code ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Index TAL code", str(e))
    
    # --- Test: Index Python Code ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        chunks = pipeline.index_content(
            SAMPLE_PYTHON_CODE.encode('utf-8'),
            "test.py",
            SourceType.CODE
        )
        assert len(chunks) > 0
        results.ok(f"Index Python code ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Index Python code", str(e))
    
    # --- Test: Index Log Entries ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        chunks = pipeline.index_content(
            SAMPLE_LOG_ENTRIES.encode('utf-8'),
            "test.log",
            SourceType.LOG
        )
        assert len(chunks) > 0
        results.ok(f"Index log entries ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Index log entries", str(e))
    
    # --- Test: Search Returns Results ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        results_list = pipeline.search("wire transfer", top_k=5)
        assert len(results_list) > 0
        results.ok(f"Search returns results ({len(results_list)} found)")
    except Exception as e:
        results.fail("Search returns results", str(e))
    
    # --- Test: Search Finds Domain Concepts ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        results_list = pipeline.search("OFAC", top_k=5)
        assert len(results_list) > 0
        # Check that OFAC-related content is found
        found_ofac = any("ofac" in r.chunk.text.lower() or "sanction" in r.chunk.text.lower() 
                        for r in results_list)
        assert found_ofac, "Should find OFAC-related content"
        results.ok("Search finds domain concepts (OFAC)")
    except Exception as e:
        results.fail("Search finds domain concepts", str(e))
    
    # --- Test: Save and Load Index ---
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
            pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
            
            # Save
            pipeline.save(tmpdir)
            
            # Load into new pipeline
            pipeline2 = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
            pipeline2.load(tmpdir)
            
            # Verify search works
            results_list = pipeline2.search("wire", top_k=5)
            assert len(results_list) > 0
            results.ok("Save and load index")
    except Exception as e:
        results.fail("Save and load index", str(e))
    
    # --- Test: Multiple Embedder Types ---
    for embedder_type in ["hash", "tfidf", "bm25", "domain"]:
        try:
            pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type=embedder_type)
            chunks = pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
            assert len(chunks) > 0
            results.ok(f"Embedder type: {embedder_type}")
        except Exception as e:
            results.fail(f"Embedder type: {embedder_type}", str(e))
    
    # --- Test: Search with Type Filter ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        pipeline.index_content(SAMPLE_LOG_ENTRIES.encode('utf-8'), "test.log", SourceType.LOG)
        
        # Search only code
        code_results = pipeline.search("wire", top_k=5, source_types=[SourceType.CODE])
        # Search only logs
        log_results = pipeline.search("wire", top_k=5, source_types=[SourceType.LOG])
        
        results.ok(f"Search with type filter (code:{len(code_results)}, log:{len(log_results)})")
    except Exception as e:
        results.fail("Search with type filter", str(e))
    
    # --- Test: Vocabulary Matching ---
    try:
        vocab = DomainVocabulary()
        vocab.load_from_data(SAMPLE_VOCABULARY)
        matches = vocab.match_text("Check OFAC sanctions screening")
        assert len(matches) > 0
        assert any("compliance" in m.capabilities for m in matches)
        results.ok(f"Vocabulary matching ({len(matches)} matches)")
    except Exception as e:
        results.fail("Vocabulary matching", str(e))
    
    # --- Test: Statistics ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        pipeline.index_content(SAMPLE_PYTHON_CODE.encode('utf-8'), "test.py", SourceType.CODE)
        
        stats = pipeline.get_statistics()
        assert stats['pipeline']['total_chunks'] > 0
        assert stats['pipeline']['files_processed'] == 2
        results.ok(f"Statistics ({stats['pipeline']['total_chunks']} chunks, {stats['pipeline']['files_processed']} files)")
    except Exception as e:
        results.fail("Statistics", str(e))


# =============================================================================
# NEGATIVE TESTS - Error Handling
# =============================================================================

def test_negative_cases(results: TestResult):
    """Test error handling and invalid inputs."""
    print("\n" + "="*60)
    print("NEGATIVE TESTS - Error Handling")
    print("="*60)
    
    # --- Test: Empty Vocabulary ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=[], embedder_type="hash")
        # Should still work, just with no domain matching
        chunks = pipeline.index_content(b"some content", "test.txt", SourceType.DOCUMENT)
        results.ok("Empty vocabulary (graceful handling)")
    except Exception as e:
        results.fail("Empty vocabulary", str(e))
    
    # --- Test: Invalid Embedder Type ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="invalid_type")
        results.fail("Invalid embedder type", "Should have raised exception")
    except (ValueError, KeyError) as e:
        results.ok("Invalid embedder type (raises exception)")
    except Exception as e:
        results.fail("Invalid embedder type", f"Wrong exception type: {type(e)}")
    
    # --- Test: Empty Content ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        chunks = pipeline.index_content(b"", "empty.txt", SourceType.DOCUMENT)
        # Should return empty list, not crash
        assert isinstance(chunks, list)
        results.ok(f"Empty content (returns {len(chunks)} chunks)")
    except Exception as e:
        results.fail("Empty content", str(e))
    
    # --- Test: Binary/Invalid UTF-8 Content ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        binary_content = bytes([0x80, 0x81, 0x82, 0xFF, 0xFE])
        chunks = pipeline.index_content(binary_content, "binary.bin", SourceType.DOCUMENT)
        # Should handle gracefully
        results.ok("Binary content (graceful handling)")
    except Exception as e:
        results.fail("Binary content", str(e))
    
    # --- Test: Search Empty Index ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        results_list = pipeline.search("anything", top_k=5)
        assert isinstance(results_list, list)
        assert len(results_list) == 0
        results.ok("Search empty index (returns empty list)")
    except Exception as e:
        results.fail("Search empty index", str(e))
    
    # --- Test: Search with Empty Query ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        results_list = pipeline.search("", top_k=5)
        # Should handle gracefully
        results.ok(f"Empty query (returns {len(results_list)} results)")
    except Exception as e:
        results.fail("Empty query", str(e))
    
    # --- Test: Load Non-existent Index ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.load("/non/existent/path/to/index")
        results.fail("Load non-existent index", "Should have raised exception")
    except (FileNotFoundError, OSError, Exception) as e:
        results.ok("Load non-existent index (raises exception)")
    
    # --- Test: Invalid Vocabulary Format ---
    try:
        invalid_vocab = [{"wrong_key": "value"}]  # Missing 'keywords'
        pipeline = IndexingPipeline(vocabulary_data=invalid_vocab, embedder_type="hash")
        # Should handle gracefully (skip invalid entries)
        results.ok("Invalid vocabulary format (graceful handling)")
    except Exception as e:
        results.fail("Invalid vocabulary format", str(e))
    
    # --- Test: Very Large top_k ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        results_list = pipeline.search("wire", top_k=10000)
        # Should return all available, not crash
        results.ok(f"Large top_k (returns {len(results_list)} results)")
    except Exception as e:
        results.fail("Large top_k", str(e))
    
    # --- Test: Negative top_k ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        results_list = pipeline.search("wire", top_k=-1)
        # Should handle gracefully
        results.ok(f"Negative top_k (handled gracefully)")
    except Exception as e:
        # Also acceptable to raise exception
        results.ok(f"Negative top_k (raises exception)")


# =============================================================================
# EDGE CASES - Boundary Conditions
# =============================================================================

def test_edge_cases(results: TestResult):
    """Test boundary conditions and unusual inputs."""
    print("\n" + "="*60)
    print("EDGE CASES - Boundary Conditions")
    print("="*60)
    
    # --- Test: Single Character Content ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        chunks = pipeline.index_content(b"X", "single.txt", SourceType.DOCUMENT)
        results.ok(f"Single character content ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Single character content", str(e))
    
    # --- Test: Very Long Single Line ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        long_line = ("wire transfer OFAC screening " * 1000).encode('utf-8')
        chunks = pipeline.index_content(long_line, "long.txt", SourceType.DOCUMENT)
        assert len(chunks) > 0
        results.ok(f"Very long single line ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Very long single line", str(e))
    
    # --- Test: Unicode Content ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        unicode_content = "Wire transfer 电汇 électronique überweisen 送金".encode('utf-8')
        chunks = pipeline.index_content(unicode_content, "unicode.txt", SourceType.DOCUMENT)
        results.ok(f"Unicode content ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Unicode content", str(e))
    
    # --- Test: Special Characters in Query ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        results_list = pipeline.search("wire && transfer || OFAC", top_k=5)
        results.ok(f"Special characters in query ({len(results_list)} results)")
    except Exception as e:
        results.fail("Special characters in query", str(e))
    
    # --- Test: Query with Only Stopwords ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        results_list = pipeline.search("the a an is are", top_k=5)
        results.ok(f"Stopwords-only query ({len(results_list)} results)")
    except Exception as e:
        results.fail("Stopwords-only query", str(e))
    
    # --- Test: Exact top_k Match ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        results_list = pipeline.search("wire", top_k=1)
        assert len(results_list) <= 1
        results.ok(f"Exact top_k=1 ({len(results_list)} result)")
    except Exception as e:
        results.fail("Exact top_k match", str(e))
    
    # --- Test: Duplicate Content ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        # Index same content twice with different names
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test1.tal", SourceType.CODE)
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test2.tal", SourceType.CODE)
        stats = pipeline.get_statistics()
        assert stats['pipeline']['files_processed'] == 2
        results.ok(f"Duplicate content ({stats['pipeline']['files_processed']} files)")
    except Exception as e:
        results.fail("Duplicate content", str(e))
    
    # --- Test: Case Sensitivity ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        results_upper = pipeline.search("WIRE TRANSFER", top_k=5)
        results_lower = pipeline.search("wire transfer", top_k=5)
        results_mixed = pipeline.search("Wire Transfer", top_k=5)
        results.ok(f"Case sensitivity (upper:{len(results_upper)}, lower:{len(results_lower)}, mixed:{len(results_mixed)})")
    except Exception as e:
        results.fail("Case sensitivity", str(e))
    
    # --- Test: Whitespace Variations ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        results1 = pipeline.search("wire transfer", top_k=5)
        results2 = pipeline.search("wire  transfer", top_k=5)  # double space
        results3 = pipeline.search(" wire transfer ", top_k=5)  # leading/trailing
        results.ok(f"Whitespace variations ({len(results1)}, {len(results2)}, {len(results3)} results)")
    except Exception as e:
        results.fail("Whitespace variations", str(e))
    
    # --- Test: Very Short File Name ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        # Use .tal extension so parser is found
        chunks = pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "x.tal", SourceType.CODE)
        assert len(chunks) > 0
        results.ok(f"Very short filename ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Very short filename", str(e))
    
    # --- Test: Long File Path ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        long_path = "/very/long/path/" + "subdir/" * 50 + "file.tal"
        chunks = pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), long_path, SourceType.CODE)
        assert len(chunks) > 0
        results.ok(f"Long file path ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Long file path", str(e))
    
    # --- Test: Vocabulary with Single Entry ---
    try:
        single_vocab = [{"keywords": "test", "capability": "testing"}]
        pipeline = IndexingPipeline(vocabulary_data=single_vocab, embedder_type="hash")
        chunks = pipeline.index_content(b"This is a test document", "test.txt", SourceType.DOCUMENT)
        results.ok(f"Single vocabulary entry ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Single vocabulary entry", str(e))
    
    # --- Test: Vocabulary with Many Keywords ---
    try:
        many_keywords = {"keywords": ",".join([f"keyword{i}" for i in range(100)]), "capability": "test"}
        pipeline = IndexingPipeline(vocabulary_data=[many_keywords], embedder_type="hash")
        chunks = pipeline.index_content(b"keyword50 keyword51", "test.txt", SourceType.DOCUMENT)
        results.ok(f"Many keywords in vocab ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Many keywords in vocab", str(e))
    
    # --- Test: Content with Only Newlines ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        chunks = pipeline.index_content(b"\n\n\n\n\n", "newlines.txt", SourceType.DOCUMENT)
        results.ok(f"Only newlines ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Only newlines", str(e))
    
    # --- Test: Mixed Source Types in Same Index ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "code.tal", SourceType.CODE)
        pipeline.index_content(b"Wire transfer documentation about OFAC", "doc.txt", SourceType.DOCUMENT)
        pipeline.index_content(SAMPLE_LOG_ENTRIES.encode('utf-8'), "app.log", SourceType.LOG)
        
        stats = pipeline.get_statistics()
        results.ok(f"Mixed source types ({stats['pipeline']['files_processed']} files, {stats['pipeline']['total_chunks']} chunks)")
    except Exception as e:
        results.fail("Mixed source types", str(e))
    
    # --- Test: Re-indexing Same File ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        stats1 = pipeline.get_statistics()
        
        # Re-index same file
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        stats2 = pipeline.get_statistics()
        
        results.ok(f"Re-indexing same file (before:{stats1['pipeline']['total_chunks']}, after:{stats2['pipeline']['total_chunks']})")
    except Exception as e:
        results.fail("Re-indexing same file", str(e))


# =============================================================================
# PARSER-SPECIFIC TESTS
# =============================================================================

def test_parsers(results: TestResult):
    """Test parsing through pipeline."""
    print("\n" + "="*60)
    print("PARSER TESTS")
    print("="*60)
    
    # --- Test: TAL Parser (via pipeline) ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        chunks = pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "test.tal", SourceType.CODE)
        assert len(chunks) > 0
        results.ok(f"TAL parsing via pipeline ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("TAL parsing via pipeline", str(e))
    
    # --- Test: Python Parser (via pipeline) ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        chunks = pipeline.index_content(SAMPLE_PYTHON_CODE.encode('utf-8'), "test.py", SourceType.CODE)
        assert len(chunks) > 0
        results.ok(f"Python parsing via pipeline ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Python parsing via pipeline", str(e))
    
    # --- Test: Document Parser (via pipeline) ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        doc_content = "Wire Transfer Guide\n\nThis document describes OFAC screening procedures."
        chunks = pipeline.index_content(doc_content.encode('utf-8'), "guide.txt", SourceType.DOCUMENT)
        assert len(chunks) > 0
        results.ok(f"Document parsing via pipeline ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Document parsing via pipeline", str(e))
    
    # --- Test: Log Parser (via pipeline) ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        chunks = pipeline.index_content(SAMPLE_LOG_ENTRIES.encode('utf-8'), "app.log", SourceType.LOG)
        assert len(chunks) > 0
        results.ok(f"Log parsing via pipeline ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("Log parsing via pipeline", str(e))
    
    # --- Test: COBOL-like code in .tal file (via pipeline) ---
    try:
        # Note: Native COBOL parsing (.cob, .cbl) not yet implemented
        # Using TAL file to test parser handles non-standard content
        cobol_like_code = """
-- COBOL-like syntax test
PROC WIREPROC;
BEGIN
    -- WS-WIRE-AMOUNT handling
    CALL VALIDATE_WIRE;
END;
        """
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        chunks = pipeline.index_content(cobol_like_code.encode('utf-8'), "wire.tal", SourceType.CODE)
        results.ok(f"COBOL-like parsing via pipeline ({len(chunks)} chunks)")
    except Exception as e:
        results.fail("COBOL-like parsing via pipeline", str(e))


# =============================================================================
# EMBEDDER TESTS
# =============================================================================

def test_embedders(results: TestResult):
    """Test individual embedders."""
    print("\n" + "="*60)
    print("EMBEDDER TESTS")
    print("="*60)
    
    vocab = DomainVocabulary()
    vocab.load_from_data(SAMPLE_VOCABULARY)
    test_texts = [
        "wire transfer processing",
        "OFAC sanctions screening",
        "validate BIC code routing"
    ]
    
    # --- Test: Hash Embedder ---
    try:
        embedder = create_embedder("hash", vocab)
        vectors = [embedder.transform(text) for text in test_texts]
        assert all(len(v) > 0 for v in vectors)
        results.ok(f"Hash Embedder (dim={len(vectors[0])})")
    except Exception as e:
        results.fail("Hash Embedder", str(e))
    
    # --- Test: TF-IDF Embedder ---
    try:
        embedder = create_embedder("tfidf", vocab)
        # TF-IDF needs fitting
        embedder.fit(test_texts)
        vectors = [embedder.transform(text) for text in test_texts]
        assert all(len(v) > 0 for v in vectors)
        results.ok(f"TF-IDF Embedder (dim={len(vectors[0])})")
    except Exception as e:
        results.fail("TF-IDF Embedder", str(e))
    
    # --- Test: BM25 Embedder ---
    try:
        embedder = create_embedder("bm25", vocab)
        embedder.fit(test_texts)
        vectors = [embedder.transform(text) for text in test_texts]
        assert all(len(v) > 0 for v in vectors)
        results.ok(f"BM25 Embedder (dim={len(vectors[0])})")
    except Exception as e:
        results.fail("BM25 Embedder", str(e))
    
    # --- Test: Domain Concept Embedder ---
    try:
        embedder = create_embedder("domain", vocab)
        vectors = [embedder.transform(text) for text in test_texts]
        assert all(len(v) > 0 for v in vectors)
        results.ok(f"Domain Embedder (dim={len(vectors[0])})")
    except Exception as e:
        results.fail("Domain Embedder", str(e))
    
    # --- Test: Embedder Consistency ---
    try:
        embedder = create_embedder("hash", vocab)
        v1 = embedder.transform("wire transfer")
        v2 = embedder.transform("wire transfer")
        assert list(v1) == list(v2), "Same input should produce same output"
        results.ok("Embedder consistency (deterministic)")
    except Exception as e:
        results.fail("Embedder consistency", str(e))


# =============================================================================
# INDEX TESTS
# =============================================================================

def test_index(results: TestResult):
    """Test index functionality via pipeline."""
    print("\n" + "="*60)
    print("INDEX TESTS")
    print("="*60)
    
    # --- Test: Add and Retrieve ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(
            b"Wire transfer processing with OFAC screening",
            "test.tal",
            SourceType.CODE
        )
        
        results_list = pipeline.search("wire transfer", top_k=5)
        assert len(results_list) > 0
        results.ok(f"Add and retrieve ({len(results_list)} results)")
    except Exception as e:
        results.fail("Add and retrieve", str(e))
    
    # --- Test: Search via pipeline.index ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(b"Wire transfer processing", "test.tal", SourceType.CODE)
        
        # Use pipeline search
        results_list = pipeline.search("wire", top_k=5)
        results.ok(f"Search via pipeline ({len(results_list)} results)")
    except Exception as e:
        results.fail("Search via pipeline", str(e))
    
    # --- Test: Index Statistics ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        for i in range(5):
            pipeline.index_content(f"Content {i} about wire transfer".encode('utf-8'), f"file{i}.tal", SourceType.CODE)
        
        stats = pipeline.get_statistics()
        assert stats['pipeline']['total_chunks'] >= 5
        results.ok(f"Index statistics ({stats['pipeline']['total_chunks']} chunks)")
    except Exception as e:
        results.fail("Index statistics", str(e))


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_integration(results: TestResult):
    """Test full pipeline integration scenarios."""
    print("\n" + "="*60)
    print("INTEGRATION TESTS")
    print("="*60)
    
    # --- Test: Full Workflow ---
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create pipeline
            pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
            
            # Index multiple files
            pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "wire.tal", SourceType.CODE)
            pipeline.index_content(SAMPLE_PYTHON_CODE.encode('utf-8'), "payment.py", SourceType.CODE)
            pipeline.index_content(SAMPLE_LOG_ENTRIES.encode('utf-8'), "app.log", SourceType.LOG)
            
            # Save
            pipeline.save(tmpdir)
            
            # Load fresh
            pipeline2 = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
            pipeline2.load(tmpdir)
            
            # Search
            search_results = pipeline2.search("OFAC screening", top_k=10)
            
            # Verify
            assert len(search_results) > 0, "Search should return results"
            stats = pipeline2.get_statistics()
            # After load, check index stats (pipeline stats are reset)
            total_indexed = stats['index']['total_indexed']
            assert total_indexed > 0, "Should have indexed chunks"
            
            results.ok(f"Full workflow ({total_indexed} indexed, {len(search_results)} search results)")
    except Exception as e:
        results.fail("Full workflow", str(e))
    
    # --- Test: Cross-Type Search ---
    try:
        pipeline = IndexingPipeline(vocabulary_data=SAMPLE_VOCABULARY, embedder_type="hash")
        pipeline.index_content(SAMPLE_TAL_CODE.encode('utf-8'), "wire.tal", SourceType.CODE)
        pipeline.index_content(SAMPLE_LOG_ENTRIES.encode('utf-8'), "app.log", SourceType.LOG)
        
        # Search for error - should find in both code and logs
        error_results = pipeline.search("error", top_k=10)
        sources = set(r.chunk.source_type for r in error_results)
        
        results.ok(f"Cross-type search ({len(error_results)} results, {len(sources)} source types)")
    except Exception as e:
        results.fail("Cross-type search", str(e))


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("="*60)
    print("UNIFIED INDEXER - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    results = TestResult()
    
    # Run all test groups
    test_positive_cases(results)
    test_negative_cases(results)
    test_edge_cases(results)
    test_parsers(results)
    test_embedders(results)
    test_index(results)
    test_integration(results)
    
    # Summary
    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
