#!/usr/bin/env python3
"""
Knowledge Extractor - Build domain vocabulary with cross-referenced concepts

================================================================================
OVERVIEW
================================================================================

This module extracts domain-specific terminology from documentation (PDFs) and
source code (TAL/COBOL), cross-references them to identify high-value terms,
calculates TF-IDF statistics, and builds a knowledge graph linking concepts.

================================================================================
ARCHITECTURE & CLASS DESIGN
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                           KNOWLEDGE EXTRACTOR                                │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐ │
│  │    PDFs     │    │    Code     │    │        KnowledgeExtractor       │ │
│  │  (Business) │    │  (Technical)│    │                                 │ │
│  └──────┬──────┘    └──────┬──────┘    │  • pdf_terms: Dict[str, Term]   │ │
│         │                  │           │  • code_terms: Dict[str, Term]  │ │
│         ▼                  ▼           │  • merged_terms: Dict[str, Term]│ │
│  ┌──────────────┐   ┌──────────────┐   │  • relationships: List[Rel]     │ │
│  │ LLM Extract  │   │ Pattern Match│   │  • document_term_sets: List     │ │
│  │ (top 30/doc) │   │ (domain only)│   │                                 │ │
│  └──────┬───────┘   └──────┬───────┘   └─────────────────────────────────┘ │
│         │                  │                                                │
│         └────────┬─────────┘                                                │
│                  ▼                                                          │
│         ┌───────────────────┐                                               │
│         │  Cross-Reference  │  ◄─── Merge terms, boost confidence          │
│         │  & TF-IDF Calc    │       for terms in BOTH sources              │
│         └─────────┬─────────┘                                               │
│                   │                                                         │
│     ┌─────────────┼─────────────┐                                           │
│     ▼             ▼             ▼                                           │
│ ┌────────┐  ┌──────────┐  ┌──────────┐                                     │
│ │ Vocab  │  │ Knowledge│  │ TF-IDF   │                                     │
│ │ .json  │  │ Graph    │  │ Stats    │                                     │
│ └────────┘  └──────────┘  └──────────┘                                     │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
DATA CLASSES
================================================================================

ExtractedTerm
─────────────
Represents a single extracted term with all its metadata and statistics.

    Fields:
    ├── term: str              # Original term text (e.g., "WIRE_MESSAGE")
    ├── normalized: str        # Lowercase normalized form for matching
    ├── term_type: TermType    # STRUCTURE, PROCEDURE, CONSTANT, etc.
    ├── source: TermSource     # PDF, CODE, or BOTH
    ├── source_files: List     # Files where term was found
    ├── context: str           # Description from LLM or comments
    ├── related_terms: Set     # Terms that co-occur or are linked
    ├── code_references: List  # [{file, line, type}, ...] for code terms
    ├── confidence: float      # Boosted for multiple occurrences
    │
    │   TF-IDF Statistics:
    ├── term_frequency: int    # Total occurrences across all documents
    ├── document_frequency: int # Number of documents containing term
    ├── idf_score: float       # log(total_docs / doc_frequency)
    ├── tf_idf_score: float    # Combined TF-IDF importance score
    └── co_occurrences: Dict   # {term: count} for co-occurring terms

Relationship
────────────
Represents a relationship between two terms in the knowledge graph.

    Fields:
    ├── source_term: str       # Source node
    ├── target_term: str       # Target node
    ├── relationship_type: str # contains, implements, co_occurs_with, related_to
    └── evidence: str          # File:line or "Co-occurred in N documents"

================================================================================
MAIN CLASS: KnowledgeExtractor
================================================================================

Methods (in processing order):
─────────────────────────────

1. EXTRACTION
   ├── extract_from_pdf(file_path)     # LLM extracts business terms
   │   └── _llm_extract_terms()        # Send to LLM, parse JSON response
   │   └── _heuristic_extract_from_doc # Fallback regex patterns
   │
   └── extract_from_code(file_path)    # Pattern-based code extraction
       ├── _extract_from_tal()         # TAL: STRUCT, PROC, DEFINE
       ├── _extract_from_cobol()       # COBOL: records, paragraphs
       └── _extract_from_c_like()      # C/Java: functions, structs

2. FILTERING
   └── is_domain_relevant(name)        # Check against DOMAIN_PATTERNS
                                       # Reject GENERIC_NAMES (i, j, tmp, etc.)

3. TF-IDF CALCULATION
   ├── record_document_terms()         # Track terms per document
   ├── calculate_tf_idf()              # Compute IDF and TF-IDF scores
   └── calculate_co_occurrences()      # Find terms appearing together

4. CROSS-REFERENCING
   └── cross_reference()               # Merge PDF + code terms
       ├── _merge_terms()              # Combine duplicate terms
       └── _discover_relationships()   # Find implements, contains links

5. OUTPUT GENERATION
   ├── generate_vocabulary()           # keywords.json compatible format
   ├── generate_knowledge_graph()      # Nodes + edges JSON
   ├── get_statistics()                # TF-IDF statistics JSON
   └── print_summary()                 # Console summary

================================================================================
TERM CONFIDENCE SCORING
================================================================================

Base confidence: 1.0

Boosts:
  +0.25  Term appears in additional PDF document
  +0.10  Term appears in additional code file
  +0.50  Term is cross-referenced (found in BOTH PDF and code)

Example:
  "wire_transfer" in 3 PDFs + 2 code files + cross-referenced
  = 1.0 + (2 × 0.25) + (1 × 0.10) + 0.50 = 2.10 confidence

================================================================================
TF-IDF SCORING
================================================================================

Term Frequency (TF):
  Raw count of term occurrences across all documents.
  Log-normalized: tf = 1 + log(raw_count) if raw_count > 0

Inverse Document Frequency (IDF):
  idf = log(total_documents / (1 + document_frequency))
  High IDF = rare/distinctive term
  Low IDF = common/universal term

TF-IDF Score:
  tf_idf = tf × idf
  Used to rank terms by distinctiveness/importance

================================================================================
RELATIONSHIP TYPES
================================================================================

  contains      Structure contains field (WIRE_MESSAGE → SENDER_BIC)
  implements    Procedure implements concept (VALIDATE_BIC → bic_validation)
  co_occurs_with Terms frequently appear together in same documents
  related_to    Generic relationship from shared vocabulary

================================================================================
USAGE
================================================================================

    # Full extraction with LLM
    python knowledge_extractor.py \\
        --docs ./payment_specs \\
        --code ./tal_code \\
        --output vocabulary.json \\
        --graph knowledge_graph.json \\
        --stats term_statistics.json

    # Offline mode (no LLM, pattern-based only)
    python knowledge_extractor.py \\
        --code ./legacy_code \\
        --no-llm \\
        --output vocab.json

================================================================================
OUTPUT FILES
================================================================================

vocabulary.json     Compatible with unified_indexer keywords.json format
                    Includes _tf_idf_score, _term_frequency, _document_frequency

knowledge_graph.json  Nodes (terms) + Edges (relationships)
                      Each node has TF-IDF stats and co-occurrence list

term_statistics.json  Aggregate statistics:
                      - top_by_tf_idf (distinctive terms)
                      - top_by_frequency (common terms)
                      - top_by_document_frequency (universal terms)
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================

# LLM provider for intelligent PDF extraction
try:
    from llm_provider import create_provider, LLMProvider, LLMResponse
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: llm_provider not found.")

# PyMuPDF for PDF text extraction
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. PDF extraction will be limited.")


# =============================================================================
# ENUMS - Term classification
# =============================================================================

class TermSource(Enum):
    """
    Where a term was discovered.
    
    BOTH indicates cross-referenced terms (highest value).
    """
    PDF = "pdf"      # Found in documentation
    CODE = "code"    # Found in source code
    BOTH = "both"    # Found in both (cross-referenced)


class TermType(Enum):
    """
    Classification of term type.
    
    Determines how the term is categorized in the vocabulary output.
    """
    BUSINESS_CONCEPT = "business_concept"  # From PDF - business terminology
    STRUCTURE = "structure"                 # TAL STRUCT, COBOL record
    PROCEDURE = "procedure"                 # TAL PROC, COBOL paragraph
    CONSTANT = "constant"                   # DEFINE, LITERAL, 88 level
    FIELD = "field"                        # Structure field
    MESSAGE_TYPE = "message_type"          # MT-103, MT-202, etc.
    ERROR_CODE = "error_code"              # Error codes
    ACRONYM = "acronym"                    # BIC, IBAN, OFAC, etc.


# =============================================================================
# DATA CLASSES - Core data structures
# =============================================================================

@dataclass
class ExtractedTerm:
    """
    Represents a single extracted term with all metadata and statistics.
    
    This is the primary data structure used throughout the extraction pipeline.
    Terms are keyed by their normalized form for deduplication and matching.
    
    Attributes:
        term: Original term text as found in source (preserves case)
        normalized: Lowercase, underscore-separated form for matching
        term_type: Classification (STRUCTURE, PROCEDURE, CONSTANT, etc.)
        source: Where found (PDF, CODE, or BOTH if cross-referenced)
        source_files: List of files where this term was discovered
        context: Description from LLM or extracted from comments
        related_terms: Set of terms that are linked/co-occur
        code_references: List of {file, line, type} for code locations
        confidence: Score boosted by multiple occurrences and cross-referencing
        
    TF-IDF Statistics:
        term_frequency: Raw count of occurrences across all documents
        document_frequency: Number of unique documents containing term
        idf_score: Inverse document frequency = log(total_docs / df)
        tf_idf_score: Combined score for ranking importance
        co_occurrences: Dict mapping co-occurring terms to counts
    """
    # Core identification
    term: str                                           # Original text
    normalized: str                                     # Normalized for matching
    term_type: TermType                                # Classification
    source: TermSource                                 # PDF, CODE, or BOTH
    
    # Source tracking
    source_files: List[str] = field(default_factory=list)      # Files containing term
    context: str = ""                                          # Description/definition
    related_terms: Set[str] = field(default_factory=set)       # Linked terms
    code_references: List[Dict] = field(default_factory=list)  # [{file, line, type}]
    
    # Confidence scoring
    confidence: float = 1.0  # Boosted: +0.25/PDF, +0.10/code file, +0.50 if cross-ref
    
    # TF-IDF statistics (calculated after all documents processed)
    term_frequency: int = 0           # Total occurrences across all docs
    document_frequency: int = 0       # Number of docs containing this term
    idf_score: float = 0.0           # log(total_docs / doc_frequency)
    tf_idf_score: float = 0.0        # tf * idf - overall importance score
    
    # Co-occurrence tracking (for relationship discovery)
    co_occurrences: Dict[str, int] = field(default_factory=dict)  # {term: count}


@dataclass 
class Relationship:
    """
    Represents a directed relationship between two terms in the knowledge graph.
    
    Relationships are discovered through:
    - Code structure analysis (contains, implements)
    - Co-occurrence analysis (co_occurs_with)
    - Name pattern matching (related_to)
    
    Attributes:
        source_term: The source node of the relationship
        target_term: The target node of the relationship
        relationship_type: One of: contains, implements, co_occurs_with, related_to
        evidence: Supporting information (e.g., "file.tal:42" or "Co-occurred in 5 docs")
    """
    source_term: str           # From node
    target_term: str           # To node
    relationship_type: str     # contains | implements | co_occurs_with | related_to
    evidence: str = ""         # File:line or occurrence count


# =============================================================================
# MAIN CLASS: KnowledgeExtractor
# =============================================================================


class KnowledgeExtractor:
    """
    Main class for extracting and cross-referencing domain knowledge.
    
    This class orchestrates the entire extraction pipeline:
    1. Extract terms from PDFs using LLM (or heuristics as fallback)
    2. Extract terms from code using pattern matching
    3. Cross-reference to identify high-value terms
    4. Calculate TF-IDF statistics
    5. Build relationships for knowledge graph
    6. Generate output files
    
    Attributes:
        llm: Optional LLM provider for intelligent PDF extraction
        pdf_terms: Dict of terms extracted from PDF documents
        code_terms: Dict of terms extracted from source code
        merged_terms: Dict of all terms after cross-referencing
        relationships: List of discovered relationships
        total_documents: Count of processed documents for TF-IDF
        document_term_sets: Terms per document for co-occurrence analysis
    
    Example:
        extractor = KnowledgeExtractor(llm_provider)
        
        # Extract from sources
        for pdf in pdfs:
            terms = extractor.extract_from_pdf(pdf)
            extractor.record_document_terms(terms, pdf)
            
        for code in code_files:
            terms = extractor.extract_from_code(code)
            extractor.record_document_terms(terms, code)
        
        # Cross-reference and calculate statistics
        extractor.cross_reference()
        
        # Generate outputs
        vocab = extractor.generate_vocabulary()
        graph = extractor.generate_knowledge_graph()
        stats = extractor.get_statistics()
    """
    
    # -------------------------------------------------------------------------
    # Domain-relevant patterns for filtering code symbols
    # These regex patterns identify payment/banking domain terminology
    # -------------------------------------------------------------------------
    DOMAIN_PATTERNS = [
        # Payment networks and protocols
        r'wire', r'transfer', r'payment', r'message', r'swift', r'fedwire',
        r'chips', r'ach', r'ofac', r'sanction', r'screen', r'compliance',
        
        # Financial identifiers
        r'bic', r'iban', r'aba', r'routing', r'account', r'beneficiary',
        r'originator', r'sender', r'receiver', r'correspondent', r'intermediary',
        
        # Settlement and processing
        r'settlement', r'clearing', r'netting', r'reconcil', r'balance',
        r'currency', r'amount', r'rate', r'fee', r'charge',
        
        # Operations
        r'validate', r'verify', r'check', r'error', r'reject', r'repair',
        r'queue', r'route', r'process', r'transaction', r'reference',
        
        # Message types and IDs
        r'mt\d{3}',  # SWIFT MT messages (MT-103, MT-202, etc.)
        r'uetr',     # Unique End-to-end Transaction Reference
        r'uti',      # Unique Transaction Identifier
        r'lei',      # Legal Entity Identifier
    ]
    
    # -------------------------------------------------------------------------
    # Generic programming names to exclude
    # These are common variable names that add noise to the vocabulary
    # -------------------------------------------------------------------------
    GENERIC_NAMES = {
        # Loop variables
        'i', 'j', 'k', 'n', 'm', 'x', 'y', 'z',
        # Temporary storage
        'tmp', 'temp', 'buf', 'buffer', 'ptr', 'len', 'length', 'size', 'count',
        # Return values
        'result', 'status', 'ret', 'retval', 'rc', 'err', 'error',
        # Generic types
        'str', 'string', 'num', 'int', 'val', 'value', 'data', 'info',
        # Common names
        'flag', 'flags', 'index', 'idx', 'pos', 'offset',
        'begin', 'end', 'start', 'stop', 'first', 'last', 'next', 'prev',
        'input', 'output', 'in', 'out', 'src', 'dst', 'source', 'dest',
        # Boolean/null
        'true', 'false', 'null', 'none', 'ok', 'fail',
    }
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the knowledge extractor.
        
        Args:
            llm_provider: Optional LLM provider for intelligent PDF extraction.
                         If None, falls back to heuristic pattern matching.
        """
        self.llm = llm_provider
        
        # Term storage (keyed by normalized form)
        self.pdf_terms: Dict[str, ExtractedTerm] = {}    # Terms from documents
        self.code_terms: Dict[str, ExtractedTerm] = {}   # Terms from code
        self.merged_terms: Dict[str, ExtractedTerm] = {} # Combined after cross-ref
        
        # Relationship storage
        self.relationships: List[Relationship] = []
        
        # TF-IDF tracking
        self.total_documents: int = 0                    # Document count
        self.document_term_sets: List[Set[str]] = []     # Terms per doc for co-occurrence
    
    def load_existing_state(self, graph_path: str, stats_path: Optional[str] = None):
        """
        Load existing extraction state from graph and stats files.
        
        This allows appending new code extractions without re-processing PDFs.
        
        Args:
            graph_path: Path to existing knowledge_graph.json
            stats_path: Path to existing term_statistics.json (optional)
        """
        print(f"\nLoading existing extraction from: {graph_path}")
        
        with open(graph_path, 'r') as f:
            graph = json.load(f)
        
        # Restore terms from graph nodes
        nodes = graph.get('nodes', {})
        pdf_count = 0
        code_count = 0
        
        for term_id, node in nodes.items():
            # Reconstruct ExtractedTerm from node data
            source_type = node.get('source_type', 'both')
            
            # Map source_type to TermSource enum
            if source_type == 'pdf':
                term_source = TermSource.PDF
            elif source_type == 'code':
                term_source = TermSource.CODE
            else:
                term_source = TermSource.BOTH
            
            # Map category to TermType enum
            category = node.get('category', 'concept').upper()
            try:
                term_type = TermType[category]
            except KeyError:
                term_type = TermType.CONCEPT
            
            term = ExtractedTerm(
                term=node.get('original', term_id),
                normalized=term_id,
                term_type=term_type,
                source=term_source,
                source_files=node.get('source_files', []),
                context=node.get('definition', ''),
                code_references=node.get('code_references', []),
                document_frequency=node.get('document_frequency', 1),
                term_frequency=node.get('term_frequency', 1),
                confidence=node.get('confidence', 0.5)
            )
            
            # Store in appropriate dict based on source
            if source_type in ('pdf', 'both'):
                self.pdf_terms[term_id] = term
                pdf_count += 1
            if source_type in ('code', 'both'):
                self.code_terms[term_id] = term
                code_count += 1
        
        # Restore relationships from edges
        for edge in graph.get('edges', []):
            rel = Relationship(
                source_term=edge.get('source', ''),
                target_term=edge.get('target', ''),
                relationship_type=edge.get('type', 'related_to'),
                weight=edge.get('weight', 1.0),
                evidence=edge.get('evidence', '')
            )
            self.relationships.append(rel)
        
        # Restore TF-IDF stats if available
        if stats_path and os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.total_documents = stats.get('total_documents', 0)
            # Note: document_term_sets can't be fully restored, but total_documents is key
        else:
            # Estimate from graph
            self.total_documents = graph.get('statistics', {}).get('total_sources', 0)
        
        print(f"  Loaded {pdf_count} PDF terms, {code_count} code terms")
        print(f"  Loaded {len(self.relationships)} relationships")
        print(f"  Total documents: {self.total_documents}")
    
    # =========================================================================
    # FILTERING METHODS
    # =========================================================================
    
    def is_domain_relevant(self, name: str) -> bool:
        """
        Check if a symbol name is domain-relevant (worth including in vocabulary).
        
        A name is considered domain-relevant if:
        1. It's not in the GENERIC_NAMES exclusion list
        2. It's at least 3 characters long
        3. It matches at least one DOMAIN_PATTERNS regex
        
        Args:
            name: The symbol name to check (e.g., "WIRE_MESSAGE", "process_payment")
            
        Returns:
            True if the name should be included in the vocabulary
            
        Examples:
            is_domain_relevant("WIRE_MESSAGE")     -> True  (matches 'wire', 'message')
            is_domain_relevant("tmp")              -> False (in GENERIC_NAMES)
            is_domain_relevant("VALIDATE_BIC")     -> True  (matches 'validate', 'bic')
            is_domain_relevant("i")                -> False (too short)
        """
        name_lower = name.lower()
        
        # Exclude generic names (loop vars, temp vars, etc.)
        if name_lower in self.GENERIC_NAMES:
            return False
        
        # Reject names that are too short to be meaningful
        if len(name_lower) < 3:
            return False
        
        # Check if name matches any domain pattern
        for pattern in self.DOMAIN_PATTERNS:
            if re.search(pattern, name_lower):
                return True
        
        # For compound names (WIRE_MESSAGE, validatePayment), check each part
        parts = name_lower.replace('-', '_').split('_')
        for part in parts:
            if len(part) > 3 and part not in self.GENERIC_NAMES:
                for pattern in self.DOMAIN_PATTERNS:
                    if re.search(pattern, part):
                        return True
        
        return False
    
    def normalize_term(self, term: str) -> str:
        """
        Normalize a term for comparison and deduplication.
        
        Normalization ensures that "WIRE_MESSAGE", "wire-message", and 
        "Wire Message" all map to the same key: "wire_message"
        
        Args:
            term: Raw term text
            
        Returns:
            Normalized lowercase string with underscores as separators
        """
        # Convert to lowercase
        normalized = term.lower()
        # Replace hyphens and spaces with underscores
        normalized = re.sub(r'[-\s]+', '_', normalized)
        # Remove any remaining special characters
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        return normalized
    
    # =========================================================================
    # PDF EXTRACTION - Extract business terms from documentation
    # =========================================================================
    
    def extract_from_pdf(self, file_path: str) -> List[ExtractedTerm]:
        """Extract business terms from PDF using LLM"""
        
        if not PDF_AVAILABLE:
            return self._extract_from_text_file(file_path)
        
        try:
            doc = fitz.open(file_path)
            text_parts = []
            for page_num in range(min(len(doc), 50)):
                text_parts.append(doc[page_num].get_text())
            doc.close()
            content = "\n".join(text_parts)
        except Exception as e:
            print(f"    Error reading PDF: {e}")
            return []
        
        return self._extract_terms_from_text(content, file_path, "pdf")
    
    def _extract_from_text_file(self, file_path: str) -> List[ExtractedTerm]:
        """Extract from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return self._extract_terms_from_text(content, file_path, "text")
        except Exception as e:
            print(f"    Error reading file: {e}")
            return []
    
    def _extract_terms_from_text(self, content: str, file_path: str, file_type: str) -> List[ExtractedTerm]:
        """Extract terms from text content using LLM or heuristics"""
        
        if self.llm and len(content) > 100:
            return self._llm_extract_terms(content, file_path)
        else:
            return self._heuristic_extract_from_doc(content, file_path)
    
    def _llm_extract_terms(self, content: str, file_path: str) -> List[ExtractedTerm]:
        """Use LLM to extract business terms"""
        
        # Truncate if needed
        if len(content) > 12000:
            content = content[:6000] + "\n\n...[truncated]...\n\n" + content[-4000:]
        
        prompt = """Extract domain-specific terminology from this document.

Focus on:
1. Payment/banking terms (wire transfer, SWIFT, settlement, etc.)
2. Message types (MT-103, MT-202, ISO 20022, etc.)
3. Compliance terms (OFAC, sanctions, AML, KYC, etc.)
4. Technical terms specific to financial messaging
5. Acronyms and their meanings
6. Business processes and workflows

Return a JSON object:
{
    "terms": [
        {
            "term": "wire transfer",
            "type": "business_concept",
            "description": "Electronic transfer of funds between banks"
        },
        {
            "term": "MT-103",
            "type": "message_type", 
            "description": "SWIFT single customer credit transfer"
        }
    ]
}

Extract up to 30 most important domain terms. Types: business_concept, message_type, acronym, error_code, process"""

        try:
            response = self.llm.invoke_llm(
                system_prompt=prompt,
                user_prompt=f"Document: {Path(file_path).name}\n\nContent:\n{content}",
                temperature=0.3,
                max_tokens=2000
            )
            
            if not response.success:
                return self._heuristic_extract_from_doc(content, file_path)
            
            # Parse JSON
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                data = json.loads(json_match.group())
                terms = []
                for item in data.get('terms', []):
                    term_type = TermType.BUSINESS_CONCEPT
                    if item.get('type') == 'message_type':
                        term_type = TermType.MESSAGE_TYPE
                    elif item.get('type') == 'acronym':
                        term_type = TermType.ACRONYM
                    elif item.get('type') == 'error_code':
                        term_type = TermType.ERROR_CODE
                    
                    terms.append(ExtractedTerm(
                        term=item['term'],
                        normalized=self.normalize_term(item['term']),
                        term_type=term_type,
                        source=TermSource.PDF,
                        source_files=[file_path],
                        context=item.get('description', '')
                    ))
                return terms
        except Exception as e:
            print(f"    LLM extraction failed: {e}")
        
        return self._heuristic_extract_from_doc(content, file_path)
    
    def _heuristic_extract_from_doc(self, content: str, file_path: str) -> List[ExtractedTerm]:
        """Heuristic extraction from document text"""
        terms = []
        found = set()
        
        # Look for domain patterns
        patterns = [
            (r'\b(MT[-\s]?\d{3})\b', TermType.MESSAGE_TYPE),
            (r'\b(wire\s+transfer|funds\s+transfer|credit\s+transfer)\b', TermType.BUSINESS_CONCEPT),
            (r'\b(SWIFT|OFAC|CHIPS|FEDWIRE|ACH|SEPA|ISO\s*20022)\b', TermType.ACRONYM),
            (r'\b(BIC|IBAN|ABA|UETR|UTI|LEI)\b', TermType.ACRONYM),
            (r'\b(sanctions?\s+screening|compliance\s+check|AML|KYC)\b', TermType.BUSINESS_CONCEPT),
            (r'\b(beneficiary|originator|correspondent|intermediary)\s*(bank|institution)?\b', TermType.BUSINESS_CONCEPT),
            (r'\b(settlement|clearing|netting|reconciliation)\b', TermType.BUSINESS_CONCEPT),
            (r'\berror\s+code\s+(\d{4})\b', TermType.ERROR_CODE),
        ]
        
        for pattern, term_type in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                term = match.group(0).strip()
                normalized = self.normalize_term(term)
                if normalized not in found:
                    found.add(normalized)
                    terms.append(ExtractedTerm(
                        term=term,
                        normalized=normalized,
                        term_type=term_type,
                        source=TermSource.PDF,
                        source_files=[file_path]
                    ))
        
        return terms[:30]
    
    # =========================================================================
    # CODE EXTRACTION
    # =========================================================================
    
    def detect_language(self, content: str, file_path: str) -> str:
        """
        Detect programming language from file content.
        
        Returns one of: 'tal', 'cobol', 'c', 'python', 'java', 'unknown'
        """
        ext = Path(file_path).suffix.lower()
        
        # First check extension for obvious cases
        if ext in ('.tal', '.tacl', '.ddl'):
            return 'tal'
        elif ext in ('.cob', '.cbl', '.cobol', '.cpy'):
            return 'cobol'
        elif ext in ('.py',):
            return 'python'
        elif ext in ('.java',):
            return 'java'
        elif ext in ('.c', '.h', '.cpp', '.hpp', '.cc'):
            return 'c'
        
        # Content-based detection for .txt and unknown extensions
        content_upper = content.upper()
        content_sample = content[:5000]  # Sample first 5KB
        
        # TAL indicators (Transaction Application Language)
        tal_score = 0
        if re.search(r'\bPROC\s+\w+', content_upper):
            tal_score += 3
        if re.search(r'\bSUBPROC\s+\w+', content_upper):
            tal_score += 3
        if re.search(r'\bSTRUCT\s+\w+', content_upper):
            tal_score += 2
        if re.search(r'\bDEFINE\s+\w+', content_upper):
            tal_score += 2
        if re.search(r'\bINT\s*\(\s*\d+\s*\)', content_upper):
            tal_score += 2
        if re.search(r'\bSTRING\s+\.\w+', content_upper):
            tal_score += 2
        if re.search(r'\bCALL\s+\w+\s*\^', content_upper):
            tal_score += 2
        if re.search(r'\^\w+', content):  # Caret notation
            tal_score += 1
        if re.search(r'^!', content, re.MULTILINE):  # TAL comments
            tal_score += 1
        if 'LITERAL' in content_upper or 'FORWARD' in content_upper:
            tal_score += 1
            
        # COBOL indicators
        cobol_score = 0
        if re.search(r'\bIDENTIFICATION\s+DIVISION', content_upper):
            cobol_score += 5
        if re.search(r'\bDATA\s+DIVISION', content_upper):
            cobol_score += 5
        if re.search(r'\bPROCEDURE\s+DIVISION', content_upper):
            cobol_score += 5
        if re.search(r'\bWORKING-STORAGE\s+SECTION', content_upper):
            cobol_score += 3
        if re.search(r'\b\d{2}\s+\w+.*PIC\s+', content_upper):
            cobol_score += 3
        if re.search(r'\bPERFORM\s+\w+', content_upper):
            cobol_score += 2
        if re.search(r'\bMOVE\s+\w+\s+TO\s+', content_upper):
            cobol_score += 2
        if re.search(r'^\s{6}', content, re.MULTILINE):  # Column 7 start
            cobol_score += 1
        if re.search(r'^\d{6}', content, re.MULTILINE):  # Line numbers
            cobol_score += 1
            
        # Python indicators
        python_score = 0
        if re.search(r'^def\s+\w+\s*\(', content_sample, re.MULTILINE):
            python_score += 3
        if re.search(r'^class\s+\w+.*:', content_sample, re.MULTILINE):
            python_score += 3
        if re.search(r'^import\s+\w+', content_sample, re.MULTILINE):
            python_score += 2
        if re.search(r'^from\s+\w+\s+import', content_sample, re.MULTILINE):
            python_score += 2
        if re.search(r'^\s+def\s+__\w+__', content_sample, re.MULTILINE):
            python_score += 2
        if '"""' in content_sample or "'''" in content_sample:
            python_score += 1
            
        # Java indicators  
        java_score = 0
        if re.search(r'\bpublic\s+class\s+\w+', content_sample):
            java_score += 4
        if re.search(r'\bprivate\s+(static\s+)?\w+\s+\w+', content_sample):
            java_score += 2
        if re.search(r'\bimport\s+java\.', content_sample):
            java_score += 3
        if re.search(r'\bpackage\s+[\w.]+;', content_sample):
            java_score += 2
        if re.search(r'\bpublic\s+static\s+void\s+main', content_sample):
            java_score += 3
            
        # C/C++ indicators
        c_score = 0
        if re.search(r'^#include\s*[<"]', content_sample, re.MULTILINE):
            c_score += 3
        if re.search(r'^#define\s+\w+', content_sample, re.MULTILINE):
            c_score += 2
        if re.search(r'\bint\s+main\s*\(', content_sample):
            c_score += 3
        if re.search(r'\b(typedef|struct|union)\s+\w+', content_sample):
            c_score += 2
        if re.search(r'\w+\s*\*\s*\w+', content_sample):  # Pointers
            c_score += 1
        if re.search(r'->\w+', content_sample):  # Arrow operator
            c_score += 1
            
        # Pick highest score
        scores = {
            'tal': tal_score,
            'cobol': cobol_score,
            'python': python_score,
            'java': java_score,
            'c': c_score
        }
        
        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]
        
        # Require minimum confidence
        if best_score >= 3:
            return best_lang
        
        return 'unknown'
    
    def extract_from_code(self, file_path: str) -> List[ExtractedTerm]:
        """Extract domain-relevant symbols from code file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"    Error reading code: {e}")
            return []
        
        # Auto-detect language from content
        lang = self.detect_language(content, file_path)
        
        if lang == 'tal':
            return self._extract_from_tal(content, file_path)
        elif lang == 'cobol':
            return self._extract_from_cobol(content, file_path)
        elif lang in ('c', 'java', 'python'):
            return self._extract_from_c_like(content, file_path)
        else:
            # Try generic extraction for unknown
            return self._extract_from_c_like(content, file_path)
    
    def _extract_from_tal(self, content: str, file_path: str) -> List[ExtractedTerm]:
        """Extract from TAL code"""
        terms = []
        lines = content.split('\n')
        
        # Track current structure for field relationships
        current_struct = None
        in_struct = False
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Skip comments
            if stripped.startswith('!') or stripped.startswith('--'):
                continue
            
            # STRUCT definitions
            struct_match = re.match(r'STRUCT\s+(\w+)', stripped, re.IGNORECASE)
            if struct_match:
                name = struct_match.group(1)
                if self.is_domain_relevant(name):
                    term = ExtractedTerm(
                        term=name,
                        normalized=self.normalize_term(name),
                        term_type=TermType.STRUCTURE,
                        source=TermSource.CODE,
                        source_files=[file_path],
                        code_references=[{'file': file_path, 'line': line_num, 'type': 'definition'}]
                    )
                    terms.append(term)
                    current_struct = name
                    in_struct = True
                continue
            
            # End of struct
            if in_struct and re.match(r'^\s*END\s*;', stripped, re.IGNORECASE):
                in_struct = False
                current_struct = None
                continue
            
            # Structure fields (while in struct)
            if in_struct:
                field_match = re.match(r'(?:INT|STRING|FIXED|UNSIGNED|REAL)\s+\.?(\w+)', stripped, re.IGNORECASE)
                if field_match:
                    name = field_match.group(1)
                    if self.is_domain_relevant(name):
                        term = ExtractedTerm(
                            term=name,
                            normalized=self.normalize_term(name),
                            term_type=TermType.FIELD,
                            source=TermSource.CODE,
                            source_files=[file_path],
                            code_references=[{'file': file_path, 'line': line_num, 'type': 'field'}]
                        )
                        if current_struct:
                            term.related_terms.add(current_struct.lower())
                            self.relationships.append(Relationship(
                                source_term=current_struct,
                                target_term=name,
                                relationship_type='contains',
                                evidence=f"{file_path}:{line_num}"
                            ))
                        terms.append(term)
            
            # PROC/SUBPROC definitions
            proc_match = re.match(r'(PROC|SUBPROC)\s+(\w+)', stripped, re.IGNORECASE)
            if proc_match:
                name = proc_match.group(2)
                if self.is_domain_relevant(name):
                    term = ExtractedTerm(
                        term=name,
                        normalized=self.normalize_term(name),
                        term_type=TermType.PROCEDURE,
                        source=TermSource.CODE,
                        source_files=[file_path],
                        code_references=[{'file': file_path, 'line': line_num, 'type': 'definition'}]
                    )
                    terms.append(term)
                continue
            
            # DEFINE/LITERAL constants
            const_match = re.match(r'(DEFINE|LITERAL)\s+(\w+)', stripped, re.IGNORECASE)
            if const_match:
                name = const_match.group(2)
                if self.is_domain_relevant(name):
                    term = ExtractedTerm(
                        term=name,
                        normalized=self.normalize_term(name),
                        term_type=TermType.CONSTANT,
                        source=TermSource.CODE,
                        source_files=[file_path],
                        code_references=[{'file': file_path, 'line': line_num, 'type': 'definition'}]
                    )
                    terms.append(term)
        
        return terms
    
    def _extract_from_cobol(self, content: str, file_path: str) -> List[ExtractedTerm]:
        """Extract from COBOL code"""
        terms = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip sequence/indicator columns in fixed format
            if len(line) > 7:
                line = line[6:]
            
            stripped = line.strip()
            
            # Skip comments
            if stripped.startswith('*'):
                continue
            
            # Data items (01-49 levels)
            data_match = re.match(r'(\d{2})\s+(\w+(?:-\w+)*)', stripped)
            if data_match:
                level = int(data_match.group(1))
                name = data_match.group(2)
                
                if level <= 49 and self.is_domain_relevant(name):
                    term_type = TermType.STRUCTURE if level == 1 else TermType.FIELD
                    terms.append(ExtractedTerm(
                        term=name,
                        normalized=self.normalize_term(name),
                        term_type=term_type,
                        source=TermSource.CODE,
                        source_files=[file_path],
                        code_references=[{'file': file_path, 'line': line_num, 'type': 'definition'}]
                    ))
            
            # Paragraph/section names
            para_match = re.match(r'^(\w+(?:-\w+)*)\s*\.\s*$', stripped)
            if para_match:
                name = para_match.group(1)
                if self.is_domain_relevant(name):
                    terms.append(ExtractedTerm(
                        term=name,
                        normalized=self.normalize_term(name),
                        term_type=TermType.PROCEDURE,
                        source=TermSource.CODE,
                        source_files=[file_path],
                        code_references=[{'file': file_path, 'line': line_num, 'type': 'definition'}]
                    ))
        
        return terms
    
    def _extract_from_c_like(self, content: str, file_path: str) -> List[ExtractedTerm]:
        """Extract from C/C++/Java code"""
        terms = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Skip comments
            if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                continue
            
            # Function definitions
            func_match = re.match(r'(?:static\s+)?(?:void|int|char|bool|[\w:]+)\s+(\w+)\s*\(', stripped)
            if func_match:
                name = func_match.group(1)
                if self.is_domain_relevant(name) and name not in ('if', 'for', 'while', 'switch'):
                    terms.append(ExtractedTerm(
                        term=name,
                        normalized=self.normalize_term(name),
                        term_type=TermType.PROCEDURE,
                        source=TermSource.CODE,
                        source_files=[file_path],
                        code_references=[{'file': file_path, 'line': line_num, 'type': 'definition'}]
                    ))
            
            # Struct definitions
            struct_match = re.match(r'(?:typedef\s+)?struct\s+(\w+)', stripped)
            if struct_match:
                name = struct_match.group(1)
                if self.is_domain_relevant(name):
                    terms.append(ExtractedTerm(
                        term=name,
                        normalized=self.normalize_term(name),
                        term_type=TermType.STRUCTURE,
                        source=TermSource.CODE,
                        source_files=[file_path],
                        code_references=[{'file': file_path, 'line': line_num, 'type': 'definition'}]
                    ))
            
            # #define constants
            define_match = re.match(r'#define\s+(\w+)', stripped)
            if define_match:
                name = define_match.group(1)
                if self.is_domain_relevant(name):
                    terms.append(ExtractedTerm(
                        term=name,
                        normalized=self.normalize_term(name),
                        term_type=TermType.CONSTANT,
                        source=TermSource.CODE,
                        source_files=[file_path],
                        code_references=[{'file': file_path, 'line': line_num, 'type': 'definition'}]
                    ))
        
        return terms
    
    # =========================================================================
    # TF-IDF AND CO-OCCURRENCE STATISTICS
    # 
    # TF-IDF (Term Frequency - Inverse Document Frequency) measures how
    # important a term is to a document collection. It helps distinguish:
    # - Common terms (low IDF): appear in many documents (e.g., "payment")
    # - Distinctive terms (high IDF): appear in few documents (e.g., "UETR")
    # 
    # Co-occurrence analysis finds terms that frequently appear together,
    # which helps discover implicit relationships between concepts.
    # =========================================================================
    
    def record_document_terms(self, terms: List[ExtractedTerm], file_path: str):
        """
        Record terms from a document for TF-IDF calculation.
        
        This must be called for each document BEFORE cross_reference() to ensure
        accurate TF-IDF statistics.
        
        Args:
            terms: List of terms extracted from this document
            file_path: Path to the source document (for tracking)
        """
        self.total_documents += 1
        
        # Collect normalized terms for this document (deduplicated)
        doc_terms = set()
        for term in terms:
            doc_terms.add(term.normalized)
            term.term_frequency += 1  # Increment raw count
        
        # Store for later document frequency and co-occurrence calculation
        self.document_term_sets.append(doc_terms)
    
    def calculate_tf_idf(self):
        """
        Calculate TF-IDF scores for all merged terms.
        
        TF-IDF Formula:
            TF = 1 + log(term_frequency)  if term_frequency > 0, else 0
            IDF = log(total_documents / (1 + document_frequency))
            TF-IDF = TF × IDF
        
        Interpretation:
            High TF-IDF: Term is frequent in few documents → distinctive
            Low TF-IDF: Term is rare OR appears in many documents → less distinctive
            
        Called automatically by cross_reference() after merging terms.
        """
        import math
        
        if self.total_documents == 0:
            return
        
        for term in self.merged_terms.values():
            # Document Frequency (DF): count of documents containing this term
            doc_count = sum(1 for doc_terms in self.document_term_sets 
                           if term.normalized in doc_terms)
            term.document_frequency = doc_count
            
            # Inverse Document Frequency (IDF)
            # Higher IDF = more distinctive (appears in fewer documents)
            # Add 1 to denominator to avoid division by zero for unseen terms
            term.idf_score = math.log(self.total_documents / (1 + doc_count))
            
            # Term Frequency with logarithmic dampening
            # This prevents very frequent terms from dominating
            if term.term_frequency > 0:
                tf = 1 + math.log(term.term_frequency)
            else:
                tf = 0
            
            # Combined TF-IDF score
            term.tf_idf_score = tf * term.idf_score
    
    def calculate_co_occurrences(self):
        """
        Calculate which terms frequently appear together in the same documents.
        
        Co-occurrence is bidirectional: if A and B appear in the same document,
        both A→B and B→A are recorded.
        
        Terms that co-occur in 2+ documents are added to each other's 
        related_terms and a 'co_occurs_with' relationship is created.
        
        This helps discover implicit relationships like:
        - "OFAC" and "sanctions" often appear together
        - "MT-103" and "beneficiary" are related concepts
        """
        # Build co-occurrence matrix from document term sets
        for doc_terms in self.document_term_sets:
            term_list = list(doc_terms)
            # Compare all pairs of terms in this document
            for i, term1 in enumerate(term_list):
                if term1 not in self.merged_terms:
                    continue
                for term2 in term_list[i+1:]:
                    if term2 not in self.merged_terms:
                        continue
                    # Record bidirectional co-occurrence
                    self.merged_terms[term1].co_occurrences[term2] = \
                        self.merged_terms[term1].co_occurrences.get(term2, 0) + 1
                    self.merged_terms[term2].co_occurrences[term1] = \
                        self.merged_terms[term2].co_occurrences.get(term1, 0) + 1
        
        # Create relationships for significant co-occurrences
        for term in self.merged_terms.values():
            if term.co_occurrences:
                # Get top 5 most frequently co-occurring terms
                sorted_cooc = sorted(term.co_occurrences.items(), 
                                    key=lambda x: -x[1])[:5]
                for related_term, count in sorted_cooc:
                    # Require at least 2 co-occurrences to create relationship
                    # This filters out coincidental single-document co-occurrences
                    if count >= 2:
                        term.related_terms.add(related_term)
                        self.relationships.append(Relationship(
                            source_term=term.term,
                            target_term=related_term,
                            relationship_type='co_occurs_with',
                            evidence=f"Co-occurred in {count} documents"
                        ))
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive TF-IDF and term statistics.
        
        Returns a dictionary containing:
        - total_documents: Number of documents processed
        - total_terms: Number of unique terms
        - terms_by_source: Breakdown by PDF/code/cross-referenced
        - top_by_tf_idf: Most distinctive terms
        - top_by_frequency: Most common terms
        - top_by_document_frequency: Most universal terms
        """
        stats = {
            'total_documents': self.total_documents,
            'total_terms': len(self.merged_terms),
            'terms_by_source': {
                'pdf_only': len([t for t in self.merged_terms.values() if t.source == TermSource.PDF]),
                'code_only': len([t for t in self.merged_terms.values() if t.source == TermSource.CODE]),
                'cross_referenced': len([t for t in self.merged_terms.values() if t.source == TermSource.BOTH]),
            },
            'top_by_tf_idf': [],
            'top_by_frequency': [],
            'top_by_document_frequency': [],
        }
        
        # Top terms by TF-IDF (most distinctive)
        sorted_by_tfidf = sorted(self.merged_terms.values(), 
                                 key=lambda t: -t.tf_idf_score)[:20]
        stats['top_by_tf_idf'] = [
            {'term': t.term, 'score': round(t.tf_idf_score, 3), 
             'tf': t.term_frequency, 'df': t.document_frequency}
            for t in sorted_by_tfidf
        ]
        
        # Top terms by raw frequency
        sorted_by_freq = sorted(self.merged_terms.values(),
                               key=lambda t: -t.term_frequency)[:20]
        stats['top_by_frequency'] = [
            {'term': t.term, 'frequency': t.term_frequency}
            for t in sorted_by_freq
        ]
        
        # Top terms by document frequency (universal terms)
        sorted_by_df = sorted(self.merged_terms.values(),
                             key=lambda t: -t.document_frequency)[:20]
        stats['top_by_document_frequency'] = [
            {'term': t.term, 'doc_frequency': t.document_frequency,
             'pct': round(t.document_frequency / max(1, self.total_documents) * 100, 1)}
            for t in sorted_by_df
        ]
        
        return stats
    
    # =========================================================================
    # CROSS-REFERENCING AND MERGING
    # =========================================================================
    
    def cross_reference(self) -> Dict[str, ExtractedTerm]:
        """Cross-reference PDF and code terms, create merged vocabulary"""
        
        # Collect all terms by normalized form
        all_terms: Dict[str, List[ExtractedTerm]] = defaultdict(list)
        
        for term in self.pdf_terms.values():
            all_terms[term.normalized].append(term)
        
        for term in self.code_terms.values():
            all_terms[term.normalized].append(term)
        
        # Merge terms
        for normalized, term_list in all_terms.items():
            if len(term_list) == 1:
                # Only found in one source
                self.merged_terms[normalized] = term_list[0]
            else:
                # Found in multiple sources - merge
                merged = self._merge_terms(term_list)
                merged.source = TermSource.BOTH
                merged.confidence = 1.5  # Higher confidence for cross-referenced
                self.merged_terms[normalized] = merged
        
        # Find relationships between terms
        self._discover_relationships()
        
        # Calculate TF-IDF scores
        self.calculate_tf_idf()
        
        # Calculate co-occurrences
        self.calculate_co_occurrences()
        
        return self.merged_terms
    
    def _merge_terms(self, terms: List[ExtractedTerm]) -> ExtractedTerm:
        """Merge multiple extracted terms into one"""
        # Prefer PDF term for display name (business terminology)
        primary = next((t for t in terms if t.source == TermSource.PDF), terms[0])
        
        merged = ExtractedTerm(
            term=primary.term,
            normalized=primary.normalized,
            term_type=primary.term_type,
            source=TermSource.BOTH,
            source_files=[],
            context=primary.context,
            code_references=[]
        )
        
        # Combine all sources, references, and statistics
        for term in terms:
            merged.source_files.extend(term.source_files)
            merged.code_references.extend(term.code_references)
            merged.related_terms.update(term.related_terms)
            merged.term_frequency += term.term_frequency
            # Merge co-occurrences
            for co_term, count in term.co_occurrences.items():
                merged.co_occurrences[co_term] = merged.co_occurrences.get(co_term, 0) + count
            if term.context and not merged.context:
                merged.context = term.context
        
        # Deduplicate source files
        merged.source_files = list(set(merged.source_files))
        
        return merged
    
    def _discover_relationships(self):
        """Discover relationships between terms"""
        
        # Find terms that appear in procedure names implementing concepts
        for term in self.merged_terms.values():
            if term.term_type == TermType.PROCEDURE:
                # Check if procedure name contains other terms
                proc_lower = term.normalized
                for other in self.merged_terms.values():
                    if other.normalized != term.normalized:
                        if other.normalized in proc_lower or proc_lower in other.normalized:
                            self.relationships.append(Relationship(
                                source_term=term.term,
                                target_term=other.term,
                                relationship_type='implements' if term.term_type == TermType.PROCEDURE else 'related_to'
                            ))
    
    # =========================================================================
    # OUTPUT GENERATION
    # =========================================================================
    
    def generate_vocabulary(self, existing_path: Optional[str] = None) -> Dict:
        """Generate vocabulary JSON compatible with unified indexer"""
        
        # Load existing if provided
        if existing_path and os.path.exists(existing_path):
            with open(existing_path, 'r') as f:
                vocab = json.load(f)
        else:
            vocab = {
                "version": "2.0",
                "description": "Domain vocabulary with cross-referenced terms",
                "entries": []
            }
        
        # Get existing terms to avoid duplicates
        existing_terms = set()
        for entry in vocab.get('entries', []):
            for kw in entry.get('keywords', '').split(','):
                existing_terms.add(kw.strip().lower())
        
        # Group terms by category
        by_category = defaultdict(list)
        
        for term in self.merged_terms.values():
            # Skip if already exists
            if term.normalized in existing_terms:
                continue
            
            # Categorize
            if term.source == TermSource.BOTH:
                category = 'cross-referenced'
            elif term.term_type == TermType.MESSAGE_TYPE:
                category = 'swift-mt-messages'
            elif term.term_type == TermType.ACRONYM:
                category = 'acronyms'
            elif term.term_type == TermType.ERROR_CODE:
                category = 'error-codes'
            elif term.term_type == TermType.PROCEDURE:
                category = 'procedures'
            elif term.term_type == TermType.STRUCTURE:
                category = 'data-structures'
            elif term.term_type == TermType.CONSTANT:
                category = 'constants'
            else:
                category = 'domain-terms'
            
            by_category[category].append(term)
        
        # Create entries for each category
        for category, terms in by_category.items():
            if not terms:
                continue
            
            # Sort by TF-IDF score first, then confidence
            terms.sort(key=lambda t: (-t.tf_idf_score, -t.confidence, t.source.value))
            
            # Group into entries of ~5 keywords each
            for i in range(0, len(terms), 5):
                batch = terms[i:i+5]
                main_terms = [t.term for t in batch]
                related = set()
                for t in batch:
                    related.update(t.related_terms)
                
                # Calculate aggregate stats for the batch
                avg_tfidf = sum(t.tf_idf_score for t in batch) / len(batch)
                avg_df = sum(t.document_frequency for t in batch) / len(batch)
                total_tf = sum(t.term_frequency for t in batch)
                
                entry = {
                    "keywords": ",".join(main_terms),
                    "metadata": category,
                    "description": f"{'Cross-referenced' if category == 'cross-referenced' else category.replace('-', ' ').title()} terms",
                    "related_keywords": ",".join(list(related)[:10]),
                    "business_capability": [category.replace('-', ' ').title()],
                    "_extracted": True,
                    "_source": "cross-referenced" if any(t.source == TermSource.BOTH for t in batch) else batch[0].source.value,
                    "_confidence": round(sum(t.confidence for t in batch) / len(batch), 2),
                    "_tf_idf_score": round(avg_tfidf, 3),
                    "_term_frequency": total_tf,
                    "_document_frequency": round(avg_df, 1)
                }
                
                vocab['entries'].append(entry)
        
        return vocab
    
    def generate_knowledge_graph(self) -> Dict:
        """Generate knowledge graph JSON"""
        
        nodes = []
        edges = []
        
        # Create nodes for all terms
        for term in self.merged_terms.values():
            node = {
                "id": term.normalized,
                "label": term.term,
                "type": term.term_type.value,
                "source": term.source.value,
                "confidence": round(term.confidence, 2),
                "files": term.source_files[:5],
                "context": term.context,
                # TF-IDF statistics
                "tf_idf_score": round(term.tf_idf_score, 3),
                "term_frequency": term.term_frequency,
                "document_frequency": term.document_frequency,
                # Top co-occurring terms
                "co_occurs_with": [k for k, v in sorted(term.co_occurrences.items(), 
                                                        key=lambda x: -x[1])[:5]]
            }
            nodes.append(node)
        
        # Create edges for relationships
        seen_edges = set()
        for rel in self.relationships:
            edge_key = (rel.source_term.lower(), rel.target_term.lower(), rel.relationship_type)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append({
                    "source": self.normalize_term(rel.source_term),
                    "target": self.normalize_term(rel.target_term),
                    "type": rel.relationship_type,
                    "evidence": rel.evidence
                })
        
        return {
            "version": "1.0",
            "nodes": nodes,
            "edges": edges,
            "statistics": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "cross_referenced": len([n for n in nodes if n['source'] == 'both']),
                "pdf_only": len([n for n in nodes if n['source'] == 'pdf']),
                "code_only": len([n for n in nodes if n['source'] == 'code'])
            }
        }
    
    def print_summary(self):
        """Print extraction summary"""
        print("\n" + "=" * 60)
        print("KNOWLEDGE EXTRACTION SUMMARY")
        print("=" * 60)
        
        print(f"\nTerms from PDFs: {len(self.pdf_terms)}")
        print(f"Terms from Code: {len(self.code_terms)}")
        print(f"Merged terms: {len(self.merged_terms)}")
        print(f"Relationships: {len(self.relationships)}")
        print(f"Documents processed: {self.total_documents}")
        
        # Count by source
        both = [t for t in self.merged_terms.values() if t.source == TermSource.BOTH]
        pdf_only = [t for t in self.merged_terms.values() if t.source == TermSource.PDF]
        code_only = [t for t in self.merged_terms.values() if t.source == TermSource.CODE]
        
        print(f"\n📊 Term Distribution:")
        print(f"  Cross-referenced (high value): {len(both)}")
        print(f"  PDF only (business terms): {len(pdf_only)}")
        print(f"  Code only (implementation): {len(code_only)}")
        
        if both:
            print(f"\n⭐ Top Cross-Referenced Terms:")
            for term in sorted(both, key=lambda t: -t.tf_idf_score)[:15]:
                print(f"  • {term.term} (TF-IDF: {term.tf_idf_score:.3f}, DF: {term.document_frequency})")
        
        # TF-IDF Statistics
        if self.merged_terms:
            print(f"\n📈 TF-IDF Statistics:")
            
            # Top by TF-IDF (distinctive terms)
            sorted_tfidf = sorted(self.merged_terms.values(), key=lambda t: -t.tf_idf_score)[:10]
            print(f"\n  Top 10 by TF-IDF (distinctive terms):")
            for t in sorted_tfidf:
                print(f"    {t.term}: TF-IDF={t.tf_idf_score:.3f} (TF={t.term_frequency}, DF={t.document_frequency})")
            
            # Top by Document Frequency (universal terms)
            sorted_df = sorted(self.merged_terms.values(), key=lambda t: -t.document_frequency)[:10]
            print(f"\n  Top 10 by Doc Frequency (universal terms):")
            for t in sorted_df:
                pct = t.document_frequency / max(1, self.total_documents) * 100
                print(f"    {t.term}: DF={t.document_frequency} ({pct:.0f}% of docs)")
            
            # Top by raw frequency
            sorted_tf = sorted(self.merged_terms.values(), key=lambda t: -t.term_frequency)[:10]
            print(f"\n  Top 10 by Raw Frequency:")
            for t in sorted_tf:
                print(f"    {t.term}: TF={t.term_frequency}")
        
        # Show relationship types
        rel_types = defaultdict(int)
        for rel in self.relationships:
            rel_types[rel.relationship_type] += 1
        
        if rel_types:
            print(f"\n🔗 Relationships:")
            for rel_type, count in sorted(rel_types.items(), key=lambda x: -x[1]):
                print(f"  {rel_type}: {count}")


def find_files(source_dirs: List[str], extensions: Set[str]) -> List[Tuple[str, str]]:
    """Find files with given extensions"""
    files = []
    
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if not source_path.exists():
            continue
        
        if source_path.is_file():
            ext = source_path.suffix.lower()
            if ext in extensions:
                files.append((str(source_path), ext))
        else:
            for file_path in source_path.rglob('*'):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in extensions:
                        files.append((str(file_path), ext))
    
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cross-reference domain knowledge from PDFs and code"
    )
    parser.add_argument("--docs", "-d", type=str, action='append', default=[],
                        help="Document source directory (PDFs, text files)")
    parser.add_argument("--code", "-c", type=str, action='append', default=[],
                        help="Code source directory (TAL, COBOL, C, etc.)")
    parser.add_argument("--output", "-o", type=str, default="vocabulary_augmented.json",
                        help="Output vocabulary JSON path")
    parser.add_argument("--graph", "-g", type=str, default="knowledge_graph.json",
                        help="Output knowledge graph JSON path (default: knowledge_graph.json)")
    parser.add_argument("--stats", "-s", type=str, default="term_statistics.json",
                        help="Output TF-IDF statistics JSON path (default: term_statistics.json)")
    parser.add_argument("--existing", "-e", type=str, default="keywords.json",
                        help="Existing vocabulary to augment")
    parser.add_argument("--provider", "-p", type=str, default="tachyon",
                        help="LLM provider for PDF extraction")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="LLM model name")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use heuristic extraction only")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--append", "-a", action="store_true",
                        help="Append to existing extraction (load --graph and --stats, skip --docs)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.append:
        if not args.code:
            print("Error: --append requires --code to specify code directories to add")
            sys.exit(1)
        if not os.path.exists(args.graph):
            print(f"Error: --append requires existing graph file: {args.graph}")
            sys.exit(1)
    elif not args.docs and not args.code:
        print("Error: Specify --docs and/or --code directories")
        sys.exit(1)
    
    print("=" * 60)
    print("KNOWLEDGE EXTRACTOR")
    print("=" * 60)
    
    # Create LLM provider (only needed if processing docs)
    llm_provider = None
    if not args.append and not args.no_llm and LLM_AVAILABLE:
        try:
            llm_provider = create_provider(args.provider, args.model)
            print(f"\nLLM Provider: {args.provider} ({llm_provider.model})")
        except Exception as e:
            print(f"\nWarning: LLM not available: {e}")
    
    # Create extractor
    extractor = KnowledgeExtractor(llm_provider)
    
    # If appending, load existing state first
    if args.append:
        print("\n[APPEND MODE] Loading existing extraction...")
        stats_path = args.stats if os.path.exists(args.stats) else None
        extractor.load_existing_state(args.graph, stats_path)
    
    # Find and process PDF/document files (skip if appending)
    if args.docs and not args.append:
        doc_extensions = {'.pdf', '.txt', '.md', '.rst', '.doc', '.docx'}
        doc_files = find_files(args.docs, doc_extensions)
        print(f"\nFound {len(doc_files)} document files")
        
        for i, (file_path, ext) in enumerate(doc_files):
            print(f"  [{i+1}/{len(doc_files)}] {Path(file_path).name}...", end=" ", flush=True)
            terms = extractor.extract_from_pdf(file_path)
            
            # Record for TF-IDF
            extractor.record_document_terms(terms, file_path)
            
            for term in terms:
                if term.normalized in extractor.pdf_terms:
                    # Merge with existing - term appears in multiple docs (higher confidence)
                    existing = extractor.pdf_terms[term.normalized]
                    existing.source_files.extend(term.source_files)
                    existing.source_files = list(set(existing.source_files))  # Dedupe
                    existing.related_terms.update(term.related_terms)
                    existing.term_frequency += term.term_frequency
                    existing.confidence += 0.25  # Boost confidence for each additional doc
                    if term.context and not existing.context:
                        existing.context = term.context
                else:
                    extractor.pdf_terms[term.normalized] = term
            print(f"found {len(terms)} terms")
    elif args.append:
        print("\n[APPEND MODE] Skipping document processing (using existing)")
    
    # Find and process code files
    if args.code:
        # Accept wide range of extensions - language will be auto-detected from content
        code_extensions = {
            # TAL/TACL
            '.tal', '.tacl', '.ddl',
            # COBOL
            '.cob', '.cbl', '.cobol', '.cpy',
            # C/C++
            '.c', '.h', '.cpp', '.hpp', '.cc', '.hh',
            # Java
            '.java',
            # Python
            '.py',
            # Other common
            '.js', '.ts', '.go', '.rs', '.rb', '.cs',
            # Text files (language auto-detected)
            '.txt', '.src', '.inc', '.include',
        }
        code_files = find_files(args.code, code_extensions)
        print(f"\nFound {len(code_files)} code files")
        
        # Track language stats
        lang_counts = {}
        
        for i, (file_path, ext) in enumerate(code_files):
            print(f"  [{i+1}/{len(code_files)}] {Path(file_path).name}...", end=" ", flush=True)
            
            # Read file to detect language
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                lang = extractor.detect_language(content, file_path)
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            except:
                lang = 'unknown'
            
            terms = extractor.extract_from_code(file_path)
            
            # Record for TF-IDF
            extractor.record_document_terms(terms, file_path)
            
            for term in terms:
                if term.normalized in extractor.code_terms:
                    # Merge with existing - term appears in multiple files
                    existing = extractor.code_terms[term.normalized]
                    existing.source_files.extend(term.source_files)
                    existing.source_files = list(set(existing.source_files))  # Dedupe
                    existing.code_references.extend(term.code_references)
                    existing.term_frequency += term.term_frequency
                    existing.confidence += 0.1  # Small boost for each additional file
                else:
                    extractor.code_terms[term.normalized] = term
            print(f"[{lang}] {len(terms)} terms")
        
        # Show language summary
        if lang_counts:
            print("\n  Language detection summary:")
            for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
                print(f"    {lang}: {count} files")
    
    # Cross-reference
    print("\nCross-referencing terms...")
    extractor.cross_reference()
    
    # Generate outputs
    print(f"\nGenerating vocabulary: {args.output}")
    vocab = extractor.generate_vocabulary(args.existing)
    with open(args.output, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"  Total entries: {len(vocab['entries'])}")
    
    if args.graph:
        print(f"Generating knowledge graph: {args.graph}")
        graph = extractor.generate_knowledge_graph()
        with open(args.graph, 'w') as f:
            json.dump(graph, f, indent=2)
        print(f"  Nodes: {graph['statistics']['total_nodes']}")
        print(f"  Edges: {graph['statistics']['total_edges']}")
    
    if args.stats:
        print(f"Generating TF-IDF statistics: {args.stats}")
        stats = extractor.get_statistics()
        with open(args.stats, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Total terms: {stats['total_terms']}")
    
    # Print summary
    extractor.print_summary()
    
    print(f"\n✅ Done!")
    print(f"   Vocabulary: {args.output}")
    if args.graph:
        print(f"   Knowledge graph: {args.graph}")
    if args.stats:
        print(f"   TF-IDF statistics: {args.stats}")


if __name__ == "__main__":
    main()
