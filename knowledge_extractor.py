#!/usr/bin/env python3
"""
Knowledge Extractor - Build domain vocabulary with cross-referenced concepts

This script:
1. Extracts business terminology from PDFs using LLM
2. Extracts domain-relevant symbols from TAL/COBOL code using smart filtering
3. Cross-references to find high-value terms appearing in both
4. Builds a tiered vocabulary with relationship links
5. Outputs augmented keywords.json and knowledge graph

Usage:
    python knowledge_extractor.py --docs ./pdfs --code ./tal_code --output vocabulary.json
    python knowledge_extractor.py --docs ./specs --code ./src --graph knowledge_graph.json
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

# Import LLM provider
try:
    from llm_provider import create_provider, LLMProvider, LLMResponse
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: llm_provider not found.")

# PDF support
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. PDF extraction will be limited.")


class TermSource(Enum):
    """Where a term was found"""
    PDF = "pdf"
    CODE = "code"
    BOTH = "both"


class TermType(Enum):
    """Type of term"""
    BUSINESS_CONCEPT = "business_concept"      # From PDF - business terminology
    STRUCTURE = "structure"                     # TAL STRUCT, COBOL record
    PROCEDURE = "procedure"                     # TAL PROC, COBOL paragraph
    CONSTANT = "constant"                       # DEFINE, LITERAL, 88 level
    FIELD = "field"                            # Structure field
    MESSAGE_TYPE = "message_type"              # MT-103, MT-202, etc.
    ERROR_CODE = "error_code"                  # Error codes
    ACRONYM = "acronym"                        # BIC, IBAN, OFAC, etc.


@dataclass
class ExtractedTerm:
    """A term extracted from documents or code"""
    term: str
    normalized: str  # Lowercase, normalized form
    term_type: TermType
    source: TermSource
    source_files: List[str] = field(default_factory=list)
    context: str = ""  # Brief context/description
    related_terms: Set[str] = field(default_factory=set)
    code_references: List[Dict] = field(default_factory=list)  # {file, line, type}
    confidence: float = 1.0


@dataclass 
class Relationship:
    """Relationship between terms for knowledge graph"""
    source_term: str
    target_term: str
    relationship_type: str  # implements, contains, calls, related_to, defined_in
    evidence: str = ""  # Where this relationship was found


class KnowledgeExtractor:
    """
    Extract and cross-reference domain knowledge from PDFs and code.
    """
    
    # Domain-relevant patterns to look for in code
    DOMAIN_PATTERNS = [
        r'wire', r'transfer', r'payment', r'message', r'swift', r'fedwire',
        r'chips', r'ach', r'ofac', r'sanction', r'screen', r'compliance',
        r'bic', r'iban', r'aba', r'routing', r'account', r'beneficiary',
        r'originator', r'sender', r'receiver', r'correspondent', r'intermediary',
        r'settlement', r'clearing', r'netting', r'reconcil', r'balance',
        r'currency', r'amount', r'rate', r'fee', r'charge',
        r'validate', r'verify', r'check', r'error', r'reject', r'repair',
        r'queue', r'route', r'process', r'transaction', r'reference',
        r'mt\d{3}', r'uetr', r'uti', r'lei',  # Message types and IDs
    ]
    
    # Generic names to exclude
    GENERIC_NAMES = {
        'i', 'j', 'k', 'n', 'm', 'x', 'y', 'z',
        'tmp', 'temp', 'buf', 'buffer', 'ptr', 'len', 'length', 'size', 'count',
        'result', 'status', 'ret', 'retval', 'rc', 'err', 'error',
        'str', 'string', 'num', 'int', 'val', 'value', 'data', 'info',
        'flag', 'flags', 'index', 'idx', 'pos', 'offset',
        'begin', 'end', 'start', 'stop', 'first', 'last', 'next', 'prev',
        'input', 'output', 'in', 'out', 'src', 'dst', 'source', 'dest',
        'true', 'false', 'null', 'none', 'ok', 'fail',
    }
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider
        self.pdf_terms: Dict[str, ExtractedTerm] = {}
        self.code_terms: Dict[str, ExtractedTerm] = {}
        self.merged_terms: Dict[str, ExtractedTerm] = {}
        self.relationships: List[Relationship] = []
    
    def is_domain_relevant(self, name: str) -> bool:
        """Check if a name appears domain-relevant"""
        name_lower = name.lower()
        
        # Exclude generic names
        if name_lower in self.GENERIC_NAMES:
            return False
        
        # Too short
        if len(name_lower) < 3:
            return False
        
        # Check against domain patterns
        for pattern in self.DOMAIN_PATTERNS:
            if re.search(pattern, name_lower):
                return True
        
        # Check for meaningful structure (contains underscore with domain word)
        parts = name_lower.replace('-', '_').split('_')
        for part in parts:
            if len(part) > 3 and part not in self.GENERIC_NAMES:
                for pattern in self.DOMAIN_PATTERNS:
                    if re.search(pattern, part):
                        return True
        
        return False
    
    def normalize_term(self, term: str) -> str:
        """Normalize a term for comparison"""
        # Lowercase
        normalized = term.lower()
        # Replace separators with underscore
        normalized = re.sub(r'[-\s]+', '_', normalized)
        # Remove special chars
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        return normalized
    
    # =========================================================================
    # PDF EXTRACTION
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
    
    def extract_from_code(self, file_path: str) -> List[ExtractedTerm]:
        """Extract domain-relevant symbols from code file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"    Error reading code: {e}")
            return []
        
        ext = Path(file_path).suffix.lower()
        
        if ext in ('.tal', '.tacl', '.ddl'):
            return self._extract_from_tal(content, file_path)
        elif ext in ('.cob', '.cbl', '.cobol', '.cpy'):
            return self._extract_from_cobol(content, file_path)
        else:
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
        
        # Combine all sources and references
        for term in terms:
            merged.source_files.extend(term.source_files)
            merged.code_references.extend(term.code_references)
            merged.related_terms.update(term.related_terms)
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
            
            # Sort by confidence and source
            terms.sort(key=lambda t: (-t.confidence, t.source.value))
            
            # Group into entries of ~5 keywords each
            for i in range(0, len(terms), 5):
                batch = terms[i:i+5]
                main_terms = [t.term for t in batch]
                related = set()
                for t in batch:
                    related.update(t.related_terms)
                
                entry = {
                    "keywords": ",".join(main_terms),
                    "metadata": category,
                    "description": f"{'Cross-referenced' if category == 'cross-referenced' else category.replace('-', ' ').title()} terms",
                    "related_keywords": ",".join(list(related)[:10]),
                    "business_capability": [category.replace('-', ' ').title()],
                    "_extracted": True,
                    "_source": "cross-referenced" if any(t.source == TermSource.BOTH for t in batch) else batch[0].source.value,
                    "_confidence": sum(t.confidence for t in batch) / len(batch)
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
                "confidence": term.confidence,
                "files": term.source_files[:5],
                "context": term.context
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
            for term in sorted(both, key=lambda t: -t.confidence)[:15]:
                print(f"  • {term.term} ({term.term_type.value})")
        
        # Show relationship types
        rel_types = defaultdict(int)
        for rel in self.relationships:
            rel_types[rel.relationship_type] += 1
        
        if rel_types:
            print(f"\n🔗 Relationships:")
            for rel_type, count in rel_types.items():
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
    parser.add_argument("--graph", "-g", type=str, default=None,
                        help="Output knowledge graph JSON path")
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
    
    args = parser.parse_args()
    
    if not args.docs and not args.code:
        print("Error: Specify --docs and/or --code directories")
        sys.exit(1)
    
    print("=" * 60)
    print("KNOWLEDGE EXTRACTOR")
    print("=" * 60)
    
    # Create LLM provider
    llm_provider = None
    if not args.no_llm and LLM_AVAILABLE:
        try:
            llm_provider = create_provider(args.provider, args.model)
            print(f"\nLLM Provider: {args.provider} ({llm_provider.model})")
        except Exception as e:
            print(f"\nWarning: LLM not available: {e}")
    
    # Create extractor
    extractor = KnowledgeExtractor(llm_provider)
    
    # Find and process PDF/document files
    if args.docs:
        doc_extensions = {'.pdf', '.txt', '.md', '.rst', '.doc', '.docx'}
        doc_files = find_files(args.docs, doc_extensions)
        print(f"\nFound {len(doc_files)} document files")
        
        for i, (file_path, ext) in enumerate(doc_files):
            print(f"  [{i+1}/{len(doc_files)}] {Path(file_path).name}...", end=" ", flush=True)
            terms = extractor.extract_from_pdf(file_path)
            for term in terms:
                extractor.pdf_terms[term.normalized] = term
            print(f"found {len(terms)} terms")
    
    # Find and process code files
    if args.code:
        code_extensions = {'.tal', '.tacl', '.ddl', '.cob', '.cbl', '.cobol', '.cpy',
                          '.c', '.h', '.cpp', '.hpp', '.java', '.py'}
        code_files = find_files(args.code, code_extensions)
        print(f"\nFound {len(code_files)} code files")
        
        for i, (file_path, ext) in enumerate(code_files):
            print(f"  [{i+1}/{len(code_files)}] {Path(file_path).name}...", end=" ", flush=True)
            terms = extractor.extract_from_code(file_path)
            for term in terms:
                if term.normalized in extractor.code_terms:
                    # Merge with existing
                    existing = extractor.code_terms[term.normalized]
                    existing.source_files.extend(term.source_files)
                    existing.code_references.extend(term.code_references)
                else:
                    extractor.code_terms[term.normalized] = term
            print(f"found {len(terms)} terms")
    
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
    
    # Print summary
    extractor.print_summary()
    
    print(f"\n✅ Done!")
    print(f"   Vocabulary: {args.output}")
    if args.graph:
        print(f"   Knowledge graph: {args.graph}")


if __name__ == "__main__":
    main()
