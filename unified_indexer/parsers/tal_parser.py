"""
TAL Code Parser - Extracts indexable chunks from TAL source files

Integrates with the existing tal_proc_parser and tal_enhanced_parser
to leverage AST-aware chunking for TAL procedures.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .base import ContentParser
from ..models import (
    IndexableChunk,
    SourceType,
    SemanticType,
    SourceReference,
    DomainMatch
)
from ..vocabulary import DomainVocabulary
from ..call_graph import CallExtractor, extract_calls


class TalCodeParser(ContentParser):
    """
    Parser for TAL (Transaction Application Language) source files
    
    Uses the existing TAL parsers for AST-aware chunking, then applies
    domain concept matching to create richly annotated chunks.
    """
    
    SOURCE_TYPE = SourceType.CODE
    SUPPORTED_EXTENSIONS = ['.tal', '.tacl', '.txt']
    
    def __init__(self, vocabulary: DomainVocabulary, tal_parser_path: Optional[str] = None):
        """
        Initialize the TAL parser
        
        Args:
            vocabulary: Domain vocabulary for concept matching
            tal_parser_path: Path to directory containing tal_proc_parser.py
        """
        super().__init__(vocabulary)
        
        self.tal_proc_parser = None
        self.tal_enhanced_parser = None
        self.call_extractor = CallExtractor()
        
        # Try to import the TAL parsers
        self._load_tal_parsers(tal_parser_path)
    
    def _load_tal_parsers(self, parser_path: Optional[str] = None):
        """Load the TAL parser modules"""
        if parser_path:
            sys.path.insert(0, parser_path)
        
        try:
            import tal_proc_parser
            self.tal_proc_parser = tal_proc_parser
        except ImportError:
            print("Warning: tal_proc_parser not found, using fallback parsing")
        
        try:
            import tal_enhanced_parser
            self.tal_enhanced_parser = tal_enhanced_parser
        except ImportError:
            print("Warning: tal_enhanced_parser not found, using basic parsing")
    
    def can_parse(self, file_path: str) -> bool:
        """Check if file is a TAL source file"""
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def parse(self, content: bytes, source_path: str) -> List[IndexableChunk]:
        """
        Parse TAL content and extract chunks
        
        Uses the enhanced TAL parser when available for AST-aware chunking,
        falls back to regex-based parsing otherwise.
        """
        text = content.decode('utf-8', errors='replace')
        
        if self.tal_enhanced_parser:
            return self._parse_with_enhanced_parser(text, source_path)
        elif self.tal_proc_parser:
            return self._parse_with_proc_parser(text, source_path)
        else:
            return self._parse_with_fallback(text, source_path)
    
    def _parse_with_enhanced_parser(self, text: str, source_path: str) -> List[IndexableChunk]:
        """Parse using the enhanced TAL parser"""
        chunks = []
        
        try:
            # Use the enhanced parser
            parser = self.tal_enhanced_parser.EnhancedTALParser()
            
            # Parse directly from string
            result = parser.parse(text, source_path)
            
            # Extract chunks from ParseResult
            # Procedures
            for proc in result.procedures:
                chunk = self._procedure_detail_to_chunk(proc, text, source_path)
                if chunk:
                    chunks.append(chunk)
            
            # Structs
            for struct in result.structs:
                chunk = self._struct_info_to_chunk(struct, text, source_path)
                if chunk:
                    chunks.append(chunk)
            
            # DEFINEs (as a group or individually for important ones)
            if result.defines:
                chunk = self._defines_to_chunk(result.defines, text, source_path)
                if chunk:
                    chunks.append(chunk)
            
            # Literals
            if result.literals:
                chunk = self._literals_to_chunk(result.literals, text, source_path)
                if chunk:
                    chunks.append(chunk)
                    
        except Exception as e:
            import traceback
            print(f"Enhanced parser failed: {e}")
            traceback.print_exc()
            print("Falling back to basic parser")
            return self._parse_with_fallback(text, source_path)
        
        # If no chunks extracted, use fallback
        if not chunks:
            return self._parse_with_fallback(text, source_path)
        
        return chunks
    
    def _procedure_detail_to_chunk(self, proc, text: str, source_path: str) -> Optional[IndexableChunk]:
        """Convert a ProcedureDetail to an IndexableChunk"""
        try:
            # Extract procedure body from source
            lines = text.split('\n')
            
            if proc.body_start_line and proc.body_end_line:
                start_line = proc.location.line
                end_line = proc.body_end_line
                proc_lines = lines[start_line - 1:end_line]
                proc_text = '\n'.join(proc_lines)
            else:
                # Forward/external - just use declaration
                start_line = proc.location.line
                end_line = start_line + 5
                proc_lines = lines[start_line - 1:end_line]
                proc_text = '\n'.join(proc_lines)
            
            # Build metadata
            metadata = {
                'procedure_name': proc.name,
                'is_main': proc.is_main,
                'is_forward': proc.is_forward,
                'is_external': proc.is_external,
                'complexity': proc.complexity,
                'parameters': [p['name'] for p in proc.parameters],
                'calls': [c.callee for c in proc.calls],
                'called_by': proc.called_by,
                'local_vars': [v['name'] for v in proc.local_vars],
                'subprocs': proc.subprocs,
                'attributes': proc.attributes
            }
            
            if proc.return_type:
                metadata['return_type'] = proc.return_type.value
            
            # Build embedding text with semantic info
            embedding_parts = [
                f"TAL procedure {proc.name}",
                f"parameters: {', '.join(p['name'] for p in proc.parameters)}" if proc.parameters else "",
                f"calls: {', '.join(c.callee for c in proc.calls)}" if proc.calls else "",
                f"complexity: {proc.complexity}",
            ]
            if proc.is_main:
                embedding_parts.append("MAIN entry point")
            embedding_text = ' '.join(filter(None, embedding_parts))
            
            # Domain analysis
            domain_matches = self.vocabulary.match_text(proc_text) if self.vocabulary else []
            
            return IndexableChunk(
                chunk_id=self.generate_chunk_id(source_path, proc_text, f"proc_{proc.name}"),
                text=proc_text,
                embedding_text=embedding_text + '\n' + proc_text[:500],
                source_type=SourceType.CODE,
                semantic_type=SemanticType.PROCEDURE,
                source_ref=SourceReference(
                    file_path=source_path,
                    line_start=start_line,
                    line_end=end_line,
                ),
                domain_matches=domain_matches,
                metadata=metadata
            )
        except Exception as e:
            print(f"Error converting procedure {proc.name}: {e}")
            return None
    
    def _struct_info_to_chunk(self, struct, text: str, source_path: str) -> Optional[IndexableChunk]:
        """Convert a StructInfo to an IndexableChunk"""
        try:
            lines = text.split('\n')
            start_line = struct.location.line
            
            # Find END of struct
            end_line = start_line
            for i in range(start_line - 1, min(start_line + 50, len(lines))):
                if 'END' in lines[i].upper() or lines[i].strip().endswith(';'):
                    end_line = i + 1
                    break
            
            struct_lines = lines[start_line - 1:end_line]
            struct_text = '\n'.join(struct_lines)
            
            # Build metadata
            field_info = [
                {'name': f.name, 'type': f.field_type.value, 'is_array': f.is_array}
                for f in struct.fields
            ]
            
            metadata = {
                'struct_name': struct.name,
                'fields': field_info,
                'field_count': len(struct.fields),
                'total_size': struct.total_size,
                'is_referral': struct.is_referral
            }
            
            # Embedding text
            field_names = ', '.join(f.name for f in struct.fields)
            embedding_text = f"TAL STRUCT {struct.name} with fields: {field_names}"
            
            domain_matches = self.vocabulary.match_text(struct_text) if self.vocabulary else []
            
            return IndexableChunk(
                chunk_id=self.generate_chunk_id(source_path, struct_text, f"struct_{struct.name}"),
                text=struct_text,
                embedding_text=embedding_text + '\n' + struct_text,
                source_type=SourceType.CODE,
                semantic_type=SemanticType.STRUCT_DEF,
                source_ref=SourceReference(
                    file_path=source_path,
                    line_start=start_line,
                    line_end=end_line,
                ),
                domain_matches=domain_matches,
                metadata=metadata
            )
        except Exception as e:
            print(f"Error converting struct {struct.name}: {e}")
            return None
    
    def _defines_to_chunk(self, defines: list, text: str, source_path: str) -> Optional[IndexableChunk]:
        """Convert DEFINEs to an IndexableChunk"""
        try:
            # Group defines into a single chunk
            define_texts = []
            for d in defines[:50]:  # Limit to first 50
                if d.is_macro:
                    define_texts.append(f"DEFINE {d.name}({', '.join(d.params)}) = {d.value}")
                else:
                    define_texts.append(f"DEFINE {d.name} = {d.value}")
            
            define_text = '\n'.join(define_texts)
            
            metadata = {
                'define_count': len(defines),
                'define_names': [d.name for d in defines],
                'macros': [d.name for d in defines if d.is_macro]
            }
            
            embedding_text = f"TAL DEFINE constants and macros: {', '.join(d.name for d in defines[:20])}"
            
            domain_matches = self.vocabulary.match_text(define_text) if self.vocabulary else []
            
            return IndexableChunk(
                chunk_id=self.generate_chunk_id(source_path, define_text, "defines"),
                text=define_text,
                embedding_text=embedding_text,
                source_type=SourceType.CODE,
                semantic_type=SemanticType.VARIABLE_DECL,
                source_ref=SourceReference(
                    file_path=source_path,
                    line_start=defines[0].location.line if defines else 1,
                    line_end=defines[-1].location.line if defines else 1,
                ),
                domain_matches=domain_matches,
                metadata=metadata
            )
        except Exception as e:
            print(f"Error converting defines: {e}")
            return None
    
    def _literals_to_chunk(self, literals: list, text: str, source_path: str) -> Optional[IndexableChunk]:
        """Convert LITERALs to an IndexableChunk"""
        try:
            literal_texts = [f"LITERAL {l.name} = {l.value}" for l in literals[:50]]
            literal_text = '\n'.join(literal_texts)
            
            metadata = {
                'literal_count': len(literals),
                'literal_names': [l.name for l in literals]
            }
            
            embedding_text = f"TAL LITERAL constants: {', '.join(l.name for l in literals[:20])}"
            
            domain_matches = self.vocabulary.match_text(literal_text) if self.vocabulary else []
            
            return IndexableChunk(
                chunk_id=self.generate_chunk_id(source_path, literal_text, "literals"),
                text=literal_text,
                embedding_text=embedding_text,
                source_type=SourceType.CODE,
                semantic_type=SemanticType.VARIABLE_DECL,
                source_ref=SourceReference(
                    file_path=source_path,
                    line_start=literals[0].location.line if literals else 1,
                    line_end=literals[-1].location.line if literals else 1,
                ),
                domain_matches=domain_matches,
                metadata=metadata
            )
        except Exception as e:
            print(f"Error converting literals: {e}")
            return None
    
    def _parse_with_proc_parser(self, text: str, source_path: str) -> List[IndexableChunk]:
        """Parse using the basic procedure parser"""
        chunks = []
        
        try:
            symbol_table = self.tal_proc_parser.SymbolTable()
            procedures, errors = self.tal_proc_parser.parse_multiple_procedures(
                text, source_path, symbol_table
            )
            
            # Get procedure declarations for line ranges
            proc_declarations = self.tal_proc_parser.find_procedure_declarations(text)
            lines = text.split('\n')
            
            for proc_node in procedures:
                chunk = self._procedure_node_to_chunk(
                    proc_node, proc_declarations, lines, source_path
                )
                if chunk:
                    chunks.append(chunk)
                    
        except Exception as e:
            print(f"Proc parser failed: {e}, falling back")
            return self._parse_with_fallback(text, source_path)
        
        return chunks
    
    def _parse_with_fallback(self, text: str, source_path: str) -> List[IndexableChunk]:
        """Fallback regex-based parsing when TAL parsers unavailable"""
        chunks = []
        lines = text.split('\n')
        
        # Find procedure declarations with regex
        proc_pattern = re.compile(
            r'^\s*(?:(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING|FIXED|UNSIGNED(?:\([^)]*\))?)\s+)?'
            r'PROC\s+([a-zA-Z_][a-zA-Z0-9_\^]*)',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Find all procedure starts
        proc_starts = []
        for i, line in enumerate(lines):
            match = proc_pattern.match(line)
            if match:
                return_type = match.group(1)
                proc_name = match.group(2)
                proc_starts.append((i, proc_name, return_type))
        
        # Extract each procedure
        for idx, (start_line, proc_name, return_type) in enumerate(proc_starts):
            # Find end of procedure (next procedure or end of file)
            if idx + 1 < len(proc_starts):
                end_line = proc_starts[idx + 1][0]
            else:
                end_line = len(lines)
            
            # Extract procedure text
            proc_text = '\n'.join(lines[start_line:end_line])
            
            # Create chunk
            chunk = self._create_procedure_chunk(
                proc_name=proc_name,
                proc_text=proc_text,
                source_path=source_path,
                line_start=start_line + 1,
                line_end=end_line,
                return_type=return_type
            )
            chunks.append(chunk)
        
        # If no procedures found, chunk the whole file
        if not chunks:
            chunks = self._chunk_file_content(text, source_path)
        
        return chunks
    
    def _procedure_to_chunk(self, 
                            proc_node, 
                            full_text: str,
                            source_path: str) -> Optional[IndexableChunk]:
        """Convert an enhanced parser procedure node to a chunk"""
        try:
            proc_name = proc_node.name
            location = proc_node.location
            
            # Get procedure attributes
            return_type = proc_node.attributes.get('return_type')
            is_main = proc_node.attributes.get('is_main', False)
            
            # Extract the procedure text from full source
            # This is approximate - ideally we'd have line ranges
            proc_text = proc_node.to_sexp()  # Use S-expression as representation
            
            # Get actual source if possible
            if hasattr(location, 'line') and location.line > 0:
                lines = full_text.split('\n')
                # Estimate procedure length
                proc_lines = []
                in_proc = False
                for i, line in enumerate(lines):
                    if i + 1 >= location.line:
                        in_proc = True
                    if in_proc:
                        proc_lines.append(line)
                        if re.search(r'\bEND\b', line, re.IGNORECASE):
                            break
                proc_text = '\n'.join(proc_lines)
            
            return self._create_procedure_chunk(
                proc_name=proc_name,
                proc_text=proc_text,
                source_path=source_path,
                line_start=location.line if hasattr(location, 'line') else None,
                line_end=None,
                return_type=return_type,
                is_main=is_main,
                ast_node=proc_node
            )
            
        except Exception as e:
            print(f"Error converting procedure node: {e}")
            return None
    
    def _procedure_node_to_chunk(self,
                                  proc_node,
                                  proc_declarations: List[Tuple],
                                  lines: List[str],
                                  source_path: str) -> Optional[IndexableChunk]:
        """Convert a proc_parser node to a chunk"""
        try:
            proc_name = proc_node.name
            
            # Find the declaration for this procedure
            proc_decl = None
            for start_line, name, declaration in proc_declarations:
                if name == proc_name:
                    proc_decl = (start_line, declaration)
                    break
            
            if not proc_decl:
                return None
            
            start_line, declaration = proc_decl
            
            # Estimate end line
            decl_lines = declaration.count('\n') + 1
            end_line = start_line + decl_lines + 50  # Rough estimate
            
            # Get actual procedure text
            proc_text = '\n'.join(lines[start_line-1:min(end_line, len(lines))])
            
            return self._create_procedure_chunk(
                proc_name=proc_name,
                proc_text=proc_text,
                source_path=source_path,
                line_start=start_line,
                line_end=end_line,
                return_type=proc_node.attributes.get('return_type'),
                is_main=proc_node.attributes.get('is_main', False),
                ast_node=proc_node
            )
            
        except Exception as e:
            print(f"Error converting procedure node: {e}")
            return None
    
    def _create_procedure_chunk(self,
                                 proc_name: str,
                                 proc_text: str,
                                 source_path: str,
                                 line_start: Optional[int] = None,
                                 line_end: Optional[int] = None,
                                 return_type: Optional[str] = None,
                                 is_main: bool = False,
                                 ast_node = None) -> IndexableChunk:
        """Create a chunk from procedure information"""
        
        # Match domain concepts
        domain_matches = self.match_domain_concepts(proc_text)
        
        # Extract additional keywords from code
        code_keywords = self._extract_tal_keywords(proc_text)
        keyword_matches = self.vocabulary.match_keywords(code_keywords)
        
        # Merge matches (avoiding duplicates)
        seen_canonical = set(m.canonical_term for m in domain_matches)
        for km in keyword_matches:
            if km.canonical_term not in seen_canonical:
                domain_matches.append(km)
                seen_canonical.add(km.canonical_term)
        
        # Build metadata
        metadata = {
            'procedure_name': proc_name,
            'language': 'TAL',
            'is_main': is_main
        }
        if return_type:
            metadata['return_type'] = return_type
        
        # Extract called procedures
        calls = []
        system_funcs = []
        
        if ast_node:
            # Use AST-based extraction
            calls = self._extract_procedure_calls(ast_node)
            system_funcs = self._extract_system_functions(ast_node)
        else:
            # Use regex-based extraction as fallback
            call_infos = self.call_extractor.extract_calls(
                proc_text, 'tal', procedure_name=proc_name
            )
            calls = [c.target for c in call_infos if c.call_type.value != 'system']
            system_funcs = [c.target for c in call_infos if c.call_type.value == 'system']
        
        if calls:
            metadata['calls'] = calls
        if system_funcs:
            metadata['system_functions'] = system_funcs
        
        # Create source reference
        source_ref = SourceReference(
            file_path=source_path,
            line_start=line_start,
            line_end=line_end,
            procedure_name=proc_name
        )
        
        # Create embedding text - include calls for semantic matching
        embedding_text = self.create_embedding_text(
            proc_text,
            SemanticType.PROCEDURE,
            domain_matches,
            metadata
        )
        
        # Add calls to embedding text if present
        if calls:
            embedding_text += f"\nCalls: {', '.join(calls[:20])}"
        
        return IndexableChunk(
            chunk_id=self.generate_chunk_id(source_path, proc_text, proc_name),
            text=proc_text,
            embedding_text=embedding_text,
            source_type=SourceType.CODE,
            semantic_type=SemanticType.PROCEDURE,
            source_ref=source_ref,
            domain_matches=domain_matches,
            metadata=metadata
        )
    
    def _struct_to_chunk(self, struct_node, text: str, source_path: str) -> Optional[IndexableChunk]:
        """Convert a structure definition to a chunk"""
        try:
            struct_name = struct_node.name or struct_node.attributes.get('struct_name', 'unknown')
            struct_text = struct_node.value or struct_node.to_sexp()
            
            domain_matches = self.match_domain_concepts(struct_text)
            
            metadata = {
                'structure_name': struct_name,
                'language': 'TAL'
            }
            
            source_ref = SourceReference(
                file_path=source_path,
                line_start=struct_node.location.line if hasattr(struct_node, 'location') else None
            )
            
            embedding_text = self.create_embedding_text(
                struct_text,
                SemanticType.STRUCT_DEF,
                domain_matches,
                metadata
            )
            
            return IndexableChunk(
                chunk_id=self.generate_chunk_id(source_path, struct_text, struct_name),
                text=struct_text,
                embedding_text=embedding_text,
                source_type=SourceType.CODE,
                semantic_type=SemanticType.STRUCT_DEF,
                source_ref=source_ref,
                domain_matches=domain_matches,
                metadata=metadata
            )
        except Exception as e:
            print(f"Error converting struct node: {e}")
            return None
    
    def _literal_to_chunk(self, literal_node, text: str, source_path: str) -> Optional[IndexableChunk]:
        """Convert a literal declaration to a chunk"""
        try:
            literal_text = literal_node.value or literal_node.to_sexp()
            
            domain_matches = self.match_domain_concepts(literal_text)
            
            metadata = {
                'type': 'literal_declaration',
                'language': 'TAL'
            }
            
            source_ref = SourceReference(
                file_path=source_path,
                line_start=literal_node.location.line if hasattr(literal_node, 'location') else None
            )
            
            embedding_text = self.create_embedding_text(
                literal_text,
                SemanticType.VARIABLE_DECL,
                domain_matches,
                metadata
            )
            
            return IndexableChunk(
                chunk_id=self.generate_chunk_id(source_path, literal_text, 'literal'),
                text=literal_text,
                embedding_text=embedding_text,
                source_type=SourceType.CODE,
                semantic_type=SemanticType.VARIABLE_DECL,
                source_ref=source_ref,
                domain_matches=domain_matches,
                metadata=metadata
            )
        except Exception as e:
            print(f"Error converting literal node: {e}")
            return None
    
    def _chunk_file_content(self, text: str, source_path: str) -> List[IndexableChunk]:
        """Chunk file content when no procedures found"""
        chunks = []
        text_chunks = self.chunk_text(text, max_chunk_size=1500, overlap=150)
        
        for i, chunk_text in enumerate(text_chunks):
            domain_matches = self.match_domain_concepts(chunk_text)
            
            source_ref = SourceReference(file_path=source_path)
            
            embedding_text = self.create_embedding_text(
                chunk_text,
                SemanticType.UNKNOWN,
                domain_matches,
                {'chunk_index': i}
            )
            
            chunks.append(IndexableChunk(
                chunk_id=self.generate_chunk_id(source_path, chunk_text, str(i)),
                text=chunk_text,
                embedding_text=embedding_text,
                source_type=SourceType.CODE,
                semantic_type=SemanticType.UNKNOWN,
                source_ref=source_ref,
                domain_matches=domain_matches,
                metadata={'chunk_index': i, 'language': 'TAL'}
            ))
        
        return chunks
    
    def _extract_tal_keywords(self, code: str) -> List[str]:
        """Extract TAL-specific keywords and identifiers"""
        keywords = []
        
        # TAL system functions
        system_funcs = re.findall(r'\$[A-Z][A-Z0-9_]*', code)
        keywords.extend(system_funcs)
        
        # Procedure calls
        calls = re.findall(r'CALL\s+([A-Za-z_][A-Za-z0-9_\^]*)', code, re.IGNORECASE)
        keywords.extend(calls)
        
        # Variable names with potential domain meaning
        # Look for variables that might have meaningful names
        identifiers = re.findall(r'\b([A-Z][A-Z0-9_]{3,})\b', code)
        keywords.extend(identifiers)
        
        # String literals that might contain domain terms
        strings = re.findall(r'"([^"]+)"', code)
        keywords.extend(strings)
        
        return list(set(keywords))
    
    def _extract_procedure_calls(self, ast_node) -> List[str]:
        """Extract procedure names called by this procedure"""
        calls = []
        
        def find_calls(node):
            if hasattr(node, 'type'):
                if node.type in ['call_stmt', 'system_function_call']:
                    func = node.attributes.get('function') or node.name
                    if func:
                        calls.append(func)
            
            if hasattr(node, 'children'):
                for child in node.children:
                    find_calls(child)
        
        find_calls(ast_node)
        return list(set(calls))
    
    def _extract_system_functions(self, ast_node) -> List[str]:
        """Extract system functions used in this procedure"""
        sys_funcs = []
        
        def find_sys_funcs(node):
            if hasattr(node, 'type'):
                if node.type == 'system_function':
                    name = node.attributes.get('original_name') or node.name
                    if name:
                        sys_funcs.append(name)
            
            if hasattr(node, 'children'):
                for child in node.children:
                    find_sys_funcs(child)
        
        find_sys_funcs(ast_node)
        return list(set(sys_funcs))
