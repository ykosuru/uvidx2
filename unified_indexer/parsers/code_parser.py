"""
Generic Code Parser

Handles common programming languages:
- C (.c, .h)
- C++ (.cpp, .hpp, .cc, .cxx)
- Java (.java)
- Python (.py)
- JavaScript/TypeScript (.js, .ts, .jsx, .tsx)
- C# (.cs)
- Go (.go)
- Rust (.rs)

Provides basic function/method extraction and intelligent chunking.
Includes call graph extraction for tracking function calls.
"""

import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .base import ContentParser
from ..models import (
    IndexableChunk,
    SourceType,
    SemanticType,
    SourceReference,
    DomainMatch
)
from ..call_graph import CallExtractor, extract_calls


@dataclass
class CodeFunction:
    """Represents a parsed function/method"""
    name: str
    signature: str
    body: str
    start_line: int
    end_line: int
    language: str
    class_name: Optional[str] = None
    docstring: Optional[str] = None
    
    @property
    def full_name(self) -> str:
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name


class GenericCodeParser(ContentParser):
    """
    Parser for common programming languages.
    
    Uses regex-based extraction for broad compatibility.
    For more accurate parsing, integrate with tree-sitter.
    """
    
    SOURCE_TYPE = SourceType.CODE
    
    SUPPORTED_EXTENSIONS = [
        # C/C++
        '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.c++', '.h++',
        # Java
        '.java',
        # Python
        '.py', '.pyw',
        # JavaScript/TypeScript
        '.js', '.jsx', '.ts', '.tsx', '.mjs',
        # C#
        '.cs',
        # Go
        '.go',
        # Rust
        '.rs',
        # Ruby
        '.rb',
        # PHP
        '.php',
        # Swift
        '.swift',
        # Kotlin
        '.kt', '.kts',
        # Scala
        '.scala',
    ]
    
    # Language detection by extension
    LANGUAGE_MAP = {
        '.c': 'c', '.h': 'c',
        '.cpp': 'cpp', '.hpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
        '.hxx': 'cpp', '.c++': 'cpp', '.h++': 'cpp',
        '.java': 'java',
        '.py': 'python', '.pyw': 'python',
        '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
        '.ts': 'typescript', '.tsx': 'typescript',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin', '.kts': 'kotlin',
        '.scala': 'scala',
    }
    
    # Function patterns by language family
    FUNCTION_PATTERNS = {
        'c': [
            # Standard single-line: int main(void) {
            r'^[\w\s\*]+\s+(\w+)\s*\([^)]*\)\s*\{',
            # PostgreSQL style: return type on separate line
            # void\nexec_simple_query(...)
            r'^(\w+)\s*\([^)]*\)\s*$',
        ],
        'cpp': [
            r'^[\w\s\*\&<>:]+\s+(\w+::)?(\w+)\s*\([^)]*\)\s*(const)?\s*\{',
            r'template\s*<[^>]*>\s*[\w\s\*\&<>:]+\s+(\w+)\s*\([^)]*\)\s*\{',
        ],
        'java': [
            r'^\s*(public|private|protected)?\s*(static)?\s*[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(throws\s+[\w,\s]+)?\s*\{',
        ],
        'python': [
            r'^\s*def\s+(\w+)\s*\([^)]*\)\s*(->\s*[^:]+)?\s*:',
            r'^\s*async\s+def\s+(\w+)\s*\([^)]*\)\s*(->\s*[^:]+)?\s*:',
        ],
        'javascript': [
            r'^\s*function\s+(\w+)\s*\([^)]*\)\s*\{',
            r'^\s*(const|let|var)\s+(\w+)\s*=\s*function\s*\([^)]*\)\s*\{',
            r'^\s*(const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{',
            r'^\s*(\w+)\s*\([^)]*\)\s*\{',
        ],
        'typescript': [
            r'^\s*function\s+(\w+)\s*(<[^>]*>)?\s*\([^)]*\)\s*(:\s*[^{]+)?\s*\{',
            r'^\s*(const|let|var)\s+(\w+)\s*:\s*[^=]+=\s*\([^)]*\)\s*=>\s*\{',
        ],
        'csharp': [
            r'^\s*(public|private|protected|internal)?\s*(static|virtual|override|async)?\s*[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{',
        ],
        'go': [
            r'^\s*func\s+(\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\([^)]*\)\s*[^{]*\{',
        ],
        'rust': [
            r'^\s*(pub\s+)?fn\s+(\w+)\s*(<[^>]*>)?\s*\([^)]*\)\s*(->\s*[^{]+)?\s*\{',
        ],
        'ruby': [
            r'^\s*def\s+(\w+[?!]?)\s*(\([^)]*\))?',
        ],
        'php': [
            r'^\s*(public|private|protected)?\s*(static)?\s*function\s+(\w+)\s*\([^)]*\)\s*(:\s*\??\w+)?\s*\{',
        ],
        'swift': [
            r'^\s*(public|private|internal|fileprivate|open)?\s*func\s+(\w+)\s*\([^)]*\)\s*(->\s*[^{]+)?\s*\{',
        ],
        'kotlin': [
            r'^\s*(public|private|protected|internal)?\s*fun\s+(\w+)\s*\([^)]*\)\s*(:\s*[^{]+)?\s*\{',
        ],
        'scala': [
            r'^\s*def\s+(\w+)\s*(\[[^\]]*\])?\s*\([^)]*\)\s*(:\s*[^=]+)?\s*=?\s*\{',
        ],
    }
    
    # Class patterns by language
    CLASS_PATTERNS = {
        'java': r'^\s*(public|private|protected)?\s*(abstract|final)?\s*class\s+(\w+)',
        'python': r'^\s*class\s+(\w+)\s*(\([^)]*\))?\s*:',
        'javascript': r'^\s*class\s+(\w+)\s*(extends\s+\w+)?\s*\{',
        'typescript': r'^\s*(export\s+)?(abstract\s+)?class\s+(\w+)',
        'csharp': r'^\s*(public|private|protected|internal)?\s*(abstract|sealed|static)?\s*class\s+(\w+)',
        'cpp': r'^\s*class\s+(\w+)\s*(:\s*(public|private|protected)\s+\w+)?\s*\{',
        'kotlin': r'^\s*(data\s+|sealed\s+|open\s+)?class\s+(\w+)',
        'scala': r'^\s*(case\s+|sealed\s+|abstract\s+)?class\s+(\w+)',
        'swift': r'^\s*(public|private|internal|fileprivate|open)?\s*class\s+(\w+)',
        'rust': r'^\s*(pub\s+)?struct\s+(\w+)',
        'go': r'^\s*type\s+(\w+)\s+struct\s*\{',
    }
    
    def __init__(self, vocabulary=None, chunk_size: int = 2000, chunk_overlap: int = 200):
        """Initialize parser - vocabulary is optional for generic code"""
        self.vocabulary = vocabulary
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.call_extractor = CallExtractor()
    
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the file"""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        return self.LANGUAGE_MAP.get(ext, 'unknown')
    
    def parse(self, content: bytes, source_path: str) -> List[IndexableChunk]:
        """Parse code file into indexable chunks"""
        try:
            text = content.decode('utf-8', errors='replace')
        except:
            text = content.decode('latin-1', errors='replace')
        
        language = self.detect_language(source_path)
        chunks = []
        
        # Try to extract functions/methods
        functions = self._extract_functions(text, language)
        
        if functions:
            # Create chunks from functions
            for func in functions:
                chunk = self._function_to_chunk(func, source_path, language)
                if chunk:
                    chunks.append(chunk)
        
        if not chunks:
            # Fall back to line-based chunking
            chunks = self._chunk_by_lines(text, source_path, language)
        
        return chunks
    
    def _extract_functions(self, text: str, language: str) -> List[CodeFunction]:
        """Extract functions/methods from code"""
        functions = []
        lines = text.split('\n')
        
        # Get patterns for this language
        patterns = self.FUNCTION_PATTERNS.get(language, [])
        if not patterns and language == 'typescript':
            patterns = self.FUNCTION_PATTERNS.get('javascript', [])
        
        # Track current class for methods
        current_class = None
        class_pattern = self.CLASS_PATTERNS.get(language)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for class definition
            if class_pattern:
                class_match = re.match(class_pattern, line)
                if class_match:
                    groups = [g for g in class_match.groups() if g]
                    if groups:
                        current_class = groups[-1]
            
            # Check for function definition
            func_found = False
            for pattern in patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    groups = [g for g in match.groups() if g and not g.startswith('(')]
                    func_name = groups[-1] if groups else None
                    
                    if func_name and func_name not in ['if', 'for', 'while', 'switch', 'catch']:
                        start_line = i
                        
                        # For PostgreSQL-style (return type on previous line), 
                        # check if brace is on next line
                        if language in ('c', 'cpp') and '{' not in line:
                            # Look for opening brace on next line(s)
                            for j in range(i + 1, min(i + 3, len(lines))):
                                if lines[j].strip() == '{':
                                    break
                                elif lines[j].strip().startswith('{'):
                                    break
                            # Also include return type from previous line if it exists
                            if i > 0 and lines[i-1].strip() and not lines[i-1].strip().startswith(('/', '*', '#')):
                                prev = lines[i-1].strip()
                                if re.match(r'^[\w\s\*]+$', prev):
                                    start_line = i - 1
                        
                        end_line = self._find_block_end(lines, i, language)
                        
                        body_lines = lines[start_line:end_line + 1]
                        body = '\n'.join(body_lines)
                        
                        docstring = self._extract_docstring(lines, start_line, language)
                        
                        functions.append(CodeFunction(
                            name=func_name,
                            signature=line.strip(),
                            body=body,
                            start_line=start_line + 1,
                            end_line=end_line + 1,
                            docstring=docstring,
                            class_name=current_class,
                            language=language
                        ))
                        
                        i = end_line
                        func_found = True
                        break
            
            if not func_found:
                i += 1
            else:
                i += 1
        
        return functions
    
    def _find_block_end(self, lines: List[str], start: int, language: str) -> int:
        """Find the end of a code block"""
        if language == 'python':
            return self._find_python_block_end(lines, start)
        elif language == 'ruby':
            return self._find_ruby_block_end(lines, start)
        else:
            return self._find_brace_block_end(lines, start)
    
    def _find_brace_block_end(self, lines: List[str], start: int) -> int:
        """Find matching closing brace"""
        brace_count = 0
        in_string = False
        string_char = None
        
        for i in range(start, len(lines)):
            line = lines[i]
            j = 0
            while j < len(line):
                char = line[j]
                
                if char in '"\'`' and (j == 0 or line[j-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            return i
                j += 1
        
        return len(lines) - 1
    
    def _find_python_block_end(self, lines: List[str], start: int) -> int:
        """Find end of Python block by indentation"""
        if start >= len(lines):
            return start
        
        first_line = lines[start]
        base_indent = len(first_line) - len(first_line.lstrip())
        
        for i in range(start + 1, len(lines)):
            line = lines[i]
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent:
                return i - 1
        
        return len(lines) - 1
    
    def _find_ruby_block_end(self, lines: List[str], start: int) -> int:
        """Find end of Ruby block"""
        depth = 0
        keywords = ['def', 'class', 'module', 'if', 'unless', 'case', 'while', 'until', 'for', 'do', 'begin']
        
        for i in range(start, len(lines)):
            line = lines[i].strip()
            
            for kw in keywords:
                if re.match(rf'\b{kw}\b', line):
                    depth += 1
                    break
            
            if re.match(r'\bend\b', line):
                depth -= 1
                if depth == 0:
                    return i
        
        return len(lines) - 1
    
    def _extract_docstring(self, lines: List[str], func_start: int, language: str) -> Optional[str]:
        """Extract docstring/comment before function"""
        docstring_lines = []
        
        if language == 'python':
            for i in range(func_start + 1, min(func_start + 5, len(lines))):
                line = lines[i].strip()
                if '"""' in line or "'''" in line:
                    docstring_lines.append(line)
                    if line.count('"""') >= 2 or line.count("'''") >= 2:
                        break
                    for j in range(i + 1, len(lines)):
                        docstring_lines.append(lines[j])
                        if '"""' in lines[j] or "'''" in lines[j]:
                            break
                    break
                elif line and not line.startswith('#'):
                    break
        else:
            for i in range(func_start - 1, max(-1, func_start - 20), -1):
                line = lines[i].strip()
                if line.startswith('//') or line.startswith('*') or line.startswith('/*'):
                    docstring_lines.insert(0, line)
                elif line.startswith('*/'):
                    docstring_lines.insert(0, line)
                elif line == '':
                    if docstring_lines:
                        break
                elif line.startswith('#') and language in ['ruby', 'php']:
                    docstring_lines.insert(0, line)
                else:
                    break
        
        return '\n'.join(docstring_lines) if docstring_lines else None
    
    def _function_to_chunk(self, func: CodeFunction, source_path: str, language: str) -> IndexableChunk:
        """Convert a CodeFunction to an IndexableChunk"""
        
        # Match domain concepts if vocabulary available
        domain_matches = []
        if self.vocabulary:
            domain_matches = self.vocabulary.match_text(func.body, deduplicate=True)
        
        # Extract function calls
        call_infos = self.call_extractor.extract_calls(
            func.body, language, procedure_name=func.name
        )
        calls = [c.target for c in call_infos]
        
        # Separate system calls from user calls
        system_calls = [c.target for c in call_infos if c.call_type.value == 'system']
        user_calls = [c.target for c in call_infos if c.call_type.value != 'system']
        
        # Build metadata
        metadata = {
            'language': language,
            'function_name': func.name,
            'full_name': func.full_name,
            'signature': func.signature,
            'has_docstring': bool(func.docstring),
        }
        if func.class_name:
            metadata['class_name'] = func.class_name
        
        # Add call information
        if calls:
            metadata['calls'] = calls
        if system_calls:
            metadata['system_calls'] = system_calls
        if user_calls:
            metadata['user_calls'] = user_calls
        
        # Create source reference
        source_ref = SourceReference(
            file_path=source_path,
            line_start=func.start_line,
            line_end=func.end_line,
            procedure_name=func.full_name
        )
        
        # Create embedding text - include calls for semantic matching
        embedding_parts = [
            f"Function: {func.full_name}",
            f"Language: {language}",
            f"Signature: {func.signature}",
        ]
        if func.docstring:
            embedding_parts.append(f"Documentation: {func.docstring[:500]}")
        if user_calls:
            embedding_parts.append(f"Calls: {', '.join(user_calls[:20])}")
        embedding_parts.append(func.body[:1000])
        embedding_text = '\n'.join(embedding_parts)
        
        # Generate chunk ID
        content_hash = hashlib.md5(func.body.encode()).hexdigest()[:12]
        chunk_id = f"code_{func.full_name}_{content_hash}"
        
        return IndexableChunk(
            chunk_id=chunk_id,
            text=func.body,
            embedding_text=embedding_text,
            source_type=SourceType.CODE,
            semantic_type=SemanticType.PROCEDURE,
            source_ref=source_ref,
            domain_matches=domain_matches,
            metadata=metadata
        )
    
    def _chunk_by_lines(self, text: str, source_path: str, language: str) -> List[IndexableChunk]:
        """Fall back to line-based chunking"""
        chunks = []
        lines = text.split('\n')
        file_name = Path(source_path).name
        
        current_chunk_lines = []
        current_size = 0
        chunk_start = 1
        chunk_idx = 0
        
        for i, line in enumerate(lines):
            current_chunk_lines.append(line)
            current_size += len(line) + 1
            
            if current_size >= self.chunk_size:
                chunk_content = '\n'.join(current_chunk_lines)
                
                # Create embedding text
                embedding_text = f"File: {file_name}\nLanguage: {language}\n{chunk_content[:500]}"
                
                # Match domain concepts if vocabulary available
                domain_matches = []
                if self.vocabulary:
                    domain_matches = self.vocabulary.match_text(chunk_content, deduplicate=True)
                
                # Generate chunk ID
                content_hash = hashlib.md5(chunk_content.encode()).hexdigest()[:12]
                chunk_id = f"code_{file_name}_{chunk_idx}_{content_hash}"
                
                source_ref = SourceReference(
                    file_path=source_path,
                    line_start=chunk_start,
                    line_end=i + 1
                )
                
                chunk = IndexableChunk(
                    chunk_id=chunk_id,
                    text=chunk_content,
                    embedding_text=embedding_text,
                    source_type=SourceType.CODE,
                    semantic_type=SemanticType.TEXT_BLOCK,
                    source_ref=source_ref,
                    domain_matches=domain_matches,
                    metadata={
                        'language': language,
                        'chunk_type': 'lines',
                    }
                )
                chunks.append(chunk)
                chunk_idx += 1
                
                # Keep overlap
                overlap_lines = current_chunk_lines[-10:] if len(current_chunk_lines) > 10 else []
                current_chunk_lines = overlap_lines
                current_size = sum(len(l) + 1 for l in current_chunk_lines)
                chunk_start = i + 2 - len(overlap_lines)
        
        # Final chunk
        if current_chunk_lines and len('\n'.join(current_chunk_lines).strip()) > 20:
            chunk_content = '\n'.join(current_chunk_lines)
            
            embedding_text = f"File: {file_name}\nLanguage: {language}\n{chunk_content[:500]}"
            
            domain_matches = []
            if self.vocabulary:
                domain_matches = self.vocabulary.match_text(chunk_content, deduplicate=True)
            
            content_hash = hashlib.md5(chunk_content.encode()).hexdigest()[:12]
            chunk_id = f"code_{file_name}_{chunk_idx}_{content_hash}"
            
            source_ref = SourceReference(
                file_path=source_path,
                line_start=chunk_start,
                line_end=len(lines)
            )
            
            chunk = IndexableChunk(
                chunk_id=chunk_id,
                text=chunk_content,
                embedding_text=embedding_text,
                source_type=SourceType.CODE,
                semantic_type=SemanticType.TEXT_BLOCK,
                source_ref=source_ref,
                domain_matches=domain_matches,
                metadata={
                    'language': language,
                    'chunk_type': 'lines',
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Return parser statistics"""
        return {
            'supported_languages': list(set(self.LANGUAGE_MAP.values())),
            'supported_extensions': self.SUPPORTED_EXTENSIONS,
        }
