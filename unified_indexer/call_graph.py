"""
Call Extractor - Extracts function/procedure calls from code

This module is used internally by parsers to identify what functions
are called within each code block.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Optional


class CallType(Enum):
    """Type of call"""
    USER = "user"        # User-defined procedure
    SYSTEM = "system"    # System/library call
    UNKNOWN = "unknown"


@dataclass
class CallInfo:
    """Information about a function call"""
    target: str           # Name of called function
    call_type: CallType   # Type of call
    line_number: int = 0  # Line where call occurs
    context: str = ""     # Surrounding code


class CallExtractor:
    """
    Extract function/procedure calls from source code.
    
    Supports multiple languages with language-specific patterns.
    """
    
    # Common system/library functions by language
    SYSTEM_FUNCTIONS = {
        'tal': {
            'CALL', 'RETURN', 'STOP', 'WRITE', 'READ', 'WRITEX', 'READX',
            'OPEN', 'CLOSE', 'AWAITIO', 'POSITION', 'CONTROL', 'SETMODE',
            'FILEINFO', 'TOSVERSION', 'PROCESS_GETINFO_', 'PROCESS_SPAWN_',
            'PROCESS_STOP_', 'LASTRECEIVE', 'LASTPARAM', 'RECEIVE', 'SEND',
            'REPLY', 'DELAY', 'SUSPENDPROCESS', 'ACTIVATEPROCESS',
        },
        'c': {
            'printf', 'sprintf', 'fprintf', 'scanf', 'sscanf', 'malloc',
            'calloc', 'realloc', 'free', 'memcpy', 'memset', 'memmove',
            'strlen', 'strcpy', 'strncpy', 'strcmp', 'strncmp', 'strcat',
            'fopen', 'fclose', 'fread', 'fwrite', 'fseek', 'ftell',
            'sizeof', 'assert', 'exit', 'abort', 'atexit',
        },
        'python': {
            'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict',
            'set', 'tuple', 'open', 'read', 'write', 'close', 'input',
            'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr',
            'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
        },
        'java': {
            'System.out.println', 'System.err.println', 'String.format',
            'Math.abs', 'Math.max', 'Math.min', 'Math.sqrt', 'Math.pow',
            'Arrays.sort', 'Collections.sort', 'Objects.equals',
        },
    }
    
    # Call patterns by language
    CALL_PATTERNS = {
        'tal': [
            r'CALL\s+(\w+)',
            r'CALL\s+(\w+)\s*\(',
            r'\^(\w+)\s*\(',
            r'(\w+)\s*\(',
        ],
        'c': [
            r'\b(\w+)\s*\(',
        ],
        'cpp': [
            r'\b(\w+)\s*\(',
            r'\b(\w+)\s*<[^>]*>\s*\(',
            r'\b(\w+)::(\w+)\s*\(',
        ],
        'python': [
            r'\b(\w+)\s*\(',
            r'\b(\w+)\.(\w+)\s*\(',
        ],
        'java': [
            r'\b(\w+)\s*\(',
            r'\b(\w+)\.(\w+)\s*\(',
            r'new\s+(\w+)\s*\(',
        ],
        'javascript': [
            r'\b(\w+)\s*\(',
            r'\b(\w+)\.(\w+)\s*\(',
        ],
    }
    
    # Keywords to exclude
    KEYWORDS = {
        'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'return',
        'break', 'continue', 'try', 'catch', 'finally', 'throw', 'throws',
        'new', 'delete', 'sizeof', 'typeof', 'instanceof', 'in', 'of',
        'class', 'struct', 'enum', 'interface', 'extends', 'implements',
        'public', 'private', 'protected', 'static', 'final', 'const',
        'void', 'int', 'char', 'float', 'double', 'bool', 'boolean',
        'string', 'var', 'let', 'def', 'async', 'await', 'yield',
        'import', 'export', 'from', 'as', 'with', 'assert', 'pass',
        'BEGIN', 'END', 'THEN', 'ELSE', 'DO', 'UNTIL', 'WHILE', 'FOR',
        'PROC', 'SUBPROC', 'DEFINE', 'LITERAL', 'STRUCT', 'INT', 'STRING',
    }
    
    def __init__(self):
        pass
    
    def extract_calls(self, 
                     code: str, 
                     language: str,
                     procedure_name: str = "") -> List[CallInfo]:
        """
        Extract all function/procedure calls from code.
        
        Args:
            code: Source code text
            language: Programming language
            procedure_name: Name of containing procedure (to exclude self-calls)
            
        Returns:
            List of CallInfo objects
        """
        calls = []
        seen = set()
        
        # Get patterns for language
        patterns = self.CALL_PATTERNS.get(language, self.CALL_PATTERNS.get('c', []))
        system_funcs = self.SYSTEM_FUNCTIONS.get(language, set())
        
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('/*') or \
               stripped.startswith('*') or stripped.startswith('#') or \
               stripped.startswith('!') or stripped.startswith('--'):
                continue
            
            # Apply patterns
            for pattern in patterns:
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    groups = [g for g in match.groups() if g]
                    if not groups:
                        continue
                    
                    # Get the function name (last group)
                    func_name = groups[-1]
                    
                    # Skip keywords
                    if func_name.upper() in self.KEYWORDS:
                        continue
                    
                    # Skip self-reference
                    if procedure_name and func_name.upper() == procedure_name.upper():
                        continue
                    
                    # Skip duplicates
                    key = func_name.upper()
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    # Determine call type
                    if func_name.upper() in {s.upper() for s in system_funcs}:
                        call_type = CallType.SYSTEM
                    else:
                        call_type = CallType.USER
                    
                    calls.append(CallInfo(
                        target=func_name,
                        call_type=call_type,
                        line_number=line_num,
                        context=stripped[:100]
                    ))
        
        return calls


def extract_calls(code: str, language: str, procedure_name: str = "") -> List[CallInfo]:
    """Convenience function to extract calls"""
    extractor = CallExtractor()
    return extractor.extract_calls(code, language, procedure_name)


# Minimal CallGraphBuilder for internal use
class CallGraphBuilder:
    """Build and query call graph relationships"""
    
    def __init__(self):
        self.calls = {}      # procedure -> [called procedures]
        self.callers = {}    # procedure -> [calling procedures]
    
    def add_procedure(self, name: str, calls: List[str]):
        """Add procedure with its calls"""
        name_upper = name.upper()
        self.calls[name_upper] = [c.upper() for c in calls]
        
        # Update reverse mapping
        for called in calls:
            called_upper = called.upper()
            if called_upper not in self.callers:
                self.callers[called_upper] = []
            if name_upper not in self.callers[called_upper]:
                self.callers[called_upper].append(name_upper)
    
    def get_calls(self, procedure: str) -> List[str]:
        """Get what this procedure calls"""
        return self.calls.get(procedure.upper(), [])
    
    def get_callers(self, procedure: str) -> List[str]:
        """Get what calls this procedure"""
        return self.callers.get(procedure.upper(), [])


__all__ = ['CallExtractor', 'CallInfo', 'CallType', 'extract_calls', 'CallGraphBuilder']
