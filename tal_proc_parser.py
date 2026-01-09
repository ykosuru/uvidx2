"""
TAL Procedure Parser - Foundation parsing for Transaction Application Language (TAL)
Stub module - user should provide their own tal_proc_parser.py
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

class TALType(Enum):
    INT = "INT"
    INT16 = "INT(16)"
    INT32 = "INT(32)"
    INT64 = "INT(64)"
    STRING = "STRING"
    REAL = "REAL"
    REAL32 = "REAL(32)"
    REAL64 = "REAL(64)"
    FIXED = "FIXED"
    UNSIGNED = "UNSIGNED"
    BYTE = "BYTE"
    CHAR = "CHAR"
    STRUCT = "STRUCT"
    POINTER = "POINTER"
    UNKNOWN = "UNKNOWN"

class ErrorSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

@dataclass
class SourceLocation:
    filename: str = ""
    line: int = 0
    column: int = 0
    length: int = 0
    
    def __str__(self):
        return f"{self.filename}:{self.line}:{self.column}"

@dataclass
class ParseError:
    message: str
    location: SourceLocation
    severity: ErrorSeverity
    context_lines: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    error_code: str = ""
    
    def __str__(self):
        return f"{self.severity.value.upper()}: {self.message} at {self.location}"

@dataclass
class Symbol:
    name: str
    symbol_type: TALType
    location: SourceLocation
    scope: str = ""
    is_pointer: bool = False
    is_array: bool = False

@dataclass
class TALNode:
    type: str
    name: str = ""
    value: Any = None
    children: List['TALNode'] = field(default_factory=list)
    location: SourceLocation = field(default_factory=SourceLocation)
    attributes: Dict[str, Any] = field(default_factory=dict)
    symbol: Optional[Symbol] = None
    semantic_type: Optional[TALType] = None
    
    def add_child(self, child):
        if isinstance(child, TALNode):
            self.children.append(child)

@dataclass
class ProcedureInfo:
    name: str
    return_type: Optional[TALType] = None
    parameters: List[Symbol] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    is_forward: bool = False
    is_external: bool = False
    location: SourceLocation = field(default_factory=SourceLocation)

class SymbolTable:
    def __init__(self):
        self.scopes: Dict[str, Dict[str, Symbol]] = {"global": {}}
        self.current_scope = "global"
        self.scope_stack: List[str] = ["global"]
    
    def enter_scope(self, scope_name: str):
        self.scope_stack.append(scope_name)
        self.current_scope = scope_name
        if scope_name not in self.scopes:
            self.scopes[scope_name] = {}
    
    def exit_scope(self):
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()
            self.current_scope = self.scope_stack[-1]
    
    def declare_symbol(self, symbol: Symbol) -> Optional[ParseError]:
        symbol.scope = self.current_scope
        if self.current_scope not in self.scopes:
            self.scopes[self.current_scope] = {}
        self.scopes[self.current_scope][symbol.name] = symbol
        return None
    
    def lookup_symbol(self, name: str) -> Optional[Symbol]:
        for scope in reversed(self.scope_stack):
            if scope in self.scopes and name in self.scopes[scope]:
                return self.scopes[scope][name]
        return None

def parse_tal_type(type_str: str) -> TALType:
    type_str = type_str.upper().strip()
    if type_str.startswith("INT("):
        if "32" in type_str: return TALType.INT32
        elif "64" in type_str: return TALType.INT64
        elif "16" in type_str: return TALType.INT16
        return TALType.INT
    elif type_str.startswith("REAL("):
        if "64" in type_str: return TALType.REAL64
        elif "32" in type_str: return TALType.REAL32
        return TALType.REAL
    elif type_str.startswith("UNSIGNED"):
        return TALType.UNSIGNED
    try:
        return TALType(type_str)
    except ValueError:
        return TALType.UNKNOWN

def find_procedure_declarations(tal_code: str) -> List[Tuple[int, str, str]]:
    lines = tal_code.split('\n')
    procedures = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        comment_pos = line.find('!')
        if comment_pos >= 0:
            code_part = line[:comment_pos].strip()
        else:
            code_part = line
        
        proc_match = re.search(
            r'\b(?:(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING|FIXED|UNSIGNED(?:\([^)]*\))?)\s+)?'
            r'PROC\s+([a-zA-Z_][a-zA-Z0-9_^]*)', 
            code_part, re.IGNORECASE
        )
        
        if proc_match:
            start_line = i + 1
            proc_name = proc_match.group(2)
            declaration_lines = []
            j = i
            found_semicolon = False
            while j < len(lines) and not found_semicolon:
                current_line = lines[j]
                declaration_lines.append(current_line)
                cp = current_line.find('!')
                code = current_line[:cp] if cp >= 0 else current_line
                if ';' in code:
                    found_semicolon = True
                j += 1
            full_declaration = '\n'.join(declaration_lines)
            procedures.append((start_line, proc_name, full_declaration))
            i = j
        else:
            i += 1
    return procedures

def extract_parameters_from_declaration(declaration: str) -> List[str]:
    clean_decl = re.sub(r'!.*$', '', declaration, flags=re.MULTILINE)
    clean_decl = ' '.join(clean_decl.split())
    paren_match = re.search(r'\(([^)]*)\)', clean_decl)
    if not paren_match:
        return []
    param_string = paren_match.group(1).strip()
    if not param_string:
        return []
    return [p.strip() for p in param_string.split(',') if p.strip()]

def parse_procedure_declaration(proc_name: str, declaration: str, start_line: int, 
                                filename: str, symbol_table: SymbolTable) -> Tuple[Optional[TALNode], List[ParseError]]:
    errors = []
    location = SourceLocation(filename, start_line, 1)
    proc_node = TALNode('procedure', name=proc_name, location=location)
    symbol_table.enter_scope(proc_name)
    
    return_type_match = re.search(
        r'\b(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING|FIXED|UNSIGNED(?:\([^)]*\))?)\s+PROC', 
        declaration, re.IGNORECASE
    )
    if return_type_match:
        proc_node.attributes['return_type'] = parse_tal_type(return_type_match.group(1)).value
    
    parameters = extract_parameters_from_declaration(declaration)
    if parameters:
        params_node = TALNode('parameters', location=location)
        for param_name in parameters:
            param_name = param_name.strip()
            if param_name:
                param_node = TALNode('parameter', name=param_name, location=location)
                param_node.attributes['type'] = TALType.UNKNOWN.value
                params_node.add_child(param_node)
        proc_node.add_child(params_node)
    
    if re.search(r'\bMAIN\b', declaration, re.IGNORECASE):
        proc_node.attributes['is_main'] = True
        proc_node.attributes['attributes'] = proc_node.attributes.get('attributes', []) + ['MAIN']
    if re.search(r'\bFORWARD\b', declaration, re.IGNORECASE):
        proc_node.attributes['is_forward'] = True
        proc_node.attributes['attributes'] = proc_node.attributes.get('attributes', []) + ['FORWARD']
    if re.search(r'\bEXTERNAL\b', declaration, re.IGNORECASE):
        proc_node.attributes['is_external'] = True
        proc_node.attributes['attributes'] = proc_node.attributes.get('attributes', []) + ['EXTERNAL']
    
    return proc_node, errors

def parse_multiple_procedures(tal_code: str, filename: str, 
                              symbol_table: SymbolTable) -> Tuple[List[TALNode], List[ParseError]]:
    procedures = []
    errors = []
    proc_declarations = find_procedure_declarations(tal_code)
    for start_line, proc_name, declaration in proc_declarations:
        proc_node, proc_errors = parse_procedure_declaration(
            proc_name, declaration, start_line, filename, symbol_table
        )
        if proc_node:
            procedures.append(proc_node)
        errors.extend(proc_errors)
        symbol_table.exit_scope()
    return procedures, errors
