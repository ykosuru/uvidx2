"""
TAL Enhanced Parser - Rich parsing for Transaction Application Language (TAL)

This module extends tal_proc_parser.py with comprehensive parsing for full TAL programs:
- DEFINE statements (constants and macros)
- STRUCT definitions with field analysis
- LITERAL declarations
- Global variable declarations
- Procedure body parsing with control flow analysis
- CALL graph extraction
- Business logic pattern detection
- Cyclomatic complexity calculation

Architecture:
- Imports foundation classes from tal_proc_parser.py
- Adds lexer-based tokenization for richer parsing
- Extracts semantic information for code analysis
- Generates comprehensive AST for downstream processing

Usage:
    from tal_enhanced_parser import EnhancedTALParser, parse_tal_file
    
    parser = EnhancedTALParser()
    result = parser.parse(tal_source_code)
    
    for proc in result.procedures:
        print(f"Procedure: {proc.name}, Complexity: {proc.complexity}")
    for define in result.defines:
        print(f"Define: {define.name} = {define.value}")
"""

import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum

# Import foundation classes from tal_proc_parser
from tal_proc_parser import (
    TALType, TALNode, Symbol, SymbolTable, SourceLocation,
    ParseError, ErrorSeverity, ProcedureInfo,
    parse_tal_type, find_procedure_declarations,
    parse_procedure_declaration, parse_multiple_procedures,
    extract_parameters_from_declaration
)


# =============================================================================
# PARSED STRUCTURES
# =============================================================================

@dataclass
class DefineInfo:
    """Information about a DEFINE statement."""
    name: str
    value: str
    params: List[str] = field(default_factory=list)  # For macro parameters
    location: SourceLocation = field(default_factory=SourceLocation)
    is_macro: bool = False
    expansion: str = ""  # Expanded form for macros
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'params': self.params,
            'is_macro': self.is_macro,
            'line': self.location.line
        }


@dataclass
class StructField:
    """Information about a STRUCT field."""
    name: str
    field_type: TALType
    offset: int = 0
    size: int = 0
    is_array: bool = False
    array_bounds: Optional[Tuple[int, int]] = None
    is_pointer: bool = False
    is_filler: bool = False
    redefines: Optional[str] = None
    subfields: List['StructField'] = field(default_factory=list)  # For nested structs
    
    @property
    def array_size(self) -> int:
        if self.array_bounds:
            return self.array_bounds[1] - self.array_bounds[0] + 1
        return 0


@dataclass
class StructInfo:
    """Information about a STRUCT definition."""
    name: str
    fields: List[StructField] = field(default_factory=list)
    total_size: int = 0
    location: SourceLocation = field(default_factory=SourceLocation)
    is_referral: bool = False  # STRUCT .name (referral/pointer)
    is_template: bool = False  # STRUCT * (template)
    base_struct: Optional[str] = None  # For STRUCT LIKE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'fields': [
                {
                    'name': f.name, 
                    'type': f.field_type.value, 
                    'offset': f.offset, 
                    'size': f.size,
                    'is_array': f.is_array,
                    'array_bounds': f.array_bounds
                }
                for f in self.fields
            ],
            'total_size': self.total_size,
            'is_referral': self.is_referral,
            'line': self.location.line
        }
    
    def get_field(self, name: str) -> Optional[StructField]:
        """Find a field by name."""
        for f in self.fields:
            if f.name == name:
                return f
        return None


@dataclass
class LiteralInfo:
    """Information about a LITERAL declaration."""
    name: str
    value: Any
    literal_type: TALType = TALType.INT
    location: SourceLocation = field(default_factory=SourceLocation)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'type': self.literal_type.value,
            'line': self.location.line
        }


@dataclass
class GlobalInfo:
    """Information about a global variable declaration."""
    name: str
    var_type: TALType
    is_array: bool = False
    array_bounds: Optional[Tuple[int, int]] = None
    is_pointer: bool = False
    struct_type: Optional[str] = None
    initial_value: Any = None
    location: SourceLocation = field(default_factory=SourceLocation)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.var_type.value,
            'is_array': self.is_array,
            'array_bounds': self.array_bounds,
            'is_pointer': self.is_pointer,
            'struct_type': self.struct_type,
            'line': self.location.line
        }


@dataclass
class CallInfo:
    """Information about a procedure/function call."""
    callee: str  # Name of called procedure
    caller: str  # Name of calling procedure
    arguments: List[str] = field(default_factory=list)
    location: SourceLocation = field(default_factory=SourceLocation)
    is_external: bool = False
    call_type: str = "CALL"  # CALL, function-style, or PCAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'callee': self.callee,
            'caller': self.caller,
            'arguments': self.arguments,
            'call_type': self.call_type,
            'line': self.location.line
        }


@dataclass
class ProcedureDetail:
    """Detailed information about a parsed procedure."""
    name: str
    return_type: Optional[TALType] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    local_vars: List[Dict[str, Any]] = field(default_factory=list)
    subprocs: List[str] = field(default_factory=list)  # SUBPROC names
    calls: List[CallInfo] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    location: SourceLocation = field(default_factory=SourceLocation)
    end_location: Optional[SourceLocation] = None
    attributes: List[str] = field(default_factory=list)
    is_main: bool = False
    is_forward: bool = False
    is_external: bool = False
    is_interrupt: bool = False
    is_resident: bool = False
    body_text: str = ""
    body_start_line: int = 0
    body_end_line: int = 0
    complexity: int = 1  # Cyclomatic complexity
    ast_node: Optional[TALNode] = None  # Reference to parsed AST node
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'return_type': self.return_type.value if self.return_type else None,
            'parameters': self.parameters,
            'local_vars': self.local_vars,
            'subprocs': self.subprocs,
            'calls': [c.callee for c in self.calls],
            'called_by': self.called_by,
            'attributes': self.attributes,
            'is_main': self.is_main,
            'is_forward': self.is_forward,
            'complexity': self.complexity,
            'line': self.location.line,
            'body_lines': (self.body_start_line, self.body_end_line)
        }


@dataclass 
class SubprocInfo:
    """Information about a SUBPROC (nested procedure)."""
    name: str
    parent_proc: str
    return_type: Optional[TALType] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    location: SourceLocation = field(default_factory=SourceLocation)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'parent_proc': self.parent_proc,
            'return_type': self.return_type.value if self.return_type else None,
            'parameters': self.parameters,
            'line': self.location.line
        }


@dataclass
class ParseResult:
    """Complete result of parsing a TAL file."""
    procedures: List[ProcedureDetail] = field(default_factory=list)
    subprocs: List[SubprocInfo] = field(default_factory=list)
    defines: List[DefineInfo] = field(default_factory=list)
    structs: List[StructInfo] = field(default_factory=list)
    literals: List[LiteralInfo] = field(default_factory=list)
    globals: List[GlobalInfo] = field(default_factory=list)
    calls: List[CallInfo] = field(default_factory=list)
    errors: List[ParseError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    filename: str = ""
    source_lines: int = 0
    
    # Computed properties
    @property
    def call_graph(self) -> Dict[str, List[str]]:
        """Build call graph: caller -> [callees]."""
        graph: Dict[str, List[str]] = {}
        for call in self.calls:
            if call.caller not in graph:
                graph[call.caller] = []
            if call.callee not in graph[call.caller]:
                graph[call.caller].append(call.callee)
        return graph
    
    @property
    def reverse_call_graph(self) -> Dict[str, List[str]]:
        """Build reverse call graph: callee -> [callers]."""
        graph: Dict[str, List[str]] = {}
        for call in self.calls:
            if call.callee not in graph:
                graph[call.callee] = []
            if call.caller not in graph[call.callee]:
                graph[call.callee].append(call.caller)
        return graph
    
    @property
    def entry_points(self) -> List[str]:
        """Find procedures that are never called (entry points)."""
        called = set(c.callee for c in self.calls)
        return [p.name for p in self.procedures if p.name not in called]
    
    @property
    def external_calls(self) -> List[CallInfo]:
        """Find calls to procedures not defined in this file."""
        defined = set(p.name for p in self.procedures)
        return [c for c in self.calls if c.callee not in defined]
    
    def get_procedure(self, name: str) -> Optional[ProcedureDetail]:
        """Find procedure by name."""
        for p in self.procedures:
            if p.name == name:
                return p
        return None
    
    def get_struct(self, name: str) -> Optional[StructInfo]:
        """Find struct by name."""
        for s in self.structs:
            if s.name == name:
                return s
        return None
    
    def get_define(self, name: str) -> Optional[DefineInfo]:
        """Find define by name."""
        for d in self.defines:
            if d.name == name:
                return d
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'filename': self.filename,
            'source_lines': self.source_lines,
            'procedures': [p.to_dict() for p in self.procedures],
            'subprocs': [s.to_dict() for s in self.subprocs],
            'defines': [d.to_dict() for d in self.defines],
            'structs': [s.to_dict() for s in self.structs],
            'literals': [l.to_dict() for l in self.literals],
            'globals': [g.to_dict() for g in self.globals],
            'call_graph': self.call_graph,
            'entry_points': self.entry_points,
            'errors': [str(e) for e in self.errors],
            'warnings': self.warnings
        }


# =============================================================================
# ENHANCED TAL PARSER
# =============================================================================

class EnhancedTALParser:
    """
    Enhanced parser for TAL source code.
    
    Provides comprehensive parsing of TAL programs including:
    - Procedure declarations and bodies (via tal_proc_parser)
    - DEFINE statements (constants and macros)
    - STRUCT definitions with field analysis
    - LITERAL declarations
    - Global variable declarations
    - Call graph extraction
    - Control flow analysis
    - Cyclomatic complexity calculation
    
    Usage:
        parser = EnhancedTALParser()
        result = parser.parse(source_code, filename="myfile.tal")
        
        for proc in result.procedures:
            print(f"Found procedure: {proc.name}, complexity: {proc.complexity}")
    """
    
    # TAL keywords for identification
    KEYWORDS = {
        'PROC', 'SUBPROC', 'BEGIN', 'END', 'IF', 'THEN', 'ELSE', 'WHILE', 'DO', 'FOR',
        'TO', 'DOWNTO', 'BY', 'CASE', 'OF', 'OTHERWISE', 'CALL', 'RETURN',
        'DEFINE', 'LITERAL', 'STRUCT', 'INT', 'REAL', 'STRING', 'FIXED',
        'UNSIGNED', 'FORWARD', 'EXTERNAL', 'MAIN', 'AND', 'OR', 'NOT',
        'XOR', 'LOR', 'LAND', 'USE', 'DROP', 'ASSERT', 'SCAN', 'RSCAN',
        'STORE', 'CODE', 'STACK', 'ENTRY', 'PRIV', 'RESIDENT',
        'CALLABLE', 'VARIABLE', 'EXTENSIBLE', 'INTERRUPT', 'PRIVATE',
        'NAME', 'BLOCK', 'FILLER', 'GOTO', 'LABEL', 'ASSERT', 'PCAL'
    }
    
    # TAL type keywords
    TYPE_KEYWORDS = {'INT', 'REAL', 'STRING', 'FIXED', 'UNSIGNED', 'STRUCT', 'BYTE'}
    
    # System procedures (commonly called, not defined in user code)
    SYSTEM_PROCS = {
        'INITIALIZER', 'PROCESS_CREATE_', 'PROCESS_STOP_', 'FILE_OPEN_',
        'FILE_CLOSE_', 'FILE_READ_', 'FILE_WRITE_', 'READX', 'WRITEX',
        'AWAITIO', 'DELAY', 'SHIFTSTRING', 'NUMIN', 'NUMOUT', 'CONTIME',
        'DEBUG', 'ABEND', 'STOP', 'MOVERIGHT', 'MOVELEFT', 'BADDR', 'WADDR',
        '$RECEIVE', '$SEND', 'MONITORCPUS', 'MYTERM', 'MYPID'
    }
    
    def __init__(self):
        """Initialize the enhanced parser."""
        self.source = ""
        self.filename = ""
        self.lines: List[str] = []
        self.errors: List[ParseError] = []
        self.warnings: List[str] = []
        self.symbol_table = SymbolTable()
        
        # Parsed components
        self.procedures: List[ProcedureDetail] = []
        self.subprocs: List[SubprocInfo] = []
        self.defines: List[DefineInfo] = []
        self.structs: List[StructInfo] = []
        self.literals: List[LiteralInfo] = []
        self.globals: List[GlobalInfo] = []
        self.calls: List[CallInfo] = []
        
        # Tracking state
        self.current_procedure: Optional[str] = None
        self.proc_ranges: Dict[str, Tuple[int, int]] = {}  # proc_name -> (start_line, end_line)
    
    def parse_file(self, filename: str) -> ParseResult:
        """
        Parse a TAL source file.
        
        Args:
            filename: Path to TAL file
            
        Returns:
            ParseResult containing all parsed components
        """
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
        return self.parse(source, filename)
    
    def parse(self, source: str, filename: str = "<string>") -> ParseResult:
        """
        Parse TAL source code and return structured result.
        
        Args:
            source: TAL source code string
            filename: Optional filename for error reporting
            
        Returns:
            ParseResult containing all parsed components
        """
        self.source = source
        self.filename = filename
        self.lines = source.split('\n')
        self.errors = []
        self.warnings = []
        self.symbol_table = SymbolTable()
        
        # Reset parsed components
        self.procedures = []
        self.subprocs = []
        self.defines = []
        self.structs = []
        self.literals = []
        self.globals = []
        self.calls = []
        self.proc_ranges = {}
        
        try:
            # Parse in multiple passes for proper resolution
            self._parse_defines()
            self._parse_literals()
            self._parse_structs()
            self._parse_globals()
            self._parse_procedures()  # Uses tal_proc_parser foundation
            self._find_procedure_boundaries()
            self._parse_procedure_bodies()
            self._extract_calls()
            self._extract_subprocs()
            self._build_call_relationships()
            self._calculate_complexity()
            
        except Exception as e:
            self.errors.append(ParseError(
                f"Parse error: {str(e)}",
                SourceLocation(filename, 0, 0),
                ErrorSeverity.FATAL
            ))
        
        return ParseResult(
            procedures=self.procedures,
            subprocs=self.subprocs,
            defines=self.defines,
            structs=self.structs,
            literals=self.literals,
            globals=self.globals,
            calls=self.calls,
            errors=self.errors,
            warnings=self.warnings,
            filename=filename,
            source_lines=len(self.lines)
        )
    
    def _remove_comments(self, line: str) -> str:
        """Remove TAL comments (! to end of line) from a line."""
        in_string = False
        result = []
        i = 0
        while i < len(line):
            char = line[i]
            if char == '"' and not in_string:
                in_string = True
                result.append(char)
            elif char == '"' and in_string:
                in_string = False
                result.append(char)
            elif char == '!' and not in_string:
                # Comment starts here
                break
            else:
                result.append(char)
            i += 1
        return ''.join(result)
    
    def _get_clean_line(self, line_num: int) -> str:
        """Get a line with comments removed (1-based line number)."""
        if 1 <= line_num <= len(self.lines):
            return self._remove_comments(self.lines[line_num - 1]).strip()
        return ""
    
    # =========================================================================
    # DEFINE PARSING
    # =========================================================================
    
    def _parse_defines(self):
        """Parse all DEFINE statements."""
        current_define: Optional[Dict[str, Any]] = None
        
        for line_num, line in enumerate(self.lines, 1):
            clean_line = self._remove_comments(line).strip()
            
            if not clean_line:
                continue
            
            # Check for DEFINE start
            define_match = re.match(
                r'DEFINE\s+([a-zA-Z_][a-zA-Z0-9_^]*)(?:\s*\(([^)]*)\))?\s*=\s*(.*)$',
                clean_line,
                re.IGNORECASE
            )
            
            if define_match:
                name = define_match.group(1)
                params_str = define_match.group(2)
                value = define_match.group(3).strip()
                
                params = []
                if params_str:
                    params = [p.strip() for p in params_str.split(',') if p.strip()]
                
                # Check if value ends with terminator
                if value.endswith(';') or value.endswith('#'):
                    value = value[:-1].strip()
                    
                    define_info = DefineInfo(
                        name=name,
                        value=value,
                        params=params,
                        is_macro=len(params) > 0,
                        location=SourceLocation(self.filename, line_num, 1)
                    )
                    self.defines.append(define_info)
                else:
                    # Multi-line define
                    current_define = {
                        'name': name,
                        'params': params,
                        'value_parts': [value],
                        'start_line': line_num
                    }
            
            elif current_define:
                # Continuation of multi-line define
                if clean_line.endswith(';') or clean_line.endswith('#'):
                    current_define['value_parts'].append(clean_line[:-1].strip())
                    
                    full_value = ' '.join(current_define['value_parts'])
                    define_info = DefineInfo(
                        name=current_define['name'],
                        value=full_value,
                        params=current_define['params'],
                        is_macro=len(current_define['params']) > 0,
                        location=SourceLocation(self.filename, current_define['start_line'], 1)
                    )
                    self.defines.append(define_info)
                    current_define = None
                else:
                    current_define['value_parts'].append(clean_line)
    
    # =========================================================================
    # LITERAL PARSING
    # =========================================================================
    
    def _parse_literals(self):
        """Parse all LITERAL declarations."""
        in_literal = False
        literal_text: List[str] = []
        start_line = 0
        
        for line_num, line in enumerate(self.lines, 1):
            clean_line = self._remove_comments(line).strip()
            
            if not clean_line:
                continue
            
            if re.match(r'^\s*LITERAL\b', clean_line, re.IGNORECASE):
                in_literal = True
                start_line = line_num
                # Remove LITERAL keyword
                literal_text = [re.sub(r'^\s*LITERAL\s*', '', clean_line, flags=re.IGNORECASE)]
            elif in_literal:
                literal_text.append(clean_line)
            
            # Check for terminator
            if in_literal and (clean_line.endswith(';') or clean_line.endswith('#')):
                full_text = ' '.join(literal_text)
                if full_text.endswith(';') or full_text.endswith('#'):
                    full_text = full_text[:-1]
                
                self._parse_literal_declarations(full_text, start_line)
                in_literal = False
                literal_text = []
    
    def _parse_literal_declarations(self, text: str, start_line: int):
        """Parse individual literal declarations from text."""
        parts = self._split_declarations(text)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Match: name = value
            match = re.match(r'([a-zA-Z_][a-zA-Z0-9_^]*)\s*=\s*(.+)$', part)
            if match:
                name = match.group(1)
                value_str = match.group(2).strip()
                
                literal_type = TALType.INT
                value: Any = value_str
                
                # Determine type and parse value
                if value_str.startswith('"'):
                    literal_type = TALType.STRING
                    value = value_str.strip('"')
                elif re.match(r'^-?\d+$', value_str):
                    value = int(value_str)
                elif re.match(r'^%H[0-9A-Fa-f]+$', value_str, re.IGNORECASE):
                    value = int(value_str[2:], 16)
                elif re.match(r'^%B[01]+$', value_str, re.IGNORECASE):
                    value = int(value_str[2:], 2)
                elif re.match(r'^%[0-7]+$', value_str):
                    value = int(value_str[1:], 8)
                elif re.match(r'^-?\d+\.\d*$', value_str):
                    literal_type = TALType.REAL
                    value = float(value_str)
                
                literal_info = LiteralInfo(
                    name=name,
                    value=value,
                    literal_type=literal_type,
                    location=SourceLocation(self.filename, start_line, 1)
                )
                self.literals.append(literal_info)
    
    # =========================================================================
    # STRUCT PARSING
    # =========================================================================
    
    def _parse_structs(self):
        """Parse all STRUCT definitions."""
        in_struct = False
        struct_lines: List[str] = []
        struct_name = ""
        start_line = 0
        struct_depth = 0
        has_begin = False
        
        for line_num, line in enumerate(self.lines, 1):
            clean_line = self._remove_comments(line).strip()
            
            if not clean_line:
                continue
            
            # Check for STRUCT start (not inside a procedure)
            struct_match = re.match(
                r'^STRUCT\s+([.*a-zA-Z_][a-zA-Z0-9_^]*)',
                clean_line,
                re.IGNORECASE
            )
            
            if struct_match and not in_struct:
                in_struct = True
                struct_name = struct_match.group(1)
                start_line = line_num
                struct_lines = [clean_line]
                struct_depth = 0
                has_begin = False
                
                # Check if it's a struct pointer/reference like "STRUCT .ptr(other);"
                # These don't have BEGIN/END blocks
                if re.search(r'\([^)]+\)\s*;?\s*$', clean_line):
                    struct_info = self._parse_struct_definition(struct_name, '\n'.join(struct_lines), start_line)
                    if struct_info:
                        self.structs.append(struct_info)
                    in_struct = False
                    struct_lines = []
                    
            elif in_struct:
                struct_lines.append(clean_line)
                
                # Track BEGIN to know when struct body starts
                if re.search(r'\bBEGIN\b', clean_line, re.IGNORECASE):
                    struct_depth += 1
                    has_begin = True
                
                # Track END to know when struct body ends
                if re.search(r'\bEND\b', clean_line, re.IGNORECASE):
                    struct_depth -= 1
                    if struct_depth == 0 and has_begin:
                        struct_text = '\n'.join(struct_lines)
                        struct_info = self._parse_struct_definition(struct_name, struct_text, start_line)
                        if struct_info:
                            self.structs.append(struct_info)
                        in_struct = False
                        struct_lines = []
                        has_begin = False
    
    def _parse_struct_definition(self, name: str, text: str, start_line: int) -> Optional[StructInfo]:
        """Parse a single STRUCT definition."""
        # Determine if it's a referral or template
        is_referral = name.startswith('.')
        is_template = name == '*'
        clean_name = name.lstrip('.*')
        
        struct_info = StructInfo(
            name=clean_name,
            is_referral=is_referral,
            is_template=is_template,
            location=SourceLocation(self.filename, start_line, 1)
        )
        
        # Parse fields
        lines = text.split('\n')
        current_offset = 0
        
        for line in lines[1:]:  # Skip the STRUCT declaration line
            line = self._remove_comments(line).strip()
            if not line or line.upper().startswith('END'):
                continue
            if line.endswith(';'):
                line = line[:-1].strip()
            if not line:
                continue
            
            # Parse field: TYPE name or TYPE name[bounds]
            field_match = re.match(
                r'(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING(?:\([^)]*\))?|FIXED(?:\([^)]*\))?|'
                r'UNSIGNED(?:\([^)]*\))?|STRUCT|FILLER|BEGIN)\s*'
                r'([.*a-zA-Z_][a-zA-Z0-9_^]*)?\s*(?:\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\])?',
                line,
                re.IGNORECASE
            )
            
            # Skip BEGIN keyword
            if line.upper() == 'BEGIN':
                continue
            
            if field_match:
                field_type_str = field_match.group(1)
                field_name = field_match.group(2)
                array_start = field_match.group(3)
                array_end = field_match.group(4)
                
                # Skip if no field name (like BEGIN keyword)
                if not field_name:
                    continue
                
                field = StructField(
                    name=field_name.lstrip('.*'),
                    field_type=parse_tal_type(field_type_str),
                    offset=current_offset,
                    is_pointer=field_name.startswith('.'),
                    is_filler=field_name.upper() == 'FILLER' or field_type_str.upper() == 'FILLER'
                )
                
                # Handle arrays
                if array_start is not None and array_end is not None:
                    field.is_array = True
                    field.array_bounds = (int(array_start), int(array_end))
                
                # Calculate size
                field.size = self._get_type_size(field.field_type)
                if field.is_array and field.array_bounds:
                    field.size *= (field.array_bounds[1] - field.array_bounds[0] + 1)
                
                struct_info.fields.append(field)
                current_offset += field.size
        
        struct_info.total_size = current_offset
        return struct_info
    
    # =========================================================================
    # GLOBAL VARIABLE PARSING
    # =========================================================================
    
    def _parse_globals(self):
        """Parse global variable declarations."""
        # Track procedure regions to avoid parsing inside them
        proc_lines = set()
        for proc_decl in find_procedure_declarations(self.source):
            start_line, _, _ = proc_decl
            # Mark this line and following as potentially inside a procedure
            # (We'll be more precise after _find_procedure_boundaries)
        
        for line_num, line in enumerate(self.lines, 1):
            clean_line = self._remove_comments(line).strip()
            
            if not clean_line:
                continue
            
            # Skip if it's a known declaration type
            if re.match(r'^\s*(DEFINE|LITERAL|STRUCT|PROC|SUBPROC)\b', clean_line, re.IGNORECASE):
                continue
            
            # Look for variable declarations: TYPE name, name2;
            var_match = re.match(
                r'^(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING(?:\([^)]*\))?|FIXED(?:\([^)]*\))?|'
                r'UNSIGNED(?:\([^)]*\))?)\s+(.+?)\s*;',
                clean_line,
                re.IGNORECASE
            )
            
            if var_match:
                var_type_str = var_match.group(1)
                var_names = var_match.group(2)
                var_type = parse_tal_type(var_type_str)
                
                # Parse each variable name
                for var in var_names.split(','):
                    var = var.strip()
                    if not var:
                        continue
                    
                    # Check for array syntax [start:end]
                    array_match = re.match(
                        r'([.*a-zA-Z_][a-zA-Z0-9_^]*)\s*\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]',
                        var
                    )
                    if array_match:
                        var_name = array_match.group(1)
                        array_bounds = (int(array_match.group(2)), int(array_match.group(3)))
                        is_array = True
                    else:
                        var_name = var
                        array_bounds = None
                        is_array = False
                    
                    global_info = GlobalInfo(
                        name=var_name.lstrip('.*'),
                        var_type=var_type,
                        is_array=is_array,
                        array_bounds=array_bounds,
                        is_pointer=var_name.startswith('.'),
                        location=SourceLocation(self.filename, line_num, 1)
                    )
                    self.globals.append(global_info)
    
    # =========================================================================
    # PROCEDURE PARSING (uses tal_proc_parser foundation)
    # =========================================================================
    
    def _parse_procedures(self):
        """Parse all procedure declarations using tal_proc_parser foundation."""
        # Use the foundation parser
        proc_nodes, parse_errors = parse_multiple_procedures(
            self.source, self.filename, self.symbol_table
        )
        
        self.errors.extend(parse_errors)
        
        # Convert TALNode results to ProcedureDetail
        for node in proc_nodes:
            proc = ProcedureDetail(
                name=node.name,
                location=node.location,
                ast_node=node
            )
            
            # Extract return type
            if 'return_type' in node.attributes:
                proc.return_type = parse_tal_type(node.attributes['return_type'])
            
            # Extract attributes
            proc.is_main = node.attributes.get('is_main', False)
            proc.is_forward = node.attributes.get('is_forward', False)
            proc.is_external = node.attributes.get('is_external', False)
            if 'attributes' in node.attributes:
                proc.attributes = node.attributes['attributes']
            
            # Extract parameters
            for child in node.children:
                if child.type == 'parameters':
                    for param_node in child.children:
                        if param_node.type == 'parameter':
                            proc.parameters.append({
                                'name': param_node.name,
                                'type': param_node.attributes.get('type', 'UNKNOWN'),
                                'is_pointer': param_node.attributes.get('pointer', False)
                            })
            
            self.procedures.append(proc)
    
    def _find_procedure_boundaries(self):
        """Find the start and end lines of each procedure body."""
        proc_starts: Dict[str, int] = {}
        
        # Find all PROC declarations
        for line_num, line in enumerate(self.lines, 1):
            clean_line = self._remove_comments(line).strip()
            proc_match = re.search(r'\bPROC\s+([a-zA-Z_][a-zA-Z0-9_^]*)', clean_line, re.IGNORECASE)
            if proc_match:
                proc_name = proc_match.group(1)
                proc_starts[proc_name] = line_num
        
        # Sort procedures by start line
        sorted_procs = sorted(
            [(name, line) for name, line in proc_starts.items()],
            key=lambda x: x[1]
        )
        
        # Find BEGIN/END pairs for each procedure
        for idx, (proc_name, start_line) in enumerate(sorted_procs):
            # Find the corresponding procedure detail
            proc = None
            for p in self.procedures:
                if p.name == proc_name:
                    proc = p
                    break
            
            if not proc or proc.is_forward or proc.is_external:
                continue
            
            # Determine end boundary (next PROC or end of file)
            if idx + 1 < len(sorted_procs):
                end_boundary = sorted_procs[idx + 1][1]
            else:
                end_boundary = len(self.lines) + 1
            
            begin_line = 0
            end_line = 0
            
            # Find first BEGIN (procedure body start)
            for line_num in range(start_line, end_boundary):
                clean_line = self._get_clean_line(line_num).upper()
                if re.search(r'\bBEGIN\b', clean_line):
                    begin_line = line_num
                    break
            
            # Find last END; before boundary (procedure body end)
            # This works because TAL procedures end with END;
            # and control structures also end with END; but the last one
            # in a procedure is always the procedure's closing END
            if begin_line > 0:
                for line_num in range(end_boundary - 1, begin_line, -1):
                    clean_line = self._get_clean_line(line_num).upper()
                    if re.search(r'\bEND\s*;', clean_line):
                        end_line = line_num
                        break
            
            if begin_line > 0 and end_line > 0:
                proc.body_start_line = begin_line
                proc.body_end_line = end_line
                self.proc_ranges[proc.name] = (begin_line, end_line)
    
    def _parse_procedure_bodies(self):
        """Parse procedure bodies for local variables and detailed analysis."""
        for proc in self.procedures:
            if proc.is_forward or proc.is_external:
                continue
            
            if proc.body_start_line == 0 or proc.body_end_line == 0:
                continue
            
            # Extract body text
            body_lines = self.lines[proc.body_start_line - 1:proc.body_end_line]
            proc.body_text = '\n'.join(body_lines)
            
            # Parse local variable declarations
            self._parse_local_variables(proc)
    
    def _parse_local_variables(self, proc: ProcedureDetail):
        """Parse local variable declarations from procedure body."""
        if not proc.body_text:
            return
        
        # Local vars are typically declared right after BEGIN
        lines = proc.body_text.split('\n')
        
        for line in lines:
            line = self._remove_comments(line).strip()
            if not line:
                continue
            
            # Stop at first executable statement (heuristic)
            if re.match(r'^(IF|WHILE|FOR|CALL|CASE|RETURN)\b', line, re.IGNORECASE):
                break
            
            # Look for type declarations
            var_match = re.match(
                r'^(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING(?:\([^)]*\))?|FIXED(?:\([^)]*\))?|'
                r'UNSIGNED(?:\([^)]*\))?)\s+(.+?)\s*;',
                line,
                re.IGNORECASE
            )
            
            if var_match:
                type_str = var_match.group(1)
                names_str = var_match.group(2)
                
                for name in names_str.split(','):
                    name = name.strip()
                    if not name:
                        continue
                    
                    # Check for array
                    array_match = re.match(r'([.*a-zA-Z_][a-zA-Z0-9_^]*)\s*\[', name)
                    if array_match:
                        var_name = array_match.group(1)
                    else:
                        var_name = name
                    
                    proc.local_vars.append({
                        'name': var_name.lstrip('.*'),
                        'type': type_str.upper(),
                        'is_pointer': var_name.startswith('.')
                    })
    
    # =========================================================================
    # CALL EXTRACTION
    # =========================================================================
    
    def _extract_calls(self):
        """Extract all procedure calls from the source."""
        current_proc: Optional[str] = None
        
        for line_num, line in enumerate(self.lines, 1):
            clean_line = self._remove_comments(line).strip()
            
            # Track current procedure
            proc_match = re.search(r'\bPROC\s+([a-zA-Z_][a-zA-Z0-9_^]*)', clean_line, re.IGNORECASE)
            if proc_match:
                current_proc = proc_match.group(1)
                continue
            
            if not current_proc:
                continue
            
            # Find CALL statements
            for match in re.finditer(
                r'\bCALL\s+([a-zA-Z_][a-zA-Z0-9_^]*)\s*(?:\(([^)]*)\))?',
                clean_line,
                re.IGNORECASE
            ):
                callee = match.group(1)
                args_str = match.group(2) or ""
                args = [a.strip() for a in args_str.split(',') if a.strip()]
                
                call_info = CallInfo(
                    callee=callee,
                    caller=current_proc,
                    arguments=args,
                    location=SourceLocation(self.filename, line_num, match.start()),
                    call_type="CALL"
                )
                self.calls.append(call_info)
            
            # Find PCAL statements (privileged call)
            for match in re.finditer(
                r'\bPCAL\s+([a-zA-Z_][a-zA-Z0-9_^]*)\s*(?:\(([^)]*)\))?',
                clean_line,
                re.IGNORECASE
            ):
                callee = match.group(1)
                args_str = match.group(2) or ""
                args = [a.strip() for a in args_str.split(',') if a.strip()]
                
                call_info = CallInfo(
                    callee=callee,
                    caller=current_proc,
                    arguments=args,
                    location=SourceLocation(self.filename, line_num, match.start()),
                    call_type="PCAL"
                )
                self.calls.append(call_info)
            
            # Find function-style calls: name(args)
            for match in re.finditer(
                r'(?<![a-zA-Z0-9_])([a-zA-Z_][a-zA-Z0-9_^]*)\s*\(([^)]*)\)',
                clean_line
            ):
                potential_callee = match.group(1)
                
                # Skip keywords, types, current procedure
                if potential_callee.upper() in self.KEYWORDS:
                    continue
                if potential_callee.upper() in self.TYPE_KEYWORDS:
                    continue
                if potential_callee == current_proc:
                    continue
                
                # Skip if preceded by CALL/PCAL (already captured)
                before_pos = match.start()
                prefix = clean_line[:before_pos].strip().upper()
                if prefix.endswith('CALL') or prefix.endswith('PCAL'):
                    continue
                
                # Skip if it looks like an array access
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_^]*\s*\[', clean_line[match.start():]):
                    continue
                
                args_str = match.group(2) or ""
                args = [a.strip() for a in args_str.split(',') if a.strip()]
                
                call_info = CallInfo(
                    callee=potential_callee,
                    caller=current_proc,
                    arguments=args,
                    location=SourceLocation(self.filename, line_num, match.start()),
                    call_type="function"
                )
                self.calls.append(call_info)
    
    def _extract_subprocs(self):
        """Extract SUBPROC declarations."""
        current_proc: Optional[str] = None
        
        for line_num, line in enumerate(self.lines, 1):
            clean_line = self._remove_comments(line).strip()
            
            # Track current procedure
            proc_match = re.search(r'\bPROC\s+([a-zA-Z_][a-zA-Z0-9_^]*)', clean_line, re.IGNORECASE)
            if proc_match:
                current_proc = proc_match.group(1)
                continue
            
            if not current_proc:
                continue
            
            # Find SUBPROC declarations
            subproc_match = re.match(
                r'(?:(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING|FIXED|UNSIGNED(?:\([^)]*\))?)\s+)?'
                r'SUBPROC\s+([a-zA-Z_][a-zA-Z0-9_^]*)',
                clean_line,
                re.IGNORECASE
            )
            
            if subproc_match:
                return_type_str = subproc_match.group(1)
                subproc_name = subproc_match.group(2)
                
                subproc_info = SubprocInfo(
                    name=subproc_name,
                    parent_proc=current_proc,
                    return_type=parse_tal_type(return_type_str) if return_type_str else None,
                    location=SourceLocation(self.filename, line_num, 1)
                )
                self.subprocs.append(subproc_info)
                
                # Also track in parent procedure
                for proc in self.procedures:
                    if proc.name == current_proc:
                        proc.subprocs.append(subproc_name)
                        break
    
    def _build_call_relationships(self):
        """Build call relationships between procedures."""
        # Build reverse lookup
        reverse_calls: Dict[str, List[str]] = {}
        for call in self.calls:
            if call.callee not in reverse_calls:
                reverse_calls[call.callee] = []
            if call.caller not in reverse_calls[call.callee]:
                reverse_calls[call.callee].append(call.caller)
        
        # Update procedure details
        for proc in self.procedures:
            proc.calls = [c for c in self.calls if c.caller == proc.name]
            proc.called_by = reverse_calls.get(proc.name, [])
    
    # =========================================================================
    # COMPLEXITY CALCULATION
    # =========================================================================
    
    def _calculate_complexity(self):
        """Calculate cyclomatic complexity for all procedures."""
        for proc in self.procedures:
            if proc.body_text:
                proc.complexity = self._compute_cyclomatic_complexity(proc.body_text)
    
    def _compute_cyclomatic_complexity(self, body: str) -> int:
        """
        Calculate cyclomatic complexity of procedure body.
        
        Cyclomatic complexity = E - N + 2P
        Simplified: 1 + number of decision points
        """
        complexity = 1  # Base complexity
        
        # Count decision points
        decision_patterns = [
            r'\bIF\b',           # IF statements
            r'\bWHILE\b',        # WHILE loops
            r'\bFOR\b',          # FOR loops
            r'\bCASE\b',         # CASE statements
            r'\bAND\b',          # Compound conditions
            r'\bOR\b',           # Compound conditions
            r'\bLAND\b',         # Logical AND
            r'\bLOR\b',          # Logical OR
        ]
        
        # Remove strings to avoid false matches
        clean_body = re.sub(r'"[^"]*"', '""', body)
        
        for pattern in decision_patterns:
            complexity += len(re.findall(pattern, clean_body, re.IGNORECASE))
        
        return complexity
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _get_type_size(self, tal_type: TALType) -> int:
        """Get size in bytes for a TAL type."""
        sizes = {
            TALType.INT: 2,
            TALType.INT16: 2,
            TALType.INT32: 4,
            TALType.INT64: 8,
            TALType.REAL: 4,
            TALType.REAL32: 4,
            TALType.REAL64: 8,
            TALType.STRING: 1,
            TALType.FIXED: 8,
            TALType.UNSIGNED: 2,
            TALType.BYTE: 1,
            TALType.CHAR: 1,
        }
        return sizes.get(tal_type, 2)
    
    def _split_declarations(self, text: str) -> List[str]:
        """Split declarations by comma, handling nested parentheses."""
        parts = []
        current: List[str] = []
        depth = 0
        
        for char in text:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        
        if current:
            parts.append(''.join(current).strip())
        
        return parts


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_tal_file(filename: str) -> ParseResult:
    """
    Parse a TAL source file.
    
    Args:
        filename: Path to TAL file
        
    Returns:
        ParseResult with all parsed components
    """
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        source = f.read()
    
    parser = EnhancedTALParser()
    return parser.parse(source, filename)


def parse_tal_string(source: str, filename: str = "<string>") -> ParseResult:
    """
    Parse TAL source code from a string.
    
    Args:
        source: TAL source code
        filename: Optional filename for error reporting
        
    Returns:
        ParseResult with all parsed components
    """
    parser = EnhancedTALParser()
    return parser.parse(source, filename)


def get_call_graph(source: str) -> Dict[str, List[str]]:
    """
    Extract call graph from TAL source.
    
    Args:
        source: TAL source code
        
    Returns:
        Dictionary mapping caller -> list of callees
    """
    result = parse_tal_string(source)
    return result.call_graph


def get_procedure_complexity(source: str) -> Dict[str, int]:
    """
    Get cyclomatic complexity for all procedures.
    
    Args:
        source: TAL source code
        
    Returns:
        Dictionary mapping procedure name -> complexity
    """
    result = parse_tal_string(source)
    return {p.name: p.complexity for p in result.procedures}


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Command-line interface for TAL enhanced parser."""
    import sys
    import os
    import json
    
    if len(sys.argv) < 2:
        print("TAL Enhanced Parser")
        print("=" * 60)
        print("\nUsage: python tal_enhanced_parser.py <tal_file> [options]")
        print("\nOptions:")
        print("  --json        Output results as JSON")
        print("  --calls       Show call graph only")
        print("  --complexity  Show complexity metrics only")
        print("  --defines     Show DEFINEs only")
        print("  --structs     Show STRUCTs only")
        print("\nExamples:")
        print("  python tal_enhanced_parser.py myprogram.tal")
        print("  python tal_enhanced_parser.py myprogram.tal --json > results.json")
        print("  python tal_enhanced_parser.py myprogram.tal --calls")
        sys.exit(1)
    
    tal_file = sys.argv[1]
    options = sys.argv[2:]
    
    output_json = '--json' in options
    show_calls = '--calls' in options
    show_complexity = '--complexity' in options
    show_defines = '--defines' in options
    show_structs = '--structs' in options
    
    if not os.path.exists(tal_file):
        print(f"Error: File '{tal_file}' not found")
        sys.exit(1)
    
    try:
        result = parse_tal_file(tal_file)
        
        if output_json:
            print(json.dumps(result.to_dict(), indent=2))
            return
        
        # Filtered output modes
        if show_calls:
            print(f"Call Graph for {tal_file}")
            print("=" * 60)
            for caller, callees in sorted(result.call_graph.items()):
                print(f"  {caller} -> {', '.join(callees)}")
            print(f"\nEntry points: {', '.join(result.entry_points)}")
            return
        
        if show_complexity:
            print(f"Complexity Metrics for {tal_file}")
            print("=" * 60)
            for proc in sorted(result.procedures, key=lambda p: -p.complexity):
                print(f"  {proc.name}: {proc.complexity}")
            return
        
        if show_defines:
            print(f"DEFINEs in {tal_file}")
            print("=" * 60)
            for d in result.defines:
                if d.is_macro:
                    print(f"  {d.name}({', '.join(d.params)}) = {d.value[:60]}...")
                else:
                    print(f"  {d.name} = {d.value[:60]}")
            return
        
        if show_structs:
            print(f"STRUCTs in {tal_file}")
            print("=" * 60)
            for s in result.structs:
                print(f"  {s.name} ({len(s.fields)} fields, {s.total_size} bytes)")
                for f in s.fields:
                    arr_info = f"[{f.array_bounds[0]}:{f.array_bounds[1]}]" if f.is_array else ""
                    print(f"    {f.field_type.value} {f.name}{arr_info}")
            return
        
        # Full output
        print(f"Parsing: {tal_file}")
        print("=" * 60)
        
        # Summary
        print(f"\nSummary:")
        print(f"  Source lines: {result.source_lines}")
        print(f"  Procedures:   {len(result.procedures)}")
        print(f"  SUBPROCs:     {len(result.subprocs)}")
        print(f"  DEFINEs:      {len(result.defines)}")
        print(f"  STRUCTs:      {len(result.structs)}")
        print(f"  LITERALs:     {len(result.literals)}")
        print(f"  Globals:      {len(result.globals)}")
        print(f"  Calls:        {len(result.calls)}")
        
        # Procedures
        if result.procedures:
            print(f"\nProcedures:")
            print("-" * 40)
            for proc in result.procedures:
                attrs = f" [{', '.join(proc.attributes)}]" if proc.attributes else ''
                params = ', '.join(p['name'] for p in proc.parameters)
                ret = f"{proc.return_type.value} " if proc.return_type else ""
                print(f"  {ret}{proc.name}({params}){attrs}")
                if proc.local_vars:
                    print(f"    Local vars: {len(proc.local_vars)}")
                if proc.calls:
                    calls_str = ', '.join(set(c.callee for c in proc.calls))
                    print(f"    Calls: {calls_str}")
                if proc.subprocs:
                    print(f"    SUBPROCs: {', '.join(proc.subprocs)}")
                print(f"    Complexity: {proc.complexity}")
                print(f"    Lines: {proc.body_start_line}-{proc.body_end_line}")
        
        # DEFINEs
        if result.defines:
            print(f"\nDEFINEs ({len(result.defines)}):")
            print("-" * 40)
            for d in result.defines[:10]:  # Show first 10
                if d.is_macro:
                    print(f"  {d.name}({', '.join(d.params)}) = {d.value[:50]}...")
                else:
                    val = d.value[:50] + "..." if len(d.value) > 50 else d.value
                    print(f"  {d.name} = {val}")
            if len(result.defines) > 10:
                print(f"  ... and {len(result.defines) - 10} more")
        
        # STRUCTs
        if result.structs:
            print(f"\nSTRUCTs ({len(result.structs)}):")
            print("-" * 40)
            for s in result.structs:
                print(f"  {s.name} ({len(s.fields)} fields, {s.total_size} bytes)")
        
        # Call Graph
        if result.call_graph:
            print(f"\nCall Graph:")
            print("-" * 40)
            for caller, callees in sorted(result.call_graph.items())[:10]:
                print(f"  {caller} -> {', '.join(callees)}")
            if len(result.call_graph) > 10:
                print(f"  ... and {len(result.call_graph) - 10} more")
        
        # Entry Points
        if result.entry_points:
            print(f"\nEntry Points (not called internally):")
            print(f"  {', '.join(result.entry_points)}")
        
        # External Calls
        external = result.external_calls
        if external:
            ext_names = sorted(set(c.callee for c in external))
            print(f"\nExternal/System Calls ({len(ext_names)} unique):")
            print(f"  {', '.join(ext_names[:15])}")
            if len(ext_names) > 15:
                print(f"  ... and {len(ext_names) - 15} more")
        
        # Errors
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            print("-" * 40)
            for error in result.errors:
                print(f"  {error}")
        
        # Warnings
        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            print("-" * 40)
            for warning in result.warnings:
                print(f"  {warning}")
    
    except Exception as e:
        print(f"Error parsing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
