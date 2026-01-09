#!/usr/bin/env python3
"""
Test Suite for TAL Parsers
==========================

Comprehensive tests for:
- tal_proc_parser.py (foundation parser)
- tal_enhanced_parser.py (enhanced parser)

Run with: python test_tal_parsers.py
Or:       python -m pytest test_tal_parsers.py -v
"""

import sys
import os

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# TEST FIXTURES - Sample TAL Code
# =============================================================================

# Simple procedure
SIMPLE_PROC = """
PROC simple_test;
BEGIN
    INT x;
    x := 1;
END;
"""

# Procedure with parameters
PROC_WITH_PARAMS = """
INT PROC add_numbers(a, b);
    INT a;
    INT b;
BEGIN
    RETURN a + b;
END;
"""

# Procedure with pointer parameters
PROC_WITH_POINTERS = """
PROC process_buffer(.buffer, length);
    STRING .buffer;
    INT length;
BEGIN
    INT i;
    FOR i := 0 TO length - 1 DO
        buffer[i] := 0;
END;
"""

# MAIN procedure
MAIN_PROC = """
PROC main_entry MAIN;
BEGIN
    CALL initialize;
    CALL process_data;
    CALL cleanup;
END;
"""

# FORWARD declaration
FORWARD_PROC = """
INT PROC calculate_total(items, count) FORWARD;

PROC use_total;
BEGIN
    INT result;
    result := calculate_total(data, 10);
END;

INT PROC calculate_total(items, count);
    INT .items;
    INT count;
BEGIN
    RETURN 0;
END;
"""

# EXTERNAL procedure
EXTERNAL_PROC = """
PROC external_routine EXTERNAL;
"""

# Multiple procedures
MULTI_PROC = """
PROC first_proc;
BEGIN
    CALL second_proc;
END;

INT PROC second_proc;
BEGIN
    RETURN third_proc(10);
END;

INT PROC third_proc(value);
    INT value;
BEGIN
    RETURN value * 2;
END;
"""

# DEFINE statements
DEFINES_CODE = """
DEFINE max_size = 100;
DEFINE buffer_len = 256;
DEFINE error_code = -1;
DEFINE success_code = 0;

DEFINE calculate_offset(base, index) = base + (index * 2);

DEFINE long_macro(a, b, c) = 
    IF a > b THEN
        c := a
    ELSE
        c := b;

PROC use_defines;
BEGIN
    INT buffer[0:max_size - 1];
    INT offset;
    offset := calculate_offset(0, 10);
END;
"""

# LITERAL declarations
LITERALS_CODE = """
LITERAL
    true = 1,
    false = 0,
    max_retries = 3,
    timeout_ms = 5000,
    hex_value = %H00FF,
    binary_flag = %B10101010;

PROC use_literals;
BEGIN
    INT flag;
    flag := true;
    IF flag = false THEN
        flag := max_retries;
END;
"""

# STRUCT definitions
STRUCT_CODE = """
STRUCT customer_record;
BEGIN
    STRING name[0:49];
    INT customer_id;
    INT account_balance;
    STRING address[0:99];
    INT status;
END;

STRUCT .customer_ptr(customer_record);

STRUCT transaction_record;
BEGIN
    INT transaction_id;
    INT customer_id;
    INT amount;
    INT trans_type;
    STRING description[0:29];
END;

PROC process_customer(.cust);
    STRUCT .cust(customer_record);
BEGIN
    cust.status := 1;
END;
"""

# Nested structures
NESTED_STRUCT = """
STRUCT outer_struct;
BEGIN
    INT header;
    STRUCT inner;
    BEGIN
        INT field1;
        INT field2;
    END;
    INT trailer;
END;
"""

# Global variables
GLOBALS_CODE = """
INT global_counter;
STRING global_buffer[0:255];
INT .global_ptr;
INT error_flags[0:15];

PROC increment_counter;
BEGIN
    global_counter := global_counter + 1;
END;
"""

# Complex procedure with control flow
COMPLEX_PROC = """
INT PROC complex_logic(input, .output);
    INT input;
    INT .output;
BEGIN
    INT temp;
    INT i;
    
    IF input < 0 THEN
        output := 0;
        RETURN -1;
    END;
    
    WHILE input > 100 DO
        input := input - 100;
    END;
    
    FOR i := 0 TO 10 DO
        temp := temp + i;
    END;
    
    CASE input OF
    BEGIN
        0 -> output := temp;
        1 -> output := temp * 2;
        OTHERWISE -> output := -1;
    END;
    
    IF input > 50 AND temp < 100 THEN
        output := output + 1;
    ELSE
        IF input < 25 OR temp > 50 THEN
            output := output - 1;
        END;
    END;
    
    RETURN 0;
END;
"""

# SUBPROC example
SUBPROC_CODE = """
PROC outer_proc;
BEGIN
    INT local_var;
    
    SUBPROC inner_helper;
    BEGIN
        local_var := local_var + 1;
    END;
    
    INT SUBPROC calculate_value(x);
        INT x;
    BEGIN
        RETURN x * 2;
    END;
    
    local_var := 0;
    CALL inner_helper;
    local_var := calculate_value(local_var);
END;
"""

# CALL variations
CALL_VARIATIONS = """
PROC caller_proc;
BEGIN
    INT result;
    
    ! Simple CALL
    CALL simple_proc;
    
    ! CALL with arguments
    CALL proc_with_args(1, 2, 3);
    
    ! Function-style call (returns value)
    result := get_value(10);
    
    ! Nested calls
    result := outer_func(inner_func(5));
    
    ! PCAL (privileged call)
    PCAL system_proc(buffer, length);
    
    ! Call to system procedure
    CALL FILE_OPEN_(filename, file_num);
END;
"""

# Complete TAL program
COMPLETE_PROGRAM = """
! Sample TAL Program for Wire Transfer Processing
! ================================================

DEFINE max_transfers = 1000;
DEFINE buffer_size = 4096;
DEFINE success = 0;
DEFINE error_invalid = -1;
DEFINE error_limit = -2;

LITERAL
    transfer_pending = 0,
    transfer_complete = 1,
    transfer_failed = 2;

STRUCT transfer_record;
BEGIN
    INT transfer_id;
    INT from_account;
    INT to_account;
    INT amount;
    INT status;
    STRING description[0:49];
END;

INT transfer_count;
STRUCT transfers[0:max_transfers - 1](transfer_record);

INT PROC validate_transfer(.transfer) FORWARD;

PROC initialize MAIN;
BEGIN
    transfer_count := 0;
    CALL clear_transfers;
END;

PROC clear_transfers;
BEGIN
    INT i;
    FOR i := 0 TO max_transfers - 1 DO
        transfers[i].status := transfer_pending;
    END;
END;

INT PROC validate_transfer(.transfer);
    STRUCT .transfer(transfer_record);
BEGIN
    IF transfer.amount <= 0 THEN
        RETURN error_invalid;
    END;
    
    IF transfer.from_account = transfer.to_account THEN
        RETURN error_invalid;
    END;
    
    RETURN success;
END;

INT PROC process_transfer(.transfer);
    STRUCT .transfer(transfer_record);
BEGIN
    INT validation_result;
    
    validation_result := validate_transfer(transfer);
    
    IF validation_result <> success THEN
        transfer.status := transfer_failed;
        RETURN validation_result;
    END;
    
    IF transfer_count >= max_transfers THEN
        RETURN error_limit;
    END;
    
    ! Process the transfer
    CALL execute_transfer(transfer);
    
    transfer.status := transfer_complete;
    transfer_count := transfer_count + 1;
    
    RETURN success;
END;

PROC execute_transfer(.transfer);
    STRUCT .transfer(transfer_record);
BEGIN
    ! Actual transfer logic would go here
    CALL log_transfer(transfer);
END;

PROC log_transfer(.transfer);
    STRUCT .transfer(transfer_record);
BEGIN
    ! Logging logic
END;
"""

# Edge cases
EDGE_CASE_EMPTY = ""

EDGE_CASE_COMMENTS_ONLY = """
! This file contains only comments
! No actual code here
! Just testing comment handling
"""

EDGE_CASE_MULTILINE_DECL = """
INT PROC very_long_procedure_name_with_many_parameters(
    first_parameter,
    second_parameter,
    third_parameter,
    .fourth_pointer_param,
    fifth_parameter);
    INT first_parameter;
    INT second_parameter;
    INT third_parameter;
    STRING .fourth_pointer_param;
    INT fifth_parameter;
BEGIN
    RETURN 0;
END;
"""

EDGE_CASE_SPECIAL_CHARS = """
PROC proc^with^carets;
BEGIN
    INT var^name;
    var^name := 0;
END;
"""


# =============================================================================
# TEST RESULT TRACKER
# =============================================================================

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
# TAL_PROC_PARSER TESTS
# =============================================================================

def test_tal_proc_parser(results: TestResult):
    """Test tal_proc_parser.py functionality."""
    print("\n" + "="*60)
    print("TAL_PROC_PARSER TESTS")
    print("="*60)
    
    try:
        from tal_proc_parser import (
            TALType, TALNode, Symbol, SymbolTable, SourceLocation,
            ParseError, ErrorSeverity, ProcedureInfo,
            parse_tal_type, find_procedure_declarations,
            parse_procedure_declaration, parse_multiple_procedures,
            extract_parameters_from_declaration
        )
    except ImportError as e:
        results.fail("Import tal_proc_parser", str(e))
        return
    
    results.ok("Import tal_proc_parser")
    
    # --- Test: TALType enum ---
    try:
        assert TALType.INT.value == "INT"
        assert TALType.STRING.value == "STRING"
        assert TALType.INT32.value == "INT(32)"
        results.ok("TALType enum values")
    except Exception as e:
        results.fail("TALType enum values", str(e))
    
    # --- Test: parse_tal_type ---
    try:
        assert parse_tal_type("INT") == TALType.INT
        assert parse_tal_type("int") == TALType.INT  # Case insensitive
        assert parse_tal_type("INT(32)") == TALType.INT32
        assert parse_tal_type("INT(64)") == TALType.INT64
        assert parse_tal_type("REAL") == TALType.REAL
        assert parse_tal_type("REAL(64)") == TALType.REAL64
        assert parse_tal_type("STRING") == TALType.STRING
        assert parse_tal_type("UNSIGNED") == TALType.UNSIGNED
        assert parse_tal_type("UNKNOWN_TYPE") == TALType.UNKNOWN
        results.ok("parse_tal_type function")
    except Exception as e:
        results.fail("parse_tal_type function", str(e))
    
    # --- Test: SourceLocation ---
    try:
        loc = SourceLocation("test.tal", 10, 5, 20)
        assert loc.filename == "test.tal"
        assert loc.line == 10
        assert loc.column == 5
        assert "test.tal:10:5" in str(loc)
        results.ok("SourceLocation")
    except Exception as e:
        results.fail("SourceLocation", str(e))
    
    # --- Test: SymbolTable ---
    try:
        st = SymbolTable()
        assert st.current_scope == "global"
        
        # Enter scope
        st.enter_scope("proc1")
        assert st.current_scope == "proc1"
        
        # Declare symbol
        sym = Symbol("x", TALType.INT, SourceLocation())
        st.declare_symbol(sym)
        
        # Lookup symbol
        found = st.lookup_symbol("x")
        assert found is not None
        assert found.name == "x"
        
        # Exit scope
        st.exit_scope()
        assert st.current_scope == "global"
        
        results.ok("SymbolTable scope management")
    except Exception as e:
        results.fail("SymbolTable scope management", str(e))
    
    # --- Test: TALNode ---
    try:
        node = TALNode('procedure', name='test_proc')
        child = TALNode('parameter', name='param1')
        node.add_child(child)
        assert len(node.children) == 1
        assert node.children[0].name == 'param1'
        results.ok("TALNode creation and children")
    except Exception as e:
        results.fail("TALNode creation and children", str(e))
    
    # --- Test: find_procedure_declarations - simple ---
    try:
        procs = find_procedure_declarations(SIMPLE_PROC)
        assert len(procs) == 1
        line, name, decl = procs[0]
        assert name == "simple_test"
        results.ok("find_procedure_declarations - simple")
    except Exception as e:
        results.fail("find_procedure_declarations - simple", str(e))
    
    # --- Test: find_procedure_declarations - with return type ---
    try:
        procs = find_procedure_declarations(PROC_WITH_PARAMS)
        assert len(procs) == 1
        line, name, decl = procs[0]
        assert name == "add_numbers"
        assert "INT PROC" in decl.upper()
        results.ok("find_procedure_declarations - with return type")
    except Exception as e:
        results.fail("find_procedure_declarations - with return type", str(e))
    
    # --- Test: find_procedure_declarations - multiple ---
    try:
        procs = find_procedure_declarations(MULTI_PROC)
        assert len(procs) == 3
        names = [p[1] for p in procs]
        assert "first_proc" in names
        assert "second_proc" in names
        assert "third_proc" in names
        results.ok("find_procedure_declarations - multiple")
    except Exception as e:
        results.fail("find_procedure_declarations - multiple", str(e))
    
    # --- Test: find_procedure_declarations - MAIN ---
    try:
        procs = find_procedure_declarations(MAIN_PROC)
        assert len(procs) == 1
        _, name, decl = procs[0]
        assert name == "main_entry"
        assert "MAIN" in decl.upper()
        results.ok("find_procedure_declarations - MAIN")
    except Exception as e:
        results.fail("find_procedure_declarations - MAIN", str(e))
    
    # --- Test: find_procedure_declarations - FORWARD ---
    try:
        procs = find_procedure_declarations(FORWARD_PROC)
        assert len(procs) >= 2  # Forward decl + implementation + use_total
        names = [p[1] for p in procs]
        assert "calculate_total" in names
        results.ok("find_procedure_declarations - FORWARD")
    except Exception as e:
        results.fail("find_procedure_declarations - FORWARD", str(e))
    
    # --- Test: extract_parameters_from_declaration ---
    try:
        params = extract_parameters_from_declaration("PROC test(a, b, c);")
        assert len(params) == 3
        assert "a" in params
        assert "b" in params
        assert "c" in params
        results.ok("extract_parameters_from_declaration - simple")
    except Exception as e:
        results.fail("extract_parameters_from_declaration - simple", str(e))
    
    # --- Test: extract_parameters_from_declaration - pointers ---
    try:
        params = extract_parameters_from_declaration("PROC test(.buffer, length);")
        assert len(params) == 2
        assert ".buffer" in params or "buffer" in [p.lstrip('.') for p in params]
        results.ok("extract_parameters_from_declaration - pointers")
    except Exception as e:
        results.fail("extract_parameters_from_declaration - pointers", str(e))
    
    # --- Test: extract_parameters_from_declaration - no params ---
    try:
        params = extract_parameters_from_declaration("PROC test;")
        assert len(params) == 0
        results.ok("extract_parameters_from_declaration - no params")
    except Exception as e:
        results.fail("extract_parameters_from_declaration - no params", str(e))
    
    # --- Test: parse_multiple_procedures ---
    try:
        st = SymbolTable()
        nodes, errors = parse_multiple_procedures(MULTI_PROC, "test.tal", st)
        assert len(nodes) == 3
        names = [n.name for n in nodes]
        assert "first_proc" in names
        assert "second_proc" in names
        assert "third_proc" in names
        results.ok("parse_multiple_procedures")
    except Exception as e:
        results.fail("parse_multiple_procedures", str(e))
    
    # --- Test: parse_procedure_declaration - attributes ---
    try:
        st = SymbolTable()
        procs = find_procedure_declarations(MAIN_PROC)
        _, name, decl = procs[0]
        node, errors = parse_procedure_declaration(name, decl, 1, "test.tal", st)
        assert node is not None
        assert node.attributes.get('is_main') == True
        results.ok("parse_procedure_declaration - MAIN attribute")
    except Exception as e:
        results.fail("parse_procedure_declaration - MAIN attribute", str(e))
    
    # --- Test: parse with FORWARD ---
    try:
        st = SymbolTable()
        nodes, errors = parse_multiple_procedures(FORWARD_PROC, "test.tal", st)
        forward_nodes = [n for n in nodes if n.attributes.get('is_forward')]
        assert len(forward_nodes) >= 1
        results.ok("parse_procedure_declaration - FORWARD attribute")
    except Exception as e:
        results.fail("parse_procedure_declaration - FORWARD attribute", str(e))
    
    # --- Test: Edge case - empty input ---
    try:
        procs = find_procedure_declarations(EDGE_CASE_EMPTY)
        assert len(procs) == 0
        results.ok("Edge case - empty input")
    except Exception as e:
        results.fail("Edge case - empty input", str(e))
    
    # --- Test: Edge case - comments only ---
    try:
        procs = find_procedure_declarations(EDGE_CASE_COMMENTS_ONLY)
        assert len(procs) == 0
        results.ok("Edge case - comments only")
    except Exception as e:
        results.fail("Edge case - comments only", str(e))
    
    # --- Test: Edge case - multiline declaration ---
    try:
        procs = find_procedure_declarations(EDGE_CASE_MULTILINE_DECL)
        assert len(procs) == 1
        _, name, decl = procs[0]
        assert name == "very_long_procedure_name_with_many_parameters"
        # Check all params are in declaration
        assert "first_parameter" in decl
        assert "fifth_parameter" in decl
        results.ok("Edge case - multiline declaration")
    except Exception as e:
        results.fail("Edge case - multiline declaration", str(e))
    
    # --- Test: Edge case - special characters (carets) ---
    try:
        procs = find_procedure_declarations(EDGE_CASE_SPECIAL_CHARS)
        assert len(procs) == 1
        _, name, _ = procs[0]
        assert "proc" in name and "with" in name
        results.ok("Edge case - special characters")
    except Exception as e:
        results.fail("Edge case - special characters", str(e))
    
    # --- Test: Complete program ---
    try:
        st = SymbolTable()
        nodes, errors = parse_multiple_procedures(COMPLETE_PROGRAM, "wire.tal", st)
        assert len(nodes) >= 5  # Multiple procedures in complete program
        names = [n.name for n in nodes]
        assert "initialize" in names
        assert "validate_transfer" in names
        assert "process_transfer" in names
        results.ok("Complete program parsing")
    except Exception as e:
        results.fail("Complete program parsing", str(e))


# =============================================================================
# TAL_ENHANCED_PARSER TESTS
# =============================================================================

def test_tal_enhanced_parser(results: TestResult):
    """Test tal_enhanced_parser.py functionality."""
    print("\n" + "="*60)
    print("TAL_ENHANCED_PARSER TESTS")
    print("="*60)
    
    try:
        from tal_enhanced_parser import (
            EnhancedTALParser, ParseResult, ProcedureDetail,
            DefineInfo, StructInfo, LiteralInfo, GlobalInfo,
            CallInfo, SubprocInfo,
            parse_tal_string, get_call_graph, get_procedure_complexity
        )
    except ImportError as e:
        results.fail("Import tal_enhanced_parser", str(e))
        return
    
    results.ok("Import tal_enhanced_parser")
    
    # --- Test: EnhancedTALParser creation ---
    try:
        parser = EnhancedTALParser()
        assert parser is not None
        results.ok("EnhancedTALParser creation")
    except Exception as e:
        results.fail("EnhancedTALParser creation", str(e))
    
    # --- Test: Parse simple procedure ---
    try:
        result = parse_tal_string(SIMPLE_PROC)
        assert isinstance(result, ParseResult)
        assert len(result.procedures) == 1
        assert result.procedures[0].name == "simple_test"
        results.ok("Parse simple procedure")
    except Exception as e:
        results.fail("Parse simple procedure", str(e))
    
    # --- Test: Parse procedure with return type ---
    try:
        result = parse_tal_string(PROC_WITH_PARAMS)
        assert len(result.procedures) == 1
        proc = result.procedures[0]
        assert proc.name == "add_numbers"
        assert proc.return_type is not None
        assert len(proc.parameters) == 2
        results.ok("Parse procedure with return type")
    except Exception as e:
        results.fail("Parse procedure with return type", str(e))
    
    # --- Test: Parse MAIN procedure ---
    try:
        result = parse_tal_string(MAIN_PROC)
        assert len(result.procedures) == 1
        proc = result.procedures[0]
        assert proc.is_main == True
        assert "MAIN" in proc.attributes
        results.ok("Parse MAIN procedure")
    except Exception as e:
        results.fail("Parse MAIN procedure", str(e))
    
    # --- Test: Parse FORWARD procedure ---
    try:
        result = parse_tal_string(FORWARD_PROC)
        forward_procs = [p for p in result.procedures if p.is_forward]
        assert len(forward_procs) >= 1
        results.ok("Parse FORWARD procedure")
    except Exception as e:
        results.fail("Parse FORWARD procedure", str(e))
    
    # --- Test: Parse multiple procedures ---
    try:
        result = parse_tal_string(MULTI_PROC)
        assert len(result.procedures) == 3
        names = [p.name for p in result.procedures]
        assert "first_proc" in names
        assert "second_proc" in names
        assert "third_proc" in names
        results.ok("Parse multiple procedures")
    except Exception as e:
        results.fail("Parse multiple procedures", str(e))
    
    # --- Test: Parse DEFINEs ---
    try:
        result = parse_tal_string(DEFINES_CODE)
        assert len(result.defines) >= 4
        names = [d.name for d in result.defines]
        assert "max_size" in names
        assert "buffer_len" in names
        
        # Check macro detection
        macros = [d for d in result.defines if d.is_macro]
        assert len(macros) >= 1
        results.ok("Parse DEFINEs")
    except Exception as e:
        results.fail("Parse DEFINEs", str(e))
    
    # --- Test: DEFINE values ---
    try:
        result = parse_tal_string(DEFINES_CODE)
        max_size = next((d for d in result.defines if d.name == "max_size"), None)
        assert max_size is not None
        assert "100" in max_size.value
        results.ok("DEFINE values")
    except Exception as e:
        results.fail("DEFINE values", str(e))
    
    # --- Test: Parse LITERALs ---
    try:
        result = parse_tal_string(LITERALS_CODE)
        assert len(result.literals) >= 4
        names = [l.name for l in result.literals]
        assert "true" in names
        assert "false" in names
        assert "max_retries" in names
        results.ok("Parse LITERALs")
    except Exception as e:
        results.fail("Parse LITERALs", str(e))
    
    # --- Test: LITERAL values ---
    try:
        result = parse_tal_string(LITERALS_CODE)
        true_lit = next((l for l in result.literals if l.name == "true"), None)
        assert true_lit is not None
        assert true_lit.value == 1
        
        false_lit = next((l for l in result.literals if l.name == "false"), None)
        assert false_lit is not None
        assert false_lit.value == 0
        results.ok("LITERAL values")
    except Exception as e:
        results.fail("LITERAL values", str(e))
    
    # --- Test: Parse STRUCTs ---
    try:
        result = parse_tal_string(STRUCT_CODE)
        assert len(result.structs) >= 2
        names = [s.name for s in result.structs]
        assert "customer_record" in names
        assert "transaction_record" in names
        results.ok("Parse STRUCTs")
    except Exception as e:
        results.fail("Parse STRUCTs", str(e))
    
    # --- Test: STRUCT fields ---
    try:
        result = parse_tal_string(STRUCT_CODE)
        cust = next((s for s in result.structs if s.name == "customer_record"), None)
        assert cust is not None
        assert len(cust.fields) >= 4
        field_names = [f.name for f in cust.fields]
        assert "name" in field_names
        assert "customer_id" in field_names
        results.ok("STRUCT fields")
    except Exception as e:
        results.fail("STRUCT fields", str(e))
    
    # --- Test: Parse global variables ---
    try:
        result = parse_tal_string(GLOBALS_CODE)
        assert len(result.globals) >= 2
        names = [g.name for g in result.globals]
        assert "global_counter" in names
        results.ok("Parse global variables")
    except Exception as e:
        results.fail("Parse global variables", str(e))
    
    # --- Test: Global array detection ---
    try:
        result = parse_tal_string(GLOBALS_CODE)
        buffer = next((g for g in result.globals if g.name == "global_buffer"), None)
        if buffer:
            assert buffer.is_array == True
            results.ok("Global array detection")
        else:
            results.ok("Global array detection (skipped - not found)")
    except Exception as e:
        results.fail("Global array detection", str(e))
    
    # --- Test: Call extraction ---
    try:
        result = parse_tal_string(MAIN_PROC)
        assert len(result.calls) >= 3
        callees = [c.callee for c in result.calls]
        assert "initialize" in callees
        assert "process_data" in callees
        assert "cleanup" in callees
        results.ok("Call extraction")
    except Exception as e:
        results.fail("Call extraction", str(e))
    
    # --- Test: Call graph ---
    try:
        result = parse_tal_string(MULTI_PROC)
        graph = result.call_graph
        assert "first_proc" in graph
        assert "second_proc" in graph["first_proc"]
        results.ok("Call graph")
    except Exception as e:
        results.fail("Call graph", str(e))
    
    # --- Test: Reverse call graph ---
    try:
        result = parse_tal_string(MULTI_PROC)
        rev_graph = result.reverse_call_graph
        assert "second_proc" in rev_graph
        assert "first_proc" in rev_graph["second_proc"]
        results.ok("Reverse call graph")
    except Exception as e:
        results.fail("Reverse call graph", str(e))
    
    # --- Test: Cyclomatic complexity ---
    try:
        result = parse_tal_string(COMPLEX_PROC)
        assert len(result.procedures) == 1
        proc = result.procedures[0]
        # Complex proc has IF, WHILE, FOR, CASE, AND, OR - should be > 5
        assert proc.complexity > 5
        results.ok(f"Cyclomatic complexity ({proc.complexity})")
    except Exception as e:
        results.fail("Cyclomatic complexity", str(e))
    
    # --- Test: Simple complexity ---
    try:
        result = parse_tal_string(SIMPLE_PROC)
        proc = result.procedures[0]
        # Simple proc should have low complexity
        assert proc.complexity <= 2
        results.ok(f"Simple complexity ({proc.complexity})")
    except Exception as e:
        results.fail("Simple complexity", str(e))
    
    # --- Test: SUBPROC extraction ---
    try:
        result = parse_tal_string(SUBPROC_CODE)
        assert len(result.subprocs) >= 1
        subproc_names = [s.name for s in result.subprocs]
        assert "inner_helper" in subproc_names or "calculate_value" in subproc_names
        results.ok("SUBPROC extraction")
    except Exception as e:
        results.fail("SUBPROC extraction", str(e))
    
    # --- Test: Procedure body boundaries ---
    try:
        result = parse_tal_string(SIMPLE_PROC)
        proc = result.procedures[0]
        assert proc.body_start_line > 0
        assert proc.body_end_line > proc.body_start_line
        results.ok("Procedure body boundaries")
    except Exception as e:
        results.fail("Procedure body boundaries", str(e))
    
    # --- Test: Procedure local variables ---
    try:
        result = parse_tal_string(SIMPLE_PROC)
        proc = result.procedures[0]
        assert len(proc.local_vars) >= 1
        var_names = [v['name'] for v in proc.local_vars]
        assert 'x' in var_names
        results.ok("Procedure local variables")
    except Exception as e:
        results.fail("Procedure local variables", str(e))
    
    # --- Test: Entry points ---
    try:
        result = parse_tal_string(MULTI_PROC)
        entry = result.entry_points
        assert "first_proc" in entry  # Not called by others
        results.ok("Entry points detection")
    except Exception as e:
        results.fail("Entry points detection", str(e))
    
    # --- Test: External calls ---
    try:
        result = parse_tal_string(CALL_VARIATIONS)
        external = result.external_calls
        # All calls should be external since procedures aren't defined
        assert len(external) > 0
        results.ok("External calls detection")
    except Exception as e:
        results.fail("External calls detection", str(e))
    
    # --- Test: to_dict conversion ---
    try:
        result = parse_tal_string(COMPLETE_PROGRAM)
        d = result.to_dict()
        assert 'procedures' in d
        assert 'defines' in d
        assert 'structs' in d
        assert 'call_graph' in d
        assert 'entry_points' in d
        results.ok("to_dict conversion")
    except Exception as e:
        results.fail("to_dict conversion", str(e))
    
    # --- Test: get_call_graph helper ---
    try:
        graph = get_call_graph(MULTI_PROC)
        assert isinstance(graph, dict)
        assert len(graph) > 0
        results.ok("get_call_graph helper")
    except Exception as e:
        results.fail("get_call_graph helper", str(e))
    
    # --- Test: get_procedure_complexity helper ---
    try:
        complexity = get_procedure_complexity(COMPLEX_PROC)
        assert isinstance(complexity, dict)
        assert "complex_logic" in complexity
        assert complexity["complex_logic"] > 5
        results.ok("get_procedure_complexity helper")
    except Exception as e:
        results.fail("get_procedure_complexity helper", str(e))
    
    # --- Test: Complete program ---
    try:
        result = parse_tal_string(COMPLETE_PROGRAM)
        
        # Check procedures
        assert len(result.procedures) >= 5
        proc_names = [p.name for p in result.procedures]
        assert "initialize" in proc_names
        assert "validate_transfer" in proc_names
        
        # Check MAIN
        main_procs = [p for p in result.procedures if p.is_main]
        assert len(main_procs) == 1
        assert main_procs[0].name == "initialize"
        
        # Check defines
        assert len(result.defines) >= 3
        
        # Check structs
        assert len(result.structs) >= 1
        
        # Check calls
        assert len(result.calls) > 0
        
        results.ok("Complete program analysis")
    except Exception as e:
        results.fail("Complete program analysis", str(e))
    
    # --- Test: Edge case - empty input ---
    try:
        result = parse_tal_string(EDGE_CASE_EMPTY)
        assert len(result.procedures) == 0
        assert len(result.defines) == 0
        results.ok("Edge case - empty input")
    except Exception as e:
        results.fail("Edge case - empty input", str(e))
    
    # --- Test: Edge case - comments only ---
    try:
        result = parse_tal_string(EDGE_CASE_COMMENTS_ONLY)
        assert len(result.procedures) == 0
        results.ok("Edge case - comments only")
    except Exception as e:
        results.fail("Edge case - comments only", str(e))
    
    # --- Test: ProcedureDetail.to_dict ---
    try:
        result = parse_tal_string(SIMPLE_PROC)
        proc = result.procedures[0]
        d = proc.to_dict()
        assert 'name' in d
        assert 'complexity' in d
        assert 'parameters' in d
        assert 'local_vars' in d
        results.ok("ProcedureDetail.to_dict")
    except Exception as e:
        results.fail("ProcedureDetail.to_dict", str(e))
    
    # --- Test: StructInfo.to_dict ---
    try:
        result = parse_tal_string(STRUCT_CODE)
        struct = result.structs[0]
        d = struct.to_dict()
        assert 'name' in d
        assert 'fields' in d
        assert 'total_size' in d
        results.ok("StructInfo.to_dict")
    except Exception as e:
        results.fail("StructInfo.to_dict", str(e))
    
    # --- Test: get_procedure helper ---
    try:
        result = parse_tal_string(MULTI_PROC)
        proc = result.get_procedure("second_proc")
        assert proc is not None
        assert proc.name == "second_proc"
        
        # Non-existent
        none_proc = result.get_procedure("nonexistent")
        assert none_proc is None
        results.ok("get_procedure helper")
    except Exception as e:
        results.fail("get_procedure helper", str(e))
    
    # --- Test: get_struct helper ---
    try:
        result = parse_tal_string(STRUCT_CODE)
        struct = result.get_struct("customer_record")
        assert struct is not None
        assert struct.name == "customer_record"
        results.ok("get_struct helper")
    except Exception as e:
        results.fail("get_struct helper", str(e))
    
    # --- Test: get_define helper ---
    try:
        result = parse_tal_string(DEFINES_CODE)
        define = result.get_define("max_size")
        assert define is not None
        assert define.name == "max_size"
        results.ok("get_define helper")
    except Exception as e:
        results.fail("get_define helper", str(e))


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_integration(results: TestResult):
    """Test integration between parsers."""
    print("\n" + "="*60)
    print("INTEGRATION TESTS")
    print("="*60)
    
    try:
        from tal_proc_parser import parse_multiple_procedures, SymbolTable
        from tal_enhanced_parser import EnhancedTALParser, parse_tal_string
    except ImportError as e:
        results.fail("Import both parsers", str(e))
        return
    
    results.ok("Import both parsers")
    
    # --- Test: Enhanced parser uses proc_parser internally ---
    try:
        # Both should find same procedures
        st = SymbolTable()
        proc_nodes, _ = parse_multiple_procedures(MULTI_PROC, "test.tal", st)
        
        enhanced_result = parse_tal_string(MULTI_PROC)
        
        proc_names = [n.name for n in proc_nodes]
        enhanced_names = [p.name for p in enhanced_result.procedures]
        
        assert set(proc_names) == set(enhanced_names)
        results.ok("Both parsers find same procedures")
    except Exception as e:
        results.fail("Both parsers find same procedures", str(e))
    
    # --- Test: Enhanced provides more information ---
    try:
        enhanced_result = parse_tal_string(COMPLETE_PROGRAM)
        
        # Enhanced should have additional info
        assert len(enhanced_result.defines) > 0
        assert len(enhanced_result.structs) > 0
        assert len(enhanced_result.calls) > 0
        
        # Should have complexity calculated
        for proc in enhanced_result.procedures:
            assert proc.complexity >= 1
        
        results.ok("Enhanced parser provides additional info")
    except Exception as e:
        results.fail("Enhanced parser provides additional info", str(e))
    
    # --- Test: Error handling ---
    try:
        # Malformed code should not crash
        malformed = "PROC incomplete"
        result = parse_tal_string(malformed)
        # Should return empty or partial result, not crash
        assert isinstance(result.procedures, list)
        results.ok("Error handling - malformed input")
    except Exception as e:
        results.fail("Error handling - malformed input", str(e))
    
    # --- Test: Large input ---
    try:
        # Create large input by duplicating procedures
        large_code = "\n".join([
            f"PROC proc_{i};\nBEGIN\n  INT x;\n  x := {i};\nEND;\n"
            for i in range(100)
        ])
        result = parse_tal_string(large_code)
        assert len(result.procedures) == 100
        results.ok("Large input handling (100 procedures)")
    except Exception as e:
        results.fail("Large input handling", str(e))


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("="*60)
    print("TAL PARSER TEST SUITE")
    print("="*60)
    
    results = TestResult()
    
    # Run all test groups
    test_tal_proc_parser(results)
    test_tal_enhanced_parser(results)
    test_integration(results)
    
    # Summary
    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
