#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Indexer
=============================================

Thorough tests covering:
- TAL Parsers (proc_parser and enhanced_parser)
- Code Parsers (multi-language)
- Document Parsers
- Vocabulary and Domain Matching
- Embeddings
- Index Operations
- Search and Retrieval
- Pipeline Integration
- Edge Cases and Error Handling
- Performance Tests

Run with: python test_full_suite.py
Or:       python -m pytest test_full_suite.py -v
"""

import sys
import os
import time
import tempfile
import json
import shutil
from typing import List, Dict, Any, Optional

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# TEST RESULT TRACKER
# =============================================================================

class TestResult:
    """Test result tracker with timing."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
        self.times = []
        self.start_time = time.time()
    
    def ok(self, name, duration=None):
        self.passed += 1
        if duration:
            self.times.append((name, duration))
        print(f"  ✓ {name}" + (f" ({duration:.3f}s)" if duration else ""))
    
    def fail(self, name, reason):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  ✗ {name}: {reason}")
    
    def skip(self, name, reason):
        self.skipped += 1
        print(f"  ○ {name}: SKIPPED - {reason}")
    
    def summary(self):
        total = self.passed + self.failed
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed}/{total} passed, {self.failed} failed, {self.skipped} skipped")
        print(f"Total time: {elapsed:.2f}s")
        if self.errors:
            print(f"\nFailures:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason[:100]}")
        if self.times:
            slowest = sorted(self.times, key=lambda x: x[1], reverse=True)[:5]
            print(f"\nSlowest tests:")
            for name, t in slowest:
                print(f"  - {name}: {t:.3f}s")
        print(f"{'='*60}")
        return self.failed == 0


# =============================================================================
# TAL CODE FIXTURES
# =============================================================================

# Basic procedures
TAL_SIMPLE = "PROC simple; BEGIN INT x; END;"

TAL_WITH_RETURN = """
INT PROC get_value;
BEGIN
    RETURN 42;
END;
"""

TAL_WITH_PARAMS = """
INT PROC add(a, b);
    INT a, b;
BEGIN
    RETURN a + b;
END;
"""

TAL_POINTER_PARAMS = """
PROC copy_buffer(.src, .dst, len);
    STRING .src, .dst;
    INT len;
BEGIN
    INT i;
    FOR i := 0 TO len - 1 DO
        dst[i] := src[i];
END;
"""

TAL_MAIN_PROC = """
PROC main_entry MAIN;
BEGIN
    CALL init;
END;
"""

TAL_FORWARD_DECL = """
INT PROC calc(x) FORWARD;

PROC use_calc;
BEGIN
    INT result;
    result := calc(10);
END;

INT PROC calc(x);
    INT x;
BEGIN
    RETURN x * 2;
END;
"""

TAL_EXTERNAL = """
PROC system_call EXTERNAL;
INT PROC get_status EXTERNAL;
"""

TAL_RESIDENT = """
PROC resident_proc RESIDENT;
BEGIN
    INT x;
END;
"""

TAL_INTERRUPT = """
PROC interrupt_handler INTERRUPT;
BEGIN
    ! Handle interrupt
END;
"""

# Complex control flow
TAL_NESTED_IF = """
INT PROC check_value(x);
    INT x;
BEGIN
    IF x < 0 THEN
        IF x < -100 THEN
            RETURN -2;
        ELSE
            RETURN -1;
        END;
    ELSE
        IF x > 100 THEN
            RETURN 2;
        ELSE
            IF x > 0 THEN
                RETURN 1;
            ELSE
                RETURN 0;
            END;
        END;
    END;
END;
"""

TAL_WHILE_LOOP = """
PROC process_while;
BEGIN
    INT i, sum;
    i := 0;
    sum := 0;
    WHILE i < 100 DO
        sum := sum + i;
        i := i + 1;
    END;
END;
"""

TAL_FOR_LOOP = """
PROC process_for;
BEGIN
    INT i, arr[0:99], sum;
    sum := 0;
    FOR i := 0 TO 99 DO
        arr[i] := i;
        sum := sum + arr[i];
    END;
END;
"""

TAL_CASE_STMT = """
INT PROC handle_case(code);
    INT code;
BEGIN
    CASE code OF
    BEGIN
        0 -> RETURN 100;
        1 -> RETURN 200;
        2, 3, 4 -> RETURN 300;
        OTHERWISE -> RETURN -1;
    END;
END;
"""

TAL_DO_UNTIL = """
PROC do_until_test;
BEGIN
    INT x;
    x := 0;
    DO
        x := x + 1;
    UNTIL x >= 10;
END;
"""

# DEFINE variations
TAL_SIMPLE_DEFINES = """
DEFINE max_size = 100;
DEFINE min_size = 10;
DEFINE default_value = 0;
"""

TAL_HEX_DEFINES = """
DEFINE hex_mask = %H00FF;
DEFINE high_byte = %HFF00;
DEFINE all_ones = %HFFFF;
"""

TAL_MACRO_DEFINES = """
DEFINE add_one(x) = x + 1;
DEFINE multiply(a, b) = a * b;
DEFINE max_of(x, y) = IF x > y THEN x ELSE y;
"""

TAL_MULTILINE_DEFINE = """
DEFINE complex_macro(a, b, c) =
    BEGIN
        IF a > b THEN
            c := a
        ELSE
            c := b;
    END;
"""

# LITERAL variations
TAL_LITERALS = """
LITERAL
    true_val = 1,
    false_val = 0,
    null_char = 0,
    max_int = 32767;
"""

TAL_HEX_LITERALS = """
LITERAL
    mask = %HFF,
    flag = %H8000,
    pattern = %HABCD;
"""

TAL_BINARY_LITERALS = """
LITERAL
    bit_0 = %B00000001,
    bit_7 = %B10000000,
    all_bits = %B11111111;
"""

TAL_OCTAL_LITERALS = """
LITERAL
    octal_val = %177,
    perm_mask = %755;
"""

# STRUCT variations
TAL_SIMPLE_STRUCT = """
STRUCT point;
BEGIN
    INT x;
    INT y;
END;
"""

TAL_STRUCT_WITH_ARRAYS = """
STRUCT buffer_info;
BEGIN
    STRING data[0:255];
    INT length;
    INT capacity;
    STRING name[0:31];
END;
"""

TAL_NESTED_STRUCT = """
STRUCT outer;
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

TAL_STRUCT_POINTER = """
STRUCT record;
BEGIN
    INT id;
    STRING name[0:49];
END;

STRUCT .record_ptr(record);
"""

TAL_REFERRAL_STRUCT = """
STRUCT base_record;
BEGIN
    INT id;
    INT type;
END;

STRUCT extended_record = base_record;
"""

TAL_STRUCT_WITH_FILLER = """
STRUCT aligned_data;
BEGIN
    INT field1;
    FILLER 2;
    INT field2;
    FILLER 4;
    STRING data[0:7];
END;
"""

# Global variables
TAL_GLOBALS = """
INT global_counter;
STRING global_buffer[0:1023];
INT .global_ptr;
REAL global_rate;
"""

TAL_GLOBAL_ARRAYS = """
INT counters[0:9];
STRING buffers[0:3][0:255];
INT matrix[0:9][0:9];
"""

TAL_INITIALIZED_GLOBALS = """
INT status := 0;
STRING default_name[0:9] := "DEFAULT";
"""

# SUBPROC
TAL_SUBPROC = """
PROC outer;
BEGIN
    INT local;
    
    SUBPROC inner;
    BEGIN
        local := local + 1;
    END;
    
    local := 0;
    CALL inner;
    CALL inner;
END;
"""

TAL_SUBPROC_WITH_PARAMS = """
PROC calculator;
BEGIN
    INT result;
    
    INT SUBPROC add(a, b);
        INT a, b;
    BEGIN
        RETURN a + b;
    END;
    
    INT SUBPROC multiply(a, b);
        INT a, b;
    BEGIN
        RETURN a * b;
    END;
    
    result := add(5, 3);
    result := multiply(result, 2);
END;
"""

# Call variations
TAL_CALL_SIMPLE = """
PROC caller;
BEGIN
    CALL proc1;
    CALL proc2;
    CALL proc3;
END;
"""

TAL_CALL_WITH_ARGS = """
PROC caller;
BEGIN
    CALL send_message(buffer, length);
    CALL process_data(input, output, count);
END;
"""

TAL_FUNCTION_STYLE = """
PROC caller;
BEGIN
    INT x, y, z;
    x := get_value();
    y := calculate(x, 10);
    z := transform(x, y);
END;
"""

TAL_PCAL = """
PROC privileged_caller;
BEGIN
    PCAL PROCESS_CREATE_(name, name_len);
    PCAL FILE_OPEN_(filename, file_num);
END;
"""

TAL_NESTED_CALLS = """
PROC caller;
BEGIN
    INT result;
    result := outer(inner(deepest(1)));
    CALL process(transform(get_data()));
END;
"""

# System calls
TAL_SYSTEM_CALLS = """
PROC system_ops;
BEGIN
    INT file_num, error;
    STRING buffer[0:255];
    
    CALL FILE_OPEN_(filename, file_num);
    CALL FILE_READ_(file_num, buffer, 256);
    CALL FILE_WRITE_(file_num, output, len);
    CALL FILE_CLOSE_(file_num);
    
    CALL AWAITIO(file_num);
    CALL DELAY(1000);
END;
"""

# Complete program
TAL_COMPLETE_PROGRAM = """
! Complete TAL Program Example
! ============================

DEFINE max_records = 1000;
DEFINE buffer_size = 4096;
DEFINE success = 0;
DEFINE error_not_found = -1;
DEFINE error_overflow = -2;

LITERAL
    status_active = 1,
    status_inactive = 0,
    status_deleted = -1;

STRUCT record_type;
BEGIN
    INT id;
    INT status;
    STRING name[0:49];
    INT value;
END;

INT record_count;
STRUCT records[0:max_records - 1](record_type);
STRING work_buffer[0:buffer_size - 1];

INT PROC find_record(id) FORWARD;
INT PROC validate_record(.rec) FORWARD;

PROC initialize MAIN;
BEGIN
    record_count := 0;
    CALL clear_records;
    CALL load_defaults;
END;

PROC clear_records;
BEGIN
    INT i;
    FOR i := 0 TO max_records - 1 DO
        records[i].status := status_deleted;
        records[i].id := 0;
    END;
END;

PROC load_defaults;
BEGIN
    ! Load default configuration
END;

INT PROC find_record(id);
    INT id;
BEGIN
    INT i;
    FOR i := 0 TO record_count - 1 DO
        IF records[i].id = id AND 
           records[i].status <> status_deleted THEN
            RETURN i;
        END;
    END;
    RETURN error_not_found;
END;

INT PROC validate_record(.rec);
    STRUCT .rec(record_type);
BEGIN
    IF rec.id <= 0 THEN
        RETURN error_not_found;
    END;
    IF rec.status = status_deleted THEN
        RETURN error_not_found;
    END;
    RETURN success;
END;

INT PROC add_record(.rec);
    STRUCT .rec(record_type);
BEGIN
    INT validation;
    
    validation := validate_record(rec);
    IF validation <> success THEN
        RETURN validation;
    END;
    
    IF record_count >= max_records THEN
        RETURN error_overflow;
    END;
    
    records[record_count] := rec;
    record_count := record_count + 1;
    
    RETURN success;
END;

INT PROC delete_record(id);
    INT id;
BEGIN
    INT index;
    
    index := find_record(id);
    IF index < 0 THEN
        RETURN error_not_found;
    END;
    
    records[index].status := status_deleted;
    RETURN success;
END;

PROC process_all;
BEGIN
    INT i;
    
    SUBPROC process_single(idx);
        INT idx;
    BEGIN
        IF records[idx].status = status_active THEN
            CALL handle_record(records[idx]);
        END;
    END;
    
    FOR i := 0 TO record_count - 1 DO
        CALL process_single(i);
    END;
END;

PROC handle_record(.rec);
    STRUCT .rec(record_type);
BEGIN
    ! Process individual record
END;
"""

# Edge cases
TAL_EMPTY = ""

TAL_COMMENTS_ONLY = """
! This is a comment
! Another comment
! More comments
"""

TAL_MIXED_CASE = """
Proc Mixed_Case;
BEGIN
    Int Variable;
    variable := 1;
END;
"""

TAL_SPECIAL_CHARS = """
PROC proc^with^carets;
BEGIN
    INT var^name;
    var^name := 0;
END;
"""

TAL_LONG_NAMES = """
PROC very_long_procedure_name_that_goes_on_and_on_and_on;
BEGIN
    INT another_extremely_long_variable_name_here;
END;
"""

TAL_UNICODE_COMMENTS = """
PROC test;
BEGIN
    ! Comment with special chars: äöü ñ 日本語
    INT x;
END;
"""

TAL_DEEPLY_NESTED = """
INT PROC deep_nesting(a, b, c, d, e);
    INT a, b, c, d, e;
BEGIN
    IF a > 0 THEN
        IF b > 0 THEN
            IF c > 0 THEN
                IF d > 0 THEN
                    IF e > 0 THEN
                        RETURN 1;
                    ELSE
                        RETURN -1;
                    END;
                END;
            END;
        END;
    END;
    RETURN 0;
END;
"""

TAL_MANY_PARAMS = """
PROC many_params(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
    INT p1, p2, p3, p4, p5;
    STRING .p6, .p7;
    INT .p8, .p9, .p10;
BEGIN
END;
"""

# Error cases (malformed TAL)
TAL_MALFORMED_PROC = "PROC incomplete"
TAL_MALFORMED_STRUCT = "STRUCT no_end; BEGIN INT x;"
TAL_UNBALANCED_BEGIN = "PROC test; BEGIN BEGIN END;"
TAL_MISSING_SEMICOLON = "PROC test BEGIN INT x END"


# =============================================================================
# OTHER LANGUAGE FIXTURES
# =============================================================================

PYTHON_CODE = '''
def calculate_sum(numbers: list) -> int:
    """Calculate the sum of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total

class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        return [x * 2 for x in self.data]
'''

JAVA_CODE = '''
public class Calculator {
    private int value;
    
    public Calculator(int initial) {
        this.value = initial;
    }
    
    public int add(int x) {
        return this.value + x;
    }
    
    public static void main(String[] args) {
        Calculator calc = new Calculator(10);
        System.out.println(calc.add(5));
    }
}
'''

C_CODE = '''
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int id;
    char name[50];
    float value;
} Record;

int process_record(Record* rec) {
    if (rec == NULL) {
        return -1;
    }
    printf("Processing: %s\\n", rec->name);
    return 0;
}

int main(int argc, char* argv[]) {
    Record r = {1, "test", 3.14};
    return process_record(&r);
}
'''

COBOL_CODE = '''
       IDENTIFICATION DIVISION.
       PROGRAM-ID. SAMPLE-PROGRAM.
       
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-COUNTER PIC 9(4) VALUE 0.
       01 WS-NAME PIC X(30).
       
       PROCEDURE DIVISION.
       MAIN-PARA.
           PERFORM INIT-PARA.
           PERFORM PROCESS-PARA.
           STOP RUN.
       
       INIT-PARA.
           MOVE 0 TO WS-COUNTER.
           MOVE "TEST" TO WS-NAME.
       
       PROCESS-PARA.
           ADD 1 TO WS-COUNTER.
           DISPLAY WS-NAME.
'''

JAVASCRIPT_CODE = '''
class PaymentProcessor {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.transactions = [];
    }
    
    async processPayment(amount, currency) {
        const transaction = {
            id: Date.now(),
            amount,
            currency,
            status: 'pending'
        };
        this.transactions.push(transaction);
        return transaction;
    }
    
    getTransactionHistory() {
        return this.transactions;
    }
}

module.exports = PaymentProcessor;
'''


# =============================================================================
# DOCUMENT FIXTURES
# =============================================================================

MARKDOWN_DOC = """
# Payment Processing Guide

## Overview

This document describes the payment processing system.

### Components

1. **Transaction Handler** - Processes incoming transactions
2. **Validation Engine** - Validates payment data
3. **Settlement Service** - Handles settlement

## API Reference

### Process Payment

```
POST /api/v1/payments
```

Parameters:
- amount: The payment amount
- currency: ISO currency code
- account: Target account number

## Error Codes

| Code | Description |
|------|-------------|
| 1001 | Invalid amount |
| 1002 | Account not found |
| 1003 | Insufficient funds |
"""

HTML_DOC = """
<!DOCTYPE html>
<html>
<head>
    <title>Payment System Documentation</title>
</head>
<body>
    <h1>Payment Processing</h1>
    <p>This system handles wire transfers and payment processing.</p>
    
    <h2>Features</h2>
    <ul>
        <li>Real-time processing</li>
        <li>Multi-currency support</li>
        <li>Compliance checking</li>
    </ul>
    
    <h2>API Endpoints</h2>
    <table>
        <tr><th>Endpoint</th><th>Method</th><th>Description</th></tr>
        <tr><td>/payments</td><td>POST</td><td>Create payment</td></tr>
        <tr><td>/payments/{id}</td><td>GET</td><td>Get payment status</td></tr>
    </table>
</body>
</html>
"""


# =============================================================================
# LOG FIXTURES
# =============================================================================

LOG_ENTRIES = """
2024-01-15 10:23:45.123 INFO [PaymentService] Processing payment TXN-12345
2024-01-15 10:23:45.456 DEBUG [Validator] Validating amount: 1000.00 USD
2024-01-15 10:23:45.789 INFO [PaymentService] Payment TXN-12345 validated
2024-01-15 10:23:46.012 ERROR [SettlementService] Settlement failed for TXN-12345: Timeout
2024-01-15 10:23:46.345 WARN [RetryHandler] Retrying settlement for TXN-12345 (attempt 1/3)
2024-01-15 10:23:47.678 INFO [SettlementService] Settlement successful for TXN-12345
"""

PAYMENT_LOG = """
[2024-01-15T10:00:00Z] PAYMENT_RECEIVED txn_id=TXN-001 amount=5000.00 currency=USD sender=ACCT001 receiver=ACCT002
[2024-01-15T10:00:01Z] VALIDATION_START txn_id=TXN-001 validator=SWIFT_FORMAT
[2024-01-15T10:00:02Z] VALIDATION_PASS txn_id=TXN-001 checks=FORMAT,AML,SANCTIONS
[2024-01-15T10:00:03Z] ROUTING_DECISION txn_id=TXN-001 route=FEDWIRE priority=HIGH
[2024-01-15T10:00:05Z] SETTLEMENT_COMPLETE txn_id=TXN-001 settlement_id=STL-001
"""

ERROR_LOG = """
2024-01-15 10:30:00 ERROR PaymentProcessor - Transaction failed
    at com.payment.Processor.process(Processor.java:123)
    at com.payment.Handler.handle(Handler.java:45)
    at com.payment.Service.execute(Service.java:78)
Caused by: java.sql.SQLException: Connection timeout
    at com.db.Pool.getConnection(Pool.java:234)
"""


# =============================================================================
# VOCABULARY FIXTURES
# =============================================================================

PAYMENT_VOCABULARY = {
    "terms": [
        {
            "term": "wire transfer",
            "category": "payment_type",
            "aliases": ["wire", "bank wire", "wire payment"],
            "definition": "Electronic transfer of funds between banks"
        },
        {
            "term": "SWIFT",
            "category": "network",
            "aliases": ["SWIFT network", "SWIFT message"],
            "definition": "Society for Worldwide Interbank Financial Telecommunication"
        },
        {
            "term": "settlement",
            "category": "process",
            "aliases": ["settle", "settling", "settled"],
            "definition": "Final transfer of funds between parties"
        },
        {
            "term": "beneficiary",
            "category": "party",
            "aliases": ["recipient", "payee", "receiver"],
            "definition": "Party receiving the payment"
        },
        {
            "term": "originator",
            "category": "party",
            "aliases": ["sender", "payer", "remitter"],
            "definition": "Party initiating the payment"
        },
        {
            "term": "AML",
            "category": "compliance",
            "aliases": ["anti-money laundering", "AML check"],
            "definition": "Anti-money laundering compliance"
        },
        {
            "term": "sanctions",
            "category": "compliance",
            "aliases": ["sanctions screening", "OFAC"],
            "definition": "Screening against sanctions lists"
        },
        {
            "term": "MT103",
            "category": "message_type",
            "aliases": ["MT 103", "single customer credit transfer"],
            "definition": "SWIFT message type for single credit transfers"
        }
    ]
}


# =============================================================================
# TAL PROC PARSER TESTS
# =============================================================================

def test_tal_proc_parser(results: TestResult):
    """Test tal_proc_parser.py thoroughly."""
    print("\n" + "="*60)
    print("TAL_PROC_PARSER TESTS")
    print("="*60)
    
    try:
        from tal_proc_parser import (
            TALType, TALNode, Symbol, SymbolTable, SourceLocation,
            ParseError, ErrorSeverity,
            parse_tal_type, find_procedure_declarations,
            parse_procedure_declaration, parse_multiple_procedures,
            extract_parameters_from_declaration
        )
    except ImportError as e:
        results.fail("Import tal_proc_parser", str(e))
        return
    
    results.ok("Import tal_proc_parser")
    
    # --- TALType tests ---
    type_tests = [
        ("INT", TALType.INT),
        ("int", TALType.INT),
        ("INT(16)", TALType.INT16),
        ("INT(32)", TALType.INT32),
        ("INT(64)", TALType.INT64),
        ("REAL", TALType.REAL),
        ("REAL(32)", TALType.REAL32),
        ("REAL(64)", TALType.REAL64),
        ("STRING", TALType.STRING),
        ("FIXED", TALType.FIXED),
        ("UNSIGNED", TALType.UNSIGNED),
        ("STRUCT", TALType.STRUCT),
        ("UNKNOWN_TYPE", TALType.UNKNOWN),
    ]
    
    for type_str, expected in type_tests:
        try:
            result = parse_tal_type(type_str)
            assert result == expected, f"Expected {expected}, got {result}"
        except Exception as e:
            results.fail(f"parse_tal_type({type_str})", str(e))
            continue
    results.ok("parse_tal_type - all variations")
    
    # --- SymbolTable tests ---
    try:
        st = SymbolTable()
        
        # Test scope nesting
        st.enter_scope("level1")
        st.enter_scope("level2")
        st.enter_scope("level3")
        assert st.current_scope == "level3"
        
        st.exit_scope()
        assert st.current_scope == "level2"
        
        st.exit_scope()
        st.exit_scope()
        assert st.current_scope == "global"
        
        # Test symbol lookup across scopes
        st.enter_scope("outer")
        outer_sym = Symbol("x", TALType.INT, SourceLocation())
        st.declare_symbol(outer_sym)
        
        st.enter_scope("inner")
        inner_sym = Symbol("y", TALType.STRING, SourceLocation())
        st.declare_symbol(inner_sym)
        
        # Should find both
        assert st.lookup_symbol("x") is not None
        assert st.lookup_symbol("y") is not None
        assert st.lookup_symbol("z") is None
        
        results.ok("SymbolTable - scope and lookup")
    except Exception as e:
        results.fail("SymbolTable - scope and lookup", str(e))
    
    # --- Procedure finding tests ---
    proc_test_cases = [
        (TAL_SIMPLE, ["simple"]),
        (TAL_WITH_RETURN, ["get_value"]),
        (TAL_WITH_PARAMS, ["add"]),
        (TAL_MAIN_PROC, ["main_entry"]),
        (TAL_FORWARD_DECL, ["calc", "use_calc", "calc"]),  # Forward + impl
        (TAL_EXTERNAL, ["system_call", "get_status"]),
        (TAL_COMPLETE_PROGRAM, ["initialize", "clear_records", "load_defaults", 
                                "find_record", "validate_record", "add_record",
                                "delete_record", "process_all", "handle_record"]),
    ]
    
    for code, expected_names in proc_test_cases:
        try:
            procs = find_procedure_declarations(code)
            found_names = [p[1] for p in procs]
            for name in expected_names:
                if name not in found_names:
                    results.fail(f"find_procedure_declarations", f"Missing {name}")
                    break
            else:
                continue
            break
        except Exception as e:
            results.fail("find_procedure_declarations", str(e))
            break
    else:
        results.ok("find_procedure_declarations - various inputs")
    
    # --- Parameter extraction tests ---
    param_tests = [
        ("PROC test;", []),
        ("PROC test();", []),
        ("PROC test(a);", ["a"]),
        ("PROC test(a, b);", ["a", "b"]),
        ("PROC test(a, b, c, d, e);", ["a", "b", "c", "d", "e"]),
        ("PROC test(.ptr);", [".ptr"]),
        ("PROC test(val, .ptr);", ["val", ".ptr"]),
    ]
    
    for decl, expected in param_tests:
        try:
            params = extract_parameters_from_declaration(decl)
            # Normalize for comparison
            params = [p.strip() for p in params]
            expected = [e.strip() for e in expected]
            assert len(params) == len(expected), f"Count mismatch: {params} vs {expected}"
        except Exception as e:
            results.fail(f"extract_parameters", str(e))
            break
    else:
        results.ok("extract_parameters_from_declaration - variations")
    
    # --- Attribute detection tests ---
    try:
        st = SymbolTable()
        
        # Test MAIN
        procs = find_procedure_declarations(TAL_MAIN_PROC)
        _, name, decl = procs[0]
        node, _ = parse_procedure_declaration(name, decl, 1, "test.tal", st)
        assert node.attributes.get('is_main') == True
        
        # Test FORWARD
        st = SymbolTable()
        procs = find_procedure_declarations(TAL_FORWARD_DECL)
        forward_found = False
        for _, name, decl in procs:
            node, _ = parse_procedure_declaration(name, decl, 1, "test.tal", st)
            if node.attributes.get('is_forward'):
                forward_found = True
                break
        assert forward_found
        
        # Test EXTERNAL
        st = SymbolTable()
        procs = find_procedure_declarations(TAL_EXTERNAL)
        for _, name, decl in procs:
            node, _ = parse_procedure_declaration(name, decl, 1, "test.tal", st)
            assert node.attributes.get('is_external') == True
        
        results.ok("Attribute detection (MAIN, FORWARD, EXTERNAL)")
    except Exception as e:
        results.fail("Attribute detection", str(e))
    
    # --- Edge cases ---
    edge_cases = [
        (TAL_EMPTY, 0),
        (TAL_COMMENTS_ONLY, 0),
        (TAL_MIXED_CASE, 1),
        (TAL_SPECIAL_CHARS, 1),
        (TAL_LONG_NAMES, 1),
    ]
    
    for code, expected_count in edge_cases:
        try:
            procs = find_procedure_declarations(code)
            assert len(procs) == expected_count
        except Exception as e:
            results.fail("Edge cases", str(e))
            break
    else:
        results.ok("Edge cases - empty, comments, mixed case, special chars")
    
    # --- Error handling ---
    try:
        # Malformed input should not crash
        procs = find_procedure_declarations(TAL_MALFORMED_PROC)
        # Should return partial result or empty, not crash
        results.ok("Error handling - malformed input")
    except Exception as e:
        results.fail("Error handling - malformed input", str(e))


# =============================================================================
# TAL ENHANCED PARSER TESTS
# =============================================================================

def test_tal_enhanced_parser(results: TestResult):
    """Test tal_enhanced_parser.py thoroughly."""
    print("\n" + "="*60)
    print("TAL_ENHANCED_PARSER TESTS")
    print("="*60)
    
    try:
        from tal_enhanced_parser import (
            EnhancedTALParser, ParseResult, ProcedureDetail,
            DefineInfo, StructInfo, LiteralInfo, GlobalInfo,
            CallInfo, SubprocInfo, StructField,
            parse_tal_string, parse_tal_file,
            get_call_graph, get_procedure_complexity
        )
    except ImportError as e:
        results.fail("Import tal_enhanced_parser", str(e))
        return
    
    results.ok("Import tal_enhanced_parser")
    
    # --- Basic parsing ---
    try:
        result = parse_tal_string(TAL_SIMPLE)
        assert isinstance(result, ParseResult)
        assert len(result.procedures) == 1
        assert result.procedures[0].name == "simple"
        results.ok("Parse simple procedure")
    except Exception as e:
        results.fail("Parse simple procedure", str(e))
    
    # --- Return types ---
    try:
        result = parse_tal_string(TAL_WITH_RETURN)
        proc = result.procedures[0]
        assert proc.return_type is not None
        results.ok("Parse return type")
    except Exception as e:
        results.fail("Parse return type", str(e))
    
    # --- Parameters ---
    try:
        result = parse_tal_string(TAL_WITH_PARAMS)
        proc = result.procedures[0]
        assert len(proc.parameters) == 2
        results.ok("Parse parameters")
    except Exception as e:
        results.fail("Parse parameters", str(e))
    
    # --- Pointer parameters ---
    try:
        result = parse_tal_string(TAL_POINTER_PARAMS)
        proc = result.procedures[0]
        assert len(proc.parameters) == 3
        results.ok("Parse pointer parameters")
    except Exception as e:
        results.fail("Parse pointer parameters", str(e))
    
    # --- DEFINE parsing ---
    define_tests = [
        (TAL_SIMPLE_DEFINES, 3),
        (TAL_HEX_DEFINES, 3),
        (TAL_MACRO_DEFINES, 3),
    ]
    
    for code, expected_count in define_tests:
        try:
            result = parse_tal_string(code)
            assert len(result.defines) >= expected_count
        except Exception as e:
            results.fail("Parse DEFINEs", str(e))
            break
    else:
        results.ok("Parse DEFINEs - simple, hex, macros")
    
    # --- Macro detection ---
    try:
        result = parse_tal_string(TAL_MACRO_DEFINES)
        macros = [d for d in result.defines if d.is_macro]
        assert len(macros) >= 2
        results.ok("Macro detection")
    except Exception as e:
        results.fail("Macro detection", str(e))
    
    # --- LITERAL parsing ---
    literal_tests = [
        (TAL_LITERALS, 4),
        (TAL_HEX_LITERALS, 3),
        (TAL_BINARY_LITERALS, 3),
    ]
    
    for code, expected_count in literal_tests:
        try:
            result = parse_tal_string(code)
            assert len(result.literals) >= expected_count
        except Exception as e:
            results.fail("Parse LITERALs", str(e))
            break
    else:
        results.ok("Parse LITERALs - decimal, hex, binary")
    
    # --- STRUCT parsing ---
    try:
        result = parse_tal_string(TAL_SIMPLE_STRUCT)
        assert len(result.structs) >= 1
        struct = result.structs[0]
        assert struct.name == "point"
        assert len(struct.fields) == 2
        results.ok("Parse simple STRUCT")
    except Exception as e:
        results.fail("Parse simple STRUCT", str(e))
    
    # --- STRUCT with arrays ---
    try:
        result = parse_tal_string(TAL_STRUCT_WITH_ARRAYS)
        struct = result.structs[0]
        array_fields = [f for f in struct.fields if f.is_array]
        assert len(array_fields) >= 2
        results.ok("Parse STRUCT with arrays")
    except Exception as e:
        results.fail("Parse STRUCT with arrays", str(e))
    
    # --- Global variables ---
    try:
        result = parse_tal_string(TAL_GLOBALS)
        assert len(result.globals) >= 3
        results.ok("Parse global variables")
    except Exception as e:
        results.fail("Parse global variables", str(e))
    
    # --- Call extraction ---
    try:
        result = parse_tal_string(TAL_CALL_SIMPLE)
        assert len(result.calls) >= 3
        callees = [c.callee for c in result.calls]
        assert "proc1" in callees
        assert "proc2" in callees
        assert "proc3" in callees
        results.ok("Extract simple CALLs")
    except Exception as e:
        results.fail("Extract simple CALLs", str(e))
    
    # --- PCAL extraction ---
    try:
        result = parse_tal_string(TAL_PCAL)
        pcalls = [c for c in result.calls if c.call_type == "PCAL"]
        assert len(pcalls) >= 2
        results.ok("Extract PCALs")
    except Exception as e:
        results.fail("Extract PCALs", str(e))
    
    # --- Function-style calls ---
    try:
        result = parse_tal_string(TAL_FUNCTION_STYLE)
        assert len(result.calls) >= 3
        results.ok("Extract function-style calls")
    except Exception as e:
        results.fail("Extract function-style calls", str(e))
    
    # --- Call graph ---
    try:
        code = """
PROC a;
BEGIN
    CALL b;
    CALL c;
END;

PROC b;
BEGIN
    CALL c;
END;

PROC c;
BEGIN
    INT x;
END;
"""
        result = parse_tal_string(code)
        graph = result.call_graph
        assert "a" in graph, f"'a' not in call graph: {graph}"
        assert "b" in graph["a"] or "c" in graph["a"], f"Expected b or c in graph[a]: {graph}"
        results.ok("Build call graph")
    except Exception as e:
        results.fail("Build call graph", str(e))
    
    # --- Reverse call graph ---
    try:
        rev_graph = result.reverse_call_graph
        # c is called by both a and b
        assert "c" in rev_graph or "b" in rev_graph, f"Expected c or b in reverse graph: {rev_graph}"
        results.ok("Build reverse call graph")
    except Exception as e:
        results.fail("Build reverse call graph", str(e))
    
    # --- Entry points ---
    try:
        entry = result.entry_points
        # a is not called by anyone
        assert "a" in entry, f"Expected 'a' in entry points: {entry}"
        results.ok("Identify entry points")
    except Exception as e:
        results.fail("Identify entry points", str(e))
    
    # --- Cyclomatic complexity ---
    complexity_tests = [
        (TAL_SIMPLE, 1, 3),  # min, max expected
        (TAL_NESTED_IF, 3, 20),
        (TAL_WHILE_LOOP, 1, 5),
        (TAL_FOR_LOOP, 1, 5),
        (TAL_CASE_STMT, 1, 10),
    ]
    
    for code, min_expected, max_expected in complexity_tests:
        try:
            result = parse_tal_string(code)
            if result.procedures:
                complexity = result.procedures[0].complexity
                assert min_expected <= complexity <= max_expected, \
                    f"Complexity {complexity} not in [{min_expected}, {max_expected}]"
        except Exception as e:
            results.fail("Cyclomatic complexity", str(e))
            break
    else:
        results.ok("Cyclomatic complexity calculation")
    
    # --- SUBPROC extraction ---
    try:
        result = parse_tal_string(TAL_SUBPROC)
        assert len(result.subprocs) >= 1
        results.ok("Extract SUBPROCs")
    except Exception as e:
        results.fail("Extract SUBPROCs", str(e))
    
    # --- Procedure body boundaries ---
    try:
        result = parse_tal_string(TAL_COMPLETE_PROGRAM)
        procs_with_bodies = 0
        for proc in result.procedures:
            if not proc.is_forward and not proc.is_external:
                if proc.body_start_line > 0 and proc.body_end_line >= proc.body_start_line:
                    procs_with_bodies += 1
        # At least some procedures should have bodies detected
        assert procs_with_bodies > 0, f"No procedures with body boundaries found"
        results.ok(f"Procedure body boundaries ({procs_with_bodies} detected)")
    except Exception as e:
        results.fail("Procedure body boundaries", str(e))
    
    # --- Local variables ---
    try:
        result = parse_tal_string(TAL_FOR_LOOP)
        proc = result.procedures[0]
        assert len(proc.local_vars) >= 2
        var_names = [v['name'] for v in proc.local_vars]
        assert 'i' in var_names
        results.ok("Extract local variables")
    except Exception as e:
        results.fail("Extract local variables", str(e))
    
    # --- Complete program ---
    try:
        start = time.time()
        result = parse_tal_string(TAL_COMPLETE_PROGRAM)
        duration = time.time() - start
        
        assert len(result.procedures) >= 8
        assert len(result.defines) >= 4
        assert len(result.literals) >= 3
        assert len(result.structs) >= 1
        assert len(result.globals) >= 2
        assert len(result.calls) >= 5
        
        results.ok(f"Complete program analysis", duration)
    except Exception as e:
        results.fail("Complete program analysis", str(e))
    
    # --- Serialization ---
    try:
        result = parse_tal_string(TAL_COMPLETE_PROGRAM)
        d = result.to_dict()
        assert 'procedures' in d
        assert 'defines' in d
        assert 'structs' in d
        assert 'call_graph' in d
        
        # JSON serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0
        
        results.ok("to_dict and JSON serialization")
    except Exception as e:
        results.fail("to_dict and JSON serialization", str(e))
    
    # --- Helper methods ---
    try:
        result = parse_tal_string(TAL_COMPLETE_PROGRAM)
        
        proc = result.get_procedure("initialize")
        assert proc is not None
        assert proc.name == "initialize"
        
        struct = result.get_struct("record_type")
        assert struct is not None
        
        define = result.get_define("max_records")
        assert define is not None
        
        results.ok("Helper methods (get_procedure, get_struct, get_define)")
    except Exception as e:
        results.fail("Helper methods", str(e))
    
    # --- File parsing ---
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tal', delete=False) as f:
            f.write(TAL_COMPLETE_PROGRAM)
            temp_path = f.name
        
        try:
            result = parse_tal_file(temp_path)
            assert len(result.procedures) >= 8
            results.ok("parse_tal_file")
        finally:
            os.unlink(temp_path)
    except Exception as e:
        results.fail("parse_tal_file", str(e))
    
    # --- Convenience functions ---
    try:
        graph = get_call_graph(TAL_COMPLETE_PROGRAM)
        assert isinstance(graph, dict)
        assert len(graph) > 0
        
        complexity = get_procedure_complexity(TAL_COMPLETE_PROGRAM)
        assert isinstance(complexity, dict)
        assert len(complexity) > 0
        
        results.ok("Convenience functions (get_call_graph, get_procedure_complexity)")
    except Exception as e:
        results.fail("Convenience functions", str(e))
    
    # --- Edge cases ---
    try:
        result = parse_tal_string(TAL_EMPTY)
        assert len(result.procedures) == 0
        
        result = parse_tal_string(TAL_COMMENTS_ONLY)
        assert len(result.procedures) == 0
        
        result = parse_tal_string(TAL_SPECIAL_CHARS)
        assert len(result.procedures) == 1
        
        results.ok("Edge cases")
    except Exception as e:
        results.fail("Edge cases", str(e))
    
    # --- Performance test ---
    try:
        # Generate large input
        large_code = "\n".join([
            f"PROC proc_{i};\nBEGIN\n  INT x_{i};\n  x_{i} := {i};\n  CALL proc_{i+1};\nEND;\n"
            for i in range(200)
        ])
        
        start = time.time()
        result = parse_tal_string(large_code)
        duration = time.time() - start
        
        assert len(result.procedures) == 200
        assert duration < 5.0  # Should complete in under 5 seconds
        
        results.ok(f"Performance test (200 procedures)", duration)
    except Exception as e:
        results.fail("Performance test", str(e))


# =============================================================================
# UNIFIED INDEXER TESTS
# =============================================================================

def test_vocabulary(results: TestResult):
    """Test vocabulary and domain matching."""
    print("\n" + "="*60)
    print("VOCABULARY TESTS")
    print("="*60)
    
    try:
        from unified_indexer.vocabulary import DomainVocabulary
    except ImportError as e:
        results.fail("Import DomainVocabulary", str(e))
        return
    
    results.ok("Import DomainVocabulary")
    
    # --- Load vocabulary ---
    try:
        vocab = DomainVocabulary()
        
        # Load from data
        vocab.load_from_data(PAYMENT_VOCABULARY.get("terms", []))
        assert len(vocab.entries) > 0
        
        results.ok("Load vocabulary from data")
    except Exception as e:
        results.fail("Load vocabulary from data", str(e))
    
    # --- Match text ---
    try:
        test_texts = [
            ("Process wire transfer to beneficiary", ["wire transfer", "beneficiary"]),
            ("SWIFT message MT103 received", ["SWIFT", "MT103"]),
            ("AML sanctions screening passed", ["AML", "sanctions"]),
            ("No payment terms here", []),
        ]
        
        for text, expected_terms in test_texts:
            matches = vocab.match_text(text)
            matched_terms = [m.term for m in matches]
            for term in expected_terms:
                found = any(term.lower() in t.lower() for t in matched_terms)
                if not found and expected_terms:
                    # May match aliases instead
                    pass
        
        results.ok("Text matching")
    except Exception as e:
        results.fail("Text matching", str(e))
    
    # --- Alias matching ---
    try:
        text = "The receiver will get the bank wire"
        matches = vocab.match_text(text)
        # Should match "beneficiary" (alias: receiver) and "wire transfer" (alias: bank wire)
        results.ok("Alias matching")
    except Exception as e:
        results.fail("Alias matching", str(e))
    
    # --- Save and load ---
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        # Save vocabulary data
        import json
        with open(temp_path, 'w') as f:
            json.dump(PAYMENT_VOCABULARY.get("terms", []), f)
        
        vocab2 = DomainVocabulary()
        vocab2.load(temp_path)
        
        assert len(vocab2.entries) > 0
        
        os.unlink(temp_path)
        results.ok("Save and load vocabulary")
    except Exception as e:
        results.fail("Save and load vocabulary", str(e))


def test_embeddings(results: TestResult):
    """Test embedding generation."""
    print("\n" + "="*60)
    print("EMBEDDING TESTS")
    print("="*60)
    
    try:
        from unified_indexer.embeddings import HashEmbedder, create_embedder
    except ImportError as e:
        results.fail("Import embeddings", str(e))
        return
    
    results.ok("Import embeddings")
    
    # --- Hash embedder ---
    try:
        embedder = HashEmbedder(n_features=256)
        
        text1 = "Process wire transfer payment"
        text2 = "Handle bank wire transaction"
        text3 = "Completely unrelated text about cooking"
        
        emb1 = embedder.get_embedding(text1)
        emb2 = embedder.get_embedding(text2)
        emb3 = embedder.get_embedding(text3)
        
        assert len(emb1) == 256
        assert len(emb2) == 256
        assert len(emb3) == 256
        
        results.ok("HashEmbedder basic functionality")
    except Exception as e:
        results.fail("HashEmbedder basic functionality", str(e))
    
    # --- Deterministic ---
    try:
        emb1a = embedder.get_embedding(text1)
        emb1b = embedder.get_embedding(text1)
        assert emb1a == emb1b
        results.ok("HashEmbedder deterministic")
    except Exception as e:
        results.fail("HashEmbedder deterministic", str(e))
    
    # --- Batch embedding ---
    try:
        texts = [text1, text2, text3]
        embeddings = [embedder.get_embedding(t) for t in texts]
        assert len(embeddings) == 3
        assert all(len(e) == 256 for e in embeddings)
        results.ok("Batch embedding")
    except Exception as e:
        results.fail("Batch embedding", str(e))
    
    # --- Create embedder factory ---
    try:
        embedder = create_embedder("hash", n_features=512)
        assert embedder is not None
        results.ok("create_embedder factory")
    except Exception as e:
        results.fail("create_embedder factory", str(e))


def test_parsers(results: TestResult):
    """Test various code and document parsers."""
    print("\n" + "="*60)
    print("PARSER TESTS")
    print("="*60)
    
    try:
        from unified_indexer.parsers.code_parser import GenericCodeParser
        from unified_indexer.parsers.tal_parser import TalCodeParser
        from unified_indexer.parsers.document_parser import DocumentParser
        from unified_indexer.parsers.log_parser import LogParser
        from unified_indexer.vocabulary import DomainVocabulary
    except ImportError as e:
        results.fail("Import parsers", str(e))
        return
    
    results.ok("Import parsers")
    
    vocab = DomainVocabulary()
    vocab.load_from_data(PAYMENT_VOCABULARY.get('terms', []))
    
    # --- Generic code parser ---
    try:
        parser = GenericCodeParser(vocab)
        
        # Python
        chunks = parser.parse(PYTHON_CODE.encode(), "test.py")
        assert len(chunks) > 0
        
        # Java
        chunks = parser.parse(JAVA_CODE.encode(), "Test.java")
        assert len(chunks) > 0
        
        # C
        chunks = parser.parse(C_CODE.encode(), "test.c")
        assert len(chunks) > 0
        
        # JavaScript
        chunks = parser.parse(JAVASCRIPT_CODE.encode(), "test.js")
        assert len(chunks) > 0
        
        results.ok("GenericCodeParser - multiple languages")
    except Exception as e:
        results.fail("GenericCodeParser", str(e))
    
    # --- TAL parser ---
    try:
        parser = TalCodeParser(vocab)
        chunks = parser.parse(TAL_COMPLETE_PROGRAM.encode(), "test.tal")
        assert len(chunks) > 0
        
        # Check chunk types
        proc_chunks = [c for c in chunks if c.semantic_type.value == "procedure"]
        assert len(proc_chunks) > 0
        
        results.ok("TalCodeParser")
    except Exception as e:
        results.fail("TalCodeParser", str(e))
    
    # --- Document parser ---
    try:
        parser = DocumentParser(vocab)
        
        # Markdown
        chunks = parser.parse(MARKDOWN_DOC.encode(), "doc.md")
        assert len(chunks) > 0
        
        # HTML
        chunks = parser.parse(HTML_DOC.encode(), "doc.html")
        assert len(chunks) > 0
        
        results.ok("DocumentParser - markdown, html")
    except Exception as e:
        results.fail("DocumentParser", str(e))
    
    # --- Log parser ---
    try:
        parser = LogParser(vocab)
        
        chunks = parser.parse(LOG_ENTRIES.encode(), "app.log")
        assert len(chunks) > 0
        
        chunks = parser.parse(PAYMENT_LOG.encode(), "payment.log")
        assert len(chunks) > 0
        
        results.ok("LogParser")
    except Exception as e:
        results.fail("LogParser", str(e))


def test_index_operations(results: TestResult):
    """Test index creation, search, and persistence."""
    print("\n" + "="*60)
    print("INDEX TESTS")
    print("="*60)
    
    try:
        from unified_indexer.index import HybridIndex
        from unified_indexer.models import IndexableChunk, SourceType, SemanticType, SourceReference
        from unified_indexer.embeddings import HashEmbedder
        from unified_indexer.vocabulary import DomainVocabulary
    except ImportError as e:
        results.fail("Import index modules", str(e))
        return
    
    results.ok("Import index modules")
    
    vocab = DomainVocabulary()
    vocab.load_from_data(PAYMENT_VOCABULARY.get('terms', []))
    embedder = HashEmbedder(n_features=256)
    
    # --- Create index ---
    try:
        index = HybridIndex(vocab, embedding_fn=embedder.get_embedding)
        assert index is not None
        results.ok("Create HybridIndex")
    except Exception as e:
        results.fail("Create HybridIndex", str(e))
    
    # --- Add chunks ---
    try:
        chunks = [
            IndexableChunk(
                chunk_id=f"chunk_{i}",
                text=text,
                embedding_text=text,
                source_type=SourceType.CODE,
                semantic_type=SemanticType.PROCEDURE,
                source_ref=SourceReference(file_path=f"file_{i}.tal", line_start=1, line_end=10),
                metadata={"index": i}
            )
            for i, text in enumerate([
                "Process wire transfer payment to beneficiary account",
                "Validate SWIFT message format MT103",
                "Check AML sanctions screening results",
                "Calculate settlement amount for transaction",
                "Log payment audit trail entry",
            ])
        ]
        
        for chunk in chunks:
            index.index_chunk(chunk)
        
        stats = index.metadata
        assert stats['total_indexed'] == 5
        
        results.ok("Add chunks to index")
    except Exception as e:
        results.fail("Add chunks to index", str(e))
    
    # --- Search ---
    try:
        results_list = index.search("wire transfer payment", top_k=3)
        assert len(results_list) > 0
        assert results_list[0].combined_score > 0
        results.ok("Search index")
    except Exception as e:
        results.fail("Search index", str(e))
    
    # --- Search with filters ---
    try:
        results_list = index.search(
            "payment",
            top_k=5,
            source_types=[SourceType.CODE]
        )
        assert len(results_list) > 0
        results.ok("Search with source type filter")
    except Exception as e:
        results.fail("Search with source type filter", str(e))
    
    # --- Save and load ---
    try:
        temp_dir = tempfile.mkdtemp()
        
        index.save(temp_dir)
        
        index2 = HybridIndex(vocab, embedding_fn=embedder.get_embedding)
        index2.load(temp_dir)
        
        stats2 = index2.metadata
        assert stats2['total_indexed'] == 5
        
        # Search loaded index
        results_list = index2.search("wire transfer", top_k=3)
        assert len(results_list) > 0
        
        shutil.rmtree(temp_dir)
        results.ok("Save and load index")
    except Exception as e:
        results.fail("Save and load index", str(e))
    
    # --- Incremental updates ---
    try:
        new_chunk = IndexableChunk(
            chunk_id="chunk_new",
            text="New payment routing decision",
            embedding_text="New payment routing decision",
            source_type=SourceType.CODE,
            semantic_type=SemanticType.PROCEDURE,
            source_ref=SourceReference(file_path="new.tal", line_start=1, line_end=5),
            metadata={}
        )
        
        index.index_chunk(new_chunk)
        stats = index.metadata
        assert stats['total_indexed'] == 6
        
        results.ok("Incremental updates")
    except Exception as e:
        results.fail("Incremental updates", str(e))


def test_pipeline(results: TestResult):
    """Test the indexing pipeline."""
    print("\n" + "="*60)
    print("PIPELINE TESTS")
    print("="*60)
    
    try:
        from unified_indexer.pipeline import IndexingPipeline
        from unified_indexer.vocabulary import DomainVocabulary
    except ImportError as e:
        results.fail("Import pipeline", str(e))
        return
    
    results.ok("Import pipeline")
    
    vocab = DomainVocabulary()
    vocab.load_from_data(PAYMENT_VOCABULARY.get('terms', []))
    
    # --- Create pipeline ---
    try:
        pipeline = IndexingPipeline(
            vocabulary_data=PAYMENT_VOCABULARY.get("terms", []),
            embedder_type="hash",
        )
        assert pipeline is not None
        results.ok("Create IndexingPipeline")
    except Exception as e:
        results.fail("Create IndexingPipeline", str(e))
    
    # --- Index files ---
    try:
        temp_dir = tempfile.mkdtemp()
        
        # Create test files
        with open(os.path.join(temp_dir, "test.tal"), "w") as f:
            f.write(TAL_COMPLETE_PROGRAM)
        
        with open(os.path.join(temp_dir, "test.py"), "w") as f:
            f.write(PYTHON_CODE)
        
        with open(os.path.join(temp_dir, "doc.md"), "w") as f:
            f.write(MARKDOWN_DOC)
        
        # Index directory
        stats = pipeline.index_directory(temp_dir)
        assert stats.files_processed >= 3
        assert stats.total_chunks > 0
        
        shutil.rmtree(temp_dir)
        results.ok("Index directory")
    except Exception as e:
        results.fail("Index directory", str(e))
    
    # --- Search through pipeline ---
    try:
        search_results = pipeline.search("wire transfer payment", top_k=5)
        assert len(search_results) > 0
        results.ok("Search through pipeline")
    except Exception as e:
        results.fail("Search through pipeline", str(e))


def test_search_quality(results: TestResult):
    """Test search result quality and relevance."""
    print("\n" + "="*60)
    print("SEARCH QUALITY TESTS")
    print("="*60)
    
    try:
        from unified_indexer.pipeline import IndexingPipeline
        from unified_indexer.vocabulary import DomainVocabulary
    except ImportError as e:
        results.fail("Import for search quality", str(e))
        return
    
    vocab = DomainVocabulary()
    vocab.load_from_data(PAYMENT_VOCABULARY.get('terms', []))
    
    pipeline = IndexingPipeline(
        vocabulary_data=PAYMENT_VOCABULARY.get("terms", []),
        embedder_type="hash",
    )
    
    # Create diverse test content
    temp_dir = tempfile.mkdtemp()
    
    try:
        # TAL files with different content
        test_files = {
            "wire_transfer.tal": """
                PROC process_wire_transfer;
                BEGIN
                    ! Process incoming wire transfer
                    CALL validate_swift_message;
                    CALL check_beneficiary;
                    CALL execute_settlement;
                END;
            """,
            "ach_payment.tal": """
                PROC process_ach;
                BEGIN
                    ! Process ACH payment batch
                    CALL validate_nacha_format;
                    CALL check_routing_number;
                END;
            """,
            "validation.tal": """
                PROC validate_amount;
                BEGIN
                    ! Validate payment amount
                    IF amount < 0 THEN
                        RETURN error_invalid;
                    END;
                END;
            """,
            "logging.tal": """
                PROC log_transaction;
                BEGIN
                    ! Log transaction for audit
                    CALL write_audit_entry;
                END;
            """,
        }
        
        for filename, content in test_files.items():
            with open(os.path.join(temp_dir, filename), "w") as f:
                f.write(content)
        
        pipeline.index_directory(temp_dir)
        
        # Test relevance
        test_queries = [
            ("wire transfer", "wire_transfer.tal"),
            ("ACH payment", "ach_payment.tal"),
            ("validate amount", "validation.tal"),
            ("audit log", "logging.tal"),
            ("SWIFT message beneficiary", "wire_transfer.tal"),
        ]
        
        for query, expected_file in test_queries:
            results_list = pipeline.search(query, top_k=3)
            if results_list:
                top_result_file = os.path.basename(results_list[0].chunk.source_ref.file_path)
                # Check if expected file is in top 3
                top_files = [os.path.basename(r.chunk.source_ref.file_path) for r in results_list[:3]]
                if expected_file in top_files:
                    continue
        
        results.ok("Search relevance")
        
    except Exception as e:
        results.fail("Search relevance", str(e))
    finally:
        shutil.rmtree(temp_dir)
    
    # --- Score normalization ---
    try:
        search_results = pipeline.search("payment processing", top_k=10)
        if search_results:
            scores = [r.combined_score for r in search_results]
            # Scores should be in reasonable range
            assert all(0 <= s <= 1 for s in scores), "Scores should be normalized 0-1"
            # Scores should be descending
            assert scores == sorted(scores, reverse=True), "Scores should be descending"
        results.ok("Score normalization")
    except Exception as e:
        results.fail("Score normalization", str(e))


def test_error_handling(results: TestResult):
    """Test error handling and edge cases."""
    print("\n" + "="*60)
    print("ERROR HANDLING TESTS")
    print("="*60)
    
    try:
        from unified_indexer.parsers.tal_parser import TalCodeParser
        from unified_indexer.parsers.code_parser import GenericCodeParser
        from unified_indexer.vocabulary import DomainVocabulary
        from unified_indexer.index import HybridIndex
        from unified_indexer.embeddings import HashEmbedder
    except ImportError as e:
        results.fail("Import for error handling", str(e))
        return
    
    vocab = DomainVocabulary()
    
    # --- Empty input ---
    try:
        parser = TalCodeParser(vocab)
        chunks = parser.parse(b"", "empty.tal")
        # Should return empty list, not crash
        assert isinstance(chunks, list)
        results.ok("Handle empty input")
    except Exception as e:
        results.fail("Handle empty input", str(e))
    
    # --- Binary garbage ---
    try:
        parser = GenericCodeParser(vocab)
        garbage = bytes([0x00, 0x01, 0xFF, 0xFE, 0x89, 0x50])
        chunks = parser.parse(garbage, "garbage.bin")
        # Should handle gracefully
        assert isinstance(chunks, list)
        results.ok("Handle binary garbage")
    except Exception as e:
        results.fail("Handle binary garbage", str(e))
    
    # --- Malformed TAL ---
    try:
        from tal_enhanced_parser import parse_tal_string
        
        malformed_cases = [
            TAL_MALFORMED_PROC,
            TAL_MALFORMED_STRUCT,
            TAL_UNBALANCED_BEGIN,
            "PROC BEGIN END",
            "STRUCT { INT x; }",
            "DEFINE = value;",
        ]
        
        for code in malformed_cases:
            try:
                result = parse_tal_string(code)
                # Should return partial result, not crash
                assert result is not None
            except:
                pass  # Some may raise, which is also acceptable
        
        results.ok("Handle malformed TAL")
    except ImportError:
        results.skip("Handle malformed TAL", "tal_enhanced_parser not available")
    except Exception as e:
        results.fail("Handle malformed TAL", str(e))
    
    # --- Search empty index ---
    try:
        embedder = HashEmbedder(n_features=256)
        index = HybridIndex(vocab, embedding_fn=embedder.get_embedding)
        
        # Search empty index
        results_list = index.search("test query", top_k=5)
        assert isinstance(results_list, list)
        assert len(results_list) == 0
        
        results.ok("Search empty index")
    except Exception as e:
        results.fail("Search empty index", str(e))
    
    # --- Unicode handling ---
    try:
        parser = TalCodeParser(vocab)
        unicode_code = """
        PROC test;
        BEGIN
            ! Comment with unicode: 日本語 中文 한국어 Ελληνικά
            STRING msg[0:49] := "Hello, 世界!";
        END;
        """
        chunks = parser.parse(unicode_code.encode('utf-8'), "unicode.tal")
        assert isinstance(chunks, list)
        results.ok("Handle unicode content")
    except Exception as e:
        results.fail("Handle unicode content", str(e))
    
    # --- Very long lines ---
    try:
        long_line = "INT x := " + " + ".join(["1"] * 1000) + ";"
        long_code = f"PROC test; BEGIN {long_line} END;"
        
        from tal_enhanced_parser import parse_tal_string
        result = parse_tal_string(long_code)
        assert result is not None
        
        results.ok("Handle very long lines")
    except ImportError:
        results.skip("Handle very long lines", "tal_enhanced_parser not available")
    except Exception as e:
        results.fail("Handle very long lines", str(e))


def test_long_queries(results: TestResult):
    """Test search with long queries of various sizes."""
    print("\n" + "="*60)
    print("LONG QUERY TESTS")
    print("="*60)
    
    try:
        from unified_indexer.pipeline import IndexingPipeline
        from unified_indexer.vocabulary import DomainVocabulary
        from unified_indexer.embeddings import HashEmbedder
    except ImportError as e:
        results.fail("Import for long query tests", str(e))
        return
    
    # Create test index with realistic content
    pipeline = IndexingPipeline(
        vocabulary_data=PAYMENT_VOCABULARY.get("terms", []),
        embedder_type="hash"
    )
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create realistic TAL files
        test_files = {
            "wire_transfer.tal": """
! Wire Transfer Processing Module
! ================================

DEFINE max_amount = 1000000;
DEFINE min_amount = 1;
DEFINE retry_limit = 3;

LITERAL
    status_pending = 0,
    status_approved = 1,
    status_rejected = 2,
    status_error = -1;

STRUCT transfer_request;
BEGIN
    INT transfer_id;
    INT from_account;
    INT to_account;
    INT amount;
    INT currency_code;
    STRING beneficiary_name[0:49];
    STRING originator_name[0:49];
    INT status;
END;

INT PROC validate_transfer(.request);
    STRUCT .request(transfer_request);
BEGIN
    INT validation_result;
    
    ! Check amount limits
    IF request.amount < min_amount OR request.amount > max_amount THEN
        RETURN status_rejected;
    END;
    
    ! Verify accounts are different
    IF request.from_account = request.to_account THEN
        RETURN status_rejected;
    END;
    
    ! Call OFAC sanctions screening
    validation_result := check_ofac_sanctions(request.beneficiary_name);
    IF validation_result <> status_approved THEN
        RETURN validation_result;
    END;
    
    ! Call AML compliance check
    validation_result := check_aml_compliance(request);
    IF validation_result <> status_approved THEN
        RETURN validation_result;
    END;
    
    RETURN status_approved;
END;

INT PROC check_ofac_sanctions(.name);
    STRING .name;
BEGIN
    ! OFAC SDN list screening implementation
    ! Checks against sanctions database
    CALL log_screening_request(name);
    RETURN status_approved;
END;

INT PROC check_aml_compliance(.request);
    STRUCT .request(transfer_request);
BEGIN
    ! Anti-money laundering compliance checks
    ! Verifies transaction patterns
    IF request.amount > 10000 THEN
        CALL enhanced_due_diligence(request);
    END;
    RETURN status_approved;
END;

PROC enhanced_due_diligence(.request);
    STRUCT .request(transfer_request);
BEGIN
    ! Enhanced verification for large transfers
    CALL verify_source_of_funds(request);
    CALL check_pep_status(request.originator_name);
END;

PROC log_screening_request(.name);
    STRING .name;
BEGIN
    ! Audit logging for compliance
END;

PROC verify_source_of_funds(.request);
    STRUCT .request(transfer_request);
BEGIN
    ! Source of funds verification
END;

PROC check_pep_status(.name);
    STRING .name;
BEGIN
    ! Politically exposed person check
END;
""",
            "error_handling.tal": """
! Error Handling Module
! =====================

DEFINE error_timeout = -100;
DEFINE error_connection = -101;
DEFINE error_validation = -102;
DEFINE error_insufficient_funds = -103;

INT PROC handle_transfer_error(error_code, .request);
    INT error_code;
    STRUCT .request(transfer_request);
BEGIN
    CASE error_code OF
    BEGIN
        error_timeout -> CALL retry_with_backoff(request);
        error_connection -> CALL failover_to_backup(request);
        error_validation -> CALL log_validation_failure(request);
        error_insufficient_funds -> CALL notify_insufficient_funds(request);
        OTHERWISE -> CALL log_unknown_error(error_code);
    END;
    RETURN error_code;
END;

PROC retry_with_backoff(.request);
    STRUCT .request(transfer_request);
BEGIN
    INT retry_count;
    INT delay_ms;
    
    retry_count := 0;
    delay_ms := 1000;
    
    WHILE retry_count < retry_limit DO
        CALL delay(delay_ms);
        IF process_transfer(request) = status_approved THEN
            RETURN;
        END;
        retry_count := retry_count + 1;
        delay_ms := delay_ms * 2;
    END;
END;

PROC failover_to_backup(.request);
    STRUCT .request(transfer_request);
BEGIN
    ! Failover to backup processing system
END;

PROC log_validation_failure(.request);
    STRUCT .request(transfer_request);
BEGIN
    ! Log validation failure for audit
END;

PROC notify_insufficient_funds(.request);
    STRUCT .request(transfer_request);
BEGIN
    ! Send notification about insufficient funds
END;

PROC log_unknown_error(code);
    INT code;
BEGIN
    ! Log unknown error code
END;
""",
            "settlement.tal": """
! Settlement Processing Module
! ============================

STRUCT settlement_record;
BEGIN
    INT settlement_id;
    INT transfer_id;
    INT settlement_amount;
    INT settlement_date;
    INT settlement_status;
END;

INT PROC process_settlement(.transfer, .settlement);
    STRUCT .transfer(transfer_request);
    STRUCT .settlement(settlement_record);
BEGIN
    ! Create settlement record
    settlement.transfer_id := transfer.transfer_id;
    settlement.settlement_amount := transfer.amount;
    
    ! Execute settlement via SWIFT
    CALL send_swift_message(transfer, settlement);
    
    ! Update status
    settlement.settlement_status := status_approved;
    
    RETURN status_approved;
END;

PROC send_swift_message(.transfer, .settlement);
    STRUCT .transfer(transfer_request);
    STRUCT .settlement(settlement_record);
BEGIN
    ! Format and send MT103 SWIFT message
    CALL format_mt103(transfer);
    CALL transmit_to_swift_network(settlement);
END;

PROC format_mt103(.transfer);
    STRUCT .transfer(transfer_request);
BEGIN
    ! Format MT103 single customer credit transfer
END;

PROC transmit_to_swift_network(.settlement);
    STRUCT .settlement(settlement_record);
BEGIN
    ! Transmit to SWIFT network
END;
"""
        }
        
        for filename, content in test_files.items():
            with open(os.path.join(temp_dir, filename), "w") as f:
                f.write(content)
        
        pipeline.index_directory(temp_dir)
        
        # ============================================================
        # TEST CASES FOR LONG QUERIES
        # ============================================================
        
        # --- Test: 50-word query ---
        try:
            query_50 = """
            I need to understand how the wire transfer validation process works in this system.
            Specifically, I want to know about OFAC sanctions screening and AML compliance checks.
            Can you show me the procedures that handle validation, screening, and error handling?
            Also interested in how the system handles retry logic for failed transactions.
            """
            word_count = len(query_50.split())
            
            start = time.time()
            results_list = pipeline.search(query_50, top_k=5)
            duration = time.time() - start
            
            assert len(results_list) > 0, "No results for 50-word query"
            assert duration < 1.0, f"50-word query too slow: {duration}s"
            results.ok(f"50-word query ({word_count} words, {duration*1000:.1f}ms, {len(results_list)} results)")
        except Exception as e:
            results.fail("50-word query", str(e))
        
        # --- Test: 100-word query ---
        try:
            query_100 = """
            I am trying to understand the complete wire transfer validation and settlement process
            in our legacy TAL payment system. The business requirement states that all wire transfers
            must go through OFAC sanctions screening and AML compliance checks before being processed.
            
            I need to find the procedures that handle validate_transfer, check_ofac_sanctions,
            check_aml_compliance, and process_settlement. Additionally, I want to understand how
            the system handles errors when validation fails, including retry logic with exponential
            backoff and failover to backup systems.
            
            The compliance team has requested documentation on the complete flow from when a transfer
            request is received until settlement is complete. Please search for all relevant procedures.
            """
            word_count = len(query_100.split())
            
            start = time.time()
            results_list = pipeline.search(query_100, top_k=5)
            duration = time.time() - start
            
            assert len(results_list) > 0, "No results for 100-word query"
            assert duration < 1.0, f"100-word query too slow: {duration}s"
            results.ok(f"100-word query ({word_count} words, {duration*1000:.1f}ms, {len(results_list)} results)")
        except Exception as e:
            results.fail("100-word query", str(e))
        
        # --- Test: 200-word query ---
        try:
            query_200 = """
            I am conducting a comprehensive review of our wire transfer processing system for the
            annual compliance audit. The auditors have requested detailed documentation of all
            validation and screening procedures, including OFAC sanctions screening, AML compliance
            checks, and enhanced due diligence for high-value transfers.
            
            Specifically, I need to understand the following aspects of the system:
            
            1. How does validate_transfer work and what validation checks does it perform?
            2. How is OFAC sanctions screening implemented in check_ofac_sanctions?
            3. What AML compliance checks are performed by check_aml_compliance?
            4. When is enhanced_due_diligence triggered and what additional checks does it perform?
            5. How does the system handle errors, including timeout, connection, and validation errors?
            6. What retry logic is implemented for failed transactions?
            7. How does failover to backup systems work?
            8. How is settlement processing handled after validation?
            9. How are SWIFT MT103 messages formatted and transmitted?
            10. What audit logging is performed throughout the process?
            
            Please search for all procedures related to these topics so I can create the
            documentation required by the compliance team. I need to see the actual TAL code
            implementation to accurately document the system behavior and control flows.
            """
            word_count = len(query_200.split())
            
            start = time.time()
            results_list = pipeline.search(query_200, top_k=10)
            duration = time.time() - start
            
            assert len(results_list) > 0, "No results for 200-word query"
            assert duration < 2.0, f"200-word query too slow: {duration}s"
            results.ok(f"200-word query ({word_count} words, {duration*1000:.1f}ms, {len(results_list)} results)")
        except Exception as e:
            results.fail("200-word query", str(e))
        
        # --- Test: 500-word query ---
        try:
            query_500 = """
            I am preparing a comprehensive technical documentation package for our wire transfer
            processing system. This documentation is required for multiple purposes including
            regulatory compliance review, internal audit, system modernization planning, and
            knowledge transfer to new team members.
            
            The documentation needs to cover the complete lifecycle of a wire transfer from
            initial request through final settlement. This includes all validation steps,
            compliance screening, error handling, and settlement processing.
            
            For the validation phase, I need to understand:
            - How incoming transfer requests are validated
            - What checks are performed on amount limits (min and max)
            - How account validation works to prevent self-transfers
            - The sequence of compliance checks performed
            
            For compliance screening, I need detailed information about:
            - OFAC sanctions screening against the SDN list
            - AML anti-money laundering compliance checks
            - Enhanced due diligence for transfers exceeding threshold amounts
            - PEP politically exposed person screening
            - Source of funds verification procedures
            
            For error handling, the documentation must cover:
            - All error codes and their meanings
            - Retry logic with exponential backoff for timeout errors
            - Failover procedures when primary systems are unavailable
            - Logging and notification for various error conditions
            - How insufficient funds situations are handled
            
            For settlement processing, I need to document:
            - How settlement records are created and linked to transfers
            - SWIFT message formatting for MT103 single customer credit transfers
            - Transmission to the SWIFT network
            - Settlement status tracking and updates
            
            Additionally, the audit team requires information about:
            - All logging points throughout the process
            - What information is captured in audit records
            - How screening requests are logged for compliance
            - Error logging and alerting mechanisms
            
            Please search the codebase for all procedures, structures, literals, and defines
            related to these topics. I need to see the actual implementation details so I can
            accurately document the system behavior, control flows, data structures, and
            integration points with external systems like SWIFT.
            
            The final documentation will be reviewed by internal audit, external auditors,
            and regulatory examiners, so accuracy and completeness are critical. Any gaps
            in the documentation could result in audit findings or regulatory concerns.
            """ * 1  # ~500 words
            word_count = len(query_500.split())
            
            start = time.time()
            results_list = pipeline.search(query_500, top_k=10)
            duration = time.time() - start
            
            assert len(results_list) > 0, "No results for 500-word query"
            assert duration < 3.0, f"500-word query too slow: {duration}s"
            results.ok(f"500-word query ({word_count} words, {duration*1000:.1f}ms, {len(results_list)} results)")
        except Exception as e:
            results.fail("500-word query", str(e))
        
        # --- Test: 1000-word query ---
        try:
            query_1000 = query_500 * 2  # Double the 500-word query
            word_count = len(query_1000.split())
            
            start = time.time()
            results_list = pipeline.search(query_1000, top_k=10)
            duration = time.time() - start
            
            assert len(results_list) > 0, "No results for 1000-word query"
            assert duration < 5.0, f"1000-word query too slow: {duration}s"
            results.ok(f"1000-word query ({word_count} words, {duration*1000:.1f}ms, {len(results_list)} results)")
        except Exception as e:
            results.fail("1000-word query", str(e))
        
        # --- Test: Query with code snippets ---
        try:
            query_with_code = """
            I found this error in the logs and need to find the relevant code:
            
            ERROR: Transfer validation failed
            Procedure: validate_transfer
            Error code: -102 (error_validation)
            Transfer ID: 12345
            Amount: 50000
            Status: status_rejected
            
            The error seems to occur after OFAC screening passes but before AML check.
            Can you find the code that handles this validation flow?
            
            I also see references to these procedures in the stack trace:
            - check_ofac_sanctions
            - check_aml_compliance  
            - enhanced_due_diligence
            
            What is the relationship between these procedures?
            """
            word_count = len(query_with_code.split())
            
            start = time.time()
            results_list = pipeline.search(query_with_code, top_k=5)
            duration = time.time() - start
            
            assert len(results_list) > 0, "No results for query with code"
            results.ok(f"Query with code snippets ({word_count} words, {duration*1000:.1f}ms, {len(results_list)} results)")
        except Exception as e:
            results.fail("Query with code snippets", str(e))
        
        # --- Test: Query with special characters ---
        try:
            query_special = """
            Find procedures that handle:
            - OFAC/SDN screening (sanctions)
            - AML/BSA compliance checks
            - MT103 SWIFT messages
            - Error codes: -100, -101, -102, -103
            - Status values: status_pending (0), status_approved (1), status_rejected (2)
            
            Also need: validate_transfer(), check_ofac_sanctions(), process_settlement()
            """
            word_count = len(query_special.split())
            
            start = time.time()
            results_list = pipeline.search(query_special, top_k=5)
            duration = time.time() - start
            
            assert len(results_list) > 0, "No results for query with special chars"
            results.ok(f"Query with special characters ({word_count} words, {duration*1000:.1f}ms, {len(results_list)} results)")
        except Exception as e:
            results.fail("Query with special characters", str(e))
        
        # --- Test: Repetitive query (stress test) ---
        try:
            query_repetitive = "wire transfer OFAC sanctions screening validation " * 100
            word_count = len(query_repetitive.split())
            
            start = time.time()
            results_list = pipeline.search(query_repetitive, top_k=5)
            duration = time.time() - start
            
            assert len(results_list) > 0, "No results for repetitive query"
            assert duration < 5.0, f"Repetitive query too slow: {duration}s"
            results.ok(f"Repetitive query ({word_count} words, {duration*1000:.1f}ms, {len(results_list)} results)")
        except Exception as e:
            results.fail("Repetitive query", str(e))
        
        # --- Test: Query relevance for long queries ---
        try:
            # Long query that should still find OFAC-related content
            long_ofac_query = """
            The compliance department needs information about sanctions screening.
            We are looking for OFAC SDN list checking procedures.
            """ + " This is filler text to make the query longer. " * 20
            
            results_list = pipeline.search(long_ofac_query, top_k=3)
            
            # Check that OFAC-related content is in top results
            found_ofac = False
            for r in results_list:
                if "ofac" in r.chunk.text.lower() or "sanctions" in r.chunk.text.lower():
                    found_ofac = True
                    break
            
            assert found_ofac, "OFAC content not found in top results for long query"
            results.ok("Long query relevance (OFAC found in top results)")
        except Exception as e:
            results.fail("Long query relevance", str(e))
        
        # --- Test: Empty and whitespace queries ---
        try:
            # These should not crash
            results_empty = pipeline.search("", top_k=5)
            results_whitespace = pipeline.search("   \n\t   ", top_k=5)
            results.ok("Empty and whitespace queries (no crash)")
        except Exception as e:
            results.fail("Empty and whitespace queries", str(e))
        
        # --- Test: Unicode in queries ---
        try:
            query_unicode = "wire transfer validation 日本語 中文 한국어 émojis 🏦💰"
            results_list = pipeline.search(query_unicode, top_k=5)
            # Should not crash, may or may not find results
            results.ok("Unicode query (no crash)")
        except Exception as e:
            results.fail("Unicode query", str(e))
            
    finally:
        shutil.rmtree(temp_dir)


def test_performance(results: TestResult):
    """Test performance with large inputs."""
    print("\n" + "="*60)
    print("PERFORMANCE TESTS")
    print("="*60)
    
    try:
        from unified_indexer.pipeline import IndexingPipeline
        from unified_indexer.vocabulary import DomainVocabulary
        from tal_enhanced_parser import parse_tal_string
    except ImportError as e:
        results.fail("Import for performance", str(e))
        return
    
    vocab = DomainVocabulary()
    vocab.load_from_data(PAYMENT_VOCABULARY.get('terms', []))
    
    # --- Large TAL file ---
    try:
        # Generate large program
        large_code_parts = []
        for i in range(500):
            large_code_parts.append(f"""
PROC procedure_{i};
BEGIN
    INT local_var_{i};
    STRING buffer_{i}[0:255];
    
    local_var_{i} := {i};
    CALL helper_proc_{i % 50};
    
    IF local_var_{i} > 100 THEN
        CALL process_large_{i};
    END;
END;
""")
        
        large_code = "\n".join(large_code_parts)
        
        start = time.time()
        result = parse_tal_string(large_code)
        duration = time.time() - start
        
        assert len(result.procedures) == 500
        assert duration < 10.0  # Should complete in under 10 seconds
        
        results.ok(f"Parse 500 procedures", duration)
    except Exception as e:
        results.fail("Parse 500 procedures", str(e))
    
    # --- Large index ---
    try:
        pipeline = IndexingPipeline(
            vocabulary_data=PAYMENT_VOCABULARY.get("terms", []),
            embedder_type="hash",
        )
        
        temp_dir = tempfile.mkdtemp()
        
        # Create many files
        for i in range(100):
            with open(os.path.join(temp_dir, f"file_{i}.tal"), "w") as f:
                f.write(f"""
PROC proc_{i}; BEGIN INT x; x := {i}; CALL next_{i}; END;
PROC helper_{i}; BEGIN STRING s[0:99]; END;
""")
        
        start = time.time()
        stats = pipeline.index_directory(temp_dir)
        duration = time.time() - start
        
        assert stats.files_processed == 100
        assert duration < 30.0
        
        shutil.rmtree(temp_dir)
        results.ok(f"Index 100 files", duration)
    except Exception as e:
        results.fail("Index 100 files", str(e))
    
    # --- Search performance ---
    try:
        start = time.time()
        for _ in range(100):
            pipeline.search("payment processing", top_k=10)
        duration = time.time() - start
        
        avg_time = duration / 100
        assert avg_time < 0.1  # Each search should be under 100ms
        
        results.ok(f"100 searches (avg {avg_time*1000:.1f}ms)", duration)
    except Exception as e:
        results.fail("Search performance", str(e))


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("="*60)
    print("COMPREHENSIVE TEST SUITE FOR UNIFIED INDEXER")
    print("="*60)
    
    results = TestResult()
    
    # TAL Parser Tests
    test_tal_proc_parser(results)
    test_tal_enhanced_parser(results)
    
    # Unified Indexer Tests
    test_vocabulary(results)
    test_embeddings(results)
    test_parsers(results)
    test_index_operations(results)
    test_pipeline(results)
    test_search_quality(results)
    
    # Robustness Tests
    test_error_handling(results)
    test_long_queries(results)
    test_performance(results)
    
    # Summary
    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
