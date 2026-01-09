#!/usr/bin/env python3
"""
Diagnostic script to find the 'ParseResult has no attribute get' error
"""
import sys
import traceback

# Test 1: Check tal_enhanced_parser directly
print("=" * 60)
print("TEST 1: tal_enhanced_parser.py directly")
print("=" * 60)

try:
    import tal_enhanced_parser
    print(f"  Loaded from: {tal_enhanced_parser.__file__}")
    
    parser = tal_enhanced_parser.EnhancedTALParser()
    code = "PROC test; BEGIN INT x; END;"
    result = parser.parse(code, "test.tal")
    
    print(f"  Result type: {type(result)}")
    print(f"  Has .procedures: {hasattr(result, 'procedures')}")
    print(f"  Has .get: {hasattr(result, 'get')}")
    print(f"  Procedures found: {len(result.procedures)}")
    print("  ✓ PASSED")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()

# Test 2: Check tal_proc_parser
print("\n" + "=" * 60)
print("TEST 2: tal_proc_parser.py")
print("=" * 60)

try:
    import tal_proc_parser
    print(f"  Loaded from: {tal_proc_parser.__file__}")
    
    procs = tal_proc_parser.find_procedure_declarations("PROC test; BEGIN END;")
    print(f"  find_procedure_declarations works: {len(procs)} found")
    print("  ✓ PASSED")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()

# Test 3: Check unified_indexer tal_parser
print("\n" + "=" * 60)
print("TEST 3: unified_indexer/parsers/tal_parser.py")
print("=" * 60)

try:
    from unified_indexer.parsers.tal_parser import TalCodeParser
    from unified_indexer.vocabulary import DomainVocabulary
    
    # Find where it's loaded from
    import unified_indexer.parsers.tal_parser as tp_module
    print(f"  Loaded from: {tp_module.__file__}")
    
    vocab = DomainVocabulary()
    parser = TalCodeParser(vocab)
    
    print(f"  Has tal_enhanced_parser: {parser.tal_enhanced_parser is not None}")
    print(f"  Has tal_proc_parser: {parser.tal_proc_parser is not None}")
    
except Exception as e:
    print(f"  ✗ FAILED during import: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Actually parse something
print("\n" + "=" * 60)
print("TEST 4: Parse TAL code through TalCodeParser")
print("=" * 60)

try:
    code = """
PROC test;
BEGIN
    INT x;
    x := 1;
END;
"""
    
    # Monkey-patch to catch the exact error
    original_parse = parser._parse_with_enhanced_parser
    
    def debug_parse(text, source_path):
        print("  Entering _parse_with_enhanced_parser...")
        try:
            ep = parser.tal_enhanced_parser.EnhancedTALParser()
            print(f"    Created EnhancedTALParser")
            
            result = ep.parse(text, source_path)
            print(f"    parse() returned: {type(result)}")
            print(f"    result.procedures: {type(result.procedures)}")
            
            # Try accessing .get to see if that's the issue
            if hasattr(result, 'get'):
                print(f"    WARNING: result has .get method!")
            else:
                print(f"    result does NOT have .get (correct)")
            
            # Now iterate
            for proc in result.procedures:
                print(f"    Processing procedure: {proc.name}")
                
            return original_parse(text, source_path)
        except Exception as e:
            print(f"    ERROR in debug_parse: {e}")
            traceback.print_exc()
            raise
    
    parser._parse_with_enhanced_parser = debug_parse
    
    chunks = parser.parse(code.encode('utf-8'), 'test.tal')
    print(f"  Got {len(chunks)} chunks")
    print("  ✓ PASSED")
    
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()

# Test 5: Check for any .get() in the source files
print("\n" + "=" * 60)
print("TEST 5: Search for .get() patterns in source")
print("=" * 60)

import os
import re

files_to_check = [
    'tal_enhanced_parser.py',
    'tal_proc_parser.py',
    'unified_indexer/parsers/tal_parser.py'
]

for filepath in files_to_check:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Look for result.get( patterns
        matches = re.findall(r'result\.get\s*\(', content)
        if matches:
            print(f"  ⚠ {filepath}: Found {len(matches)} 'result.get(' patterns!")
            # Find line numbers
            for i, line in enumerate(content.split('\n'), 1):
                if 'result.get(' in line:
                    print(f"      Line {i}: {line.strip()[:60]}")
        else:
            print(f"  ✓ {filepath}: No 'result.get(' found")
    else:
        print(f"  ? {filepath}: File not found")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
