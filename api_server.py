"""
Astra API Server

Orchestrates:
- search_index.py: search indexed codebase
- unified_indexer: parsing, indexing
- astra: translation & documentation

SETUP:
    1. Copy this file to your uvidx2 directory (where unified_indexer/ exists)
    2. Build index: python build_index.py --tal-dir ./code --output ./index
    3. Run server: python api_server.py

ALTERNATIVE (set paths):
    export PYTHONPATH=/path/to/uvidx2:$PYTHONPATH
    export INDEX_PATH=/path/to/index
    python api_server.py

Environment:
    INDEX_PATH: Path to index directory (default: ./index)
    PORT: Server port (default: 8080)
    MOCK_MODE: Set to '1' for testing without index
"""

import os
import sys
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# Mock mode for testing without index
MOCK_MODE = os.environ.get('MOCK_MODE', '').lower() in ('1', 'true', 'yes')

# Index path - your structure uses ./index
INDEX_PATH = os.environ.get('INDEX_PATH', './index')
VOCAB_PATH = os.environ.get('VOCAB_PATH', './keywords.json')
KNOWLEDGE_GRAPH_PATH = os.environ.get('KNOWLEDGE_GRAPH_PATH', './knowledge_graph.json')

# Cache for vocabulary and knowledge graph (with timestamps)
_cache = {
    'vocabulary': None,
    'vocabulary_mtime': 0,
    'knowledge_graph': None,
    'knowledge_graph_mtime': 0,
}

# ============================================================
# Import unified_indexer
# ============================================================
INDEXER_AVAILABLE = False
IndexingPipeline = None

if MOCK_MODE:
    print("⚠️  MOCK_MODE enabled - using sample data for testing")
    INDEXER_AVAILABLE = True  # Pretend it's available
else:
    try:
        from unified_indexer.pipeline import IndexingPipeline
        INDEXER_AVAILABLE = True
        print("✓ unified_indexer.pipeline loaded")
    except ImportError as e:
        print(f"✗ unified_indexer.pipeline not found: {e}")
        print("  → Run from uvidx2 directory OR set PYTHONPATH")
        print("  → For testing: export MOCK_MODE=1")

# Check index exists
if not MOCK_MODE:
    if os.path.exists(INDEX_PATH):
        print(f"✓ Index found at: {INDEX_PATH}")
    else:
        print(f"✗ Index not found at: {INDEX_PATH}")
        print(f"  → Build with: python build_index.py --output {INDEX_PATH}")
        print(f"  → Or set: export INDEX_PATH=/path/to/index")

# Load vocabulary for IndexingPipeline
def load_vocabulary(vocab_path):
    """Load vocabulary from JSON file."""
    if not os.path.exists(vocab_path):
        return []
    try:
        with open(vocab_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'entries' in data:
                return data['entries']
            return []
    except Exception as e:
        print(f"Warning: Could not load vocabulary: {e}")
        return []


def get_vocabulary_cached():
    """
    Get vocabulary with file-based cache invalidation.
    Reloads if file has been modified.
    """
    global _cache
    
    if not os.path.exists(VOCAB_PATH):
        return {'entries': [], 'version': '1.0', 'error': 'File not found'}
    
    try:
        mtime = os.path.getmtime(VOCAB_PATH)
        
        # Check if cache is valid
        if _cache['vocabulary'] is not None and _cache['vocabulary_mtime'] >= mtime:
            return _cache['vocabulary']
        
        # Reload from file
        with open(VOCAB_PATH, 'r') as f:
            data = json.load(f)
        
        # Normalize structure
        if isinstance(data, list):
            data = {'entries': data, 'version': '1.0'}
        elif isinstance(data, dict) and 'entries' not in data:
            data = {'entries': [], 'version': '1.0', **data}
        
        # Add metadata
        data['_cached_at'] = mtime
        data['_file_path'] = VOCAB_PATH
        
        # Update cache
        _cache['vocabulary'] = data
        _cache['vocabulary_mtime'] = mtime
        
        return data
        
    except Exception as e:
        return {'entries': [], 'version': '1.0', 'error': str(e)}


def get_knowledge_graph_cached():
    """
    Get knowledge graph with file-based cache invalidation.
    Reloads if file has been modified.
    """
    global _cache
    
    if not os.path.exists(KNOWLEDGE_GRAPH_PATH):
        return {'nodes': [], 'edges': [], 'statistics': {}, 'error': 'File not found'}
    
    try:
        mtime = os.path.getmtime(KNOWLEDGE_GRAPH_PATH)
        
        # Check if cache is valid
        if _cache['knowledge_graph'] is not None and _cache['knowledge_graph_mtime'] >= mtime:
            return _cache['knowledge_graph']
        
        # Reload from file
        with open(KNOWLEDGE_GRAPH_PATH, 'r') as f:
            data = json.load(f)
        
        # Ensure expected structure
        if 'nodes' not in data:
            data['nodes'] = []
        if 'edges' not in data:
            data['edges'] = []
        if 'statistics' not in data:
            data['statistics'] = {}
        
        # Add metadata
        data['_cached_at'] = mtime
        data['_file_path'] = KNOWLEDGE_GRAPH_PATH
        
        # Update cache
        _cache['knowledge_graph'] = data
        _cache['knowledge_graph_mtime'] = mtime
        
        return data
        
    except Exception as e:
        return {'nodes': [], 'edges': [], 'statistics': {}, 'error': str(e)}


def expand_query_with_knowledge(query: str, max_expansions: int = 5) -> dict:
    """
    Use knowledge graph to expand a query with related terms.
    
    Returns:
        {
            'original': 'wire transfer',
            'keywords': ['wire', 'transfer'],
            'expanded': ['wire', 'transfer', 'fedwire', 'swift', 'payment'],
            'related_concepts': [
                {'term': 'fedwire', 'relationship': 'co_occurs_with', 'score': 0.8},
                ...
            ]
        }
    """
    kg = get_knowledge_graph_cached()
    vocab = get_vocabulary_cached()
    
    # Normalize query
    query_lower = query.lower()
    query_words = set(query_lower.replace('-', ' ').replace('_', ' ').split())
    
    # Find matching nodes in knowledge graph
    matched_nodes = []
    for node in kg.get('nodes', []):
        node_id = node.get('id', '').lower()
        node_label = node.get('label', '').lower()
        
        # Check if node matches query
        if node_id in query_lower or node_label in query_lower:
            matched_nodes.append(node)
        elif any(w in node_id or w in node_label for w in query_words if len(w) > 2):
            matched_nodes.append(node)
    
    # Collect related terms from matched nodes
    related = []
    seen = set(query_words)
    
    for node in matched_nodes:
        # Add co-occurring terms
        for co_term in node.get('co_occurs_with', []):
            co_lower = co_term.lower()
            if co_lower not in seen:
                seen.add(co_lower)
                related.append({
                    'term': co_term,
                    'relationship': 'co_occurs_with',
                    'score': node.get('tf_idf_score', 0.5),
                    'source_node': node.get('label', node.get('id'))
                })
    
    # Also check edges for relationships
    for edge in kg.get('edges', []):
        source = edge.get('source', '').lower()
        target = edge.get('target', '').lower()
        
        if source in query_lower or any(w in source for w in query_words if len(w) > 2):
            if target not in seen:
                seen.add(target)
                related.append({
                    'term': target,
                    'relationship': edge.get('type', 'related_to'),
                    'score': 0.6
                })
        elif target in query_lower or any(w in target for w in query_words if len(w) > 2):
            if source not in seen:
                seen.add(source)
                related.append({
                    'term': source,
                    'relationship': edge.get('type', 'related_to'),
                    'score': 0.6
                })
    
    # Also check vocabulary for related keywords
    for entry in vocab.get('entries', []):
        keywords = entry.get('keywords', '').lower()
        related_kw = entry.get('related_keywords', '').lower()
        
        # Check if query matches this vocabulary entry
        if any(w in keywords for w in query_words if len(w) > 2):
            # Add related keywords
            for rk in related_kw.split(','):
                rk = rk.strip()
                if rk and rk.lower() not in seen:
                    seen.add(rk.lower())
                    related.append({
                        'term': rk,
                        'relationship': 'vocabulary_related',
                        'score': 0.5
                    })
    
    # Sort by score and limit
    related.sort(key=lambda x: -x.get('score', 0))
    related = related[:max_expansions]
    
    # Build expanded keywords
    expanded = list(query_words)
    for r in related:
        term = r['term'].lower().replace('_', ' ')
        for word in term.split():
            if word not in expanded and len(word) > 2:
                expanded.append(word)
    
    return {
        'original': query,
        'keywords': list(query_words),
        'expanded': expanded[:15],  # Limit total
        'related_concepts': related,
        'matched_nodes': len(matched_nodes)
    }

# ============================================================
# Import parsers
# ============================================================
tal_parse = None
try:
    from tal_enhanced_parser import parse_tal_string
    tal_parse = parse_tal_string
    print("✓ tal_enhanced_parser loaded")
except ImportError as e:
    print(f"✗ tal_enhanced_parser not found: {e}")

# ============================================================
# Import astra (translation & docs) - with fallbacks
# ============================================================
ASTRA_AVAILABLE = False
ASTRA_TRANSLATION_AVAILABLE = False
ASTRA_DOCS_AVAILABLE = False
engine = None
enhancer = None
DocGenerator = None
diagram_gen = None

# Try importing translation components
try:
    from astra.translation import CodeGenerationEngine, TranslationEnhancer
    engine = CodeGenerationEngine()
    enhancer = TranslationEnhancer()
    ASTRA_TRANSLATION_AVAILABLE = True
    print("✓ astra.translation loaded")
except ImportError:
    try:
        # Fallback: try direct import
        from astra import CodeGenerationEngine, TranslationEnhancer
        engine = CodeGenerationEngine()
        enhancer = TranslationEnhancer()
        ASTRA_TRANSLATION_AVAILABLE = True
        print("✓ astra (translation) loaded")
    except ImportError as e:
        print(f"ℹ astra.translation not available: {e}")

# Try importing docs components
try:
    from astra.docs import DocGenerator, DiagramGenerator
    diagram_gen = DiagramGenerator()
    ASTRA_DOCS_AVAILABLE = True
    print("✓ astra.docs loaded")
except ImportError:
    try:
        # Fallback: try direct import
        from astra import DocGenerator, DiagramGenerator
        diagram_gen = DiagramGenerator()
        ASTRA_DOCS_AVAILABLE = True
        print("✓ astra (docs) loaded")
    except ImportError as e:
        print(f"ℹ astra.docs not available: {e}")

# Set overall availability
ASTRA_AVAILABLE = ASTRA_TRANSLATION_AVAILABLE or ASTRA_DOCS_AVAILABLE
if ASTRA_AVAILABLE:
    print(f"✓ astra available (translation: {ASTRA_TRANSLATION_AVAILABLE}, docs: {ASTRA_DOCS_AVAILABLE})")
else:
    print("ℹ astra not available - /translate and /enhance endpoints disabled")
    print("  (This is optional - search and generate-docs still work)")

# ============================================================
# Import LLM config (for status endpoint only)
# ============================================================
LLM_CONFIG = None

try:
    from llm_config_loader import load_config
    LLM_CONFIG = load_config()
    default_profile = LLM_CONFIG.get('default_profile', 'default')
    profile = LLM_CONFIG.get('profiles', {}).get(default_profile, {})
    print(f"✓ LLM config loaded: {profile.get('model', 'unknown')} (info only - LLM calls via Copilot)")
except ImportError:
    print("ℹ llm_config_loader not found (optional)")
except Exception as e:
    print(f"ℹ LLM config: {e} (optional)")


# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)
CORS(app)

DOCS_OUTPUT_DIR = os.environ.get('DOCS_OUTPUT_DIR', './generated_docs')


def get_mock_search_results(query, limit=10):
    """Generate mock search results for testing without index."""
    # Sample TAL code for realistic testing
    mock_chunks = [
        {
            'file_path': 'src/wire/WIRE_TRANSFER.tal',
            'procedure_name': 'VALIDATE_WIRE_TRANSFER',
            'text': '''PROC VALIDATE_WIRE_TRANSFER(wire_request, status);
BEGIN
  INT status;
  ! Validate wire transfer request
  IF wire_request.amount > MAX_WIRE_AMOUNT THEN
    status := ERROR_AMOUNT_EXCEEDED;
    RETURN;
  END;
  
  ! Check beneficiary
  CALL CHECK_BENEFICIARY(wire_request.beneficiary, status);
  IF status <> OK THEN RETURN;
  
  ! OFAC screening
  CALL OFAC_SCREEN(wire_request, status);
  IF status = OFAC_HIT THEN
    status := ERROR_OFAC_BLOCKED;
    RETURN;
  END;
  
  status := OK;
END;''',
            'score': 0.95,
            'type': 'procedure'
        },
        {
            'file_path': 'src/wire/WIRE_TRANSFER.tal',
            'procedure_name': 'EXECUTE_WIRE',
            'text': '''PROC EXECUTE_WIRE(wire_request, confirmation);
BEGIN
  STRING confirmation[0:63];
  INT status;
  
  ! Validate first
  CALL VALIDATE_WIRE_TRANSFER(wire_request, status);
  IF status <> OK THEN RETURN status;
  
  ! Debit source account
  CALL DEBIT_ACCOUNT(wire_request.source_account, wire_request.amount, status);
  IF status <> OK THEN RETURN status;
  
  ! Send to FedWire/SWIFT
  IF wire_request.network = FEDWIRE THEN
    CALL SEND_FEDWIRE(wire_request, confirmation);
  ELSE
    CALL SEND_SWIFT(wire_request, confirmation);
  END;
  
  RETURN OK;
END;''',
            'score': 0.88,
            'type': 'procedure'
        },
        {
            'file_path': 'src/compliance/OFAC_CHECK.tal',
            'procedure_name': 'OFAC_SCREEN',
            'text': '''PROC OFAC_SCREEN(request, result);
BEGIN
  INT result;
  STRING name[0:127];
  
  ! Extract beneficiary name
  name := request.beneficiary.name;
  
  ! Check SDN list
  CALL CHECK_SDN_LIST(name, result);
  IF result = SDN_MATCH THEN
    result := OFAC_HIT;
    CALL LOG_OFAC_HIT(request, name);
    RETURN;
  END;
  
  ! Check country sanctions
  CALL CHECK_COUNTRY_SANCTIONS(request.beneficiary.country, result);
  
  result := OFAC_CLEAR;
END;''',
            'score': 0.82,
            'type': 'procedure'
        },
        {
            'file_path': 'src/structs/WIRE_STRUCTS.tal',
            'procedure_name': '',
            'text': '''STRUCT WIRE_REQUEST;
BEGIN
  STRING source_account[0:19];
  STRING beneficiary_account[0:34];
  INT(32) amount;
  INT network;  ! FEDWIRE or SWIFT
  STRUCT beneficiary_info beneficiary;
  STRING reference[0:63];
  INT priority;
END;

STRUCT BENEFICIARY_INFO;
BEGIN
  STRING name[0:127];
  STRING address[0:255];
  STRING country[0:2];
  STRING bank_code[0:11];
END;''',
            'score': 0.75,
            'type': 'structure'
        },
        {
            'file_path': 'src/constants/WIRE_CONSTANTS.tal',
            'procedure_name': '',
            'text': '''LITERAL MAX_WIRE_AMOUNT = 1000000000;  ! $10M limit
LITERAL FEDWIRE = 1;
LITERAL SWIFT = 2;

LITERAL OK = 0;
LITERAL ERROR_AMOUNT_EXCEEDED = 101;
LITERAL ERROR_OFAC_BLOCKED = 102;
LITERAL ERROR_INSUFFICIENT_FUNDS = 103;
LITERAL OFAC_HIT = 1;
LITERAL OFAC_CLEAR = 0;
LITERAL SDN_MATCH = 1;''',
            'score': 0.70,
            'type': 'constant'
        }
    ]
    
    # Filter by query terms
    query_terms = query.lower().split()
    scored = []
    for chunk in mock_chunks:
        text_lower = chunk['text'].lower()
        matches = sum(1 for term in query_terms if term in text_lower)
        if matches > 0:
            scored.append({
                **chunk,
                'score': chunk['score'] * (0.5 + 0.5 * matches / len(query_terms)),
                'content_preview': truncate_to_tokens(chunk['text'], 500)
            })
    
    # Sort by score and limit
    scored.sort(key=lambda x: -x['score'])
    return scored[:limit]


def do_search(query, limit=10):
    """Perform search using IndexingPipeline or mock data"""
    
    # Mock mode for testing
    if MOCK_MODE:
        return get_mock_search_results(query, limit)
    
    if not INDEXER_AVAILABLE or IndexingPipeline is None:
        return []
    
    if not os.path.exists(INDEX_PATH):
        return []
    
    try:
        # Load vocabulary
        vocab_data = load_vocabulary(VOCAB_PATH)
        
        # Create pipeline and load index (matching search_index.py pattern)
        pipeline = IndexingPipeline(
            vocabulary_data=vocab_data,
            embedder_type=None  # Will be restored from saved index
        )
        pipeline.load(INDEX_PATH)
        
        # Execute search
        results = pipeline.search(query, top_k=limit)
        
        # Format results
        formatted = []
        for r in results:
            if hasattr(r, 'chunk'):
                chunk = r.chunk
                text = chunk.text if hasattr(chunk, 'text') else ''
                formatted.append({
                    'file_path': chunk.source_ref.file_path if hasattr(chunk, 'source_ref') and chunk.source_ref else '',
                    'procedure_name': chunk.metadata.get('procedure_name', '') if hasattr(chunk, 'metadata') and chunk.metadata else '',
                    'function_name': chunk.metadata.get('function_name', '') if hasattr(chunk, 'metadata') and chunk.metadata else '',
                    'content_preview': truncate_to_tokens(text, 500),
                    'text': text,
                    'score': r.combined_score if hasattr(r, 'combined_score') else 0,
                    'detected_language': chunk.metadata.get('language', '') if hasattr(chunk, 'metadata') and chunk.metadata else '',
                    'concepts': chunk.metadata.get('concepts', []) if hasattr(chunk, 'metadata') and chunk.metadata else [],
                })
            elif isinstance(r, dict):
                formatted.append(r)
        
        return formatted
        
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        return []


def truncate_to_tokens(text: str, max_tokens: int = 500) -> str:
    """
    Truncate text to approximately max_tokens.
    
    Uses a simple tokenization approach:
    - Split on whitespace and punctuation boundaries
    - Each word/symbol counts as ~1 token
    - More accurate than character count for LLM context
    
    For code, this tends to be conservative (code has more tokens per char).
    """
    if not text:
        return ''
    
    # Simple tokenization: split on whitespace, keeping structure
    # For code, also split on common delimiters
    import re
    tokens = re.findall(r'\S+|\n', text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Take first max_tokens tokens and rejoin
    truncated_tokens = tokens[:max_tokens]
    result = ''
    for token in truncated_tokens:
        if token == '\n':
            result += '\n'
        elif result and not result.endswith('\n'):
            result += ' ' + token
        else:
            result += token
    
    return result + '...'


# ============================================================
# Grep-based search (fallback/supplementary to index search)
# ============================================================

# Code directory for grep search - set via environment or auto-detect
CODE_DIR = os.environ.get('CODE_DIR', '')

def get_code_directory():
    """Get the code directory to search."""
    if CODE_DIR:
        if os.path.isdir(CODE_DIR):
            return CODE_DIR
        print(f"Warning: CODE_DIR={CODE_DIR} does not exist")
    
    # Try common locations relative to current directory
    candidates = [
        './code',
        './src', 
        './tal',
        './source',
        './sources',
        './cobol',
        './legacy',
        '../code',
        '../src',
        '../tal',
        '.',  # Current directory as last resort
    ]
    
    # Check for directories that exist and contain code files
    code_extensions = {'.tal', '.tacl', '.cbl', '.cob', '.cobol', '.py', '.java'}
    
    for candidate in candidates:
        if os.path.isdir(candidate):
            # For current directory, only use if it has code files
            if candidate == '.':
                has_code = False
                try:
                    for f in os.listdir(candidate):
                        if any(f.lower().endswith(ext) for ext in code_extensions):
                            has_code = True
                            break
                except:
                    pass
                if has_code:
                    return os.path.abspath(candidate)
            else:
                return os.path.abspath(candidate)
    
    return None


def do_grep_search(keywords: list, code_dir: str = None, limit: int = 20, 
                   extensions: list = None) -> list:
    """
    Perform grep-based search across codebase.
    
    Searches:
    1. File contents using grep -ri
    2. File names using find
    
    Args:
        keywords: List of search terms
        code_dir: Directory to search (auto-detected if None)
        limit: Maximum results to return
        extensions: File extensions to search (default: .tal, .cbl, .py, .java)
    
    Returns:
        List of results in same format as do_search()
    """
    import subprocess
    
    if not keywords:
        return []
    
    # Get code directory
    search_dir = code_dir or get_code_directory()
    if not search_dir or not os.path.isdir(search_dir):
        print(f"Grep search: No valid code directory found (tried: {search_dir})")
        return []
    
    # Default extensions for legacy code
    if extensions is None:
        extensions = ['.tal', '.tacl', '.cbl', '.cob', '.cobol', '.py', '.java', '.js', '.ts']
    
    results = []
    seen_files = {}  # file_path -> {matches, score, lines}
    
    # Build grep pattern - OR of all keywords
    # Use word boundaries for better matching
    pattern = '|'.join(re.escape(kw) for kw in keywords if kw)
    if not pattern:
        return []
    
    # Build include patterns for extensions
    include_args = []
    for ext in extensions:
        include_args.extend(['--include', f'*{ext}'])
    
    try:
        # Run grep -rni (recursive, case-insensitive, line numbers)
        cmd = ['grep', '-rniE', pattern] + include_args + [search_dir]
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        # Parse grep output: filename:lineno:content
        for line in proc.stdout.split('\n'):
            if not line or ':' not in line:
                continue
            
            # Parse: /path/to/file.tal:123:matched line content
            parts = line.split(':', 2)
            if len(parts) < 3:
                continue
            
            file_path = parts[0]
            try:
                line_no = int(parts[1])
            except ValueError:
                continue
            matched_content = parts[2] if len(parts) > 2 else ''
            
            # Count keyword matches in this line
            line_lower = matched_content.lower()
            match_count = sum(1 for kw in keywords if kw.lower() in line_lower)
            
            # Aggregate by file
            if file_path not in seen_files:
                seen_files[file_path] = {
                    'matches': 0,
                    'match_lines': [],
                    'first_match_line': line_no,
                }
            
            seen_files[file_path]['matches'] += match_count
            if len(seen_files[file_path]['match_lines']) < 10:  # Keep first 10 matches
                seen_files[file_path]['match_lines'].append({
                    'line_no': line_no,
                    'content': matched_content.strip()[:200]
                })
        
    except subprocess.TimeoutExpired:
        print("Grep search timed out")
    except FileNotFoundError:
        print("grep command not found")
    except Exception as e:
        print(f"Grep search error: {e}")
    
    # Also search file names
    try:
        for kw in keywords:
            if not kw:
                continue
            # Find files with keyword in name
            cmd = ['find', search_dir, '-type', 'f', '-iname', f'*{kw}*']
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            for file_path in proc.stdout.strip().split('\n'):
                if not file_path:
                    continue
                # Check extension
                if not any(file_path.lower().endswith(ext) for ext in extensions):
                    continue
                    
                if file_path not in seen_files:
                    seen_files[file_path] = {
                        'matches': 0,
                        'match_lines': [],
                        'first_match_line': 0,
                        'filename_match': True
                    }
                seen_files[file_path]['filename_match'] = True
                seen_files[file_path]['matches'] += 2  # Boost for filename match
                
    except Exception as e:
        print(f"Find search error: {e}")
    
    # Now read file contents and build results
    for file_path, info in seen_files.items():
        try:
            # Read file content
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
            
            # Extract procedure/function name if possible
            proc_name = extract_procedure_name_from_file(content, file_path)
            
            # Calculate score based on matches
            base_score = min(1.0, info['matches'] / (len(keywords) * 3))
            if info.get('filename_match'):
                base_score = min(1.0, base_score + 0.3)  # Boost filename matches
            
            # Determine language
            lang = detect_language_from_path(file_path)
            
            # Extract relevant section around matches
            if info['match_lines']:
                # Get context around first match
                first_line = info['first_match_line']
                lines = content.split('\n')
                start = max(0, first_line - 20)
                end = min(len(lines), first_line + 30)
                relevant_content = '\n'.join(lines[start:end])
            else:
                relevant_content = content[:2000]
            
            results.append({
                'file_path': file_path,
                'procedure_name': proc_name,
                'function_name': proc_name,
                'text': relevant_content,
                'content_preview': truncate_to_tokens(relevant_content, 500),
                'score': base_score,
                'detected_language': lang,
                'search_method': 'grep',
                'match_count': info['matches'],
                'match_lines': info['match_lines'][:5],  # First 5 match lines
                'filename_match': info.get('filename_match', False),
            })
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    # Sort by score and limit
    results.sort(key=lambda x: -x['score'])
    return results[:limit]


def extract_procedure_name_from_file(content: str, file_path: str) -> str:
    """Extract main procedure/function name from file content."""
    
    # TAL: PROC name
    tal_match = re.search(r'\bPROC\s+(\w+)', content, re.IGNORECASE)
    if tal_match:
        return tal_match.group(1)
    
    # COBOL: PROGRAM-ID
    cobol_match = re.search(r'PROGRAM-ID\.\s*(\w+)', content, re.IGNORECASE)
    if cobol_match:
        return cobol_match.group(1)
    
    # Python: first def or class
    py_match = re.search(r'^(?:def|class)\s+(\w+)', content, re.MULTILINE)
    if py_match:
        return py_match.group(1)
    
    # Java: class name
    java_match = re.search(r'\bclass\s+(\w+)', content)
    if java_match:
        return java_match.group(1)
    
    # Default to filename without extension
    basename = os.path.basename(file_path)
    name, _ = os.path.splitext(basename)
    return name


def detect_language_from_path(file_path: str) -> str:
    """Detect programming language from file path."""
    ext_map = {
        '.tal': 'tal',
        '.tacl': 'tacl',
        '.cbl': 'cobol',
        '.cob': 'cobol',
        '.cobol': 'cobol',
        '.py': 'python',
        '.java': 'java',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.go': 'go',
        '.sh': 'shell',
    }
    
    _, ext = os.path.splitext(file_path.lower())
    return ext_map.get(ext, 'unknown')


def combined_search(query: str, limit: int = 20, use_grep: bool = True) -> list:
    """
    Combine index search with grep search for comprehensive results.
    
    Args:
        query: Search query string
        limit: Maximum results
        use_grep: Whether to also use grep search
    
    Returns:
        Combined and deduplicated results
    """
    results = []
    seen_files = set()
    
    # First: index search (semantic, higher quality)
    index_results = do_search(query, limit=limit)
    for r in index_results:
        file_path = r.get('file_path', '')
        if file_path:
            seen_files.add(os.path.abspath(file_path) if not file_path.startswith('/') else file_path)
        r['search_method'] = r.get('search_method', 'index')
        results.append(r)
    
    # Second: grep search (keyword-based, catches files not in index)
    if use_grep:
        keywords = query.split()
        grep_results = do_grep_search(keywords, limit=limit)
        
        for r in grep_results:
            file_path = r.get('file_path', '')
            abs_path = os.path.abspath(file_path) if file_path and not file_path.startswith('/') else file_path
            
            # Skip if already in index results
            if abs_path in seen_files:
                continue
            
            # Reduce score slightly for grep-only results
            r['score'] = r.get('score', 0.5) * 0.8
            results.append(r)
            seen_files.add(abs_path)
    
    # Re-sort by score
    results.sort(key=lambda x: -x.get('score', 0))
    return results[:limit]


def extract_relevant_sections(code: str, query_terms: list, context_lines: int = 5) -> str:
    """
    Extract code sections containing query terms + surrounding context.
    
    This is a fast, non-LLM approach to compress code context.
    Keeps complete logical blocks around matches.
    
    Args:
        code: Full code content
        query_terms: List of terms to match
        context_lines: Lines to include above/below each match
    
    Returns:
        Extracted relevant sections with ... markers for gaps
    """
    if not code or not query_terms:
        return code
    
    lines = code.split('\n')
    relevant_indices = set()
    
    # Normalize query terms
    query_lower = [t.lower() for t in query_terms if t]
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Check if any query term appears in this line
        for term in query_lower:
            if term in line_lower:
                # Add this line + context above and below
                for j in range(max(0, i - context_lines), min(len(lines), i + context_lines + 1)):
                    relevant_indices.add(j)
                break
    
    if not relevant_indices:
        # No matches - return truncated original
        return truncate_to_tokens(code, 500)
    
    # Build output with gap markers
    sorted_indices = sorted(relevant_indices)
    result_lines = []
    prev_idx = -2
    
    for idx in sorted_indices:
        if idx > prev_idx + 1:
            # Gap - add marker
            if result_lines:
                result_lines.append('    ... [code omitted] ...')
        result_lines.append(lines[idx])
        prev_idx = idx
    
    return '\n'.join(result_lines)


def extract_procedure_by_name(code: str, proc_name: str) -> str:
    """
    Extract a complete procedure from TAL code by name.
    
    Finds PROC declaration and extracts until matching END.
    """
    if not code or not proc_name:
        return ''
    
    lines = code.split('\n')
    proc_pattern = re.compile(
        rf'\b(?:INT|REAL|STRING|FIXED|UNSIGNED)?\s*PROC\s+{re.escape(proc_name)}\b',
        re.IGNORECASE
    )
    
    start_idx = None
    for i, line in enumerate(lines):
        if proc_pattern.search(line):
            start_idx = i
            break
    
    if start_idx is None:
        return ''
    
    # Find END of procedure (track BEGIN/END depth)
    depth = 0
    end_idx = start_idx
    found_begin = False
    
    for i in range(start_idx, len(lines)):
        line_upper = lines[i].upper()
        
        # Count BEGIN/END (simple - doesn't handle strings/comments perfectly)
        if 'BEGIN' in line_upper:
            depth += 1
            found_begin = True
        if 'END' in line_upper:
            depth -= 1
            if found_begin and depth == 0:
                end_idx = i
                break
    
    return '\n'.join(lines[start_idx:end_idx + 1])


def aggregate_chunks_by_file(chunks: list) -> dict:
    """
    Aggregate search result chunks by file path.
    
    Returns dict: file_path -> {
        'procedures': [list of procedure names],
        'content': combined text,
        'score': max score
    }
    """
    files = {}
    
    for chunk in chunks:
        file_path = chunk.get('file_path', chunk.get('filePath', 'unknown'))
        proc_name = chunk.get('procedure_name', chunk.get('function_name', ''))
        text = chunk.get('text', chunk.get('content', ''))
        score = chunk.get('score', chunk.get('combined_score', 0))
        
        if file_path not in files:
            files[file_path] = {
                'procedures': [],
                'contents': [],
                'score': 0
            }
        
        if proc_name and proc_name not in files[file_path]['procedures']:
            files[file_path]['procedures'].append(proc_name)
        
        files[file_path]['contents'].append(text)
        files[file_path]['score'] = max(files[file_path]['score'], score)
    
    # Combine contents
    for file_path in files:
        files[file_path]['content'] = '\n\n'.join(files[file_path]['contents'])
        del files[file_path]['contents']
    
    return files


# ============================================================
# Endpoints
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    code_dir = get_code_directory()
    return jsonify({
        'status': 'ok',
        'mock_mode': MOCK_MODE,
        'indexer_available': INDEXER_AVAILABLE,
        'astra_available': ASTRA_AVAILABLE,
        'index_path': INDEX_PATH,
        'index_exists': os.path.exists(INDEX_PATH) or MOCK_MODE,
        'grep_available': code_dir is not None,
        'code_dir': code_dir,
    })


@app.route('/status', methods=['GET'])
def status():
    """Full status including LLM config info"""
    llm_status = {'available': False, 'model': None, 'provider': None, 'note': 'LLM calls handled by GitHub Copilot in VS Code'}
    
    if LLM_CONFIG:
        default_profile = LLM_CONFIG.get('default_profile', 'default')
        profile = LLM_CONFIG.get('profiles', {}).get(default_profile, {})
        llm_status = {
            'available': True,
            'model': profile.get('model', 'unknown'),
            'provider': default_profile,
            'base_url': profile.get('base_url', 'unknown'),
            'note': 'Config loaded (LLM calls handled by GitHub Copilot)',
        }
    
    index_stats = None
    if MOCK_MODE:
        index_stats = {
            'path': 'MOCK_MODE',
            'chunks': 5,
            'note': 'Using mock data for testing'
        }
    elif INDEXER_AVAILABLE and os.path.exists(INDEX_PATH):
        try:
            vocab_data = load_vocabulary(VOCAB_PATH)
            pipeline = IndexingPipeline(
                vocabulary_data=vocab_data,
                embedder_type=None
            )
            pipeline.load(INDEX_PATH)
            stats = pipeline.get_statistics()
            index_stats = {
                'path': INDEX_PATH,
                'chunks': stats.get('pipeline', {}).get('total_chunks', 'unknown'),
            }
        except Exception as e:
            index_stats = {'path': INDEX_PATH, 'error': str(e)}
    
    # Grep search info
    code_dir = get_code_directory()
    grep_info = {
        'available': code_dir is not None,
        'code_dir': code_dir,
        'note': 'Grep search works without index',
    }
    
    return jsonify({
        'status': 'ok',
        'mock_mode': MOCK_MODE,
        'indexer_available': INDEXER_AVAILABLE or MOCK_MODE,
        'astra_available': ASTRA_AVAILABLE,
        'astra_translation': ASTRA_TRANSLATION_AVAILABLE,
        'astra_docs': ASTRA_DOCS_AVAILABLE,
        'grep_available': code_dir is not None,  # Top-level for easy access
        'code_dir': code_dir,  # Top-level for easy access
        'llm': llm_status,
        'index': index_stats,
        'grep': grep_info,
    })


@app.route('/vocabulary', methods=['GET'])
def vocabulary_endpoint():
    """
    Get domain vocabulary for keyword extraction.
    
    Returns:
        {
            "entries": [...],
            "version": "1.0",
            "_cached_at": 1234567890
        }
    """
    vocab = get_vocabulary_cached()
    
    # Add summary stats
    entries = vocab.get('entries', [])
    vocab['_summary'] = {
        'total_entries': len(entries),
        'has_tf_idf': any('_tf_idf_score' in e for e in entries),
    }
    
    return jsonify(vocab)


@app.route('/knowledge-graph', methods=['GET'])
def knowledge_graph_endpoint():
    """
    Get knowledge graph for query expansion and related concepts.
    
    Returns:
        {
            "nodes": [...],
            "edges": [...],
            "statistics": {...}
        }
    """
    kg = get_knowledge_graph_cached()
    return jsonify(kg)


@app.route('/expand-query', methods=['POST'])
def expand_query_endpoint():
    """
    Expand a query using knowledge graph and vocabulary.
    
    Request:
        {"query": "wire transfer validation"}
    
    Returns:
        {
            "original": "wire transfer validation",
            "keywords": ["wire", "transfer", "validation"],
            "expanded": ["wire", "transfer", "validation", "fedwire", "swift"],
            "related_concepts": [
                {"term": "fedwire", "relationship": "co_occurs_with", "score": 0.8},
                ...
            ]
        }
    """
    data = request.get_json() or {}
    query = data.get('query', '')
    max_expansions = data.get('max_expansions', 5)
    
    if not query:
        return jsonify({'error': 'query required'}), 400
    
    result = expand_query_with_knowledge(query, max_expansions)
    return jsonify(result)


@app.route('/generate-docs', methods=['POST'])
def generate_docs_endpoint():
    """
    Gather comprehensive context for documenting a feature from indexed codebase.
    
    This is for documenting features by NAME - the server searches its index
    to find all relevant code files. For documenting user-attached files,
    the extension handles that client-side.
    
    Request:
        {
            "feature": "wire transfer validation",
            "max_files": 15,
            "include_related": true,
            "compress_context": true,      # Use smart extraction vs truncation
            "aggregate_by_file": false,    # Group chunks by file
            "max_tokens_per_chunk": 1000   # Token limit per chunk
        }
    
    Returns:
        {
            "feature": "wire transfer validation",
            "files": [
                {
                    "path": "src/wire_transfer.tal",
                    "procedure_name": "VALIDATE_WIRE",
                    "content": "PROC VALIDATE_WIRE...",
                    "type": "procedure",
                    "relevance_score": 0.95,
                    "matched_terms": ["wire", "transfer", "validate"],
                    "compressed": true
                },
                ...
            ],
            "related_concepts": [
                {"term": "OFAC", "relationship": "co_occurs_with"},
                {"term": "beneficiary", "relationship": "contains"}
            ],
            "statistics": {
                "total_files": 12,
                "procedures": 5,
                "structures": 3,
                "total_lines": 450
            },
            "suggested_sections": [
                "Wire Transfer Processing",
                "Validation Rules", 
                "OFAC Screening"
            ]
        }
    """
    # Allow mock mode for testing
    if not INDEXER_AVAILABLE and not MOCK_MODE:
        return jsonify({
            'error': 'unified_indexer not available',
            'reason': 'The unified_indexer package could not be imported',
            'hint': 'Make sure unified_indexer/ is in the Python path',
            'fix': [
                'cd /path/to/uvidx2',
                'python api_server.py',
                '# OR: export PYTHONPATH=/path/to/uvidx2:$PYTHONPATH',
                '# OR for testing: export MOCK_MODE=1'
            ]
        }), 503
    
    if not os.path.exists(INDEX_PATH) and not MOCK_MODE:
        return jsonify({
            'error': f'Index not found',
            'index_path': INDEX_PATH,
            'reason': 'No index has been built at the configured path',
            'hint': 'Build index first before using /generate-docs',
            'fix': [
                f'python build_index.py --tal-dir ./your_code --output {INDEX_PATH}',
                '# OR: export INDEX_PATH=/path/to/existing/index',
                '# OR for testing: export MOCK_MODE=1'
            ]
        }), 503
    
    data = request.get_json() or {}
    feature = data.get('feature', '')
    max_files = min(data.get('max_files', 15), 30)  # Cap at 30
    include_related = data.get('include_related', True)
    compress_context = data.get('compress_context', True)  # Smart extraction
    aggregate_by_file = data.get('aggregate_by_file', False)
    max_tokens_per_chunk = data.get('max_tokens_per_chunk', 1000)
    
    if not feature:
        return jsonify({'error': 'feature name required'}), 400
    
    # Check if grep search is enabled
    use_grep = data.get('use_grep', True)  # Default: also use grep
    code_dir = data.get('code_dir', None)  # Optional: override code directory
    
    try:
        # Step 1: Expand query using knowledge graph
        expansion = expand_query_with_knowledge(feature, max_expansions=8)
        search_terms = expansion.get('expanded', [feature])
        related_concepts = expansion.get('related_concepts', [])
        
        # Step 2: Search for main feature files (index + grep)
        main_query = ' '.join(search_terms[:6])  # Top 6 terms
        main_results = do_search(main_query, limit=max_files * 2)  # Get more for dedup
        
        # Mark index results
        for r in main_results:
            r['search_method'] = r.get('search_method', 'index')
        
        # Step 2b: Also search with grep for files not in index
        grep_results = []
        if use_grep:
            keywords = feature.split() + search_terms[:4]
            keywords = list(set(k.lower() for k in keywords if len(k) > 2))
            
            grep_hits = do_grep_search(keywords, code_dir=code_dir, limit=max_files)
            
            # Deduplicate against index results
            seen_paths = set()
            for r in main_results:
                fp = r.get('file_path', '')
                if fp:
                    seen_paths.add(os.path.abspath(fp) if not fp.startswith('/') else fp)
            
            for r in grep_hits:
                fp = r.get('file_path', '')
                abs_path = os.path.abspath(fp) if fp and not fp.startswith('/') else fp
                
                if abs_path not in seen_paths:
                    # Slightly lower score for grep-only results
                    r['score'] = r.get('score', 0.5) * 0.85
                    r['search_method'] = 'grep'
                    grep_results.append(r)
                    seen_paths.add(abs_path)
        
        # Combine index + grep results
        main_results = main_results + grep_results
        main_results.sort(key=lambda x: -x.get('score', 0))
        
        # Step 3: Search for related concept files (if enabled)
        related_results = []
        if include_related and related_concepts:
            seen_paths = set(r.get('file_path', r.get('filePath', '')) for r in main_results)
            
            for concept in related_concepts[:3]:  # Top 3 related concepts
                concept_results = do_search(concept['term'], limit=5)
                for r in concept_results:
                    path = r.get('file_path', r.get('filePath', ''))
                    if path and path not in seen_paths:
                        seen_paths.add(path)
                        r['_matched_concept'] = concept['term']
                        related_results.append(r)
        
        # Step 4: Combine results
        all_results = main_results + related_results
        
        # Step 5: Optionally aggregate by file
        if aggregate_by_file:
            aggregated = aggregate_chunks_by_file(all_results)
            # Convert back to list format, sorted by score
            all_results = []
            for file_path, file_data in sorted(aggregated.items(), key=lambda x: -x[1]['score']):
                all_results.append({
                    'file_path': file_path,
                    'procedure_name': ', '.join(file_data['procedures'][:3]),
                    'text': file_data['content'],
                    'score': file_data['score'],
                    '_aggregated': True,
                    '_procedure_count': len(file_data['procedures'])
                })
        
        # Limit results
        all_results = all_results[:max_files]
        
        # Step 6: Process each result with smart compression
        kg = get_knowledge_graph_cached()
        files_output = []
        stats = {
            'total_chunks': len(all_results),
            'procedures': 0,
            'structures': 0,
            'constants': 0,
            'total_content_length': 0,
            'compressed_length': 0
        }
        
        for r in all_results:
            file_path = r.get('file_path', r.get('filePath', 'unknown'))
            content = r.get('text', r.get('content', ''))
            proc_name = r.get('procedure_name', r.get('function_name', ''))
            
            # Determine type from content
            file_type = 'code'
            if 'PROC ' in content or 'SUBPROC ' in content:
                file_type = 'procedure'
                stats['procedures'] += 1
            elif 'STRUCT ' in content:
                file_type = 'structure'
                stats['structures'] += 1
            elif 'DEFINE ' in content or 'LITERAL ' in content:
                file_type = 'constant'
                stats['constants'] += 1
            
            # Find which search terms matched
            content_lower = content.lower()
            matched_terms = [t for t in search_terms if t.lower() in content_lower]
            
            # Apply smart compression or simple truncation
            original_length = len(content)
            
            if compress_context and matched_terms:
                # Smart extraction: keep sections with query terms + context
                compressed_content = extract_relevant_sections(
                    content, 
                    search_terms,  # Use all search terms for extraction
                    context_lines=8  # Keep 8 lines above/below matches
                )
                # Still apply token limit
                final_content = truncate_to_tokens(compressed_content, max_tokens_per_chunk)
            else:
                # Simple truncation
                final_content = truncate_to_tokens(content, max_tokens_per_chunk)
            
            stats['total_content_length'] += original_length
            stats['compressed_length'] += len(final_content)
            
            files_output.append({
                'path': file_path,
                'procedure_name': proc_name,
                'content': final_content,
                'type': file_type,
                'relevance_score': r.get('score', r.get('combined_score', 0.5)),
                'matched_terms': matched_terms[:5],
                'matched_concept': r.get('_matched_concept'),
                'compressed': compress_context and len(final_content) < original_length,
                'original_tokens': len(content.split()),
                'aggregated': r.get('_aggregated', False),
                'search_method': r.get('search_method', 'index'),
                'filename_match': r.get('filename_match', False),
            })
        
        # Track which search methods were used
        search_methods_used = list(set(f.get('search_method', 'index') for f in files_output))
        index_count = sum(1 for f in files_output if f.get('search_method') == 'index')
        grep_count = sum(1 for f in files_output if f.get('search_method') == 'grep')
        
        # Step 7: Suggest documentation sections based on what we found
        suggested_sections = []
        if stats['procedures'] > 0:
            suggested_sections.append('Key Procedures')
        if stats['structures'] > 0:
            suggested_sections.append('Data Structures')
        
        # Add sections from related concepts
        for concept in related_concepts[:3]:
            term = concept.get('term', '')
            if term:
                suggested_sections.append(f'{term.replace("_", " ").title()}')
        
        # Calculate compression ratio
        if stats['total_content_length'] > 0:
            stats['compression_ratio'] = round(
                stats['compressed_length'] / stats['total_content_length'], 2
            )
        
        # Add search method stats
        stats['search_methods'] = search_methods_used
        stats['index_results'] = index_count
        stats['grep_results'] = grep_count
        
        return jsonify({
            'feature': feature,
            'files': files_output,
            'related_concepts': related_concepts[:8],
            'search_terms_used': search_terms,
            'statistics': stats,
            'suggested_sections': suggested_sections[:6],
            'search_methods': search_methods_used,
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def search_endpoint():
    """
    Search indexed codebase with optional grep fallback.
    
    Request body:
        {
            "query": "wire transfer validation",
            "limit": 20,
            "use_grep": true,      // Also search with grep (default: true)
            "grep_only": false,    // Only use grep, skip index (default: false)
            "code_dir": "./code"   // Override code directory for grep
        }
    
    Returns:
        {
            "results": [...],
            "count": 15,
            "query": "wire transfer",
            "search_methods": ["index", "grep"]
        }
    """
    data = request.get_json() or {}
    query = data.get('query', '')
    limit = data.get('limit', 20)
    use_grep = data.get('use_grep', True)
    grep_only = data.get('grep_only', False)
    code_dir = data.get('code_dir', None)
    
    if not query:
        return jsonify({'error': 'query required'}), 400
    
    try:
        results = []
        search_methods = []
        
        # Index search (unless grep_only)
        if not grep_only:
            if INDEXER_AVAILABLE and os.path.exists(INDEX_PATH):
                index_results = do_search(query, limit)
                for r in index_results:
                    r['search_method'] = 'index'
                results.extend(index_results)
                search_methods.append('index')
            elif MOCK_MODE:
                mock_results = do_search(query, limit)
                for r in mock_results:
                    r['search_method'] = 'mock'
                results.extend(mock_results)
                search_methods.append('mock')
        
        # Grep search
        if use_grep or grep_only:
            keywords = query.split()
            grep_results = do_grep_search(keywords, code_dir=code_dir, limit=limit)
            
            # Deduplicate by file path
            seen_files = {os.path.abspath(r.get('file_path', '')) for r in results if r.get('file_path')}
            
            for r in grep_results:
                file_path = r.get('file_path', '')
                abs_path = os.path.abspath(file_path) if file_path else ''
                
                if abs_path not in seen_files:
                    # Reduce score for grep-only results
                    if not grep_only:
                        r['score'] = r.get('score', 0.5) * 0.8
                    results.append(r)
                    seen_files.add(abs_path)
            
            if grep_results:
                search_methods.append('grep')
        
        # Sort by score
        results.sort(key=lambda x: -x.get('score', 0))
        results = results[:limit]
        
        return jsonify({
            'results': results,
            'count': len(results),
            'query': query,
            'search_methods': search_methods,
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
        }), 500


@app.route('/grep', methods=['POST'])
def grep_endpoint():
    """
    Direct grep search across codebase (no index required).
    
    Searches file contents with grep -ri and filenames with find.
    
    Request body:
        {
            "keywords": ["wire", "transfer"],   // OR: "query": "wire transfer"
            "code_dir": "./code",               // Optional: override code directory
            "limit": 20,                        // Max results
            "extensions": [".tal", ".cbl"]      // Optional: filter by extension
        }
    
    Returns:
        {
            "results": [
                {
                    "file_path": "/code/wire_transfer.tal",
                    "procedure_name": "VALIDATE_WIRE",
                    "content_preview": "PROC VALIDATE_WIRE...",
                    "score": 0.85,
                    "match_count": 5,
                    "match_lines": [
                        {"line_no": 42, "content": "IF wire_amount > MAX..."}
                    ],
                    "filename_match": true
                }
            ],
            "count": 12,
            "keywords": ["wire", "transfer"],
            "code_dir": "./code"
        }
    """
    data = request.get_json() or {}
    
    # Accept either keywords list or query string
    keywords = data.get('keywords', [])
    if not keywords and data.get('query'):
        keywords = data.get('query', '').split()
    
    if not keywords:
        return jsonify({'error': 'keywords or query required'}), 400
    
    code_dir = data.get('code_dir', None)
    limit = data.get('limit', 20)
    extensions = data.get('extensions', None)
    
    # Validate code_dir if provided
    search_dir = code_dir or get_code_directory()
    if not search_dir:
        return jsonify({
            'error': 'No code directory found',
            'hint': 'Set CODE_DIR environment variable or provide code_dir in request',
            'tried': ['./code', './src', './tal', './source']
        }), 400
    
    if not os.path.isdir(search_dir):
        return jsonify({
            'error': f'Code directory not found: {search_dir}',
            'hint': 'Verify the path exists and is accessible'
        }), 400
    
    try:
        results = do_grep_search(
            keywords=keywords,
            code_dir=search_dir,
            limit=limit,
            extensions=extensions
        )
        
        return jsonify({
            'results': results,
            'count': len(results),
            'keywords': keywords,
            'code_dir': search_dir,
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
        }), 500


@app.route('/translate', methods=['POST'])
def translate_endpoint():
    """Parse query, search, build LLM prompt"""
    if not ASTRA_TRANSLATION_AVAILABLE:
        return jsonify({
            'error': 'astra.translation not available',
            'hint': 'Copy astra/translation/ to your project or install astra package'
        }), 503
    
    data = request.get_json() or {}
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'query required'}), 400
    
    try:
        parsed = engine.parse_query(query)
        search_params = engine.get_search_params(parsed)
        
        results = []
        if INDEXER_AVAILABLE and os.path.exists(INDEX_PATH):
            results = do_search(search_params.get('query', ''), 15)
        
        context = engine.build_context(parsed, results)
        prompt = engine.build_prompt(parsed, context)
        
        return jsonify({
            'parsed': {
                'intent': parsed.intent.name,
                'target_language': parsed.target_language,
                'source_languages': parsed.source_languages,
                'target_name': parsed.target_name,
                'target_type': parsed.target_type,
                'framework': parsed.framework,
                'search_terms': parsed.search_terms,
                'include_tests': parsed.include_tests,
                'include_docs': parsed.include_docs,
            },
            'search_params': search_params,
            'results': results,
            'results_count': len(results),
            'context': {
                'detected_languages': context.detected_languages,
                'detected_patterns': context.detected_patterns,
                'domain_concepts': context.domain_concepts,
            },
            'prompt': prompt,
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/enhance', methods=['POST'])
def enhance_endpoint():
    """Enhance parsed code with translation metadata"""
    if not ASTRA_TRANSLATION_AVAILABLE:
        return jsonify({
            'error': 'astra.translation not available',
            'hint': 'Copy astra/translation/ to your project or install astra package'
        }), 503
    
    data = request.get_json() or {}
    parsed_data = data.get('parsed_data', {})
    source_language = data.get('source_language', 'tal')
    
    if not parsed_data:
        return jsonify({'error': 'parsed_data required'}), 400
    
    try:
        enhanced = enhancer.enhance(parsed_data, source_language)
        return jsonify({
            'enhanced': enhanced,
            'source_language': source_language,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate-docs-files', methods=['POST'])
def generate_docs_files_endpoint():
    """Generate documentation files with Mermaid diagrams (uses astra.docs)"""
    if not ASTRA_DOCS_AVAILABLE:
        return jsonify({
            'error': 'astra.docs not available',
            'hint': 'Copy astra/docs/ to your project or install astra package'
        }), 503
    
    data = request.get_json() or {}
    parsed_data = data.get('parsed_data', {})
    project_name = data.get('project_name', 'Project')
    source_language = data.get('source_language', 'unknown')
    output_dir = data.get('output_dir', DOCS_OUTPUT_DIR)
    
    if not parsed_data:
        return jsonify({'error': 'parsed_data required'}), 400
    
    try:
        gen = DocGenerator(project_name)
        docs = gen.generate(parsed_data, source_language)
        output_path = gen.save(docs, output_dir)
        
        files = [f for f in os.listdir(output_path) if f.endswith('.md')]
        
        return jsonify({
            'output_dir': output_path,
            'files': files,
            'preview': docs.render()[:2000],
            'sections': [s.title for s in docs.sections],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate-diagram', methods=['POST'])
def generate_diagram_endpoint():
    """Generate a Mermaid diagram"""
    if not ASTRA_DOCS_AVAILABLE:
        return jsonify({
            'error': 'astra.docs not available',
            'hint': 'Copy astra/docs/ to your project or install astra package'
        }), 503
    
    data = request.get_json() or {}
    diagram_type = data.get('type', 'call_graph')
    diagram_data = data.get('data', {})
    title = data.get('title', 'Diagram')
    
    try:
        if diagram_type == 'call_graph':
            diagram = diagram_gen.call_graph(diagram_data, title)
        elif diagram_type == 'class_diagram':
            diagram = diagram_gen.class_diagram(diagram_data, title)
        elif diagram_type == 'sequence':
            diagram = diagram_gen.sequence_diagram(diagram_data, title)
        elif diagram_type == 'flowchart':
            steps = diagram_data if isinstance(diagram_data, list) else diagram_data.get('steps', [])
            diagram = diagram_gen.flowchart(steps, title)
        elif diagram_type == 'module':
            diagram = diagram_gen.module_diagram(diagram_data, title)
        else:
            return jsonify({'error': f'Unknown diagram type: {diagram_type}'}), 400
        
        return jsonify({
            'mermaid': diagram.render(),
            'type': diagram_type,
            'title': title,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/parse', methods=['POST'])
def parse_endpoint():
    """Parse TAL/COBOL code"""
    data = request.get_json() or {}
    code = data.get('code', '')
    language = data.get('language', 'tal').lower()
    
    if not code:
        return jsonify({'error': 'code required'}), 400
    
    try:
        if language == 'tal' and tal_parse:
            result = tal_parse(code)
            # Convert ParseResult to dict
            parsed = {
                'procedures': [vars(p) if hasattr(p, '__dict__') else p for p in getattr(result, 'procedures', [])],
                'structs': [vars(s) if hasattr(s, '__dict__') else s for s in getattr(result, 'structs', [])],
                'defines': getattr(result, 'defines', {}),
                'call_graph': getattr(result, 'call_graph', {}),
            }
            return jsonify({'parsed': parsed, 'language': language})
        else:
            return jsonify({'error': f'Parser not available for: {language}'}), 400
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================
# LLM Chat Endpoint
# ============================================================

# LLM Configuration (set via environment variables)
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

def call_anthropic(prompt: str, max_tokens: int = 4096) -> dict:
    """Call Anthropic Claude API."""
    import urllib.request
    import urllib.error
    
    if not ANTHROPIC_API_KEY:
        return {'error': 'ANTHROPIC_API_KEY not configured'}
    
    data = json.dumps({
        'model': 'claude-sonnet-4-20250514',
        'max_tokens': max_tokens,
        'messages': [{'role': 'user', 'content': prompt}]
    }).encode('utf-8')
    
    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data=data,
        headers={
            'Content-Type': 'application/json',
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01'
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
            text = result.get('content', [{}])[0].get('text', '')
            return {
                'success': True,
                'response': text,
                'model': 'claude-sonnet-4-20250514',
                'provider': 'anthropic'
            }
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else str(e)
        return {'error': f'Anthropic API error: {e.code} - {error_body}'}
    except Exception as e:
        return {'error': f'Anthropic API error: {str(e)}'}


def call_openai(prompt: str, max_tokens: int = 4096) -> dict:
    """Call OpenAI API."""
    import urllib.request
    import urllib.error
    
    if not OPENAI_API_KEY:
        return {'error': 'OPENAI_API_KEY not configured'}
    
    data = json.dumps({
        'model': 'gpt-4o',
        'max_tokens': max_tokens,
        'messages': [{'role': 'user', 'content': prompt}]
    }).encode('utf-8')
    
    req = urllib.request.Request(
        'https://api.openai.com/v1/chat/completions',
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
            text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            return {
                'success': True,
                'response': text,
                'model': 'gpt-4o',
                'provider': 'openai'
            }
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else str(e)
        return {'error': f'OpenAI API error: {e.code} - {error_body}'}
    except Exception as e:
        return {'error': f'OpenAI API error: {str(e)}'}


@app.route('/llm/chat', methods=['POST'])
def llm_chat_endpoint():
    """
    LLM Chat endpoint - fallback when GitHub Copilot is unavailable.
    
    POST /llm/chat
    {
        "prompt": "Explain this code...",
        "context": "optional additional context",
        "max_tokens": 4096
    }
    
    Returns:
    {
        "success": true,
        "response": "The code does...",
        "model": "claude-sonnet-4-20250514",
        "provider": "anthropic"
    }
    
    Environment:
        ANTHROPIC_API_KEY - Anthropic Claude API key (preferred)
        OPENAI_API_KEY - OpenAI API key (fallback)
    """
    data = request.get_json() or {}
    prompt = data.get('prompt', '')
    context = data.get('context', '')
    max_tokens = data.get('max_tokens', 4096)
    
    if not prompt:
        return jsonify({'error': 'prompt is required'}), 400
    
    # Combine prompt and context
    full_prompt = prompt
    if context:
        full_prompt = f"{context}\n\n---\n\n{prompt}"
    
    # Try Anthropic first, then OpenAI
    if ANTHROPIC_API_KEY:
        result = call_anthropic(full_prompt, max_tokens)
        if result.get('success'):
            return jsonify(result)
        print(f"Anthropic failed: {result.get('error')}")
    
    if OPENAI_API_KEY:
        result = call_openai(full_prompt, max_tokens)
        if result.get('success'):
            return jsonify(result)
        print(f"OpenAI failed: {result.get('error')}")
    
    # No LLM configured
    return jsonify({
        'error': 'No LLM configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.',
        'anthropic_configured': bool(ANTHROPIC_API_KEY),
        'openai_configured': bool(OPENAI_API_KEY)
    }), 503


@app.route('/llm/status', methods=['GET'])
def llm_status_endpoint():
    """Check LLM configuration status."""
    return jsonify({
        'anthropic_configured': bool(ANTHROPIC_API_KEY),
        'openai_configured': bool(OPENAI_API_KEY),
        'available': bool(ANTHROPIC_API_KEY or OPENAI_API_KEY)
    })


@app.route('/llm', methods=['POST'])
def llm_simple_endpoint():
    """
    Simple LLM endpoint - alias for /llm/chat.
    
    POST /llm
    {
        "prompt": "Your prompt here"
    }
    """
    return llm_chat_endpoint()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    index_status = '✓ Found' if os.path.exists(INDEX_PATH) else '✗ Not found'
    vocab_status = '✓ Found' if os.path.exists(VOCAB_PATH) else '✗ Not found'
    kg_status = '✓ Found' if os.path.exists(KNOWLEDGE_GRAPH_PATH) else '✗ Not found'
    code_dir = get_code_directory()
    code_dir_status = f'✓ {code_dir}' if code_dir else '✗ Not found (set CODE_DIR)'
    llm_status = '✓ Anthropic' if ANTHROPIC_API_KEY else ('✓ OpenAI' if OPENAI_API_KEY else '✗ Not configured')
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║              Astra API Server v2.0                        ║
║      Copilot first, API server LLM as fallback            ║
╠═══════════════════════════════════════════════════════════╣
║  Endpoints:                                               ║
║    POST /search          - Search index + grep            ║
║    POST /grep            - Grep search only (no index)    ║
║    POST /llm/chat        - LLM chat (Copilot fallback)    ║
║    GET  /llm/status      - LLM configuration status       ║
║    POST /expand-query    - Expand query with knowledge    ║
║    POST /generate-docs   - Gather context for feature doc ║
║    GET  /vocabulary      - Get domain vocabulary          ║
║    GET  /knowledge-graph - Get knowledge graph            ║
║    POST /translate       - Build translation prompt       ║
║    POST /generate-diagram - Generate Mermaid diagram      ║
║    POST /parse           - Parse source code              ║
║    GET  /health          - Health check                   ║
║    GET  /status          - Full status info               ║
╠═══════════════════════════════════════════════════════════╣
║  Status:                                                  ║
║    unified_indexer: {'✓ Available' if INDEXER_AVAILABLE else '✗ Not found':<36} ║
║    astra:           {'✓ Available' if ASTRA_AVAILABLE else '✗ Not found':<36} ║
║    llm:             {llm_status:<36} ║
║    index:           {index_status:<36} ║
║    code_dir:        {code_dir_status[:36]:<36} ║
║    vocab:           {vocab_status:<36} ║
║    knowledge_graph: {kg_status:<36} ║
║    port:            {port:<36} ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    if not ANTHROPIC_API_KEY and not OPENAI_API_KEY:
        print("⚠️  No LLM API key configured for fallback")
        print("   Set: export ANTHROPIC_API_KEY=sk-ant-...")
        print("   Or:  export OPENAI_API_KEY=sk-...")
        print()
    
    if not os.path.exists(INDEX_PATH):
        print(f"⚠️  Index not found at: {INDEX_PATH}")
        print(f"   Build with: python build_index.py --output {INDEX_PATH}")
        print(f"   Or use grep search: POST /grep {'{'}\"keywords\": [\"wire\", \"transfer\"]{'}'}")
        print()
    
    if not code_dir:
        print(f"⚠️  No code directory found for grep search")
        print(f"   Set CODE_DIR environment variable or create ./code, ./src, or ./tal directory")
        print()
    
    app.run(host='0.0.0.0', port=port, debug=debug)
