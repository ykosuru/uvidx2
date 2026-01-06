#!/usr/bin/env python3
"""
Index Builder - Indexes PDF documents and code files with knowledge graph support

================================================================================
OVERVIEW
================================================================================

Builds a searchable index from documents and code. Optionally integrates
knowledge graph for enhanced search capabilities:
- TF-IDF weighted vocabulary (distinctive terms get higher weight)
- Embedded knowledge graph (enables automatic query expansion)
- Pre-computed relationships (faster search-time lookups)

================================================================================
ARCHITECTURE
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                           INDEX BUILD PIPELINE                               │
│                                                                             │
│  INPUTS                          PROCESSING                    OUTPUT       │
│  ──────                          ──────────                    ──────       │
│                                                                             │
│  ┌──────────────┐                                                           │
│  │ vocabulary   │──┐                                                        │
│  │ .json        │  │         ┌─────────────────────┐                       │
│  └──────────────┘  │         │                     │                       │
│                    ├────────▶│  IndexingPipeline   │                       │
│  ┌──────────────┐  │         │                     │      ┌─────────────┐  │
│  │ knowledge    │──┤         │  • Vocabulary       │      │ ./my_index/ │  │
│  │ _graph.json  │  │         │  • Embedder         │─────▶│             │  │
│  └──────────────┘  │         │  • Concept Index    │      │ index.pkl   │  │
│                    │         │  • Vector Index     │      │ meta.json   │  │
│  ┌──────────────┐  │         │                     │      │ kg.json     │  │
│  │ PDF/Code     │──┘         └─────────────────────┘      └─────────────┘  │
│  │ Files        │                     │                                     │
│  └──────────────┘                     │                                     │
│                                       ▼                                     │
│                            ┌─────────────────────┐                         │
│                            │ For each file:      │                         │
│                            │ 1. Parse content    │                         │
│                            │ 2. Create chunks    │                         │
│                            │ 3. Extract concepts │                         │
│                            │ 4. Generate vectors │                         │
│                            │ 5. Add to index     │                         │
│                            └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
KNOWLEDGE GRAPH INTEGRATION
================================================================================

When --knowledge-graph is provided:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  knowledge_graph.json                                                       │
│  ┌─────────────────┐                                                        │
│  │ nodes:          │                                                        │
│  │   - id: "ofac"  │                                                        │
│  │     tf_idf: 2.5 │───────┐                                               │
│  │   - id: "sdn"   │       │                                               │
│  │     tf_idf: 3.0 │       │                                               │
│  │ edges:          │       │                                               │
│  │   - ofac→sdn    │───────┼───┐                                           │
│  └─────────────────┘       │   │                                           │
│                            │   │                                           │
│                            ▼   │                                           │
│                    ┌───────────────────┐                                   │
│                    │ apply_tfidf_      │                                   │
│                    │ weights()         │                                   │
│                    │                   │                                   │
│                    │ Vocab entry with  │                                   │
│                    │ "OFAC" keyword    │                                   │
│                    │ gets weight 1.3   │                                   │
│                    └───────────────────┘                                   │
│                            │   │                                           │
│                            │   ▼                                           │
│                            │  ┌───────────────────┐                        │
│                            │  │ extract_          │                        │
│                            │  │ relationships_    │                        │
│                            │  │ for_expansion()   │                        │
│                            │  │                   │                        │
│                            │  │ Creates compact   │                        │
│                            │  │ lookup map:       │                        │
│                            │  │ ofac → [sdn,      │                        │
│                            │  │         sanctions]│                        │
│                            │  └───────────────────┘                        │
│                            │           │                                   │
│                            ▼           ▼                                   │
│                    ┌───────────────────────────┐                           │
│                    │      Index Directory      │                           │
│                    │  ┌─────────────────────┐  │                           │
│                    │  │ knowledge_graph.json│  │ ← Full KG for search     │
│                    │  │ expansion_map.json  │  │ ← Fast expansion lookup  │
│                    │  │ index_meta.json     │  │ ← Flags: kg=true         │
│                    │  └─────────────────────┘  │                           │
│                    └───────────────────────────┘                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
TF-IDF WEIGHTING
================================================================================

Vocabulary entries are weighted based on TF-IDF scores from knowledge graph:

    Original vocabulary entry:
    {
        "keywords": "OFAC,SDN,sanctions",
        "metadata": "compliance",
        ...
    }

    After weighting (with KG):
    {
        "keywords": "OFAC,SDN,sanctions",
        "metadata": "compliance",
        "_tfidf_weight": 1.35,    ← Added: normalized weight (0.5 - 1.5)
        ...
    }

Weight calculation:
    1. Look up TF-IDF score for each keyword in KG nodes
    2. Average the scores for the entry
    3. Normalize to 0.5 - 1.5 range (1.0 = average importance)

Effect on search:
    - High weight entries (distinctive terms) rank higher
    - Low weight entries (common terms) rank lower

================================================================================
OUTPUT FILES
================================================================================

./my_index/
├── index.pkl           Main index (chunks, vectors, concepts)
├── index_meta.json     Build metadata:
│                         - vocabulary_file: "vocabulary.json"
│                         - embedder_type: "hash"
│                         - knowledge_graph: true
│                         - tfidf_weighted: true
│                         - stats: {pdf: {...}, tal: {...}}
├── knowledge_graph.json  Full KG (copied if provided)
└── expansion_map.json    Compact term→[related terms] lookup

================================================================================
USAGE
================================================================================

Basic:
    python build_index.py --pdf-dir ./docs --tal-dir ./code --output ./my_index

With Knowledge Graph:
    python build_index.py --pdf-dir ./docs --tal-dir ./code --output ./my_index \\
        --knowledge-graph ./knowledge_graph.json

Full Pipeline:
    # 1. Extract knowledge from docs and code
    python knowledge_extractor.py --docs ./docs --code ./code \\
        --output vocabulary.json --graph knowledge_graph.json --stats stats.json
    
    # 2. Build index with knowledge graph
    python build_index.py --pdf-dir ./docs --tal-dir ./code --output ./my_index \\
        --vocab vocabulary.json --knowledge-graph knowledge_graph.json
    
    # 3. Search (knowledge graph auto-loaded from index)
    python search_index.py --index ./my_index --query "OFAC screening" --expand-query

================================================================================
ARGUMENTS
================================================================================

    --pdf-dir           Directory containing PDF documents
    --tal-dir           Directory containing TAL code
    --code-dir          Directory containing other code (C, Java, Python, etc.)
    --output            Output directory for index (required)
    --vocab             Path to vocabulary JSON file (default: keywords.json)
    --recursive         Search directories recursively (default: True)
    --embedder          Embedder type: hash, hybrid, tfidf, domain, payment, etc.
    
    Knowledge Graph Options:
    --knowledge-graph   Path to knowledge_graph.json (from knowledge_extractor)
    --tfidf-weight      Apply TF-IDF weights to vocabulary (default: True with KG)
    --no-tfidf-weight   Don't apply TF-IDF weighting

When knowledge graph is provided:
    1. Vocabulary entries are weighted by TF-IDF scores
    2. Knowledge graph is copied to index directory
    3. Search auto-loads the graph (no --knowledge-graph needed at search time)
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_indexer import IndexingPipeline, SourceType

# Default vocabulary file - located in same directory as this script
DEFAULT_KEYWORDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keywords.json")


# =============================================================================
# KNOWLEDGE GRAPH INTEGRATION
# 
# These functions handle loading the knowledge graph and applying its
# TF-IDF scores to weight vocabulary entries for better search relevance.
# =============================================================================

def load_knowledge_graph(graph_path: str) -> Optional[Dict]:
    """
    Load knowledge graph JSON for vocabulary weighting.
    
    The knowledge graph contains:
    - nodes: Terms with TF-IDF scores, types, source files
    - edges: Relationships (co_occurs_with, implements, contains)
    - statistics: Summary counts
    
    Args:
        graph_path: Path to knowledge_graph.json
        
    Returns:
        Dict with nodes, edges, statistics or None if not found
    """
    # Validate path exists
    if not graph_path or not os.path.exists(graph_path):
        return None
    
    try:
        with open(graph_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading knowledge graph: {e}")
        return None


def apply_tfidf_weights(vocab_data: List[Dict], 
                        knowledge_graph: Dict) -> List[Dict]:
    """
    Apply TF-IDF weights to vocabulary entries based on knowledge graph.
    
    This enhances search by giving higher weight to vocabulary entries
    that contain distinctive terms (high TF-IDF = appears in few docs).
    
    Weight Range: 0.5 - 1.5
        0.5 = very common terms (low TF-IDF)
        1.0 = average terms
        1.5 = very distinctive terms (high TF-IDF)
    
    Args:
        vocab_data: List of vocabulary entries from keywords.json
        knowledge_graph: Knowledge graph with TF-IDF scores per node
        
    Returns:
        Updated vocabulary list with _tfidf_weight field added
    """
    # Step 1: Build term → TF-IDF score lookup from knowledge graph nodes
    # We index by both node ID and label for flexible matching
    tfidf_scores = {}
    for node in knowledge_graph.get('nodes', []):
        term_id = node.get('id', '').lower()      # Normalized ID (e.g., "wire_transfer")
        label = node.get('label', '').lower()      # Display label (e.g., "WIRE_TRANSFER")
        score = node.get('tf_idf_score', 0.0)
        
        # Only store terms with positive TF-IDF scores
        if score > 0:
            tfidf_scores[term_id] = score
            # Also index by label if different from ID
            if label != term_id:
                tfidf_scores[label] = score
    
    if not tfidf_scores:
        print("   No TF-IDF scores found in knowledge graph")
        # Still set default weights for consistency
        for entry in vocab_data:
            entry['_tfidf_weight'] = 1.0
        return vocab_data
    
    # Step 2: Calculate score range for normalization to 0.5-1.5
    max_score = max(tfidf_scores.values())
    min_score = min(tfidf_scores.values())
    score_range = max_score - min_score
    
    # Special case: if all scores are the same, use neutral weight (1.0)
    if score_range == 0:
        for entry in vocab_data:
            entry['_tfidf_weight'] = 1.0
        print(f"   All TF-IDF scores equal, using neutral weight for all entries")
        return vocab_data
    
    # Step 3: Apply weights to each vocabulary entry
    weighted_count = 0
    for entry in vocab_data:
        # Parse keywords from entry (may be comma-separated string or list)
        keywords = entry.get('keywords', '')
        if isinstance(keywords, str):
            keyword_list = [k.strip().lower() for k in keywords.split(',')]
        else:
            keyword_list = [str(k).lower() for k in keywords]
        
        # Collect TF-IDF scores for keywords in this entry
        weights = []
        for kw in keyword_list:
            # Try direct match first
            if kw in tfidf_scores:
                weights.append(tfidf_scores[kw])
            else:
                # Try with normalized separators (wire_transfer → wire transfer)
                normalized = kw.replace('_', ' ').replace('-', ' ')
                if normalized in tfidf_scores:
                    weights.append(tfidf_scores[normalized])
        
        # Calculate final weight for this entry
        if weights:
            # Average the TF-IDF scores of matched keywords
            avg_score = sum(weights) / len(weights)
            # Normalize to 0.5-1.5 range: 0.5 + (score - min) / range
            normalized_weight = 0.5 + (avg_score - min_score) / score_range
            entry['_tfidf_weight'] = round(normalized_weight, 3)
            weighted_count += 1
        else:
            # No TF-IDF data for this entry - use neutral weight
            entry['_tfidf_weight'] = 1.0
    
    print(f"   Applied TF-IDF weights to {weighted_count}/{len(vocab_data)} entries")
    return vocab_data


def extract_relationships_for_expansion(knowledge_graph: Dict) -> Dict:
    """
    Extract term relationships for query expansion.
    
    Creates a compact lookup structure for fast query expansion at search time.
    Only includes relationships useful for expanding queries:
    - co_occurs_with: Terms that appear together (OFAC → sanctions)
    - implements: Procedures implementing concepts (validate_bic → BIC)
    - related_to: Semantically related terms
    
    Args:
        knowledge_graph: Full knowledge graph
        
    Returns:
        Dict mapping term -> list of related terms
        Example: {"ofac": ["sanctions", "sdn", "screening"]}
    """
    expansion_map = {}
    
    # Iterate through all edges in the knowledge graph
    for edge in knowledge_graph.get('edges', []):
        source = edge.get('source', '').lower()
        target = edge.get('target', '').lower()
        rel_type = edge.get('type', '')
        
        # Skip edges with missing source or target
        if not source or not target:
            continue
        
        # Only include relationship types useful for query expansion
        # Skip 'contains' (structure→field) as it's too specific
        if rel_type in ['co_occurs_with', 'implements', 'related_to']:
            if source not in expansion_map:
                expansion_map[source] = []
            if target not in expansion_map[source]:
                expansion_map[source].append(target)
            
            # Bidirectional for co-occurrence
            if rel_type == 'co_occurs_with':
                if target not in expansion_map:
                    expansion_map[target] = []
                if source not in expansion_map[target]:
                    expansion_map[target].append(source)
    
    return expansion_map


# =============================================================================
# VOCABULARY LOADING
# =============================================================================


def load_vocabulary(vocab_path: str) -> list:
    """Load vocabulary from JSON file"""
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found: {vocab_path}")
        print(f"Please ensure 'keywords.json' exists in the same directory as this script,")
        print(f"or specify a custom vocabulary file with --vocab")
        sys.exit(1)
    
    with open(vocab_path, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: list or dict with 'entries' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get('entries', [data])
    else:
        print(f"Error: Invalid vocabulary format in {vocab_path}")
        sys.exit(1)


# =============================================================================
# INCREMENTAL INDEXING SUPPORT
# =============================================================================

def get_file_signature(file_path: str) -> dict:
    """
    Get a signature for a file to detect changes.
    
    Returns dict with path, modification time, and size.
    """
    path = Path(file_path)
    if not path.exists():
        return None
    
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime": stat.st_mtime,
        "size": stat.st_size
    }


def load_file_manifest(index_dir: str) -> dict:
    """
    Load the file manifest from an existing index.
    
    Returns dict mapping file paths to their signatures.
    """
    manifest_path = os.path.join(index_dir, "file_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return {}


def save_file_manifest(index_dir: str, manifest: dict):
    """Save file manifest to index directory."""
    manifest_path = os.path.join(index_dir, "file_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def file_needs_indexing(file_path: str, manifest: dict) -> bool:
    """
    Check if a file needs to be (re)indexed.
    
    Returns True if:
    - File is not in manifest (new file)
    - File mtime or size has changed (modified file)
    """
    sig = get_file_signature(file_path)
    if not sig:
        return False  # File doesn't exist
    
    resolved_path = sig["path"]
    
    if resolved_path not in manifest:
        return True  # New file
    
    old_sig = manifest[resolved_path]
    
    # Check if modified
    if sig["mtime"] != old_sig.get("mtime") or sig["size"] != old_sig.get("size"):
        return True
    
    return False


def detect_file_type(file_path: str) -> str:
    """Detect the type of file for indexing."""
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext in ['.pdf']:
        return 'pdf'
    elif ext in ['.tal', '.tacl']:
        return 'tal'
    elif ext in ['.txt']:
        # Could be TAL or document - check content
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(1000)
            # TAL indicators
            if any(kw in content.upper() for kw in ['PROC ', 'DEFINE ', 'LITERAL ', 'STRUCT ']):
                return 'tal'
        except:
            pass
        return 'document'
    elif ext in ['.c', '.h', '.cpp', '.hpp', '.cc', '.cxx']:
        return 'code'
    elif ext in ['.java']:
        return 'code'
    elif ext in ['.py']:
        return 'code'
    elif ext in ['.cbl', '.cob', '.cpy']:
        return 'cobol'
    elif ext in ['.md', '.rst', '.html', '.htm']:
        return 'document'
    else:
        return 'unknown'


def index_single_file(pipeline: IndexingPipeline, 
                      file_path: str,
                      manifest: dict) -> dict:
    """
    Index a single file and update manifest.
    
    Returns stats dict with files_processed, chunks_created, etc.
    """
    stats = {
        "files_processed": 0,
        "files_failed": 0,
        "chunks_created": 0,
        "skipped": 0,
        "errors": []
    }
    
    path = Path(file_path)
    if not path.exists():
        stats["errors"].append(f"File not found: {file_path}")
        stats["files_failed"] += 1
        return stats
    
    # Check if needs indexing
    if not file_needs_indexing(file_path, manifest):
        stats["skipped"] += 1
        return stats
    
    file_type = detect_file_type(file_path)
    
    try:
        print(f"  Processing: {path.name}...", end=" ", flush=True)
        
        with open(path, 'rb') as f:
            content = f.read()
        
        # Determine source type
        if file_type in ['pdf', 'document']:
            source_type = SourceType.DOCUMENT
        else:
            source_type = SourceType.CODE
        
        chunks = pipeline.index_content(content, str(path), source_type)
        
        stats["files_processed"] += 1
        stats["chunks_created"] += len(chunks)
        print(f"✓ ({len(chunks)} chunks)")
        
        # Update manifest
        sig = get_file_signature(file_path)
        if sig:
            manifest[sig["path"]] = sig
        
    except Exception as e:
        stats["files_failed"] += 1
        stats["errors"].append(f"{file_path}: {str(e)}")
        print(f"✗ Error: {str(e)[:50]}")
    
    return stats


def index_pdf_directory(pipeline: IndexingPipeline, 
                        pdf_dir: str, 
                        recursive: bool = True,
                        manifest: dict = None) -> dict:
    """Index all PDF files in a directory (with incremental support)"""
    stats = {
        "files_processed": 0,
        "files_failed": 0,
        "chunks_created": 0,
        "skipped": 0,
        "errors": []
    }
    
    if manifest is None:
        manifest = {}
    
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"Warning: PDF directory does not exist: {pdf_dir}")
        return stats
    
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(pdf_path.glob(pattern))
    # Also check for uppercase
    pdf_files.extend(pdf_path.glob(pattern.replace('.pdf', '.PDF')))
    pdf_files = list(set(pdf_files))  # Remove duplicates
    
    # Filter to only files needing indexing
    files_to_index = [f for f in pdf_files if file_needs_indexing(str(f), manifest)]
    skipped = len(pdf_files) - len(files_to_index)
    
    if skipped > 0:
        print(f"\nIndexing PDFs from {pdf_dir}: {len(files_to_index)} new/modified, {skipped} unchanged")
    else:
        print(f"\nIndexing {len(pdf_files)} PDF files from {pdf_dir}...")
    
    stats["skipped"] = skipped
    
    for pdf_file in files_to_index:
        try:
            print(f"  Processing: {pdf_file.name}...", end=" ", flush=True)
            
            with open(pdf_file, 'rb') as f:
                content = f.read()
            
            chunks = pipeline.index_content(
                content,
                str(pdf_file),
                SourceType.DOCUMENT
            )
            
            stats["files_processed"] += 1
            stats["chunks_created"] += len(chunks)
            print(f"✓ ({len(chunks)} chunks)")
            
            # Update manifest
            sig = get_file_signature(str(pdf_file))
            if sig:
                manifest[sig["path"]] = sig
            
        except Exception as e:
            stats["files_failed"] += 1
            stats["errors"].append(f"{pdf_file}: {str(e)}")
            print(f"✗ Error: {str(e)[:50]}")
    
    return stats


def index_tal_directory(pipeline: IndexingPipeline, 
                        tal_dir: str, 
                        recursive: bool = True,
                        manifest: dict = None) -> dict:
    """Index all TAL code files (.txt, .tal) in a directory (with incremental support)"""
    stats = {
        "files_processed": 0,
        "files_failed": 0,
        "chunks_created": 0,
        "skipped": 0,
        "errors": []
    }
    
    if manifest is None:
        manifest = {}
    
    tal_path = Path(tal_dir)
    if not tal_path.exists():
        print(f"Warning: TAL directory does not exist: {tal_dir}")
        return stats
    
    # Look for various extensions
    extensions = ["*.txt", "*.tal", "*.tacl", "*.TAL", "*.TXT", "*.TACL"]
    tal_files = []
    
    for ext in extensions:
        pattern = f"**/{ext}" if recursive else ext
        tal_files.extend(tal_path.glob(pattern))
    
    # Remove duplicates
    tal_files = list(set(tal_files))
    
    # Filter to only files needing indexing
    files_to_index = [f for f in tal_files if file_needs_indexing(str(f), manifest)]
    skipped = len(tal_files) - len(files_to_index)
    
    if skipped > 0:
        print(f"\nIndexing TAL from {tal_dir}: {len(files_to_index)} new/modified, {skipped} unchanged")
    else:
        print(f"\nIndexing {len(tal_files)} TAL files from {tal_dir}...")
    
    stats["skipped"] = skipped
    
    for tal_file in files_to_index:
        try:
            print(f"  Processing: {tal_file.name}...", end=" ", flush=True)
            
            with open(tal_file, 'rb') as f:
                content = f.read()
            
            chunks = pipeline.index_content(
                content,
                str(tal_file),
                SourceType.CODE
            )
            
            stats["files_processed"] += 1
            stats["chunks_created"] += len(chunks)
            print(f"✓ ({len(chunks)} chunks)")
            
            # Update manifest
            sig = get_file_signature(str(tal_file))
            if sig:
                manifest[sig["path"]] = sig
            
        except Exception as e:
            stats["files_failed"] += 1
            stats["errors"].append(f"{tal_file}: {str(e)}")
            print(f"✗ Error: {str(e)[:50]}")
    
    return stats


def index_code_directory(pipeline: IndexingPipeline, 
                         code_dir: str, 
                         recursive: bool = True,
                         manifest: dict = None) -> dict:
    """Index all code files (C, C++, Java, Python, etc.) in a directory (with incremental support)"""
    stats = {
        "files_processed": 0,
        "files_failed": 0,
        "chunks_created": 0,
        "skipped": 0,
        "errors": []
    }
    
    if manifest is None:
        manifest = {}
    
    code_path = Path(code_dir)
    if not code_path.exists():
        print(f"Warning: Code directory does not exist: {code_dir}")
        return stats
    
    # Supported code extensions
    extensions = [
        # TAL/TACL (Tandem Application Language)
        "*.tal", "*.tacl", "*.ddl", "*.txt",
        # COBOL
        "*.cob", "*.cbl", "*.cobol", "*.cpy",
        # C/C++
        "*.c", "*.h", "*.cpp", "*.hpp", "*.cc", "*.cxx", "*.hxx",
        # Java
        "*.java",
        # Python
        "*.py",
        # JavaScript/TypeScript
        "*.js", "*.jsx", "*.ts", "*.tsx",
        # C#
        "*.cs",
        # Go
        "*.go",
        # Rust
        "*.rs",
        # Ruby
        "*.rb",
        # PHP
        "*.php",
        # Swift
        "*.swift",
        # Kotlin
        "*.kt", "*.kts",
        # Scala
        "*.scala",
        # Other source files
        "*.src", "*.inc",
    ]
    
    code_files = []
    for ext in extensions:
        pattern = f"**/{ext}" if recursive else ext
        code_files.extend(code_path.glob(pattern))
    
    # Remove duplicates
    code_files = list(set(code_files))
    
    # Filter to only files needing indexing
    files_to_index = [f for f in code_files if file_needs_indexing(str(f), manifest)]
    skipped = len(code_files) - len(files_to_index)
    
    if skipped > 0:
        print(f"\nIndexing code from {code_dir}: {len(files_to_index)} new/modified, {skipped} unchanged")
    else:
        print(f"\nIndexing {len(code_files)} code files from {code_dir}...")
    
    stats["skipped"] = skipped
    
    for code_file in files_to_index:
        try:
            print(f"  Processing: {code_file.name}...", end=" ", flush=True)
            
            with open(code_file, 'rb') as f:
                content = f.read()
            
            chunks = pipeline.index_content(
                content,
                str(code_file),
                SourceType.CODE
            )
            
            stats["files_processed"] += 1
            stats["chunks_created"] += len(chunks)
            print(f"✓ ({len(chunks)} chunks)")
            
            # Update manifest
            sig = get_file_signature(str(code_file))
            if sig:
                manifest[sig["path"]] = sig
            
        except Exception as e:
            stats["files_failed"] += 1
            stats["errors"].append(f"{code_file}: {str(e)}")
            print(f"✗ Error: {str(e)[:50]}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build search index from PDF documents and code files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build new index with documents and code
  python build_index.py --docs ./pdfs --code ./code --output ./my_index
  
  # Multiple directories
  python build_index.py --docs ./pdfs --docs ./manuals --code ./tal --code ./cobol \\
      --output ./my_index
  
  # With knowledge graph and vocabulary
  python build_index.py --docs ./pdfs --code ./code --output ./my_index \\
      --knowledge-graph knowledge_graph.json --vocab vocabulary.json
  
  # Incremental update
  python build_index.py --docs ./pdfs --output ./my_index --incremental
  
  # Add single files
  python build_index.py --add-file ./new_doc.pdf --add-file ./new_code.tal \\
      --output ./my_index
        """
    )
    
    # Input directories (can specify multiple)
    parser.add_argument("--docs", type=str, action="append", default=[],
                        help="Document directory (PDFs, text files). Can specify multiple.")
    parser.add_argument("--code", "-c", type=str, action="append", default=[],
                        help="Code directory (TAL, COBOL, C, Java, Python, etc.). Can specify multiple.")
    
    # Legacy aliases for backward compatibility
    parser.add_argument("--pdf-dir", type=str, help="(Legacy) Same as --docs")
    parser.add_argument("--tal-dir", type=str, help="(Legacy) Same as --code")
    parser.add_argument("--code-dir", type=str, help="(Legacy) Same as --code")
    
    parser.add_argument("--add-file", type=str, action="append", default=[],
                        help="Add a single file to existing index (can be used multiple times)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for index")
    parser.add_argument("--incremental", "-i", action="store_true",
                        help="Incremental update: only index new/modified files")
    parser.add_argument("--vocab", "-v", type=str, default=DEFAULT_KEYWORDS_FILE,
                        help=f"Path to vocabulary JSON file (default: keywords.json)")
    parser.add_argument("--recursive", "-r", action="store_true", default=True, 
                        help="Search directories recursively (default: True)")
    parser.add_argument("--no-recursive", action="store_true", help="Don't search recursively")
    parser.add_argument("--embedder", "-e", type=str, default="hash",
                        choices=["hash", "hybrid", "tfidf", "domain", "bm25", 
                                 "payment", "payment_hybrid"],
                        help="Embedder type (default: hash)")
    parser.add_argument("--dims", "-d", type=int, default=None,
                        help="Embedding dimensions (default: 1024 for hash, 512+vocab for hybrid)")
    parser.add_argument("--domain-weight", type=float, default=0.6,
                        help="Weight for domain concepts in hybrid embedder (default: 0.6)")
    
    # Knowledge graph options
    parser.add_argument("--knowledge-graph", "-kg", type=str, default=None,
                        help="Path to knowledge_graph.json (from knowledge_extractor)")
    parser.add_argument("--tfidf-weight", action="store_true", default=True,
                        help="Apply TF-IDF weights to vocabulary (default: True when KG provided)")
    parser.add_argument("--no-tfidf-weight", action="store_true",
                        help="Don't apply TF-IDF weighting to vocabulary")
    
    # Logging options
    parser.add_argument("--verbose", "-V", action="store_true",
                        help="Verbose output: show generation updates and atomic swaps")
    
    args = parser.parse_args()
    
    # Merge legacy arguments into new lists
    if args.pdf_dir:
        args.docs.append(args.pdf_dir)
    if args.tal_dir:
        args.code.append(args.tal_dir)
    if args.code_dir:
        args.code.append(args.code_dir)
    
    # Validate inputs
    if not args.docs and not args.code and not args.add_file:
        print("Error: At least one of --docs, --code, or --add-file is required")
        sys.exit(1)
    
    # --add-file implies incremental
    if args.add_file:
        args.incremental = True
    
    recursive = not args.no_recursive
    apply_tfidf = args.tfidf_weight and not args.no_tfidf_weight
    
    print("=" * 60)
    print("UNIFIED INDEXER - BUILD INDEX")
    print("=" * 60)
    
    # Load vocabulary from keywords.json
    print(f"\nLoading vocabulary from: {args.vocab}")
    vocab_data = load_vocabulary(args.vocab)
    print(f"Vocabulary entries: {len(vocab_data)}")
    
    # Load knowledge graph if provided
    knowledge_graph = None
    if args.knowledge_graph:
        print(f"\nLoading knowledge graph from: {args.knowledge_graph}")
        knowledge_graph = load_knowledge_graph(args.knowledge_graph)
        if knowledge_graph:
            nodes = len(knowledge_graph.get('nodes', []))
            edges = len(knowledge_graph.get('edges', []))
            print(f"   Nodes: {nodes}, Edges: {edges}")
            
            # Apply TF-IDF weights to vocabulary
            if apply_tfidf:
                print(f"\nApplying TF-IDF weights to vocabulary...")
                vocab_data = apply_tfidf_weights(vocab_data, knowledge_graph)
        else:
            print(f"   ⚠️  Could not load knowledge graph")
    
    # Build embedder kwargs
    embedder_kwargs = {}
    if args.dims:
        if args.embedder == "hash":
            embedder_kwargs['n_features'] = args.dims
        elif args.embedder in ["hybrid", "payment_hybrid"]:
            embedder_kwargs['text_dim'] = args.dims
        elif args.embedder in ["tfidf", "bm25"]:
            embedder_kwargs['max_features'] = args.dims
    
    if args.embedder == "hybrid":
        embedder_kwargs['domain_weight'] = args.domain_weight
        embedder_kwargs['text_weight'] = 1.0 - args.domain_weight
    elif args.embedder == "payment_hybrid":
        embedder_kwargs['payment_weight'] = args.domain_weight
        embedder_kwargs['text_weight'] = 1.0 - args.domain_weight
    
    # Create or load pipeline
    print(f"\nInitializing pipeline with '{args.embedder}' embedder...")
    
    pipeline = IndexingPipeline(
        vocabulary_data=vocab_data,
        embedder_type=args.embedder
    )
    
    # Set embedder with custom dimensions if specified
    if embedder_kwargs:
        pipeline.set_embedder(args.embedder, **embedder_kwargs)
    
    # Handle incremental mode
    manifest = {}
    existing_chunks = 0
    
    if args.incremental and os.path.exists(args.output):
        # Check for generation-based index (preferred) or legacy index.pkl
        gen_file = os.path.join(args.output, "generation.txt")
        chunks_file = os.path.join(args.output, "chunks.json")
        has_index = os.path.exists(gen_file) or os.path.exists(chunks_file)
        
        if has_index:
            print(f"\n📂 Loading existing index for incremental update...")
            pipeline.load(args.output, verbose=args.verbose)
            existing_chunks = pipeline.index.get_statistics().get('total_chunks', 0)
            print(f"   Existing chunks: {existing_chunks}")
            
            # Load file manifest
            manifest = load_file_manifest(args.output)
            print(f"   Tracked files: {len(manifest)}")
        else:
            print(f"\n⚠️  No existing index found, creating new index")
    
    # Report dimensions
    if hasattr(pipeline.embedder, 'n_dimensions'):
        print(f"Embedding dimensions: {pipeline.embedder.n_dimensions}")
    elif hasattr(pipeline.embedder, 'n_features'):
        print(f"Embedding dimensions: {pipeline.embedder.n_features}")
    
    total_stats = {
        "pdf": {"files_processed": 0, "chunks_created": 0, "files_failed": 0, "skipped": 0},
        "tal": {"files_processed": 0, "chunks_created": 0, "files_failed": 0, "skipped": 0},
        "code": {"files_processed": 0, "chunks_created": 0, "files_failed": 0, "skipped": 0},
        "added": {"files_processed": 0, "chunks_created": 0, "files_failed": 0, "skipped": 0}
    }
    
    # Index individual files (--add-file)
    if args.add_file:
        print(f"\nAdding {len(args.add_file)} file(s) to index...")
        for file_path in args.add_file:
            file_stats = index_single_file(pipeline, file_path, manifest)
            total_stats["added"]["files_processed"] += file_stats["files_processed"]
            total_stats["added"]["chunks_created"] += file_stats["chunks_created"]
            total_stats["added"]["files_failed"] += file_stats["files_failed"]
            total_stats["added"]["skipped"] += file_stats.get("skipped", 0)
    
    # Index document directories (PDFs, text files)
    total_stats["docs"] = {"files_processed": 0, "chunks_created": 0, "files_failed": 0}
    for doc_dir in args.docs:
        if os.path.isdir(doc_dir):
            print(f"\nIndexing documents from: {doc_dir}")
            pdf_stats = index_pdf_directory(pipeline, doc_dir, recursive, manifest)
            total_stats["docs"]["files_processed"] += pdf_stats.get("files_processed", 0)
            total_stats["docs"]["chunks_created"] += pdf_stats.get("chunks_created", 0)
            total_stats["docs"]["files_failed"] += pdf_stats.get("files_failed", 0)
        else:
            print(f"Warning: Document directory not found: {doc_dir}")
    
    # Index code directories (TAL, COBOL, C, Java, Python, etc.)
    total_stats["code"] = {"files_processed": 0, "chunks_created": 0, "files_failed": 0}
    for code_dir in args.code:
        if os.path.isdir(code_dir):
            print(f"\nIndexing code from: {code_dir}")
            # Use TAL indexer for .txt files (likely TAL), generic for others
            code_stats = index_code_directory(pipeline, code_dir, recursive, manifest)
            total_stats["code"]["files_processed"] += code_stats.get("files_processed", 0)
            total_stats["code"]["chunks_created"] += code_stats.get("chunks_created", 0)
            total_stats["code"]["files_failed"] += code_stats.get("files_failed", 0)
        else:
            print(f"Warning: Code directory not found: {code_dir}")
    
    # Save index
    print(f"\nSaving index to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    pipeline.save(args.output, verbose=args.verbose)
    
    # Save file manifest for future incremental updates
    save_file_manifest(args.output, manifest)
    
    # Save knowledge graph to index directory if provided
    kg_saved = False
    if knowledge_graph and args.knowledge_graph:
        kg_dest = os.path.join(args.output, "knowledge_graph.json")
        print(f"Saving knowledge graph to index: {kg_dest}")
        
        # Save the full knowledge graph
        with open(kg_dest, 'w') as f:
            json.dump(knowledge_graph, f, indent=2)
        
        # Also save compact expansion map for faster query expansion
        expansion_map = extract_relationships_for_expansion(knowledge_graph)
        if expansion_map:
            expansion_dest = os.path.join(args.output, "expansion_map.json")
            with open(expansion_dest, 'w') as f:
                json.dump(expansion_map, f, indent=2)
            print(f"   Expansion map: {len(expansion_map)} terms")
        
        kg_saved = True
    
    # Save reference to vocabulary and settings used
    new_chunks = pipeline.index.get_statistics().get('total_chunks', 0)
    index_meta = {
        "vocabulary_file": os.path.basename(args.vocab),
        "embedder_type": args.embedder,
        "doc_dirs": args.docs,
        "code_dirs": args.code,
        "knowledge_graph": kg_saved,
        "tfidf_weighted": apply_tfidf and kg_saved,
        "stats": total_stats,
        "total_chunks": new_chunks,
        "tracked_files": len(manifest)
    }
    with open(os.path.join(args.output, "index_meta.json"), 'w') as f:
        json.dump(index_meta, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    if args.incremental:
        print("INCREMENTAL INDEX UPDATE COMPLETE")
    else:
        print("INDEX BUILD COMPLETE")
    print("=" * 60)
    
    if args.add_file:
        print(f"\nAdded Files:")
        print(f"  Files processed: {total_stats['added'].get('files_processed', 0)}")
        print(f"  Files skipped:   {total_stats['added'].get('skipped', 0)}")
        print(f"  Chunks created:  {total_stats['added'].get('chunks_created', 0)}")
    
    print(f"\nPDF Documents:")
    print(f"  Files processed: {total_stats['pdf'].get('files_processed', 0)}")
    print(f"  Files skipped:   {total_stats['pdf'].get('skipped', 0)}")
    print(f"  Files failed:    {total_stats['pdf'].get('files_failed', 0)}")
    print(f"  Chunks created:  {total_stats['pdf'].get('chunks_created', 0)}")
    
    print(f"\nTAL Code:")
    print(f"  Files processed: {total_stats['tal'].get('files_processed', 0)}")
    print(f"  Files skipped:   {total_stats['tal'].get('skipped', 0)}")
    print(f"  Files failed:    {total_stats['tal'].get('files_failed', 0)}")
    print(f"  Chunks created:  {total_stats['tal'].get('chunks_created', 0)}")
    
    print(f"\nGeneric Code (C/C++/Java/Python/etc.):")
    print(f"  Files processed: {total_stats['code'].get('files_processed', 0)}")
    print(f"  Files skipped:   {total_stats['code'].get('skipped', 0)}")
    print(f"  Files failed:    {total_stats['code'].get('files_failed', 0)}")
    print(f"  Chunks created:  {total_stats['code'].get('chunks_created', 0)}")
    
    new_chunks = (total_stats['pdf'].get('chunks_created', 0) + 
                  total_stats['tal'].get('chunks_created', 0) +
                  total_stats['code'].get('chunks_created', 0) +
                  total_stats['added'].get('chunks_created', 0))
    
    if args.incremental and existing_chunks > 0:
        print(f"\nChunks: {existing_chunks} existing + {new_chunks} new = {existing_chunks + new_chunks} total")
    else:
        print(f"\nTotal chunks indexed: {new_chunks}")
    
    print(f"Tracked files: {len(manifest)}")
    print(f"Index saved to: {args.output}")
    
    # Knowledge graph info
    if kg_saved:
        print(f"\n📊 Knowledge Graph:")
        print(f"  Embedded in index: Yes")
        print(f"  TF-IDF weighted vocabulary: {'Yes' if apply_tfidf else 'No'}")
        print(f"  Query expansion ready: Yes")
    
    # Print errors if any
    all_errors = (total_stats['pdf'].get('errors', []) + 
                  total_stats['tal'].get('errors', []) +
                  total_stats['code'].get('errors', []) +
                  total_stats['added'].get('errors', []))
    if all_errors:
        print(f"\n⚠️  Errors encountered ({len(all_errors)}):")
        for err in all_errors[:5]:
            print(f"   {err}")
        if len(all_errors) > 5:
            print(f"   ... and {len(all_errors) - 5} more")
    
    print(f"\nTo search the index, run:")
    print(f"  python search_index.py --index {args.output} --query \"your search query\"")
    if kg_saved:
        print(f"\n  Knowledge graph will be auto-loaded. For query expansion:")
        print(f"  python search_index.py --index {args.output} --query \"OFAC\" --expand-query")
    
    print(f"\nTo add more files later (incremental update):")
    print(f"  python build_index.py --output {args.output} --add-file ./new_file.pdf")
    print(f"  python build_index.py --output {args.output} --pdf-dir ./more_docs --incremental")


if __name__ == "__main__":
    main()
