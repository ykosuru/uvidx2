#!/usr/bin/env python3
"""
Keyword Extractor - Extract domain keywords from PDFs and TAL code using LLM

This script:
1. Reads PDF documents and TAL/code files from a source directory
2. Sends content to LLM to extract top 25 most important domain keywords
3. Merges extracted keywords with existing keywords.json
4. Outputs augmented keywords.json for improved indexing

Usage:
    python extract_keywords.py --source ./docs --source ./code --output keywords_augmented.json
    python extract_keywords.py --source ./my_files --provider tachyon
    python extract_keywords.py --source ./my_files --no-llm  # Heuristic mode (offline)
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

# Import LLM provider
try:
    from llm_provider import create_provider, LLMProvider, LLMResponse
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: llm_provider not found. Install or ensure it's in the path.")

# PDF support
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. PDF extraction will be skipped.")
    print("   Install with: pip install pymupdf")


@dataclass
class ExtractedKeywords:
    """Keywords extracted from a single document"""
    file_path: str
    file_type: str  # 'pdf', 'tal', 'code'
    keywords: List[str] = field(default_factory=list)
    context: str = ""  # Brief description from LLM
    error: Optional[str] = None


def read_pdf_content(file_path: str, max_pages: int = 50) -> str:
    """Extract text from PDF file"""
    if not PDF_AVAILABLE:
        return ""
    
    try:
        doc = fitz.open(file_path)
        text_parts = []
        
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            text_parts.append(page.get_text())
        
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        print(f"  Error reading PDF {file_path}: {e}")
        return ""


def read_code_content(file_path: str, max_lines: int = 2000) -> str:
    """Read code file content"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[:max_lines]
            return "".join(lines)
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return ""


def get_file_type(file_path: str) -> Optional[str]:
    """Determine file type from extension"""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.pdf':
        return 'pdf'
    elif ext in ('.tal', '.tacl', '.ddl'):
        return 'tal'
    elif ext in ('.cob', '.cbl', '.cobol', '.cpy'):
        return 'cobol'
    elif ext in ('.c', '.h', '.cpp', '.hpp', '.cc'):
        return 'c'
    elif ext in ('.java',):
        return 'java'
    elif ext in ('.py',):
        return 'python'
    elif ext in ('.txt', '.md', '.rst'):
        return 'text'
    else:
        return None


EXTRACTION_PROMPT = """You are a domain keyword extraction expert specializing in payment systems, banking, and legacy code modernization.

Analyze the following content and extract the TOP 25 most important domain-specific keywords or phrases.

Focus on:
1. Payment/banking terminology (SWIFT, ACH, wire transfer, settlement, etc.)
2. Technical terms specific to the domain (message types, error codes, etc.)
3. Business processes (validation, routing, screening, etc.)
4. System/platform names and acronyms
5. Data fields and message elements
6. Compliance and regulatory terms

DO NOT include:
- Generic programming keywords (if, else, for, while, function, etc.)
- Common English words (the, and, or, is, etc.)
- Variable names that are just abbreviations (i, j, x, tmp, etc.)

Output format - respond with ONLY a JSON object like this:
{
    "keywords": ["keyword1", "keyword2", ...],
    "context": "Brief 1-sentence description of what this document/code is about"
}

Ensure exactly 25 keywords, ordered by importance (most important first).
"""


def extract_keywords_heuristic(content: str, file_type: str) -> List[str]:
    """
    Extract keywords using heuristics when LLM is not available.
    
    Uses regex patterns to find domain-specific terms.
    """
    keywords = set()
    content_upper = content.upper()
    
    # Payment/banking terms to look for
    domain_patterns = [
        # SWIFT message types
        r'\bMT[-\s]?(\d{3})\b',
        # Common payment terms
        r'\b(WIRE\s*TRANSFER|FUNDS\s*TRANSFER|REMITTANCE)\b',
        r'\b(SWIFT|FEDWIRE|CHIPS|ACH|SEPA)\b',
        r'\b(BIC|IBAN|ABA|ROUTING)\b',
        r'\b(OFAC|SDN|SANCTIONS|SCREENING|COMPLIANCE)\b',
        r'\b(BENEFICIARY|ORDERING|CORRESPONDENT|INTERMEDIARY)\b',
        r'\b(SETTLEMENT|CLEARING|NETTING)\b',
        r'\b(UETR|UTI|LEI)\b',
        # Error codes
        r'\bERROR[_\s]*(\d{4})\b',
        r'\b(ERR|ERROR)[_\s]+(\w+)\b',
        # Message fields
        r'\b(SENDER|RECEIVER|ORIGINATOR)[_\s]*(BIC|ACCT|NAME|ADDR)\b',
        r'\b(AMOUNT|CURRENCY|REFERENCE|PRIORITY)\b',
        # Technical terms
        r'\b(VALIDATE|VERIFY|SCREEN|ROUTE|PROCESS)\b',
        r'\b(QUEUE|TRANSACTION|MESSAGE|PAYMENT)\b',
    ]
    
    for pattern in domain_patterns:
        for match in re.finditer(pattern, content_upper):
            keyword = match.group(0).strip()
            # Normalize
            keyword = re.sub(r'[\s_]+', '_', keyword).lower()
            if len(keyword) > 2:
                keywords.add(keyword)
    
    # Extract DEFINE/LITERAL values from TAL
    if file_type in ('tal', 'cobol'):
        for match in re.finditer(r'(?:DEFINE|LITERAL)\s+(\w+)', content, re.IGNORECASE):
            name = match.group(1).lower()
            if len(name) > 3 and not name.isdigit():
                keywords.add(name)
    
    # Extract struct/record names
    for match in re.finditer(r'STRUCT\s+(\w+)', content, re.IGNORECASE):
        keywords.add(match.group(1).lower())
    
    # Extract procedure names that look domain-specific
    for match in re.finditer(r'(?:PROC|SUBPROC|void|int)\s+(\w+)', content, re.IGNORECASE):
        name = match.group(1).lower()
        # Only include if it contains domain words
        domain_words = ['wire', 'payment', 'transfer', 'validate', 'route', 'screen', 
                       'ofac', 'swift', 'message', 'process', 'check', 'send', 'receive']
        if any(dw in name for dw in domain_words):
            keywords.add(name)
    
    return list(keywords)[:25]


def extract_keywords_from_content(
    content: str,
    file_path: str,
    file_type: str,
    provider: LLMProvider
) -> ExtractedKeywords:
    """Use LLM to extract keywords from content"""
    
    # Truncate content if too long (keep ~8000 chars for context window)
    if len(content) > 12000:
        # Take beginning and end
        content = content[:6000] + "\n\n... [content truncated] ...\n\n" + content[-4000:]
    
    user_prompt = f"""File: {Path(file_path).name}
Type: {file_type.upper()}

CONTENT:
```
{content}
```

Extract the top 25 domain-specific keywords from this content."""

    try:
        response = provider.invoke_llm(
            system_prompt=EXTRACTION_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=1000
        )
        
        if not response.success:
            return ExtractedKeywords(
                file_path=file_path,
                file_type=file_type,
                error=response.error
            )
        
        # Parse JSON response
        result_text = response.content.strip()
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            result_json = json.loads(json_match.group())
            keywords = result_json.get('keywords', [])
            context = result_json.get('context', '')
            
            # Clean keywords
            keywords = [k.strip().lower() for k in keywords if k.strip()]
            
            return ExtractedKeywords(
                file_path=file_path,
                file_type=file_type,
                keywords=keywords[:25],
                context=context
            )
        else:
            return ExtractedKeywords(
                file_path=file_path,
                file_type=file_type,
                error=f"Could not parse JSON from response: {result_text[:200]}"
            )
            
    except json.JSONDecodeError as e:
        return ExtractedKeywords(
            file_path=file_path,
            file_type=file_type,
            error=f"JSON parse error: {e}"
        )
    except Exception as e:
        return ExtractedKeywords(
            file_path=file_path,
            file_type=file_type,
            error=str(e)
        )


def find_files(source_dirs: List[str], extensions: Optional[Set[str]] = None) -> List[Tuple[str, str]]:
    """Find all relevant files in source directories"""
    if extensions is None:
        extensions = {'.pdf', '.tal', '.tacl', '.ddl', '.cob', '.cbl', '.cobol', 
                      '.cpy', '.c', '.h', '.cpp', '.java', '.py', '.txt', '.md'}
    
    files = []
    
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Warning: Source directory not found: {source_dir}")
            continue
        
        if source_path.is_file():
            ext = source_path.suffix.lower()
            if ext in extensions:
                file_type = get_file_type(str(source_path))
                if file_type:
                    files.append((str(source_path), file_type))
        else:
            for file_path in source_path.rglob('*'):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in extensions:
                        file_type = get_file_type(str(file_path))
                        if file_type:
                            files.append((str(file_path), file_type))
    
    return files


def load_existing_keywords(keywords_path: str) -> Dict:
    """Load existing keywords.json"""
    try:
        with open(keywords_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "version": "1.0",
            "description": "Payment Systems Domain Vocabulary",
            "entries": []
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing {keywords_path}: {e}")
        return {"version": "1.0", "entries": []}


def get_existing_keywords(keywords_data: Dict) -> Set[str]:
    """Extract all existing keywords from keywords.json"""
    existing = set()
    
    for entry in keywords_data.get('entries', []):
        # Main keywords
        if 'keywords' in entry:
            for kw in entry['keywords'].split(','):
                existing.add(kw.strip().lower())
        
        # Related keywords
        if 'related_keywords' in entry:
            for kw in entry['related_keywords'].split(','):
                existing.add(kw.strip().lower())
    
    return existing


def categorize_keyword(keyword: str) -> str:
    """Try to categorize a keyword based on patterns"""
    keyword_lower = keyword.lower()
    
    # SWIFT message types
    if re.match(r'^mt[- ]?\d{3}', keyword_lower):
        return 'swift-mt-messages'
    
    # Error codes
    if re.match(r'^(error|err|e)\d+', keyword_lower) or re.match(r'^\d{4}$', keyword_lower):
        return 'error-codes'
    
    # Compliance terms
    compliance_terms = {'ofac', 'sanctions', 'aml', 'kyc', 'compliance', 'screening', 'sdn'}
    if any(term in keyword_lower for term in compliance_terms):
        return 'compliance-fraud'
    
    # Payment terms
    payment_terms = {'wire', 'transfer', 'payment', 'remittance', 'ach', 'fed', 'chips'}
    if any(term in keyword_lower for term in payment_terms):
        return 'payment-systems'
    
    # SWIFT/messaging
    swift_terms = {'swift', 'message', 'fin', 'bic', 'iban'}
    if any(term in keyword_lower for term in swift_terms):
        return 'swift-messaging'
    
    # Banking terms
    banking_terms = {'account', 'bank', 'institution', 'routing', 'aba', 'correspondent'}
    if any(term in keyword_lower for term in banking_terms):
        return 'banking'
    
    # Technical/system terms
    if any(term in keyword_lower for term in ['system', 'process', 'queue', 'server', 'database']):
        return 'technical'
    
    return 'domain-specific'


def merge_keywords(
    keywords_data: Dict,
    extracted: List[ExtractedKeywords],
    min_occurrences: int = 1
) -> Dict:
    """Merge extracted keywords into existing keywords.json"""
    
    existing_keywords = get_existing_keywords(keywords_data)
    
    # Count keyword occurrences across documents
    keyword_counts = defaultdict(int)
    keyword_sources = defaultdict(list)
    
    for extraction in extracted:
        if extraction.error:
            continue
        
        for kw in extraction.keywords:
            kw_lower = kw.lower().strip()
            if kw_lower and len(kw_lower) > 2:  # Skip very short keywords
                keyword_counts[kw_lower] += 1
                keyword_sources[kw_lower].append(Path(extraction.file_path).name)
    
    # Find new keywords (not in existing)
    new_keywords = {}
    for kw, count in keyword_counts.items():
        if count >= min_occurrences and kw not in existing_keywords:
            new_keywords[kw] = {
                'count': count,
                'sources': keyword_sources[kw][:5],  # Keep up to 5 source references
                'category': categorize_keyword(kw)
            }
    
    # Group new keywords by category
    by_category = defaultdict(list)
    for kw, info in new_keywords.items():
        by_category[info['category']].append(kw)
    
    # Create new entries for each category
    new_entries = []
    for category, keywords in by_category.items():
        if not keywords:
            continue
        
        # Sort by occurrence count
        keywords_sorted = sorted(
            keywords,
            key=lambda k: new_keywords[k]['count'],
            reverse=True
        )
        
        # Create entry with top keywords
        main_keywords = keywords_sorted[:5]
        related_keywords = keywords_sorted[5:15]
        
        entry = {
            "keywords": ",".join(main_keywords),
            "metadata": category,
            "description": f"Keywords extracted from {category.replace('-', ' ')} domain",
            "related_keywords": ",".join(related_keywords) if related_keywords else "",
            "business_capability": [category.replace('-', ' ').title()],
            "_extracted": True,  # Mark as auto-extracted
            "_sources": list(set(
                src for kw in main_keywords 
                for src in new_keywords[kw]['sources']
            ))[:10]
        }
        new_entries.append(entry)
    
    # Add new entries to keywords data
    keywords_data['entries'].extend(new_entries)
    
    # Update version/description
    keywords_data['description'] = keywords_data.get('description', '') + ' (augmented with extracted keywords)'
    
    return keywords_data, new_keywords


def print_summary(extracted: List[ExtractedKeywords], new_keywords: Dict):
    """Print extraction summary"""
    
    successful = [e for e in extracted if not e.error]
    failed = [e for e in extracted if e.error]
    
    print("\n" + "=" * 60)
    print("KEYWORD EXTRACTION SUMMARY")
    print("=" * 60)
    
    print(f"\nFiles processed: {len(extracted)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print("\nFailed files:")
        for e in failed[:10]:
            print(f"  - {Path(e.file_path).name}: {e.error[:50]}...")
    
    print(f"\nNew keywords found: {len(new_keywords)}")
    
    # Show top keywords by occurrence
    if new_keywords:
        print("\nTop 20 new keywords by frequency:")
        sorted_kw = sorted(new_keywords.items(), key=lambda x: x[1]['count'], reverse=True)
        for kw, info in sorted_kw[:20]:
            print(f"  {kw}: {info['count']} occurrences ({info['category']})")
    
    # Group by category
    by_cat = defaultdict(int)
    for kw, info in new_keywords.items():
        by_cat[info['category']] += 1
    
    if by_cat:
        print("\nKeywords by category:")
        for cat, count in sorted(by_cat.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract domain keywords from PDFs and code files using LLM"
    )
    parser.add_argument("--source", "-s", type=str, action='append', required=True,
                        help="Source directory or file (can specify multiple)")
    parser.add_argument("--output", "-o", type=str, default="keywords_augmented.json",
                        help="Output keywords.json path")
    parser.add_argument("--existing", "-e", type=str, default="keywords.json",
                        help="Existing keywords.json to augment")
    parser.add_argument("--provider", "-p", type=str, default="tachyon",
                        help="LLM provider (tachyon, anthropic, openai, ollama)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="LLM model name")
    parser.add_argument("--min-occurrences", type=int, default=1,
                        help="Minimum occurrences to include keyword (default: 1)")
    parser.add_argument("--max-files", type=int, default=100,
                        help="Maximum files to process (default: 100)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use heuristic extraction instead of LLM (for offline use)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    if not LLM_AVAILABLE:
        print("Error: llm_provider module not available")
        sys.exit(1)
    
    print("=" * 60)
    print("KEYWORD EXTRACTOR")
    print("=" * 60)
    
    # Find files
    print(f"\nSearching for files in: {args.source}")
    files = find_files(args.source)
    
    if not files:
        print("No files found to process")
        sys.exit(1)
    
    print(f"Found {len(files)} files")
    
    # Limit files
    if len(files) > args.max_files:
        print(f"Limiting to {args.max_files} files")
        files = files[:args.max_files]
    
    # Group by type
    by_type = defaultdict(list)
    for fp, ft in files:
        by_type[ft].append(fp)
    
    print("\nFiles by type:")
    for ft, fps in by_type.items():
        print(f"  {ft}: {len(fps)}")
    
    # Create LLM provider (unless --no-llm)
    provider = None
    if not args.no_llm:
        print(f"\nInitializing LLM provider: {args.provider}")
        try:
            provider = create_provider(args.provider, args.model)
            print(f"  Model: {provider.model}")
        except Exception as e:
            print(f"Warning: LLM provider error: {e}")
            print("  Falling back to heuristic extraction")
            args.no_llm = True
    else:
        print("\nUsing heuristic extraction (--no-llm)")
    
    # Load existing keywords
    print(f"\nLoading existing keywords from: {args.existing}")
    keywords_data = load_existing_keywords(args.existing)
    existing_count = len(keywords_data.get('entries', []))
    print(f"  Existing entries: {existing_count}")
    
    # Extract keywords from each file
    print(f"\nExtracting keywords from {len(files)} files...")
    extracted = []
    
    for i, (file_path, file_type) in enumerate(files):
        filename = Path(file_path).name
        print(f"  [{i+1}/{len(files)}] {filename}...", end=" ", flush=True)
        
        # Read content
        if file_type == 'pdf':
            if not PDF_AVAILABLE:
                print("skipped (no PDF support)")
                continue
            content = read_pdf_content(file_path)
        else:
            content = read_code_content(file_path)
        
        if not content or len(content) < 100:
            print("skipped (no content)")
            continue
        
        # Extract keywords (LLM or heuristic)
        if args.no_llm:
            keywords = extract_keywords_heuristic(content, file_type)
            result = ExtractedKeywords(
                file_path=file_path,
                file_type=file_type,
                keywords=keywords,
                context="Extracted via heuristic"
            )
        else:
            result = extract_keywords_from_content(content, file_path, file_type, provider)
        
        extracted.append(result)
        
        if result.error:
            print(f"error: {result.error[:30]}...")
        else:
            print(f"found {len(result.keywords)} keywords")
            if args.verbose and result.keywords:
                print(f"       Top 5: {', '.join(result.keywords[:5])}")
    
    # Merge keywords
    print("\nMerging extracted keywords...")
    augmented_data, new_keywords = merge_keywords(
        keywords_data, 
        extracted, 
        min_occurrences=args.min_occurrences
    )
    
    # Save augmented keywords
    print(f"\nSaving augmented keywords to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(augmented_data, f, indent=2)
    
    final_count = len(augmented_data.get('entries', []))
    print(f"  Total entries: {final_count} (added {final_count - existing_count})")
    
    # Print summary
    print_summary(extracted, new_keywords)
    
    print(f"\n✅ Done! Augmented keywords saved to: {args.output}")
    print(f"   Use with: python build_index.py --vocab {args.output} ...")


if __name__ == "__main__":
    main()
