#!/usr/bin/env python3
"""
Usage Example - Unified Indexer

This demonstrates the direct import pattern for the unified indexer.
"""

# Direct imports (avoid circular dependency)
from unified_indexer.pipeline import IndexingPipeline, LLMInterface, LLMEnhancedPipeline
from unified_indexer.vocabulary import DomainVocabulary
from unified_indexer.parsers.tal_parser import TalCodeParser
from unified_indexer.parsers.document_parser import DocumentParser
from unified_indexer.parsers.log_parser import LogParser
from unified_indexer.index import HybridIndex
from unified_indexer.models import SourceType, IndexableChunk
from unified_indexer.embeddings import (
    create_embedder,
    HybridEmbedder,
    HashEmbedder,
    TFIDFEmbedder,
    DomainConceptEmbedder,
    BM25Embedder,
    cosine_similarity,
    batch_cosine_similarity
)


def main():
    """Example usage of unified indexer with direct imports."""
    print("Unified Indexer - Usage Example")
    print("=" * 40)
    
    # Example vocabulary
    vocab_data = [
        {"keywords": "wire transfer,wire,transfer", "capability": "payment_transfer"},
        {"keywords": "OFAC,sanctions,screening", "capability": "compliance"},
    ]
    
    # Create pipeline
    pipeline = IndexingPipeline(
        vocabulary_data=vocab_data,
        embedder_type="hash"
    )
    
    print(f"Pipeline created with {len(vocab_data)} vocabulary entries")
    print("Done!")


if __name__ == "__main__":
    main()
