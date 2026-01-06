"""
Unified Indexer - Cross-reference code, documents, and logs
"""

__version__ = "1.0.0"

# Lazy imports to avoid circular import issues
def __getattr__(name):
    if name == "SourceType":
        from .models import SourceType
        return SourceType
    elif name == "IndexableChunk":
        from .models import IndexableChunk
        return IndexableChunk
    elif name == "SearchResult":
        from .models import SearchResult
        return SearchResult
    elif name == "DomainMatch":
        from .models import DomainMatch
        return DomainMatch
    elif name == "VocabularyEntry":
        from .models import VocabularyEntry
        return VocabularyEntry
    elif name == "DomainVocabulary":
        from .vocabulary import DomainVocabulary
        return DomainVocabulary
    elif name == "HybridIndex":
        from .index import HybridIndex
        return HybridIndex
    elif name == "BM25Index":
        from .index import BM25Index
        return BM25Index
    elif name == "reciprocal_rank_fusion":
        from .index import reciprocal_rank_fusion
        return reciprocal_rank_fusion
    elif name == "IndexingPipeline":
        from .pipeline import IndexingPipeline
        return IndexingPipeline
    elif name == "LLMInterface":
        from .pipeline import LLMInterface
        return LLMInterface
    elif name == "LLMEnhancedPipeline":
        from .pipeline import LLMEnhancedPipeline
        return LLMEnhancedPipeline
    elif name == "create_embedder":
        from .embeddings import create_embedder
        return create_embedder
    elif name == "HashEmbedder":
        from .embeddings import HashEmbedder
        return HashEmbedder
    elif name == "ContentParser":
        from .parsers.base import ContentParser
        return ContentParser
    elif name == "TalCodeParser":
        from .parsers.tal_parser import TalCodeParser
        return TalCodeParser
    elif name == "DocumentParser":
        from .parsers.document_parser import DocumentParser
        return DocumentParser
    elif name == "LogParser":
        from .parsers.log_parser import LogParser
        return LogParser
    raise AttributeError(f"module 'unified_indexer' has no attribute '{name}'")

__all__ = [
    "SourceType",
    "IndexableChunk", 
    "SearchResult",
    "DomainMatch",
    "VocabularyEntry",
    "DomainVocabulary",
    "HybridIndex",
    "BM25Index",
    "reciprocal_rank_fusion",
    "IndexingPipeline",
    "LLMInterface",
    "LLMEnhancedPipeline",
    "create_embedder",
    "HashEmbedder",
    "ContentParser",
    "TalCodeParser",
    "DocumentParser",
    "LogParser",
]
