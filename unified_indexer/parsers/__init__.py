"""Content Parsers for the Unified Indexer"""

__all__ = [
    "ContentParser",
    "TalCodeParser", 
    "DocumentParser",
    "LogParser",
    "GenericCodeParser"
]

def __getattr__(name):
    if name == "ContentParser":
        from .base import ContentParser
        return ContentParser
    elif name == "TalCodeParser":
        from .tal_parser import TalCodeParser
        return TalCodeParser
    elif name == "DocumentParser":
        from .document_parser import DocumentParser
        return DocumentParser
    elif name == "LogParser":
        from .log_parser import LogParser
        return LogParser
    elif name == "GenericCodeParser":
        from .code_parser import GenericCodeParser
        return GenericCodeParser
    raise AttributeError(f"module 'unified_indexer.parsers' has no attribute '{name}'")
