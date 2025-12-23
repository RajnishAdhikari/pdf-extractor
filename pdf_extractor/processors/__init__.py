"""
Processors package for model grouping and RAG setup.
"""

from .model_grouper import ModelGrouper
from .rag_setup import RAGRetriever

__all__ = ["ModelGrouper", "RAGRetriever"]
