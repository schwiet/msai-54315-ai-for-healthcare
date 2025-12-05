"""
RAG Doctor - Enhanced Clinical Patient Search System

This package provides intelligent patient search using:
1. LLM-based query parsing to extract structured parameters
2. Multi-modal vector search for Subject ID lookups
3. Filtered embedding search for demographic/diagnosis queries
"""

from .config import DEVICE, MODEL_ID, DATA_DIR
from .data_loader import load_data, build_indices
from .models import load_models, get_query_embedding
from .query_parser import parse_query
from .filters import get_filtered_patient_ids
from .search import search_by_subject_id, search_filtered_embeddings, search_all_embeddings
from .generator import get_patient_text, generate_answer

__all__ = [
    # Config
    'DEVICE',
    'MODEL_ID', 
    'DATA_DIR',
    # Data
    'load_data',
    'build_indices',
    # Models
    'load_models',
    'get_query_embedding',
    # Query parsing
    'parse_query',
    # Filtering
    'get_filtered_patient_ids',
    # Search
    'search_by_subject_id',
    'search_filtered_embeddings',
    'search_all_embeddings',
    # Generation
    'get_patient_text',
    'generate_answer',
]

