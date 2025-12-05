"""
Configuration constants for RAG Doctor.
"""

import torch

# Model identifiers
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.2"
LLAMA_3_8B = "meta-llama/Meta-Llama-3-8B-Instruct"

# Active model selection
MODEL_ID = MISTRAL_7B

# BERT model for clinical embeddings
BERT_MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data paths
DATA_DIR = "./mimic-iii"
CCS_CROSSWALK_PATH = "./Single_Level_CCS_2015/$dxref 2015.csv"

# Search parameters
DEFAULT_K_NEIGHBORS = 5
KNN_INDEX_NEIGHBORS = 10

# Text processing
MAX_CONTEXT_LENGTH = 3000
MAX_TOKEN_LENGTH = 512

