"""
Configuration constants for RAG Doctor.
"""

import torch

# Model identifiers
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.2"
LLAMA_3_8B = "meta-llama/Meta-Llama-3-8B-Instruct"
OPEN_BIO = "aaditya/Llama3-OpenBioLLM-70B"

# Active model selection
MODEL_ID = OPEN_BIO

# BERT model for clinical embeddings
BERT_MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LLM device map - force all layers onto CUDA for high-VRAM systems like DGX
# Options:
#   "auto" - let transformers decide (may put some layers on CPU)
#   "cuda:0" or {"": "cuda:0"} - force everything onto GPU 0
#   {"": 0} - shorthand for cuda:0
LLM_DEVICE_MAP = {"": "cuda:0"} if torch.cuda.is_available() else "cpu"

# Data paths
DATA_DIR = "./mimic-iii"
CCS_CROSSWALK_PATH = "./Single_Level_CCS_2015/$dxref 2015.csv"

# Search parameters
DEFAULT_K_NEIGHBORS = 5
KNN_INDEX_NEIGHBORS = 10

# Text processing
# NOTE: this corresponds to Mistral 7B's v0.2 context window length
#       https://obot.ai/resources/learning-center/mistral-7b-instruct#h-mistral-7b-instruct-v0-2
# MAX_CONTEXT_LENGTH = 22000
# NOTE: this corresponds to meta-llama/Meta-Llama-3-70B context window length
#       https://huggingface.co/meta-llama/Meta-Llama-3-70B
#       then we divide by 2 to account for passing two sets of patient history to the LLM
MAX_CONTEXT_LENGTH = 3000
# this is used for query embedding with BERT. I'm not totally sure it is sufficient
MAX_TOKEN_LENGTH = 512

