"""
Model loading and embedding generation for RAG Doctor.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig

from .config import MODEL_ID, BERT_MODEL_ID, DEVICE, MAX_TOKEN_LENGTH


def load_models():
    """Load BERT and LLM models."""
    print(f"🔌 Loading Models (BERT + {MODEL_ID})...")
    models = {}
    
    # BERT for query embedding
    models['bert_tokenizer'] = AutoTokenizer.from_pretrained(BERT_MODEL_ID)
    models['bert_model'] = AutoModel.from_pretrained(BERT_MODEL_ID).to(DEVICE)
    
    # LLM for query parsing and answer generation
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    models['llm_tokenizer'] = AutoTokenizer.from_pretrained(MODEL_ID)
    models['llm_model'] = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    return models


def get_query_embedding(text, models):
    """Convert query text to BERT embedding."""
    inputs = models['bert_tokenizer'](
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=MAX_TOKEN_LENGTH
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = models['bert_model'](**inputs)
    
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

