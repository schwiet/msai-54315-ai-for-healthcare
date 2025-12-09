"""
Model loading and embedding generation for RAG Doctor.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig

from .config import MODEL_ID, BERT_MODEL_ID, DEVICE, MAX_TOKEN_LENGTH, LLM_DEVICE_MAP


def load_models():
    """Load BERT and LLM models."""
    print(f"🔌 Loading Models (BERT + {MODEL_ID})...")
    print(f"   LLM device map: {LLM_DEVICE_MAP}")
    models = {}
    use_4bit = torch.cuda.is_available()

    if use_4bit:
        print("   Using 4-bit quantization via bitsandbytes for all transformer models.")
    else:
        print("   ⚠️ CUDA not available - falling back to full-precision models on CPU.")
    
    # BERT for query embedding
    models['bert_tokenizer'] = AutoTokenizer.from_pretrained(BERT_MODEL_ID)
    bert_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    ) if use_4bit else None
    if bert_bnb_config:
        models['bert_model'] = AutoModel.from_pretrained(
            BERT_MODEL_ID,
            quantization_config=bert_bnb_config,
            device_map="auto",
        )
    else:
        models['bert_model'] = AutoModel.from_pretrained(BERT_MODEL_ID).to(DEVICE)
    
    # LLM for query parsing and answer generation
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    ) if use_4bit else None

    models['llm_tokenizer'] = AutoTokenizer.from_pretrained(MODEL_ID)
    models['llm_model'] = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=LLM_DEVICE_MAP if use_4bit else "cpu",  # Force CUDA on high-VRAM systems
    )
    
    return models


def get_llm_device(models):
    """
    Get the device for LLM inputs when using device_map='auto'.
    
    For models with device_map='auto', layers may be spread across devices.
    We need to send inputs to the device of the first layer (embed_tokens).
    """
    llm = models['llm_model']
    
    # For models with device_map, get the device of the input embeddings
    if hasattr(llm, 'hf_device_map'):
        # The model has a device map - get the first layer's device
        try:
            # Try common patterns for getting the embedding layer device
            if hasattr(llm, 'model') and hasattr(llm.model, 'embed_tokens'):
                return llm.model.embed_tokens.weight.device
            elif hasattr(llm, 'transformer') and hasattr(llm.transformer, 'wte'):
                return llm.transformer.wte.weight.device
        except Exception:
            pass
    
    # Fallback: get device from first parameter
    try:
        return next(llm.parameters()).device
    except StopIteration:
        return torch.device('cpu')


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

