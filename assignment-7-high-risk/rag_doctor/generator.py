"""
Answer generation for RAG Doctor.

Uses the LLM to generate responses based on patient notes.
"""

import torch

from .config import DEVICE, MAX_CONTEXT_LENGTH


def get_patient_text(subject_id, indices):
    """Retrieve the clinical notes for a patient."""
    row = indices['master_db'][indices['master_db']['SUBJECT_ID'] == subject_id]
    if len(row) > 0:
        return row.iloc[0]['TEXT']
    return None


def generate_answer(query, patient_text, models):
    """Generate an answer using the LLM based on patient notes."""
    context = patient_text[:MAX_CONTEXT_LENGTH] if patient_text else "No patient notes available."
    
    messages = [
        {
            "role": "system", 
            "content": "You are an expert medical AI. Answer based ONLY on the provided patient history. Be concise and clinically relevant."
        },
        {
            "role": "user", 
            "content": f"--- PATIENT HISTORY ---\n{context}\n\nQUESTION: {query}"
        }
    ]
    
    input_ids = models['llm_tokenizer'].apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        outputs = models['llm_model'].generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=models['llm_tokenizer'].eos_token_id,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )
    
    response = models['llm_tokenizer'].decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

