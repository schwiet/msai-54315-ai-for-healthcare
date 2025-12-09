"""
Answer generation for RAG Doctor.

Uses the LLM to generate responses based on patient notes.
"""

import torch

from .config import MAX_CONTEXT_LENGTH
from .models import get_llm_device


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
            "content": (
                "You are an expert medical AI. Answer based ONLY on the provided patient history. "
                "Be concise and clinically relevant. Your answer must explicitly address the clinical details "
                "mentioned in the user's QUESTION: confirm if each is present in the history (and briefly where), "
                "or state that it is not evident. If absent, call it out clearly."
            )
        },
        {
            "role": "user", 
            "content": (
                f"--- PATIENT HISTORY ---\n{context}\n\n"
                f"QUESTION: {query}\n\n"
                "Respond with:\n"
                "- Matched query details: bullet list of clinical details from the QUESTION that are present in the history (note brief evidence).\n"
                "- Missing or unclear details: bullet list of QUESTION details not supported by the history.\n"
                "- Clinical summary: brief, focused clinical assessment based on the history."
            )
        }
    ]
    
    llm_device = get_llm_device(models)
    input_ids = models['llm_tokenizer'].apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(llm_device)
    
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


def compare_patients(patient_text_a, patient_text_b, query, models):
    """
    Compare two patients and explain whether and why they are similar.
    If they are not very similar, explain the key differences.
    Also explicitly address which clinical details from the QUESTION are
    present or absent in each patient's history.
    """
    ctx_a = patient_text_a[:MAX_CONTEXT_LENGTH] if patient_text_a else "No patient notes available."
    ctx_b = patient_text_b[:MAX_CONTEXT_LENGTH] if patient_text_b else "No patient notes available."

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert medical AI. Given two patient histories, assess how similar they are. "
                "Be concise and clinically grounded. Always include the following information in distinct sections:\n"
                "- A similarity judgment (high/medium/low) and a 0-100 similarity score estimate, specifically based on whether or not the patients are similar based on ONLY the criteria in the QUESTION, not any other information.\n"
                "- For each clinical detail mentioned in the QUESTION: state for Patient A and Patient B whether it is present (with brief evidence) or not evident. Don't include extra information in this list.\n"
                "- Shared key problems/diagnoses, treatments, demographics.\n"
                "- Key differences (diagnoses, acuity, comorbidities, demographics, treatments).\n"
                "If similarity is low - again based on the criteria in the QUESTION - clearly state that they are not very similar and why."
            ),
        },
        {
            "role": "user",
            "content": (
                f"--- PATIENT A HISTORY ---\n{ctx_a}\n\n"
                f"--- PATIENT B HISTORY ---\n{ctx_b}\n\n"
                f"QUESTION: {query}\n\n"
                "Provide the comparison as:\n"
                "Similarity: <high/medium/low> (<score>/100)\n"
                "Query details:\n"
                "- Patient A: per clinical QUESTION detail <present/absent per detail with brief evidence>\n"
                "- Patient B: per clinical QUESTION detail <present/absent per detail with brief evidence>\n"
                "Shared factors: ...\n"
                "Differences: ...\n"
                "Overall judgment: ... based on the criteria in the QUESTION"
            ),
        },
    ]

    llm_device = get_llm_device(models)
    input_ids = models['llm_tokenizer'].apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(llm_device)

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
