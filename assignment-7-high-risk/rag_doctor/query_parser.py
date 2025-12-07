"""
LLM-based query parsing for RAG Doctor.

Extracts structured parameters from natural language queries.
"""

import re
import json
import torch

from .config import DEVICE


def parse_query(query, models, ccs_descriptions):
    """
    Use the LLM to extract structured parameters from the user query.
    
    Returns a dict with:
    - subject_id: int or None
    - gender: 'M', 'F', or None  
    - age_min: int or None
    - age_max: int or None
    - ethnicity: str or None (race/ethnic background)
    - religion: str or None
    - diagnoses: list of diagnosis descriptions or empty list
    - raw_query: the original query for embedding
    """
    
    # first, check for explicit Subject ID in the query
    subject_id_match = re.search(
        r'\b(?:subject[_\s]*id|patient[_\s]*id|subject|patient)[:\s#]*(\d+)\b', 
        query, 
        re.IGNORECASE
    )
    if subject_id_match:
        subject_id = int(subject_id_match.group(1))
    else:
        # also check for bare numbers that might be IDs
        bare_id_match = re.search(r'\bid\s*[:=]?\s*(\d+)\b', query, re.IGNORECASE)
        subject_id = int(bare_id_match.group(1)) if bare_id_match else None
    
    # sample of diagnosis categories for context (not all 285)
    sample_diagnoses = ccs_descriptions[:50] if len(ccs_descriptions) > 50 else ccs_descriptions
    diagnoses_context = ", ".join(sample_diagnoses[:30])
    
    # use LLM to extract filter parameters
    extraction_prompt = f"""Extract structured filters from this patient search query. Return ONLY valid JSON.

Query: "{query}"

Available diagnosis categories (examples): {diagnoses_context}

IMPORTANT DISTINCTIONS:
- ETHNICITY refers to race/ethnic background: WHITE, BLACK, ASIAN, HISPANIC, etc.
- RELIGION refers to religious affiliation: CHRISTIAN, CATHOLIC, JEWISH, MUSLIM, BUDDHIST, HINDU, NOT SPECIFIED, etc.
- These are DIFFERENT fields - do not confuse them!

Return JSON with these fields (use null for missing values):
{{
    "gender": "M" or "F" or null,
    "age_min": integer or null,
    "age_max": integer or null,  
    "ethnicity": string or null (ONLY race/ethnic background like "WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN - CHINESE"),
    "religion": string or null (religious affiliation like "CHRISTIAN", "JEWISH", "MUSLIM", "BUDDHIST", "NOT SPECIFIED"),
    "diagnoses": [list of diagnosis category names that match the query] or []
}}

Example queries and responses:
- "elderly female with diabetes" → {{"gender": "F", "age_min": 65, "age_max": null, "ethnicity": null, "religion": null, "diagnoses": ["Diabetes mellitus without complication", "Diabetes mellitus with complications"]}}
- "young male of christian faith with heart failure" → {{"gender": "M", "age_min": null, "age_max": 40, "ethnicity": null, "religion": "CHRISTIAN", "diagnoses": ["Congestive heart failure; nonhypertensive"]}}
- "asian patients with sepsis" → {{"gender": null, "age_min": null, "age_max": null, "ethnicity": "ASIAN", "religion": null, "diagnoses": ["Septicemia (except in labor)"]}}
- "jewish male with pneumonia" → {{"gender": "M", "age_min": null, "age_max": null, "ethnicity": null, "religion": "JEWISH", "diagnoses": ["Pneumonia"]}}

JSON response:"""

    # TODO: try adding system role to improve performance
    # TODO: try moving examples to agent role to see if results improve
    messages = [
        {"role": "user", "content": extraction_prompt}
    ]
    
    # this runs the messages through the LLM's chat template using the
    # tokenizer, creating input_ids suitable for model input (as a tensor)
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
            max_new_tokens=200,
            do_sample=False,  # deterministic for parsing
            temperature=0.1
        )
    
    response = models['llm_tokenizer'].decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # parse the response
    try:
        # find JSON in response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = {}
    except json.JSONDecodeError:
        parsed = {}
    
    return {
        # TODO: should I also try to extract subject ID with LLM if not already
        # determined?
        'subject_id': subject_id,
        'gender': parsed.get('gender'),
        'age_min': parsed.get('age_min'),
        'age_max': parsed.get('age_max'),
        'ethnicity': parsed.get('ethnicity'),
        'religion': parsed.get('religion'),
        'diagnoses': parsed.get('diagnoses', []),
        'raw_query': query
    }

