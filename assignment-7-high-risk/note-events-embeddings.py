import pandas as pd

chunksize = 10000
chunks = []

# iterate through the file in chunks
for chunk in pd.read_csv("./mimic-iii/NOTEEVENTS.csv.gz", chunksize=chunksize):
    # filter: keep only discharge summaries
    filtered_chunk = chunk[chunk['CATEGORY'] == 'Discharge summary']
    
    # store: add the filtered piece to our list
    chunks.append(filtered_chunk)

# glue all the pieces together
discharge_summaries = pd.concat(chunks)
discharge_summaries.info()

discharge_summaries.head()
print(discharge_summaries.shape)

# condense the dataset to one row per patient
patient_summaries = discharge_summaries.groupby('SUBJECT_ID')['TEXT'].apply(
    lambda x: '\n\n-----------\n\n'.join(x))

print(patient_summaries.shape)

from transformers import AutoTokenizer, BertModel
import torch

# 1. Define the device (Use GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Load the specific medical 'Translator' and 'Brain'
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)

def get_patient_embedding(text, tokenizer, model, device):
    """
    Splits long text into chunks, embeds each and averages them
    """
    # tokenize with 'return_overflowing_tokens' to handle long texts
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding='max_length',
        return_overflowing_tokens=True, # creates multiple chunks
        stride=100 # overlap between chunks
    )

    # move the batch of chunks to GPU
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # run the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # take the CLS token (first token) of each chunk as the summary
    chunk_embeddings = outputs.last_hidden_state[:, 0, :]

    patient_vector = torch.mean(chunk_embeddings, dim=0)

    return patient_vector.cpu().numpy()

print("Starting embedding process... hold please 🏎️")
embeddings = []
ids = []

import math
total_patients = len(patient_summaries)
one_percent = math.floor(total_patients / 100)
print(f"One percent: {one_percent}")
for subject_id, text in patient_summaries.items():
    try:
        vec = get_patient_embedding(text, tokenizer, model, device)
        embeddings.append(vec)
        ids.append(subject_id)

        if len(embeddings) % one_percent == 0:
            print(f"Processed {len(embeddings)} embeddings")

    except Exception as e:
        print(f"Error embedding patient {subject_id}: {e}")
        continue

print("Embedding process complete! 🎉")
embedding_df = pd.DataFrame(embeddings)
embedding_df['SUBJECT_ID'] = ids
embedding_df.to_csv("./mimic-iii/patient_embeddings.csv", index=False)

print("SUCCESS: saved embeddings to ./mimic-iii/patient_embeddings.csv")