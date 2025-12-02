import pandas as pd

chunksize = 10000
chunks = []

# Iterate through the file in chunks
for chunk in pd.read_csv("./mimic-iii/NOTEEVENTS.csv.gz", chunksize=chunksize):
    # FILTER: Keep only discharge summaries
    filtered_chunk = chunk[chunk['CATEGORY'] == 'Discharge summary']
    
    # STORE: Add the filtered piece to our list
    chunks.append(filtered_chunk)

# Glue all the pieces together
discharge_summaries = pd.concat(chunks)
discharge_summaries.info()

discharge_summaries.head()
print(discharge_summaries.shape)

# condense the dataset to one row per patient
patient_summaries = discharge_summaries.groupby('SUBJECT_ID')['TEXT'].apply(
    lambda x: '\n\n-----------\n\n'.join(x))

print(patient_summaries.shape)

from transformers import AutoTokenizer, AutoModel
import torch

# 1. Define the device (Use GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Load the specific medical 'Translator' and 'Brain'
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)