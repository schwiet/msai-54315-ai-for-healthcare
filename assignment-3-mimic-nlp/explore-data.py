import pandas as pd

NOTE_EVENTS_PATH = "../mimic-iii/NOTEEVENTS.csv.gz"
DIAGNOSES_PATH = "../mimic-iii/DIAGNOSES_ICD.csv.gz"
DIAG_DICT_PATH = "../mimic-iii/D_ICD_DIAGNOSES.csv.gz"

df_diagnoses_icd = pd.read_csv(DIAGNOSES_PATH, compression='gzip', dtype={'SEQ_NUM': 'Int64'})
df_diagnoses_icd.info()
df_diagnoses_icd.iloc[0]

df_diagnoses_dict = pd.read_csv(DIAG_DICT_PATH, compression='gzip', dtype={'SEQ_NUM': 'Int64'})
df_diagnoses_dict.info()

top_icd9_counts = df_diagnoses_icd['ICD9_CODE'].value_counts().head(20)
print("Top 10 ICD9_CODE counts:")
for code, count in top_icd9_counts.items():
    print(f"{code} ({df_diagnoses_dict[df_diagnoses_dict['ICD9_CODE'] == code]['SHORT_TITLE'].values[0]}): {count}")


# filter rows with ICD9_CODE
ICD_FILTER = ['99592']
df_filtered = df_diagnoses_icd[df_diagnoses_icd['ICD9_CODE'].isin(ICD_FILTER)]
admissions_with_diagnoses = df_filtered['HADM_ID'].unique()
print(f"Number of ICD-9 {ICD_FILTER} admissions: {admissions_with_diagnoses.size}")

# load note events
df_noteevents = pd.read_csv(NOTE_EVENTS_PATH, compression='gzip')
df_noteevents.info()

# filter notes for admissions with diagnoses
df_noteevents_filtered = df_noteevents[df_noteevents['HADM_ID'].isin(admissions_with_diagnoses)]
df_noteevents_filtered.info()

df_noteevents_filtered['CATEGORY'].unique()
df_diag_notes = df_noteevents_filtered[df_noteevents_filtered['CATEGORY'] == 'Respiratory ']
df_diag_notes.info()

# import spacy and load general and scientific models
import spacy
import os
from tqdm import tqdm
nlp_gen = spacy.load("en_core_web_sm")
nlp_sci_md = spacy.load("en_core_sci_md")

# grab a set of text samples from the filtered notes
NER_N = min(1000, len(df_diag_notes))
text_samples = df_diag_notes["TEXT"].head(NER_N).tolist()

###############################################################################
# 1 - extracted entities
###############################################################################

# define function to extract entities
def extract_ents(nlp, texts, batch_size=128):
    docs = []
    # only run the NER pipe for speed
    with nlp.select_pipes(enable=["ner"]):
        for doc in tqdm(
            nlp.pipe(texts, batch_size=batch_size),
            total=len(texts), desc=f"NER ({nlp.meta.get('name','model')})"
        ):
            docs.append(doc)
    return docs

ents_gen = extract_ents(nlp_gen, text_samples)
ents_sci = extract_ents(nlp_sci_md, text_samples)

# show entities for first text sample from each model
from spacy import displacy
displacy.render(ents_gen[0], style="ent", jupyter=True)
displacy.render(ents_sci[0], style="ent", jupyter=True)


###############################################################################
# 2 - word2vec
###############################################################################

import re
def get_corpus(docs):
    corpus = []
    for doc in docs:
        ents = []
        for ent in doc.ents:
            # normalize entity string -> lowercase, keep only letters/spaces,
            # replace spaces with underscores
            e = re.sub(r"[^a-zA-Z\s]", " ", ent.text.lower()).strip()
            e = re.sub(r"\s+", "_", e)
            if len(e) > 2:
                ents.append(e)
        corpus.append(ents)
    return corpus

corpus_gen = get_corpus(ents_gen)
corpus_sci = get_corpus(ents_sci)

print(len(corpus_gen[0]))
print(len(corpus_sci[0]))