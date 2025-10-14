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
# text_samples = df_diag_notes["TEXT"].head(NER_N).tolist()
text_samples = df_diag_notes["TEXT"].tolist()

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
            # normalize entity string -> lowercase, keep only letters/digits/spaces,
            e = re.sub(r"[^a-zA-Z0-9\s]", " ", ent.text.lower()).strip()
            if len(e) > 2:
                ents.append(e)
        corpus.append(ents)
    return corpus

corpus_gen = get_corpus(ents_gen)
corpus_sci = get_corpus(ents_sci)

print(len(corpus_gen[0]))
print(len(corpus_sci[0]))

from gensim.models import Word2Vec
model_gen = Word2Vec(corpus_gen, vector_size=100, window=5, min_count=1, workers=4)
model_sci = Word2Vec(corpus_sci, vector_size=100, window=5, min_count=1, workers=4)

model_gen.wv.most_similar("asthma")
model_sci.wv.most_similar("asthma")

###############################################################################
# 3 - t-SNE
###############################################################################

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def tsne_plot(model, words, showLabels=True, preTrained=False, xlim=None, ylim=None):
    labels, tokens = [], []
    for w in words:
        tokens.append(model[w] if preTrained else model.wv[w])
        if showLabels: labels.append(w)

    tokens = np.array(tokens)
    tsne_model = TSNE(perplexity=30, early_exaggeration=12, n_components=2,
                      init='pca', n_iter=1000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x, y = new_values[:,0], new_values[:,1]
    plt.figure(figsize=(16,16))
    plt.scatter(x, y, s=8)
    if showLabels:
        for i, lab in enumerate(labels):
            plt.annotate(lab, (x[i], y[i]), xytext=(5,2), textcoords='offset points',
                         ha='right', va='bottom')

    if xlim is not None: plt.xlim(*xlim)
    if ylim is not None: plt.ylim(*ylim)
    plt.show()

gen_vocabs = model_gen.wv.key_to_index.keys()
tsne_plot(model_gen,gen_vocabs, showLabels=False)
tsne_plot(model_gen,gen_vocabs, showLabels=True, xlim=(-5,5), ylim=(0,10))

sci_vocabs = model_sci.wv.key_to_index.keys()
tsne_plot(model_sci,sci_vocabs, showLabels=False)
tsne_plot(model_sci,sci_vocabs, showLabels=True, xlim=(35,45), ylim=(-42,-32))

###############################################################################
# Bonus
###############################################################################

from transformers import pipeline

# create a biomedical NER pipeline
# d4data/biomedical-ner-all is a solid, general biomedical NER model
ner_pipe = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple",  # groups wordpieces
    device=-1                       # CPU
)

# batch-run NER on text samples
def extract_ents_hf(texts, batch_size=16):
    """
    Returns list of "docs", where each doc mimics spaCy-ish .ents as a list of dicts:
      {"text": "...", "label_": "..."}
    """
    docs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="NER (HF biomedical)"):
        batch = texts[i:i+batch_size]
        # transformers pipeline supports list input directly
        outputs = ner_pipe(batch)
        # outputs is a list per input text
        for ents in outputs:
            docs.append([{"text": e["word"], "label_": e["entity_group"]} for e in ents])
    return docs

ents_hf = extract_ents_hf(text_samples)

def get_corpus_from_hf(docs):
    corpus = []
    for ents in docs:
        toks = []
        for e in ents:
            t = re.sub(r"[^a-z0-9\s]", " ", e["text"].lower()).strip()
            if len(t) > 2:
                toks.append(t)
        corpus.append(toks)
    return corpus

corpus_hf = get_corpus_from_hf(ents_hf)

# train Word2Vec and plot
model_hf = Word2Vec(corpus_hf, vector_size=100, window=5, min_count=1, workers=4)

hf_vocabs = model_hf.wv.key_to_index.keys()
tsne_plot(model_hf, hf_vocabs, showLabels=False)
tsne_plot(model_hf, hf_vocabs, showLabels=True, xlim=(-40,-30), ylim=(-5,5))