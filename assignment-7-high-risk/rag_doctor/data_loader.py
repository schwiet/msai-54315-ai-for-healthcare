"""
Data loading and index building for RAG Doctor.
"""

import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .config import DATA_DIR, CCS_CROSSWALK_PATH, DEVICE, KNN_INDEX_NEIGHBORS


def load_data():
    """Load all required data files for the RAG system."""
    print(f"🧑‍⚕️ Initializing Enhanced RAG Doctor on {DEVICE}...")
    
    data = {}
    progress_started = False

    def _announce(item):
        nonlocal progress_started
        prefix = "   📄 Loading: " if not progress_started else ", "
        print(f"{prefix}{item}", end="", flush=True)
        progress_started = True

    # load patient embeddings (text-only vectors)
    _announce("patient_embeddings.csv")
    embeddings_path = os.path.join(DATA_DIR, "patient_embeddings.csv")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Could not find {embeddings_path}! Run note-events-embeddings.py first.")
    data['embeddings'] = pd.read_csv(embeddings_path)
    
    # load multi-modal vectors if available
    _announce("multimodal_vectors.csv")
    multimodal_path = os.path.join(DATA_DIR, "multimodal_vectors.csv")
    if os.path.exists(multimodal_path):
        data['multimodal'] = pd.read_csv(multimodal_path)
    else:
        data['multimodal'] = None
    
    # load structured features for filtering
    _announce("structured_features.csv")
    structured_path = os.path.join(DATA_DIR, "structured_features.csv")
    if os.path.exists(structured_path):
        data['structured'] = pd.read_csv(structured_path)
    else:
        data['structured'] = None
    
    # load raw MIMIC-III tables for filtering
    _announce("PATIENTS.csv.gz")
    data['patients'] = pd.read_csv(os.path.join(DATA_DIR, "PATIENTS.csv.gz"))
    _announce("ADMISSIONS.csv.gz")
    data['admissions'] = pd.read_csv(os.path.join(DATA_DIR, "ADMISSIONS.csv.gz"))
    _announce("DIAGNOSES_ICD.csv.gz")
    data['diagnoses'] = pd.read_csv(os.path.join(DATA_DIR, "DIAGNOSES_ICD.csv.gz"))
    
    # load CCS crosswalk for diagnosis mapping
    if os.path.exists(CCS_CROSSWALK_PATH):
        _announce("CCS crosswalk")
        ccs_crosswalk = pd.read_csv(CCS_CROSSWALK_PATH, skiprows=1)
        # clean up the columns
        ccs_map = ccs_crosswalk[[
            "'ICD-9-CM CODE'", 
            "'CCS CATEGORY'", 
            "'CCS CATEGORY DESCRIPTION'"
        ]].copy()
        ccs_map.columns = ['ICD9_CODE', 'CCS_CATEGORY', 'CCS_DESCRIPTION']
        for col in ccs_map.columns:
            if ccs_map[col].dtype == 'object':
                ccs_map[col] = ccs_map[col].astype(str).str.strip("'\"").str.strip()
        data['ccs_map'] = ccs_map
        
        # get unique CCS descriptions for query parsing
        data['ccs_descriptions'] = ccs_map['CCS_DESCRIPTION'].dropna().unique().tolist()
    else:
        data['ccs_map'] = None
        data['ccs_descriptions'] = []
    
    # finish the progress line
    if progress_started:
        print()
    
    # load clinical notes text
    print("   📝 Reading clinical notes (this may take a moment)...")
    chunksize = 20000
    chunks = []
    notes_path = os.path.join(DATA_DIR, "NOTEEVENTS.csv.gz")
    for chunk in pd.read_csv(notes_path, chunksize=chunksize):
        filtered = chunk[chunk['CATEGORY'] == 'Discharge summary']
        chunks.append(filtered[['SUBJECT_ID', 'TEXT']])
    df_text = pd.concat(chunks)
    df_text_grouped = df_text.groupby('SUBJECT_ID')['TEXT'].apply(lambda x: '\n'.join(x)).reset_index()
    data['notes'] = df_text_grouped
    
    return data


def build_indices(data):
    """Build search indices from loaded data."""
    indices = {}
    
    # build embedding index (text-only for filtered search)
    print("   🔍 Building embedding search index...")
    df_embeddings = data['embeddings']
    feature_cols = [c for c in df_embeddings.columns if c != 'SUBJECT_ID']
    embedding_matrix = df_embeddings[feature_cols].values
    indices['embedding_matrix'] = embedding_matrix
    indices['embedding_ids'] = df_embeddings['SUBJECT_ID'].values
    
    knn_embeddings = NearestNeighbors(n_neighbors=KNN_INDEX_NEIGHBORS, metric='cosine', algorithm='brute')
    knn_embeddings.fit(embedding_matrix)
    indices['knn_embeddings'] = knn_embeddings
    
    # build multi-modal index if available
    if data['multimodal'] is not None:
        print("   🔮 Building multi-modal search index...")
        df_multimodal = data['multimodal']
        feature_cols = [c for c in df_multimodal.columns if c != 'SUBJECT_ID']
        multimodal_matrix = df_multimodal[feature_cols].values
        indices['multimodal_matrix'] = multimodal_matrix
        indices['multimodal_ids'] = df_multimodal['SUBJECT_ID'].values
        
        knn_multimodal = NearestNeighbors(n_neighbors=KNN_INDEX_NEIGHBORS, metric='cosine', algorithm='brute')
        knn_multimodal.fit(multimodal_matrix)
        indices['knn_multimodal'] = knn_multimodal
    
    # create master database with text
    print("   📚 Building master patient database...")
    master_db = pd.merge(df_embeddings, data['notes'], on='SUBJECT_ID', how='inner')
    indices['master_db'] = master_db
    
    # 4. Build diagnosis lookup
    if data['ccs_map'] is not None:
        print("   🏷️  Building diagnosis lookup...")
        diagnoses = data['diagnoses'].dropna(subset=['ICD9_CODE'])
        diagnoses['ICD9_CODE'] = diagnoses['ICD9_CODE'].astype(str).str.strip()
        diagnoses_ccs = diagnoses.merge(data['ccs_map'], on='ICD9_CODE', how='left')
        diagnoses_ccs['CCS_DESCRIPTION'] = diagnoses_ccs['CCS_DESCRIPTION'].fillna('Unmapped')
        
        # Group diagnoses by patient
        patient_diagnoses = diagnoses_ccs.groupby('SUBJECT_ID')['CCS_DESCRIPTION'].apply(list).reset_index()
        patient_diagnoses.columns = ['SUBJECT_ID', 'DIAGNOSES']
        indices['patient_diagnoses'] = patient_diagnoses
    
    return indices

