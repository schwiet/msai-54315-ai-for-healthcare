"""
Search strategies for RAG Doctor.

Provides different search methods:
- Multi-modal search (by subject ID)
- Filtered embedding search
- Full embedding search
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .models import get_query_embedding
from .config import DEFAULT_K_NEIGHBORS


def search_by_subject_id(subject_id, indices, k=DEFAULT_K_NEIGHBORS):
    """
    Search using a specific patient's multi-modal vector.
    Returns similar patients based on the full multi-modal representation.
    """
    if 'multimodal_matrix' not in indices:
        print("   ⚠️  Multi-modal vectors not available, falling back to embeddings")
        return None
    
    # Find the patient's index
    try:
        idx = np.where(indices['multimodal_ids'] == subject_id)[0][0]
    except IndexError:
        print(f"   ⚠️  Patient {subject_id} not found in multi-modal database")
        return None
    
    # Get the patient's vector
    query_vector = indices['multimodal_matrix'][idx].reshape(1, -1)
    
    # Find similar patients
    distances, result_indices = indices['knn_multimodal'].kneighbors(query_vector, n_neighbors=k+1)
    
    # Skip the first result (it's the query patient)
    results = []
    for i in range(1, min(k+1, len(result_indices[0]))):
        neighbor_idx = result_indices[0][i]
        neighbor_id = indices['multimodal_ids'][neighbor_idx]
        similarity = (1 - distances[0][i]) * 100
        results.append({
            'subject_id': int(neighbor_id),
            'similarity': similarity,
            'rank': i
        })
    
    return results


def search_filtered_embeddings(query, filtered_ids, indices, models, k=DEFAULT_K_NEIGHBORS):
    """
    Search only within a filtered set of patients using text embeddings.
    """
    # Get query embedding
    query_vec = get_query_embedding(query, models)
    
    # Get indices of filtered patients in the embedding matrix
    id_to_idx = {pid: idx for idx, pid in enumerate(indices['embedding_ids'])}
    filtered_indices = [id_to_idx[pid] for pid in filtered_ids if pid in id_to_idx]
    
    if not filtered_indices:
        return []
    
    # Extract filtered embeddings
    filtered_matrix = indices['embedding_matrix'][filtered_indices]
    filtered_patient_ids = [indices['embedding_ids'][i] for i in filtered_indices]
    
    # Build temporary KNN for filtered set
    n_neighbors = min(k, len(filtered_matrix))
    knn_filtered = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
    knn_filtered.fit(filtered_matrix)
    
    # Search
    distances, result_indices = knn_filtered.kneighbors(query_vec, n_neighbors=n_neighbors)
    
    results = []
    for i in range(len(result_indices[0])):
        neighbor_idx = result_indices[0][i]
        neighbor_id = filtered_patient_ids[neighbor_idx]
        similarity = (1 - distances[0][i]) * 100
        results.append({
            'subject_id': int(neighbor_id),
            'similarity': similarity,
            'rank': i + 1
        })
    
    return results


def search_all_embeddings(query, indices, models, k=DEFAULT_K_NEIGHBORS):
    """
    Search all patients using text embeddings (default fallback).
    """
    query_vec = get_query_embedding(query, models)
    distances, result_indices = indices['knn_embeddings'].kneighbors(query_vec, n_neighbors=k)
    
    results = []
    for i in range(len(result_indices[0])):
        neighbor_idx = result_indices[0][i]
        neighbor_id = indices['embedding_ids'][neighbor_idx]
        similarity = (1 - distances[0][i]) * 100
        results.append({
            'subject_id': int(neighbor_id),
            'similarity': similarity,
            'rank': i + 1
        })
    
    return results

