"""
Patient filtering based on demographic and clinical criteria.
"""

import pandas as pd


def get_filtered_patient_ids(parsed_query, data, indices):
    """
    Filter patients based on parsed query parameters.
    Returns a set of Subject IDs that match the filters.
    """
    # start with all patients that have embeddings
    valid_ids = set(indices['embedding_ids'])
    
    # apply gender filter
    if parsed_query['gender']:
        gender_ids = set(data['patients'][
            data['patients']['GENDER'] == parsed_query['gender']
        ]['SUBJECT_ID'].values)
        valid_ids &= gender_ids
        print(f"      Gender filter ({parsed_query['gender']}): {len(valid_ids)} patients")
    
    # apply ethnicity filter
    if parsed_query['ethnicity']:
        ethnicity_pattern = parsed_query['ethnicity'].upper()
        ethnicity_ids = set(data['admissions'][
            data['admissions']['ETHNICITY'].str.contains(ethnicity_pattern, case=False, na=False)
        ]['SUBJECT_ID'].values)
        valid_ids &= ethnicity_ids
        print(f"      Ethnicity filter ({ethnicity_pattern}): {len(valid_ids)} patients")
    
    # apply religion filter
    if parsed_query['religion']:
        religion_pattern = parsed_query['religion'].upper()
        religion_ids = set(data['admissions'][
            data['admissions']['RELIGION'].str.contains(religion_pattern, case=False, na=False)
        ]['SUBJECT_ID'].values)
        valid_ids &= religion_ids
        print(f"      Religion filter ({religion_pattern}): {len(valid_ids)} patients")
    
    # apply age filter (requires computing age from DOB and admission time)
    if parsed_query['age_min'] or parsed_query['age_max']:
        valid_ids = _apply_age_filter(parsed_query, data, valid_ids)
    
    # apply diagnosis filter
    if parsed_query['diagnoses'] and 'patient_diagnoses' in indices:
        valid_ids = _apply_diagnosis_filter(parsed_query, indices, valid_ids)
    
    return valid_ids


def _apply_age_filter(parsed_query, data, valid_ids):
    """Apply age-based filtering."""
    # compute ages
    patients_with_age = data['patients'].copy()
    patients_with_age['DOB'] = pd.to_datetime(patients_with_age['DOB'])
    
    admissions = data['admissions'].copy()
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
    
    # get first admission for each patient
    first_admission = admissions.groupby('SUBJECT_ID')['ADMITTIME'].min().reset_index()
    patients_with_age = patients_with_age.merge(first_admission, on='SUBJECT_ID', how='inner')
    
    # fix HIPAA-shifted dates (pre-2000 DOB)
    patients_with_age['DOB'] = patients_with_age['DOB'].apply(
        lambda d: d.replace(year=2000) if d.year < 2000 else d
    )
    
    # calculate age
    patients_with_age['age'] = (
        (patients_with_age['ADMITTIME'] - patients_with_age['DOB']).dt.days / 365.25
    )
    # cap shifted patients at 90
    patients_with_age['age'] = patients_with_age['age'].where(
        patients_with_age['age'] < 200, 90
    )
    
    if parsed_query['age_min']:
        age_ids = set(patients_with_age[
            patients_with_age['age'] >= parsed_query['age_min']
        ]['SUBJECT_ID'].values)
        valid_ids &= age_ids
        print(f"      Age >= {parsed_query['age_min']}: {len(valid_ids)} patients")
    
    if parsed_query['age_max']:
        age_ids = set(patients_with_age[
            patients_with_age['age'] <= parsed_query['age_max']
        ]['SUBJECT_ID'].values)
        valid_ids &= age_ids
        print(f"      Age <= {parsed_query['age_max']}: {len(valid_ids)} patients")
    
    return valid_ids


def _apply_diagnosis_filter(parsed_query, indices, valid_ids):
    """Apply diagnosis-based filtering."""
    diagnosis_matches = set()
    for _, row in indices['patient_diagnoses'].iterrows():
        patient_diags = [d.lower() for d in row['DIAGNOSES']]
        for query_diag in parsed_query['diagnoses']:
            # fuzzy match on diagnosis description
            if any(query_diag.lower() in d for d in patient_diags):
                diagnosis_matches.add(row['SUBJECT_ID'])
                break
    
    if diagnosis_matches:
        valid_ids &= diagnosis_matches
        print(f"      Diagnosis filter ({parsed_query['diagnoses']}): {len(valid_ids)} patients")
    
    return valid_ids

