import numpy as np
import pandas as pd

# Load MIMIC-III tables
patients = pd.read_csv("./mimic-iii/PATIENTS.csv.gz")
admissions = pd.read_csv("./mimic-iii/ADMISSIONS.csv.gz")

##################################################################
# Prepare demographic structured features
##################################################################

# Parse date columns
patients["DOB"] = pd.to_datetime(patients["DOB"])
admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])

# Build dataset: one row per admission
# Include admission-level demographic features
admission_demographics = [
    "SUBJECT_ID", "HADM_ID", "ADMITTIME",
    "ETHNICITY", "INSURANCE", "LANGUAGE", "RELIGION", "MARITAL_STATUS"
]
dataset = admissions[admission_demographics].copy()

# Join patient demographic information
patient_demographics = patients[
    ["SUBJECT_ID", "DOB", "GENDER"]
].copy()

dataset = dataset.merge(
    patient_demographics,
    on="SUBJECT_ID",
    how="left"
)

# to avoid int64 overflow, set all dates of birth to the year 2000 if they are before 2000
# (this addresses the age shift to comply with HIPAA for patients older than 89)
dataset['DOB'] = (
  dataset['DOB'].apply(
      lambda d: d.replace(year=2000) if d.year < 2000 else d
  )
)

# Calculate raw age at admission (in years)
dataset["age_raw"] = (
    (dataset["ADMITTIME"] - dataset["DOB"]).dt.days / 365.25
)

# Handle HIPAA-shifted patients (age > 300 means shifted by ~300 years)
# Compute true age: if raw > 200, add back the ~300-year shift to get age at first admission
# For simplicity, cap anyone with raw age > 200 at 90 (a conservative estimate)
dataset["age_at_admission"] = dataset["age_raw"].where(
    dataset["age_raw"] < 200,  # normal patients
    90  # shifted patients: set to 90 as a placeholder
)

# Drop intermediate columns used for age calculation
dataset = dataset.drop(columns=["DOB", "age_raw"])

# Display summary of demographic features
print("Dataset shape:", dataset.shape)
print("\nDemographic features:")
print(f"  Gender: {dataset['GENDER'].value_counts().to_dict()}")
print(f"  Ethnicity (top 5): {dataset['ETHNICITY'].value_counts().head().to_dict()}")
print(f"  Insurance: {dataset['INSURANCE'].value_counts().to_dict()}")

dataset.head()

dataset.info()

# assume that missing language means English
dataset["LANGUAGE"] = dataset["LANGUAGE"].fillna("ENGL")

# mark missing marital status and religion as 'Unknown'
dataset['MARITAL_STATUS'] = dataset['MARITAL_STATUS'].fillna('UNKNOWN (DEFAULT)')
dataset['MARITAL_STATUS'].value_counts()

# mark missing religion as 'NOT SPECIFIED'
dataset['RELIGION'] = dataset['RELIGION'].fillna('NOT SPECIFIED')
dataset['RELIGION'].value_counts()

dataset["LANGUAGE"].unique().size

# Condense Christian religion groups into a single category
# Common Christian denominations in MIMIC-III
christian_keywords = [
    "CATHOLIC", "PROTESTANT", "METHODIST", "EPISCOPALIAN", 
    "BAPTIST", "LUTHERAN", "PRESBYTERIAN", "CHRISTIAN", 
    "QUAKER", "UNITARIAN", "PENTECOSTAL", "ADVENTIST"
]

def condense_religion(religion):
    """Group Christian denominations into a single 'CHRISTIAN' category."""
    if pd.isna(religion):
        return "NOT SPECIFIED"
    religion_upper = str(religion).upper()
    # Check if religion contains any Christian keyword
    if any(keyword in religion_upper for keyword in christian_keywords):
        return "CHRISTIAN"
    return religion_upper

dataset["RELIGION"] = dataset["RELIGION"].apply(condense_religion)
print("\nReligion after condensing Christian groups:")
print(dataset["RELIGION"].value_counts())

# One-hot encode categorical features
categorical_features = ["GENDER", "ETHNICITY", "INSURANCE", "LANGUAGE", "RELIGION", "MARITAL_STATUS"]

# Create one-hot encoded columns
for feature in categorical_features:
    dummies = pd.get_dummies(dataset[feature], prefix=feature, dtype=int)
    dataset = pd.concat([dataset, dummies], axis=1)

# Drop original categorical columns
dataset = dataset.drop(columns=categorical_features)

print(f"\nDataset shape after one-hot encoding: {dataset.shape}")
print(f"Number of one-hot encoded columns: {dataset.shape[1] - len(['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'age_at_admission'])}")
dataset.head()
dataset.shape

##################################################################
# Prepare clinical structured features
##################################################################

# Load clinical tables
diagnoses = pd.read_csv("./mimic-iii/DIAGNOSES_ICD.csv.gz")
procedures = pd.read_csv("./mimic-iii/PROCEDURES_ICD.csv.gz")

# Drop rows where ICD9_CODE is missing
diagnoses = diagnoses.dropna(subset=['ICD9_CODE'])

# To keep diagnoses codes from exploding our dimensionality,
# we use CCS Crosswalk to collapse related diagnoses into
# a smaller set of categories.

# Skip the first row (note/comment) before the column headers
ccs_crosswalk = pd.read_csv("./Single_Level_CCS_2015/$dxref 2015.csv", skiprows=1)
print("CCS Crosswalk columns:", ccs_crosswalk.columns.tolist())
ccs_crosswalk.head()

# Create diagnoses dataframe with CCS categories
# Column names in the crosswalk file
icd_col = "'ICD-9-CM CODE'"
ccs_cat_col = "'CCS CATEGORY'"
ccs_desc_col = "'CCS CATEGORY DESCRIPTION'"

print(f"\nUsing columns:")
print(f"  ICD code column: {icd_col}")
print(f"  CCS category column: {ccs_cat_col}")
print(f"  Description column: {ccs_desc_col}")

# Prepare crosswalk: select relevant columns and strip any quotes from values
ccs_map = ccs_crosswalk[[icd_col, ccs_cat_col, ccs_desc_col]].copy()
ccs_map.columns = ['ICD9_CODE', 'CCS_CATEGORY', 'CCS_DESCRIPTION']

# Strip quotes from all string columns if present
for col in ccs_map.columns:
    if ccs_map[col].dtype == 'object':
        ccs_map[col] = ccs_map[col].astype(str).str.strip("'\"")
        # Also remove any leading/trailing whitespace
        ccs_map[col] = ccs_map[col].str.strip()

print("\nSample of ccs_map:")
print(ccs_map.head())

# Prepare diagnoses: ensure ICD9_CODE is string and matches format
diagnoses_ccs = diagnoses[['HADM_ID', 'SUBJECT_ID', 'ICD9_CODE']].copy()
diagnoses_ccs['ICD9_CODE'] = diagnoses_ccs['ICD9_CODE'].astype(str).str.strip()

print("\nSample of diagnoses ICD9_CODE:")
print(diagnoses_ccs['ICD9_CODE'].head())
print(f"\nDiagnoses ICD9_CODE format examples: {diagnoses_ccs['ICD9_CODE'].head(5).tolist()}")
print(f"CCS map ICD9_CODE format examples: {ccs_map['ICD9_CODE'].head(5).tolist()}")

# Merge diagnoses with CCS crosswalk
diagnoses_ccs = diagnoses_ccs.merge(
    ccs_map,
    on='ICD9_CODE',
    how='left'
)

diagnoses_ccs.head()

# Fill missing CCS categories (codes not in crosswalk) with 'UNMAPPED'
diagnoses_ccs['CCS_CATEGORY'] = diagnoses_ccs['CCS_CATEGORY'].fillna('UNMAPPED')
diagnoses_ccs['CCS_DESCRIPTION'] = diagnoses_ccs['CCS_DESCRIPTION'].fillna('Unmapped ICD-9 code')

print(f"\nDiagnoses with CCS mapping:")
print(f"  Total diagnoses: {len(diagnoses_ccs)}")
print(f"  Unique CCS categories: {diagnoses_ccs['CCS_CATEGORY'].nunique()}")
print(f"  Unmapped codes: {(diagnoses_ccs['CCS_CATEGORY'] == 'UNMAPPED').sum()}")
diagnoses_ccs[diagnoses_ccs['CCS_CATEGORY'] == 'UNMAPPED'].head()

diagnoses_ccs.head()

dataset.describe()

# TODO next, turn dataset and diagnoses_ccs into a single dataframe
# with a wide format, with one row per patient. for diagnoses columns,
# we'll sum the instances of each CCS category for that patient.