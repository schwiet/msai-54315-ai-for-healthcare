from operator import concat
import pandas as pd

# keep only the timestamp for labevents to find first-order times + values for
# labeling. Also include definitions of lab events
labs = pd.read_csv(
  '../mimic-iii/LABEVENTS.csv.gz',
  usecols=[
    'SUBJECT_ID',
    'HADM_ID',
    'ITEMID',
    'CHARTTIME',
    'VALUENUM',
    'VALUEUOM'
  ],
  parse_dates=['CHARTTIME']
)
d_labitems = pd.read_csv("../mimic-iii/D_LABITEMS.csv.gz")

# import patients, admissions and diagnoses tables
patients = pd.read_csv(
  '../mimic-iii/PATIENTS.csv.gz',
  parse_dates=['DOB'])
admissions = pd.read_csv(
  '../mimic-iii/ADMISSIONS.csv.gz',
  parse_dates=['ADMITTIME','DISCHTIME'])
diagnoses = pd.read_csv(
  '../mimic-iii/DIAGNOSES_ICD.csv.gz',
  usecols=['SUBJECT_ID','HADM_ID','ICD9_CODE'])
d_diag = pd.read_csv('../mimic-iii/D_ICD_DIAGNOSES.csv.gz')


#####
# get the first admission for each patient
#####

admissions.info()
first_adm = (
    admissions.sort_values(['SUBJECT_ID','ADMITTIME'])
    .assign(rn=lambda df: df.groupby('SUBJECT_ID').cumcount()+1)
    .query('rn==1')
)
first_adm.head()

#####
# get the cohort of patients with diagnosis of diabetes
#####
# TODO: might prior diagnoses only be captured in Charts? look into this
diagnoses = diagnoses.assign(
  is_diabetes = diagnoses['ICD9_CODE'].astype(str).str.startswith('250')
)
diag_labeled = diagnoses.merge(d_diag, on='ICD9_CODE', how='left')
diagnoses.head()
# verify

diab_related = diag_labeled[diag_labeled['is_diabetes'] == True]

# get a cohort of patients with a diabetes-related diagnosis
cohort_diab = diab_related['SUBJECT_ID'].unique()
cohort_diab.size

#####
# get a cohort of patients with diabetes-related lab work
#####
# To increase the sample size to include those who _should_ be tested for
# diabetes (i.e. those who are at risk due to high A1C), we want to apply a
# positive label even to undiagnosed patients if their labs indicate elevated
# risk.
#####
# First, determine which labs to evaluate.

glucose_related_labs = d_labitems[
  d_labitems['LABEL'].str.contains('glucose', case=False, na=False) &
  d_labitems['FLUID'].str.contains('blood|plasma|serum', case=False, na=False)
]
glucose_related_labs.head()

glucose_related_lab_events = labs[
  labs['ITEMID'].isin(glucose_related_labs['ITEMID']) & labs['VALUEUOM'].notna()
]
glucose_related_lab_events['VALUEUOM'].unique()
glucose_related_lab_events.head()

# TODO: should probably remove lab events for patients already in cohort A?

# Source for normal blood glucose levels:
# https://my.clevelandclinic.org/health/diagnostics/12363-blood-glucose-test
# **NOTE**: used for demonstration purposes only, not for medical decision-making
# To improve accuracy of model, medical professionals should be consulted for the
# determination of normal blood glucose levels.

lab_abnormal_glucose = glucose_related_lab_events[
  (glucose_related_lab_events['VALUENUM'] < 70) | (glucose_related_lab_events['VALUENUM'] > 110)
]
lab_abnormal_glucose.size

a1c_related_labs = d_labitems[
  d_labitems['LABEL'].str.contains(
    'a1c|hba1c|glycohemoglobin',
    case=False,
    na=False)
]
a1c_related_labs.head()

a1c_related_lab_events = labs[
  labs['ITEMID'].isin(a1c_related_labs['ITEMID']) & labs['VALUEUOM'].notna()
]
a1c_related_lab_events['VALUEUOM'].unique()
# TODO: need to interpret A1C lab events