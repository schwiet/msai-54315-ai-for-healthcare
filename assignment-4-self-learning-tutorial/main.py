import pandas as pd

RANDOM_STATE = 1138

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
cohort_abnormal_glucose = lab_abnormal_glucose['SUBJECT_ID'].unique()

# get a cohort of patients with multiple abnormal glucose lab events
lab_abnormal_glucose_multi = lab_abnormal_glucose[
    lab_abnormal_glucose['SUBJECT_ID'].isin(
        lab_abnormal_glucose['SUBJECT_ID'].value_counts()[lambda x: x > 1].index
    )
]
lab_abnormal_glucose_multi.size
cohort_abnormal_glucose_multi = lab_abnormal_glucose_multi['SUBJECT_ID'].unique()
# Visualize relationship of cohorts

import matplotlib.pyplot as plt
import seaborn as sns
def overlap_heatmap(a, b, labels=("A", "B")):
    A, B = set(a), set(b)
    inter = A & B

    df = pd.DataFrame(
        [[len(A),       len(inter)],
         [len(inter),   len(B)]],
        index=[labels[0], labels[1]],
        columns=[labels[0], labels[1]],
    )

    ax = sns.heatmap(df, annot=True, fmt="d", cbar=False)
    ax.set_title("Set sizes (diagonal) and intersection (off-diagonal)")
    plt.tight_layout()
    plt.show()

    # print the Venn-style counts to help interpret the heatmap
    print({
        f"Only {labels[0]}": len(A - B),
        "Both": len(inter),
        f"Only {labels[1]}": len(B - A),
    })

overlap_heatmap(cohort_diab, cohort_abnormal_glucose, labels=("Diagnosed Diabetes", "Abnormal Glucose"))

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
a1c_related_lab_events.head()

# Source of normal A1C levels:
# https://my.clevelandclinic.org/health/diagnostics/9731-a1c
# **NOTE**: used for demonstration purposes only, not for medical decision-making
# To improve accuracy of model, medical professionals should be consulted for the
# determination of normal A1C levels.

lab_abnormal_a1c = a1c_related_lab_events[
  (a1c_related_lab_events['VALUENUM'] >= 5.7)
]
lab_abnormal_a1c.size
cohort_abnormal_a1c = lab_abnormal_a1c['SUBJECT_ID'].unique()

overlap_heatmap(cohort_diab, cohort_abnormal_a1c, labels=("Diagnosed Diabetes", "Abnormal A1C"))
overlap_heatmap(cohort_abnormal_glucose, cohort_abnormal_a1c, labels=("Abnormal Glucose", "Abnormal A1C"))

# Positive Label Cohort by merging those diagnosed with those with abnormal glucose or A1C
positive_label_cohort = set(cohort_diab) | set(cohort_abnormal_glucose_multi) | set(cohort_abnormal_a1c)
len(positive_label_cohort)

# see if I have enough patients with nominal glucose and a1c to train a model
lab_nominal_glucose = glucose_related_lab_events[
  (glucose_related_lab_events['VALUENUM'] >= 70) & (glucose_related_lab_events['VALUENUM'] <= 110)
]
lab_nominal_glucose.size
lab_nominal_glucose.head()
cohort_nominal_glucose = lab_nominal_glucose['SUBJECT_ID'].unique()
cohort_nominal_glucose.size

lab_nominal_a1c = a1c_related_lab_events[
  (a1c_related_lab_events['VALUENUM'] < 5.7)
]
lab_nominal_a1c.size
lab_nominal_a1c.head()
cohort_nominal_a1c = lab_nominal_a1c['SUBJECT_ID'].unique()
lab_nominal_a1c.head()
cohort_nominal_a1c.size

# Negative Label Cohort by merging those with nominal glucose and a1c, but not in the positive label cohort
negative_label_cohort = (set(cohort_nominal_glucose) | set(cohort_nominal_a1c)) - positive_label_cohort
len(negative_label_cohort)

# create a new DataFrame from first_adm including only admissions for patients in one of the label cohorts
positive_and_negative_ids = set(positive_label_cohort) | set(negative_label_cohort)
first_admissions_in_label_cohorts = first_adm[first_adm['SUBJECT_ID'].isin(positive_and_negative_ids)]
first_admissions_in_label_cohorts.head()

first_admissions_in_label_cohorts['label'] = first_admissions_in_label_cohorts['SUBJECT_ID'].apply(
    lambda x: x in positive_label_cohort
)
first_admissions_in_label_cohorts['label'].value_counts()

# merge with patients table to add gender and date of birth
first_admissions_in_label_cohorts = first_admissions_in_label_cohorts.merge(
    patients[['SUBJECT_ID', 'GENDER', 'DOB']],
    on='SUBJECT_ID',
    how='left'
)


# to avoid int64 overflow, set all dates of birth to the year 2000 if they are before 2000
# (this addresses the age shift to comply with HIPAA for patients older than 89)
first_admissions_in_label_cohorts['DOB'] = (
  first_admissions_in_label_cohorts['DOB'].apply(
      lambda d: d.replace(year=2000) if d.year < 2000 else d
  )
)
# calculate age at admission
first_admissions_in_label_cohorts['age_at_admission'] = (
    (first_admissions_in_label_cohorts['ADMITTIME'] - first_admissions_in_label_cohorts['DOB']).dt.days / 365.25
).clip(lower=0, upper=120)
first_admissions_in_label_cohorts['ADMITTIME'].describe()
first_admissions_in_label_cohorts['DOB'].describe()
first_admissions_in_label_cohorts['age_at_admission'].describe()

first_admissions_in_label_cohorts.info()

# feature engineering. This is the initial feature set for the model.

cat_features = [
  'ADMISSION_TYPE',
  'ADMISSION_LOCATION',
  'INSURANCE',
  'LANGUAGE',
  'RELIGION',
  'MARITAL_STATUS',
  'ETHNICITY',
  'GENDER'
]
num_features = ['age_at_admission']
features = cat_features + num_features

first_admissions_in_label_cohorts[features].isna().sum()

# assume that missing language means English
first_admissions_in_label_cohorts['LANGUAGE'] = first_admissions_in_label_cohorts['LANGUAGE'].fillna('ENGL')
first_admissions_in_label_cohorts['LANGUAGE'].value_counts()

# mark missing marital status and religion as 'Unknown'
first_admissions_in_label_cohorts['MARITAL_STATUS'] = first_admissions_in_label_cohorts['MARITAL_STATUS'].fillna('UNKNOWN (DEFAULT)')
first_admissions_in_label_cohorts['MARITAL_STATUS'].value_counts()

# mark missing religion as 'NOT SPECIFIED'
first_admissions_in_label_cohorts['RELIGION'] = first_admissions_in_label_cohorts['RELIGION'].fillna('NOT SPECIFIED')
first_admissions_in_label_cohorts['RELIGION'].value_counts()

first_admissions_in_label_cohorts[features].isna().sum()

########################################################
# Prepare dataset for modeling DecisionTreeClassifier
########################################################

# Features
X = first_admissions_in_label_cohorts[features].copy()
# labels
y = first_admissions_in_label_cohorts['label'].copy()

# split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# one-hot encode the categorical features
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
], remainder='passthrough')

# train a shallow tree classifier to do some initial evaluation
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    max_depth=6,            # shallow tree for interpretability
    class_weight='balanced',
    random_state=RANDOM_STATE
)

model = Pipeline([
    ('preprocess', preprocess),
    ('tree', clf)
])

model.fit(X_train, y_train)

# evaluate the model
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve, f1_score

def evaluate_model(model, X_test, y_test):
  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)[:,1]

  print(classification_report(y_test, y_pred))
  print("ROC AUC:", roc_auc_score(y_test, y_prob))

  ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

  prec, rec, thresh = precision_recall_curve(y_test, y_prob, pos_label=True)

  # Example: pick threshold that achieves ~0.30 negative precision (or use F1 for 'False' class)
  # We'll choose the threshold maximizing macro-F1 as a demo:
  import numpy as np
  def macro_f1_at(t):
      yp = (y_prob >= t).astype(bool)
      from sklearn.metrics import f1_score
      return f1_score(y_test, yp, average='macro')

  best_t = max(np.linspace(0.1, 0.9, 41), key=macro_f1_at)
  print("Best threshold for macro-F1:", best_t)
  y_pred_adj = (y_prob >= best_t).astype(bool)
  print(classification_report(y_test, y_pred_adj))
  ConfusionMatrixDisplay.from_predictions(y_test, y_pred_adj)
  RocCurveDisplay.from_predictions(y_test, y_prob)
  # PrecisionRecallDisplay.from_predictions(y_test, y_prob)

evaluate_model(model, X_test, y_test)

# try a different sized tree
eval_clf = DecisionTreeClassifier(
    max_depth=4,
    class_weight='balanced',
    random_state=RANDOM_STATE
)

eval_model = Pipeline([
    ('preprocess', preprocess),
    ('tree', eval_clf)
])

eval_model.fit(X_train, y_train)

evaluate_model(eval_model, X_test, y_test)

# ChatGPT says:
# "The model shows good discrimination (AUC ≈ 0.70) using only admission-level
# features, which is acceptable for an early predictive model with minimal
# clinical data."

# Try Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,              # let trees grow, control with min_samples_leaf
    min_samples_leaf=5,          # stabilizes splits on imbalance
    max_features='sqrt',         # good default
    class_weight='balanced',     # handle ~9:1 imbalance
    n_jobs=-1,
    random_state=RANDOM_STATE
)

model = Pipeline([
    ('prep', preprocess),
    ('rf', rf)
])

model.fit(X_train, y_train)

evaluate_model(model, X_test, y_test)

# Now compare to a dummy model
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='stratified', random_state=RANDOM_STATE)
dummy.fit(X_train, y_train) 

evaluate_model(dummy, X_test, y_test)