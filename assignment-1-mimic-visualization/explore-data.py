import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
print("Current working directory:", os.getcwd())

patients = pd.read_csv("../mimic-iii/PATIENTS.csv.gz")
admissions = pd.read_csv("../mimic-iii/ADMISSIONS.csv.gz")
transfers = pd.read_csv("../mimic-iii/TRANSFERS.csv.gz")
procedures = pd.read_csv("../mimic-iii/PROCEDURES_ICD.csv.gz")
d_procedures = pd.read_csv("../mimic-iii/D_ICD_PROCEDURES.csv.gz", usecols=["ICD9_CODE", "SHORT_TITLE"])
diagnoses = pd.read_csv("../mimic-iii/DIAGNOSES_ICD.csv.gz")
d_diagnoses = pd.read_csv("../mimic-iii/D_ICD_DIAGNOSES.csv.gz", usecols=["ICD9_CODE", "SHORT_TITLE"])
diag_map = dict(zip(d_diagnoses["ICD9_CODE"], d_diagnoses["SHORT_TITLE"]))

sns.countplot(data=admissions, x="ADMISSION_TYPE")
plt.show()

admissions["DIAGNOSIS"].value_counts()
admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])
admissions["DISCHTIME"] = pd.to_datetime(admissions["DISCHTIME"])
admissions["STAYTIME_HRS"] = (admissions["DISCHTIME"] - admissions["ADMITTIME"]).dt.total_seconds()/3600
admissions["DEATHTIME"] = pd.to_datetime(admissions["DEATHTIME"])

died_in_hospital = admissions[admissions["DEATHTIME"].notna()]

def plot_duration(data, col, hue):
    plt.figure(figsize=(10,6))

    sns.histplot(
        data=data,
        x=col,
        hue=hue,
        bins=30,         # adjust as needed
        multiple="stack" # stack bars instead of overlay
    )

    plt.xlabel("Hospital Stay Duration (hours)")
    plt.ylabel("Number of Admissions")
    plt.title("Distribution of Hospital Stay Duration, by Admission Type")
    plt.show()

def duration_by_admission_type(data):
    plt.figure(figsize=(10,6))

    sns.violinplot(
        data=data,
        x="ADMISSION_TYPE",
        y="STAYTIME_HRS",
        inner="box",    # shows median and IQR inside violin
        cut=0           # avoids extending beyond observed values
    )

    plt.xlabel("Admission Type")
    plt.ylabel("Hospital Stay Duration (hours)")
    plt.title("Length of Stay, by Admission Type")
    plt.xticks(rotation=45)  # rotate labels if they overlap
    plt.show()

admissions["STAYTIME_HRS"].describe()
plot_duration(admissions, "STAYTIME_HRS", "ADMISSION_TYPE")
duration_by_admission_type(admissions)
plot_duration(died_in_hospital, "STAYTIME_HRS", "ADMISSION_TYPE")

##############################################
# Age at Admission compared to Length of Stay
##############################################

# convert DOB to datetime
patients["DOB"] = pd.to_datetime(patients["DOB"])
# merge to get DOB
adm_with_age = admissions.merge(patients[["SUBJECT_ID", "DOB", "GENDER"]], on="SUBJECT_ID", how="left")

# determine age at admission with steps to avoid int64 overflow

# take just the date part (removes time-of-day; helps avoid tz/ns corner cases)
admit_d = adm_with_age["ADMITTIME"].values.astype('datetime64[D]')
dob_d   = adm_with_age["DOB"].values.astype('datetime64[D]')

# day-precision age calculation (avoids ns overflow)
# Note: result is a numpy timedelta in days; convert to float years
age_days  = (admit_d - dob_d).astype('timedelta64[D]').astype('float')
age_years = age_days / 365.25

# age at admission (years)
adm_with_age["AGE_AT_ADMISSION"] = age_years

# filter out unrealistically old adults
realistic_patients = adm_with_age[adm_with_age["AGE_AT_ADMISSION"] < 120]
realistic_patients = realistic_patients[realistic_patients["AGE_AT_ADMISSION"] > 16]
realistic_patients["AGE_AT_ADMISSION"].describe()

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=realistic_patients,
    x="STAYTIME_HRS",
    y="AGE_AT_ADMISSION",
    hue="GENDER",
    alpha=0.6
)
plt.xlabel("Hospital Stay Duration (hours)")
plt.ylabel("Age at Admission (years)")
plt.title("Stay Duration vs. Age for Patients")
plt.show()


##############################################
# Age at Admission compared to Death count
##############################################
adm_that_died = realistic_patients[realistic_patients["DEATHTIME"].notna()]

plt.figure(figsize=(10,6))
sns.histplot(
        data=adm_that_died,
        x="AGE_AT_ADMISSION",
        hue="GENDER",
        bins=70,
        multiple="stack"
    )
plt.xlabel("Age at Admission (years)")
plt.ylabel("Number of Deaths")
plt.title("Histogram of Age at Admission for Deceased Patients")
plt.show()

##############################################
# Procedures and Mortality Rates
##############################################

# drop duplicates from procedures table
procedures = procedures.drop_duplicates(subset=["ICD9_CODE", "HADM_ID"])

# merge procedures with admissions
procedures_with_admissions = procedures.merge(admissions, on="HADM_ID", how="left")
procedures_with_admissions["DIED"] = procedures_with_admissions["DEATHTIME"].notna()

# compute mortality rates per procedure code.
procedure_mortality_rates = (
    procedures_with_admissions.groupby("ICD9_CODE").agg(
        num_admissions=("HADM_ID", "count"),
        num_deaths=("DIED", "sum"),
    ).reset_index()
)

procedure_mortality_rates["mortality_rate"] = (
    procedure_mortality_rates["num_deaths"] / procedure_mortality_rates["num_admissions"]
)
procedure_mortality_rates = procedure_mortality_rates.merge(d_procedures, on="ICD9_CODE", how="left")
top_10_procs = procedure_mortality_rates.sort_values("num_admissions", ascending=False).head(10)

plt.figure(figsize=(12,6))
sns.barplot(
    data=top_10_procs,
    x="SHORT_TITLE",
    y="mortality_rate",
    palette="viridis"
)

plt.xlabel("ICD-9 Procedure Code")
plt.ylabel("Mortality Rate")
plt.title("Mortality Rate by Most Common Procedures")
plt.xticks(rotation=45)
plt.show()


##############################################
# Diagnoses Cooccurrence
##############################################

from itertools import combinations
from collections import Counter

diagnoses = diagnoses.dropna(subset=["ICD9_CODE"])
code_counts = diagnoses["ICD9_CODE"].value_counts()

def compute_comorbidity_occurrences(data, group_col):
    # creates an empty counter that will hold pairs of diagnoses as keys and how
    # many patients had them as values.
    pairs = Counter()

    # group is the sub-dataframe of all rows (all diagnoses) for a single
    # column.
    for _, group in data.groupby(group_col):
        # pull just the ICD9 codes for this column.
        # .unique() ensures we don’t count duplicates if the same code was assigned
        # multiple times for that column.
        codes = group["ICD9_CODE"].unique()

        # skip columns with only one diagnosis
        if len(codes) <= 1:
            continue

        # produce all unique pairs of diagnoses for this column
        # sorted() ensures consistent ordering of pairs, so (A, B) and (B, A)
        # are counted as the same pair.
        for c1, c2 in combinations(sorted(codes), 2):
            pairs[(c1, c2)] += 1

    # convert to dataframe
    df_pairs = pd.DataFrame(
        [(c1, c2, count) for (c1, c2), count in pairs.items()],
        columns=["code1", "code2", "co_occurrence"]
    )

    # add short titles for readability
    df_pairs = df_pairs.merge(d_diagnoses, left_on="code1", right_on="ICD9_CODE", how="left")
    df_pairs = df_pairs.rename(columns={"SHORT_TITLE": "label1"}).drop(columns="ICD9_CODE")

    df_pairs = df_pairs.merge(d_diagnoses, left_on="code2", right_on="ICD9_CODE", how="left")
    df_pairs = df_pairs.rename(columns={"SHORT_TITLE": "label2"}).drop(columns="ICD9_CODE")

    return df_pairs

patient_comorbidities = compute_comorbidity_occurrences(diagnoses, "SUBJECT_ID")
admission_comorbidities = compute_comorbidity_occurrences(diagnoses, "HADM_ID")

def get_codes_with_most_occurrences(data, count=20):
    dict_codes = {}

    for _, row in data.iterrows():
        if row["code1"] not in dict_codes:
            dict_codes[row["code1"]] = 0
        if row["code2"] not in dict_codes:
            dict_codes[row["code2"]] = 0
        dict_codes[row["code1"]] += row["co_occurrence"]
        dict_codes[row["code2"]] += row["co_occurrence"]

    return pd.Series(dict_codes).sort_values(ascending=False).head(count).index


def make_matrix(data, count=20):
    common_codes = get_codes_with_most_occurrences(data, count)
    common_diagnoses = data[
        data["code1"].isin(common_codes) & data["code2"].isin(common_codes)
    ]

    matrix = common_diagnoses.pivot(index="code1", columns="code2", values="co_occurrence").fillna(0)

    # Symmetrize (so both upper/lower triangles filled)
    matrix_full = matrix.add(matrix.T, fill_value=0).fillna(0)

    matrix_labeled = matrix_full.rename(index=diag_map, columns=diag_map)
    return matrix_labeled

patient_mtx_full = make_matrix(patient_comorbidities, 20)
admission_mtx_full = make_matrix(admission_comorbidities, 20)

def plot_comorbidity_matrix(matrix, type):
    plt.figure(figsize=(12,10))
    sns.heatmap(matrix, cmap="Reds", square=True, cbar=True)
    plt.title(f"{type} level ICD-9 Co-occurrence Matrix")
    plt.show()

plot_comorbidity_matrix(patient_mtx_full, "Patient")
plot_comorbidity_matrix(admission_mtx_full, "Admission")
