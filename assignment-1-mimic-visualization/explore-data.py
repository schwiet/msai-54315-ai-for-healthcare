import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
print("Current working directory:", os.getcwd())

patients = pd.read_csv("../mimic-iii/PATIENTS.csv.gz")
admissions = pd.read_csv("../mimic-iii/ADMISSIONS.csv.gz")
transfers = pd.read_csv("../mimic-iii/TRANSFERS.csv.gz")

admid1 = transfers[transfers["HADM_ID"] == 163281]

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
