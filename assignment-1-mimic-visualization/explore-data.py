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
