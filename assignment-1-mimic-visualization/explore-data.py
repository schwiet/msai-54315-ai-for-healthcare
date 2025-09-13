import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
print("Current working directory:", os.getcwd())

patients = pd.read_csv("./mimic-iii/PATIENTS.csv.gz")
admissions = pd.read_csv("./mimic-iii/ADMISSIONS.csv.gz")

sns.countplot(data=admissions, x="ADMISSION_TYPE")
plt.show()

admissions["DIAGNOSIS"].value_counts()
admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])
admissions["DISCHTIME"] = pd.to_datetime(admissions["DISCHTIME"])
admissions["STAYTIME_HRS"] = (admissions["DISCHTIME"] - admissions["ADMITTIME"]).dt.seconds/3600
admissions["DEATHTIME"] = pd.to_datetime(admissions["DEATHTIME"])

died_in_hospital = admissions[admissions["DEATHTIME"].notna()]

def plot_duration(data, col, hue):
    plt.figure(figsize=(10,6))

    sns.histplot(
        data=data,
        x=col,
        hue=hue,
        bins=50,         # adjust as needed
        multiple="stack" # stack bars instead of overlay
    )

    plt.xlabel("Hospital Stay Duration (hours)")
    plt.ylabel("Number of Admissions (patients who died)")
    plt.title("Distribution of Hospital Stay Duration Until Death, by Admission Type")
    plt.show()

admissions["STAYTIME_HRS"].max()
plot_duration(admissions, "STAYTIME_HRS", "ADMISSION_TYPE")
plot_duration(died_in_hospital, "STAYTIME_HRS", "ADMISSION_TYPE")