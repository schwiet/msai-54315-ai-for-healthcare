import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

patients = pd.read_csv("../mimic-iii/PATIENTS.csv.gz")
admissions = pd.read_csv("../mimic-iii/ADMISSIONS.csv.gz")

sns.countplot(data=admissions, x="ADMISSION_TYPE")
plt.show()

admissions["DIAGNOSIS"].value_counts()
admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])
admissions["DISCHTIME"] = pd.to_datetime(admissions["DISCHTIME"])
admissions["STAYTIME_HRS"] = (admissions["DISCHTIME"] - admissions["ADMITTIME"]).dt.seconds/3600
admissions["DEATHTIME"] = pd.to_datetime(admissions["DEATHTIME"])

died_in_hospital = admissions[admissions["DEATHTIME"].notna()]

