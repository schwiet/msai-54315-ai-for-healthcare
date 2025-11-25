import getpass

# Get the OpenAI API key and confirm it is set
openai_key = getpass.getpass("MY_API_KEY: ")
if openai_key and openai_key.strip():
    print("OPENAI_KEY is non-empty")
else:
    print("OPENAI_KEY is empty or not set")

import pandas as pd
# src: https://mitre.box.com/shared/static/9iglv8kbs1pfi7z8phjl9sbpjk08spze.zip
patients = pd.read_csv("10k_synthea_covid19_csv/patients.csv")
encounters = pd.read_csv("10k_synthea_covid19_csv/encounters.csv")
conditions = pd.read_csv("10k_synthea_covid19_csv/conditions.csv")
patients.info()
patients.head()
encounters.info()
encounters.head()
conditions.info()
conditions.head()

conditions["DESCRIPTION"].value_counts()
encounters["START"].info()

parse_dates = ["START","STOP","DATE","BIRTHDATE","DEATHDATE","ONSET","RECORDED_DATE"]

# normalize date columns
for df in (patients, encounters, conditions):
    for c in [c for c in parse_dates if c in df.columns]:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_localize(None)

covid_cond = conditions[
    conditions["DESCRIPTION"].str.contains("COVID", case=False, na=False)
].copy()
first_onset = covid_cond.groupby("PATIENT")["START"].min().rename("COVID_ONSET")

encounters.info()
covid_cond.info()
first_onset.info()
first_onset.head()

enc_join_covid = encounters.merge(
    first_onset,
    left_on="PATIENT",
    right_index=True,
    how="inner")

enc_join_covid.info()

age_at_onset = (
    (
        first_onset - first_onset.to_frame().join(
            patients.set_index("Id"),
            how="left")["BIRTHDATE"]
    ).dt.days / 365.25
).round()

window = (
    enc_join_covid["START"] >= enc_join_covid["COVID_ONSET"]
) & (
    enc_join_covid["START"] <= enc_join_covid["COVID_ONSET"] +
    pd.Timedelta(days=30)
)

# List of patients with severe encounter within 30days of onset of COVID
severe_hit = enc_join_covid.loc[
    window & enc_join_covid["ENCOUNTERCLASS"].str.lower().isin(
        ["inpatient","emergency"]
    )
].groupby("PATIENT").size().gt(0)

# create a label for each patient based on whether they had a severe encounter
# within 30 days of first onset of COVID
label = severe_hit.reindex(
    first_onset.index,
    fill_value=False
).astype(
        int
).rename("severe_30d")

COMORBID_KEYWORDS = [
    "diabetes",
    "hypertension",
    "asthma",
    "copd",
    "coronary artery",
    "heart failure",
    "obesity",
    "chronic kidney",
    "ckd",
    "cancer",
    "immunodefic",
    "hyperlipid"
]
# merge other conditions with first onset of COVID
pre = conditions.merge(
    first_onset,
    left_on="PATIENT",
    right_index=True,
    how="inner"
)
# filter out conditions that occurred after onset of COVID
pre = pre[pre["START"] < pre["COVID_ONSET"]]
# for each patient, create column with flags for each comorbidity
# indicating whether the patient has the comorbidity before onset of COVID
flags = pre.assign(**{
        kw: pre["DESCRIPTION"].str.contains(kw, na=False) for kw in COMORBID_KEYWORDS
}).groupby("PATIENT")[COMORBID_KEYWORDS].max().astype(bool)
flags.info()
flags.head()

# construct our dataset
frame = pd.DataFrame(index=first_onset.index)
frame["age_at_onset"] = age_at_onset.astype("Int64")
frame["gender"] = patients.set_index("Id").reindex(frame.index)["GENDER"]
# make sure flags has an entry for every patient in frame
flags = flags.reindex(frame.index, fill_value=False)
frame = frame.join(flags, how="left", on="PATIENT")
frame = frame.join(label, how="left").dropna(subset=["severe_30d"])
frame

from openai import OpenAI
client = OpenAI(api_key=openai_key)

response = client.responses.create(
    model="gpt-5-nano",
    input="Write a short bedtime story about a unicorn."
)