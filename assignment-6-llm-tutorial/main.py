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

parse_dates = ["START","STOP","DATE","BIRTHDATE","DEATHDATE","ONSET","RECORDED_DATE"]

# normalize date columns
for df in (patients, encounters, conditions):
    for c in [c for c in parse_dates if c in df.columns]:
        df[c] = pd.to_datetime(df[c], errors="coerce")


from openai import OpenAI
client = OpenAI(api_key=openai_key)

response = client.responses.create(
    model="gpt-5-nano",
    input="Write a short bedtime story about a unicorn."
)