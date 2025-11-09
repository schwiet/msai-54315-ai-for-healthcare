MIMIC_IV = '../mimic-iv'
RANDOM_SEED = 42

import pandas as pd
admissions = pd.read_csv(f"{MIMIC_IV}/admissions.csv.gz")
patients   = pd.read_csv(f"{MIMIC_IV}/patients.csv.gz")
icustays   = pd.read_csv(f"{MIMIC_IV}/icustays.csv.gz")

# build cohort and label
#   one row per hospital admission y=1 if admission ever appears in ICU stay, y=0 otherwise

# start by labeling every admission in ICU as 1
icu_labeled = icustays[["hadm_id"]].drop_duplicates().assign(y=1)
icu_labeled.head()

# merge labeled ICU into admissions and fill missing values with 0
cohort = (
    admissions.merge(icu_labeled, on="hadm_id", how="left")
              .assign(y=lambda df: df["y"].fillna(0).astype(int))
)

# so now any admission that appears in the ICU stay is labeled as 1, otherwise it is a 0
# let's take a look at the distribution
cohort["y"].value_counts()