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

# add some basic demographic information to the cohort from the patients table
# most of the other demographic info was already in the admissions table
patients.describe()
cohort = cohort.merge(
    patients[["subject_id", "anchor_age", "gender"]],
    on="subject_id",
    how="left"
)
cohort.info()

# look at the age distribution - 18 is already the minimum age
cohort["anchor_age"].describe()

# to simplify and avoid multiple correlated samples for the same patient keep
# only the first admission per patient
cohort = (
    cohort.sort_values(["subject_id", "admittime"])
          .drop_duplicates("subject_id", keep="first")
)

# Visualize the distribution of the labels
import matplotlib.pyplot as plt

class_counts = cohort["y"].value_counts().sort_index()
plt.figure()
plt.bar(["Non-ICU (0)", "ICU (1)"], class_counts.values, color=["#ff0000", "#0000ff"])
plt.title("Class balance: ICU vs non-ICU admissions")
plt.ylabel("Number of admissions")
plt.show()
cohort["y"].value_counts(normalize=True)

# initially, keep the feature set deliberately small and interpretable:
# Numeric feature:
# - anchor_age (age)
# Categorical features:
# - gender
# - admission_type (EMERGENCY, ELECTIVE, URGENT, etc.)
# - admission_location (ED, transfer from other hospital, clinic, …)
# - insurance (Medicare, Medicaid, Private, Self Pay, …)
# - race (broad categories)
# - language (English, Spanish, French, etc.)
# - marital_status (Single, Married, Divorced, Widowed, etc.)
num_cols = ["anchor_age"]
cat_cols = [
    "gender",
    "admission_type",
    "admission_location",
    "insurance",
    "race",
    "language",
    "marital_status",
]
features = num_cols + cat_cols

data = cohort[["hadm_id", "y"] + features].copy()
data.head()

########################################################
# Feature Engineering - Handle Missing Feature Values
########################################################

# check for missing values in the numeric features
data[num_cols].isna().sum()

data[cat_cols].isna().sum()

# call missing insurance "NONE"
data["insurance"] = data["insurance"].fillna("NONE")
# assuming missing language means English
data["language"] = data["language"].fillna("English")
data["language"].value_counts()

# call missing marital status and religion "Unknown"
data["marital_status"] = data["marital_status"].fillna("Unknown")
data["marital_status"].value_counts()

# there's a single row with a missing admission location. It's admission type
# is "URGENT, so assume it was a TRANSFER FROM HOSPITAL, since that is the 
# location most often associated with urgent admissions
data[data["admission_location"].isna()]
data[data["admission_type"] == "URGENT"]["admission_location"].value_counts()
data["admission_location"] = data["admission_location"].fillna("TRANSFER FROM HOSPITAL")

##################################################################
# Feature Engineering - One-hot Encode Categorical Features
##################################################################

X = pd.get_dummies(data[features], columns=cat_cols, dummy_na=False)

# cast age to float, since we will normalize after splitting
# cast one-hot encoded columns to float, since our MLP model expects that
X = X.astype("float64")

# cast one-hot encoded columns to float, since we will normalize after splitting
y = data["y"].values
hadm_ids = data["hadm_id"].values

X.shape, y.shape

# split the dataset into training and testing sets
# using a 70/30 split for training and validation - and a 50/50 split for
# validation and testing

from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp, hadm_train, hadm_temp = train_test_split(
    X, y, hadm_ids,
    test_size=0.3,
    random_state=RANDOM_SEED,
    stratify=y
)

X_val, X_test, y_val, y_test, hadm_val, hadm_test = train_test_split(
    X_temp, y_temp, hadm_temp,
    test_size=0.5,
    random_state=RANDOM_SEED,
    stratify=y_temp
)

X_train.shape, X_val.shape, X_test.shape
y_train.shape, y_val.shape, y_test.shape

# standardize the numeric features

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# apply to each split individually, so information doesn't leak from
# test and validation sets into training
scaler.fit(X_train[num_cols])

X_train[num_cols].info()
X_train.head()

# apply to each split
for df in (X_train, X_val, X_test):
    df.loc[:, num_cols] = scaler.transform(df[num_cols])


##################################################################
# build a simple baseline model for comparison
##################################################################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)
log_reg.fit(X_train, y_train)

# validation performance
val_probs_lr = log_reg.predict_proba(X_val)[:, 1]
val_auc_lr = roc_auc_score(y_val, val_probs_lr)
val_ap_lr  = average_precision_score(y_val, val_probs_lr)

print(f"Logistic Regression - Val ROC AUC: {val_auc_lr:.3f}, AP: {val_ap_lr:.3f}")

##################################################################
# build a simple fully connected network (MLP) in PyTorch.
##################################################################

import torch
from torch.utils.data import Dataset, DataLoader

# create a small Dataset class for the training data
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# create datasets and loaders for each split
batch_size = 64

train_ds = TabularDataset(X_train.to_numpy(), y_train)
val_ds   = TabularDataset(X_val.to_numpy(), y_val)
test_ds  = TabularDataset(X_test.to_numpy(), y_test)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# use a simple MLP model with two hidden layers
# Input -> 2 hidden layers -> output
# Dropout + BatchNorm to help generalization

input_dim = X_train.shape[1]

class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(64, 1)  # output logits
        )
        
    def forward(self, x):
        return self.net(x).squeeze(1)  # (batch,)

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_dim).to(device)

criterion = torch.nn.BCEWithLogitsLoss()  # combines sigmoid + BCE
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# we’ll train for a fixed number of epochs and track validation AUC
def evaluate(model, loader):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    auc = roc_auc_score(all_targets, all_probs)
    ap  = average_precision_score(all_targets, all_probs)
    return auc, ap, all_probs, all_targets


import numpy as np
n_epochs = 20
best_val_auc = 0.0
best_state_dict = None

for epoch in range(1, n_epochs + 1):
    model.train()
    epoch_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * X_batch.size(0)
    
    epoch_loss /= len(train_ds)
    
    val_auc, val_ap, _, _ = evaluate(model, val_loader)
    
    print(f"Epoch {epoch:02d} | Train Loss: {epoch_loss:.4f} | "
          f"Val ROC AUC: {val_auc:.3f} | Val AP: {val_ap:.3f}")
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state_dict = model.state_dict()

# load the best model weights
model.load_state_dict(best_state_dict)