# README

## How the dataset is assembled with these scripts

Because the stages were run in different environments, they are not included in a single notebook.

### note-event-embedding.py
- Run inside Docker container on DGX Spark
- Script (`note-events-embeddings.py`) reads `NOTEEVENTS.csv.gz` in 10k-row chunks, filters to `Discharge summary`, and concatenates all notes per `SUBJECT_ID` into a single timeline string.
- Uses `emilyalsentzer/Bio_ClinicalBERT` (GPU if available) to chunk long texts into 512-token windows with overlap, average the CLS vectors, and produce one embedding per patient.

### ingest-and-feature-engineering.py
- Loads `PATIENTS.csv.gz` + `ADMISSIONS.csv.gz`, parses dates, merges demographics, handles HIPAA age shifting (DOB < 2000 reset; ages >200 capped at 90), fills missing language/marital/religion, collapses Christian denominations into `CHRISTIAN`, and one-hot encodes categorical fields.
- Maps ICD-9 diagnoses (`DIAGNOSES_ICD.csv.gz` + CCS crosswalk) into CCS categories, builds per-patient diagnosis frequency tables, and aggregates admission features per patient (max of one-hots, min/max age, admission count).
- Merges structured features with `patient_embeddings.csv`, scales age/admission-count, L2-normalizes BERT vectors, concatenates into a multimodal matrix, and fits a cosine `NearestNeighbors` model for similarity search.
- Saves artifacts for RAG/search: `multimodal_vectors.csv`, `structured_features.csv`, and `multimodal_metadata.json` (split index, structured column list, total feature count).

## Example Results

```
🔎 SEARCH: patients at least 90 years old who live alone, have suffered a fall and had surgery

============================================================
📥 Query: patients at least 90 years old who live alone, have suffered a fall and had surgery
============================================================

🧠 Parsing query...
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
   Parsed parameters:
      Subject ID: None
      Gender: None
      Age range: 90 - None
      Ethnicity: None
      Religion: None
      Diagnoses: ['Falls', 'Surgery']

🔍 Strategy B: Filtered embedding search
      Age >= 90: 1935 patients
   Found 1935 patients matching filters

📊 Search Results (filtered):
----------------------------------------
   #1: Patient 53131 (Similarity: 72.4%)
   #2: Patient 90740 (Similarity: 72.4%)
   #3: Patient 41983 (Similarity: 72.0%)
   #4: Patient 43787 (Similarity: 72.0%)
   #5: Patient 93483 (Similarity: 71.9%)

🧮 Similarity analysis for top 2 matches (Patients 53131 & 90740):

📝 ANALYSIS:
----------------------------------------
Similarity: low (10/100)

Query details:
- Patient A: age over 90, lived alone, suffered a fall, underwent surgery for jaundice and gallstones with a history of AAA repair, iliac stenting, CVA, and bladder cancer.
- Patient B: age over 90, lived alone, suffered a fall, underwent surgery for multiple injuries with a history of HTN, GI bleeds, kidney stones, PE, and a known 7.5cm AAA.

Shared factors: neither patient had known drug allergies or smoked.

Differences:
- Patient A was admitted for observation following ERCP with biliary stenting and sphincterotomy due to jaundice and gallstones, while Patient B was admitted for extensive internal injuries from a fall and underwent multiple chest tube placements.
- Patient A had a history of AAA repair, iliac stenting, CVA, and bladder cancer, while Patient B had a history of HTN, GI bleeds, kidney stones,
----------------------------------------
```

```
🔎 SEARCH: male patients between 30 and 40 years of age with hypertension and a history of smoking

============================================================
📥 Query: male patients between 30 and 40 years of age with hypertension and a history of smoking
============================================================

🧠 Parsing query...
   Parsed parameters:
      Subject ID: None
      Gender: M
      Age range: 30 - 40
      Ethnicity: None
      Religion: None
      Diagnoses: ['Hypertension', 'History of smoking', "Hd/nck cancr - Non-epith ca (for example, if we assume that 'history of smoking' is a diagnosis related to head and neck cancer)"]

🔍 Strategy B: Filtered embedding search
      Gender filter (M): 23199 patients
      Age >= 30: 20027 patients
      Age <= 40: 1129 patients
   Found 1129 patients matching filters

📊 Search Results (filtered):
----------------------------------------
   #1: Patient 91550 (Similarity: 64.7%)
   #2: Patient 23188 (Similarity: 64.5%)
   #3: Patient 58672 (Similarity: 64.2%)
   #4: Patient 18449 (Similarity: 64.1%)
   #5: Patient 4870 (Similarity: 64.0%)

🧮 Similarity analysis for top 2 matches (Patients 91550 & 23188):

📝 ANALYSIS:
----------------------------------------
Similarity: low (10)

Query details:
- Patient A: hypertension (present), smoking history (not evident), motor vehicle accident (present)
- Patient B: hypertension (present), smoking history (present), alcohol intoxication (present)

Shared factors: both patients are males between 30 and 40 years of age and have a history of hypertension.

Differences:
- Patient A: sustained injuries from a motor vehicle accident, no mention of alcohol intake or substance abuse, no psychiatric or mental health issues
- Patient B: admitted for alcohol intoxication, history of mood disorders and alcohol abuse, underwent psychiatric consultation and recommended for inpatient substance abuse treatment.

Overall judgment: while both patients share the factors of being male, between 30 and 40 years of age, and having a history of hypertension, the two cases are not very similar as Patient A's presentation is related to a motor vehicle accident and Patient B's is related to alcohol intoxication and substance abuse.
----------------------------------------
```

```
🔎 SEARCH: between 30 and 50, history of smoking, hypertension and diabetes

============================================================
📥 Query: between 30 and 50, history of smoking, hypertension and diabetes
============================================================

🧠 Parsing query...
   Parsed parameters:
      Subject ID: None
      Gender: None
      Age range: 30 - 50
      Ethnicity: None
      Religion: None
      Diagnoses: ['Hypertension', 'Diabetes mellitus without complication', 'Diabetes mellitus with complications', 'history of smoking']

🔍 Strategy B: Filtered embedding search
      Age >= 30: 35482 patients
      Age <= 50: 5984 patients
   Found 5984 patients matching filters

📊 Search Results (filtered):
----------------------------------------
   #1: Patient 9749 (Similarity: 75.1%)
   #2: Patient 20951 (Similarity: 75.0%)
   #3: Patient 93893 (Similarity: 74.3%)
   #4: Patient 60807 (Similarity: 74.3%)
   #5: Patient 71527 (Similarity: 74.2%)

🧮 Similarity analysis for top 2 matches (Patients 9749 & 20951):

📝 ANALYSIS:
----------------------------------------
Similarity: low (30/100)

Patient A:
- Age: not mentioned
- Smoking history: present, smoking 12-2 ppd
- Hypertension: present
- Diabetes: not mentioned

Patient B:
- Age: 42
- Smoking history: present, 10 pack year history; currently smoking
- Hypertension: not mentioned
- Diabetes: not mentioned

Shared factors: smoking history
Differences: age, presence of hypertension and diabetes
Overall judgment: The patients are not very similar based on the criteria in the QUESTION. Patient A has hypertension and an unspecified age, while Patient B is 42 years old and does not have hypertension. Both patients have a history of smoking. The differences in age and comorbidities such as hypertension and diabetes make them less similar.
----------------------------------------
```

## Python Kernel when running remotely

In VS Code, if running remotely and the environment from `.venv` is not selectable with the Jupyter extension, select it manually via **"Python: Select Interpreter"**

# GPU Embedding Preprocessing Using NVIDIA PyTorch Containers (DGX Spark)

This project runs primarily inside a standard `uv` Python environment. However, GPU-based embedding generation (Torch + Transformers) is performed **outside the uv environment** using an [**NVIDIA-provided PyTorch container**](https://build.nvidia.com/spark/pytorch-fine-tune/instructions) that is fully compatible with DGX Spark (Grace Blackwell + CUDA 13).

This avoids dependency issues on the host OS and ensures consistent, supported GPU behavior while keeping the main development environment clean.

> **NOTE**: I initially tried building dependencies from source according to [this repo](https://github.com/GuigsEvt/dgx_spark_config). While it succeeded at enabling cuda for `torch`, other libraries like `transformers` had compatibility issues with the built `torch` version.

## Why this workflow?

DGX Spark uses a Grace Blackwell architecture (ARM64 + CUDA 13).  
At the time of development, upstream PyTorch + Transformers wheels did not fully support this configuration inside standard Python environments.

NVIDIA, however, provides **tested Docker images** that include:

- CUDA 13  
- cuDNN 9  
- NVSHMEM  
- NCCL  
- PyTorch built with Blackwell support  
- A compatible Transformers setup  

Following NVIDIA’s recommended workflow, we:

1. Use the NVIDIA image **only** for GPU-heavy embedding preparation.  
2. Save embeddings as a Dataframe to a `.csv`
3. Load artifacts back in the main `uv` environment for downstream RAG workflows.

---

# 1. Create Embeddings Using NVIDIA PyTorch Container

NVIDIA publishes compatible containers here:  
https://build.nvidia.com/spark/pytorch-fine-tune/instructions

Example workflow:

```bash
docker pull nvcr.io/nvidia/pytorch:24.11-py3

docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/pytorch:24.11-py3 bash
````

Inside the container:

```bash
cd /workspace
python note-events-embeddings.py
```

Where `note-events-embeddings.py`:

* Loads your dataset
* Uses PyTorch + Transformers normally
* Generates embeddings with BERT Model
* Saves artifacts:

When the container exits, the embedding files remain on the host in your project directory.

From here, you can:

* Build vector stores
* Run RAG pipelines
* Perform Spark ETL or distributed processing
* Train CPU-based downstream models

All without needing GPU-enabled PyTorch inside uv.

# 3. Why This Split Approach Works Well

* **No fragile custom builds.** Avoids compiling PyTorch or managing CUDA runtimes manually.
* **Reproducible.** NVIDIA's container stack is stable and tested for DGX hardware.
* **Portable artifacts.** Embeddings are simple files that any environment can consume.
* **uv stays clean.** Use modern tools (`uv`, `transformers` CPU-only, Spark, etc.) without GPU constraints.
* **Aligns with NVIDIA recommendations.** Matches their official DGX Spark workflow.


# 📌 4. Summary

This project uses a **two-phase workflow**:

### **Phase A — GPU Embedding Generation (NVIDIA Docker Container)**

* Runs fully GPU-accelerated PyTorch + Transformers
* Uses NVIDIA’s supported DGX Spark software stack
* Stores embeddings as files

### **Phase B — Downstream Processing (uv Environment)**

* Loads precomputed embeddings
* Performs RAG, indexing, inference, and Spark tasks
* Does not require GPU-enabled PyTorch

This design provides strong separation of concerns, reproducibility, and avoids dependency issues inherent to cutting-edge DGX Spark GPU stacks.
