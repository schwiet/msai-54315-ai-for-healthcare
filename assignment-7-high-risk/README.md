# README

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
