# ZenML Customer Review Predictor: An MLOps Showcase

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ZenML](https://img.shields.io/badge/built%20with-ZenML-blueviolet)](https://zenml.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *"An MLOps pipeline is only as good as its organization."*

Welcome to my end-to-end Machine Learning pipeline project! I built this repository to demonstrate how to implement **production-grade MLOps best practices** using **ZenML** and **Scikit-Learn**. 

While many projects focus solely on hyper-tuning a model to get the highest accuracy, this project focuses heavily on **Artifact Management, Metadata Tracking, and Code Maintainability**. 

It trains a regression model to predict customer review scores using the famous [Olist E-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), but the real star of the show is the architecture behind it.

---

## Why I Built This

As ML projects scale, pipelines often turn into a chaotic mess of untracked Jupyter notebooks. You end up with 50 models, 200 orphaned datasets, and no idea which model version is currently in production. 

To solve this, I implemented the [ZenML Tagging & Organization Framework](https://docs.zenml.io/user-guides/best-practices/organizing-pipelines-and-models) from the ground up. If you explore the code, you'll find:

1. **Strict Artifact Tagging**: *Every* piece of data, model, and metric is tagged at birth using a centralized `Tag Registry` (using Python Enums to prevent typos!).
2. **Dynamic Evaluation Tags**: The pipeline automatically evaluates the model and tags the artifacts dynamically (e.g., `[performance-high-r2]` or `[performance-low-r2]`) based on the actual evaluation metrics.
3. **The Strategy Pattern**: The codebase uses clean OOP design. Data cleaning and model training algorithms can be easily swapped without ever touching the core ZenML `@step` definitions.
4. **Model Control Plane**: Models aren't just saved as `.pkl` files; they are tracked as first-class entities in the ZenML Model Registry.

---

## How It Works (The Architecture)

I've strictly separated the ML business logic (`src/`) from the pipeline orchestration (`steps/`). 

```text
Mlops-project-zenml/
├── pipeline/
│   └── training_pipeline.py    ← Connects the steps + links to Model Registry
├── steps/                      
│   ├── ingest_data.py          ← Loads raw CSV → tags as "artifact-raw"
│   ├── clean_data.py           ← Preprocesses + dynamic data quality tags
│   ├── train_model.py          ← Trains model → tags as "artifact-model"
│   └── evaluation.py           ← Evaluates + dynamic performance tags
├── src/                        
│   ├── data_cleaning.py        ← DataStrategy ABC (OOP logic)
│   ├── model_development.py    ← Model ABC (sklearn Pipeline + Imputers)
│   └── evaluation.py           ← Evaluation ABC (RMSE, R2, MSE)
├── utils/
│   └── tag_manager.py          ← Custom CLI to query tagged/orphaned resources
├── configuration/
│   └── config.yaml             
├── tag_registry.py             ← Central Enum registry for all metadata tags
└── run_pipeline.py             ← Main entry point
```

### The Pipeline Flow

Here is how data moves through the system, getting tagged at every stage:

```text
Load Data ──→ Clean & Split ──→ Train Model ──→ Evaluate
   │               │                │              │
   ▼               ▼                ▼              ▼
[artifact-raw]  [artifact-       [artifact-    [artifact-metric]
[domain-        processed]       model]        + dynamic tags:
ecommerce]      + dynamic:       [algorithm-   [performance-
                [quality-*]      linear-       high-r2]
                                 regression]   
```

---

## Tech Stack

I rely on a modern, lightweight MLOps stack:
- **Orchestration**: ZenML
- **Package Management**: [uv](https://github.com/astral-sh/uv) (lightning fast!)
- **ML Framework**: Scikit-Learn
- **Data Processing**: Pandas, NumPy

---

## Getting Started

Want to run this on your own machine? It takes less than 2 minutes.

### 1. Installation

This project uses `uv` for incredibly fast dependency management.

```bash
# Clone the repository
git clone https://github.com/abhilashpanda04/Mlops-project-zenml.git
cd Mlops-project-zenml

# Install dependencies and create a virtual environment instantly
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Initialize ZenML locally
zenml init
```

### 2. Run the Pipeline

Execute the full training pipeline:

```bash
python run_pipeline.py
```

### 3. See the Magic (Querying Tags)

I built a custom utility to showcase the power of ZenML tagging. Run the tag manager to easily find your best performing models and detect orphaned pipeline runs:

```bash
python -m utils.tag_manager
```

### 4. View the Dashboard

To see the visual DAG, your model artifacts, and metrics, start the local ZenML server:

```bash
zenml up
```

---

## Contributing & Feedback

If you're interested in MLOps, system design, or have feedback on my implementation of the Strategy pattern, I'd love to connect! Feel free to open an issue, submit a PR, or reach out directly.

## About Me

**Abhilash Kumar Panda**
- 📧 Email: abhilashk.isme1517@gmail.com
- 🔗 LinkedIn: [Abhilash Kumar Panda](https://www.linkedin.com/in/abhilash-kumar-panda/)
- 🌐 Portfolio: [abhilashpanda04.github.io](https://abhilashpanda04.github.io/Portfolio_site/)
- GitHub: [@abhilashpanda04](https://github.com/abhilashpanda04)

---
*If you found this architecture helpful or interesting, please consider giving the repo a star!*
