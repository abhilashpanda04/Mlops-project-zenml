# Mlops-project-zenml

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ZenML](https://img.shields.io/badge/built%20with-ZenML-blueviolet)](https://zenml.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end ML pipeline using **ZenML** for orchestration — predicting customer review scores from the Olist e-commerce dataset. Demonstrates ZenML best practices including **tagging, artifact management, Model Control Plane**, and the **Strategy design pattern**.

## 📋 Overview

This project builds a regression pipeline that predicts `review_score` from order and product features. It showcases ZenML's organization capabilities:

- 🏷️ **Tag Registry** — Centralized Enum-based tagging for consistency
- 📦 **Artifact Tagging** — Every artifact (data, model, metrics) is tagged via `ArtifactConfig`
- 🤖 **Model Control Plane** — `Model` entity tracks the model across pipeline runs
- 🔄 **Dynamic Tagging** — Performance-based tags applied automatically after evaluation
- 🔍 **Tag Manager** — Utility to query, filter, and find orphaned resources
- 🧱 **Strategy Pattern** — Clean OOP design for data cleaning, model training, and evaluation

## 🏗️ Architecture

```
run_pipeline.py                 ← Entry point (runtime tags + YAML config)
│
├── pipeline/
│   └── training_pipeline.py    ← @pipeline with tags + Model entity
│
├── steps/                      ← ZenML @step definitions with ArtifactConfig
│   ├── ingest_data.py          ← Loads raw CSV → tagged "artifact-raw"
│   ├── clean_data.py           ← Preprocesses + splits → tagged "artifact-processed"
│   ├── train_model.py          ← Trains model → tagged "artifact-model"
│   ├── evaluation.py           ← Evaluates + dynamic tags (performance-high-r2, etc.)
│   └── config.py               ← Pydantic model config
│
├── src/                        ← Business logic (Strategy pattern)
│   ├── data_cleaning.py        ← DataStrategy ABC + preprocessing/splitting strategies
│   ├── model_development.py    ← Model ABC + LinearRegression strategy
│   └── evaluation.py           ← Evaluation ABC + MSE/R2/RMSE strategies
│
├── tag_registry.py             ← Central tag registry using Python Enums
├── utils/
│   └── tag_manager.py          ← Query/filter tagged resources + orphan detection
│
├── configuration/
│   └── config.yaml             ← Pipeline parameters + config-level tags
│
└── data/
    └── olist_customers_dataset.csv
```

## 🏷️ Tagging Strategy

This project implements ZenML's tagging best practices with a centralized registry:

| Category | Tags | Applied To |
|----------|------|-----------|
| **Environment** | `environment-development`, `environment-staging`, `environment-production` | Pipelines |
| **Domain** | `domain-ecommerce`, `domain-customer-reviews` | Pipelines, Artifacts |
| **Pipeline Type** | `pipeline-training`, `pipeline-inference` | Pipelines |
| **Artifact Type** | `artifact-raw`, `artifact-processed`, `artifact-model`, `artifact-metric` | Artifacts |
| **Algorithm** | `algorithm-linear-regression` | Models, Artifacts |
| **Status** | `status-experimental`, `status-validated`, `status-production` | Models |
| **Data Quality** | `quality-complete`, `quality-incomplete` | Artifacts (dynamic) |
| **Performance** | `performance-high-r2`, `performance-low-r2` | Model artifacts (dynamic) |

```python
# Tags are defined as Enums for consistency (no typos!)
from tag_registry import Environment, Domain, ArtifactType

@pipeline(tags=[Environment.DEV.value, Domain.ECOMMERCE.value])
def train_pipeline():
    ...
```

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| **Orchestration** | ZenML |
| **ML Framework** | scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Tag Management** | ZenML Tags + Custom Enum Registry |
| **Model Tracking** | ZenML Model Control Plane |
| **Language** | Python 3.8+ |

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/abhilashpanda04/Mlops-project-zenml.git
cd Mlops-project-zenml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize ZenML
zenml init
```

## 🚀 Usage

### Run the Pipeline

```bash
# Run the training pipeline
python run_pipeline.py
```

### View Pipeline in ZenML Dashboard

```bash
# Launch ZenML dashboard
zenml up

# List pipeline runs
zenml pipeline runs list
```

### Query Tagged Resources

```bash
# Run the tag manager to find and inspect tagged resources
python -m utils.tag_manager
```

This will show:
- All training pipeline runs
- Raw and processed artifacts
- Model artifacts with performance tags
- Orphaned (untagged) resources

## 🔧 Configuration

Edit `configuration/config.yaml`:

```yaml
parameters:
  data_path: "data/olist_customers_dataset.csv"
  name: "LinearRegression"

tags:
  - "config-driven"
  - "dataset-olist"
```

## 🧱 Design Patterns

### Strategy Pattern

The project uses the **Strategy design pattern** for swappable algorithms:

```
DataStrategy (ABC)
├── DataPreProcessingStrategy   → Handles missing values, feature selection
└── DataDevideStretegy          → Train/test split

Model (ABC)
└── LinearRegressionModel       → sklearn LinearRegression

Evaluation (ABC)
├── MSE                         → Mean Squared Error
├── R2                          → R-squared Score
└── RMSE                        → Root Mean Squared Error
```

## 📁 Project Structure

```
Mlops-project-zenml/
├── pipeline/
│   └── training_pipeline.py    # Pipeline definition with tags + Model
├── steps/
│   ├── ingest_data.py          # Data ingestion with ArtifactConfig
│   ├── clean_data.py           # Cleaning + dynamic quality tags
│   ├── train_model.py          # Training with model artifact tags
│   ├── evaluation.py           # Evaluation + dynamic performance tags
│   └── config.py               # Pydantic configuration
├── src/
│   ├── data_cleaning.py        # Data cleaning strategies
│   ├── model_development.py    # Model training strategies
│   └── evaluation.py           # Evaluation metric strategies
├── utils/
│   └── tag_manager.py          # Tag query and orphan detection
├── configuration/
│   └── config.yaml             # Pipeline parameters
├── data/
│   └── olist_customers_dataset.csv
├── tag_registry.py             # Centralized tag definitions
├── run_pipeline.py             # Entry point
├── requirements.txt
└── README.md
```

## 📚 ZenML Features Demonstrated

- [x] `@pipeline` and `@step` decorators
- [x] Pipeline tags (`tags=[...]` in decorator)
- [x] Runtime tags via `.with_options(tags=[...])`
- [x] YAML config tags
- [x] `ArtifactConfig` with tags on all artifacts
- [x] `Annotated` type hints for named outputs
- [x] Dynamic tagging with `add_tags()`
- [x] `Model` entity for Model Control Plane
- [x] Tag Registry with Python Enums
- [x] Tag filtering (`startswith:`, `contains:`)
- [x] Orphaned resource detection
- [x] Strategy design pattern for extensibility

## 📈 Pipeline Flow

```
Load Data ──→ Clean & Split ──→ Train Model ──→ Evaluate
   │               │                │              │
   ▼               ▼                ▼              ▼
[artifact-raw]  [artifact-       [artifact-    [artifact-metric]
[domain-        processed]       model]        + dynamic tags:
ecommerce]      + dynamic:       [algorithm-   [performance-
                [quality-*]      linear-       high-r2] or
                                 regression]   [performance-
                                               low-r2]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Resources

- [ZenML Documentation](https://docs.zenml.io/)
- [ZenML Tagging Guide](https://docs.zenml.io/how-to/data-artifact-management/handle-data-artifacts/tagging)
- [ZenML Model Control Plane](https://docs.zenml.io/how-to/model-management-metrics/model-control-plane)
- [Organizing Pipelines & Models](https://docs.zenml.io/user-guides/best-practices/organizing-pipelines-and-models)

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👤 Author

**Abhilash Kumar Panda**
- 📧 Email: abhilashk.isme1517@gmail.com
- 🔗 LinkedIn: [Abhilash Kumar Panda](https://www.linkedin.com/in/abhilash-kumar-panda/)
- 🌐 Portfolio: [abhilashpanda04.github.io](https://abhilashpanda04.github.io/Portfolio_site/)
- GitHub: [@abhilashpanda04](https://github.com/abhilashpanda04)

---

⭐ If this project helps you, please consider giving it a star!
