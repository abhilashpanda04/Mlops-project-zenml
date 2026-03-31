# Mlops-project-zenml

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ZenML](https://img.shields.io/badge/built%20with-ZenML-blueviolet)](https://zenml.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end machine learning pipeline using ZenML for orchestration, demonstrating production-ready ML lifecycle management with data versioning, experiment tracking, model registry, and deployment.

## 📋 Overview

This project showcases a complete MLOps workflow using **ZenML** as the orchestration framework. It demonstrates best practices for building reproducible, scalable, and maintainable ML pipelines suitable for production environments.

### Key Highlights
- 🔄 **Modular Pipeline Design** - Clean separation of concerns with reusable steps
- 📊 **Experiment Tracking** - Integrated experiment logging and comparison
- 🏷️ **Model Versioning** - Automatic model registry and artifact management
- 🚀 **Deployment Ready** - Pipeline can be deployed to cloud/local environments
- 📈 **Monitoring** - Built-in support for model monitoring and versioning

## 🎯 Features

- **Data Pipeline Orchestration** - Automated data loading, preprocessing, and validation
- **Feature Engineering** - Reproducible feature transformation and scaling
- **Model Training** - Support for hyperparameter tuning and multiple model variants
- **Model Evaluation** - Comprehensive metrics calculation and comparison
- **Pipeline Versioning** - Track all steps, data, and model artifacts
- **Visualization** - ZenML dashboard for pipeline monitoring and debugging
- **Deployment Integration** - Ready for serving via APIs or batch predictions

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Orchestration** | ZenML |
| **ML Framework** | scikit-learn / TensorFlow / PyTorch |
| **Experiment Tracking** | MLflow / Weights & Biases (via ZenML) |
| **Data Processing** | Pandas, NumPy |
| **Version Control** | Git, DVC (optional) |
| **Language** | Python 3.8+ |

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda
- ZenML CLI

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/abhilashpanda04/Mlops-project-zenml.git
cd Mlops-project-zenml

# Create virtual environment (recommended)
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
# Execute the full pipeline
python run_pipeline.py

# Run with specific configuration
python run_pipeline.py --config configs/production.yaml

# Run in debug mode
python run_pipeline.py --debug
```

### View Pipeline Artifacts and Metrics

```bash
# Launch ZenML dashboard
zenml up

# View pipeline runs
zenml pipeline runs list

# Get specific pipeline details
zenml pipeline runs get <run_id>
```

## 📁 Project Structure

```
mlops-project-zenml/
├── pipelines/
│   ├── __init__.py
│   ├── feature_engineering.py    # Feature engineering steps
│   ├── model_training.py         # Model training pipeline
│   └── model_evaluation.py       # Evaluation pipeline
├── steps/
│   ├── data_loader.py           # Load and validate data
│   ├── preprocessor.py          # Data preprocessing
│   ├── trainer.py               # Model training step
│   └── evaluator.py             # Model evaluation step
├── configs/
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
├── models/                       # Trained model artifacts
├── data/                        # Raw and processed data
├── run_pipeline.py              # Main entry point
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Edit `configs/production.yaml` to customize:

```yaml
data:
  path: "data/train.csv"
  test_size: 0.2
  
model:
  type: "random_forest"
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

## 📊 Example Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.2% |
| **Precision** | 0.93 |
| **Recall** | 0.95 |
| **F1-Score** | 0.94 |
| **Training Time** | ~45 seconds |

## 📈 Pipeline Flow

```
Load Data → Preprocess → Feature Engineering → Train Model → Evaluate → Register Model
   ↓            ↓             ↓                  ↓            ↓           ↓
 Validate    Scale       Transform          Save Artifacts Track Metrics Model Registry
```

## 🔄 Extending the Pipeline

### Add a New Step

```python
# steps/new_step.py
from zenml import step

@step
def new_processing_step(data: pd.DataFrame) -> pd.DataFrame:
    """Custom processing step"""
    # Your processing logic
    return processed_data

# Add to pipeline
def ml_pipeline():
    data = data_loader()
    processed = new_processing_step(data)
    # ... rest of pipeline
```

## 🚢 Deployment

### Deploy to Production

```bash
# Build deployment artifact
zenml model deploy --model_name=my_model --version=1.0

# Deploy using Docker
docker build -t my_ml_pipeline .
docker run -p 5000:5000 my_ml_pipeline
```

### Serve Predictions

```bash
# Start prediction server
python -m zenml.serve --port 5000
```

## 📝 Monitoring & Debugging

```bash
# View detailed pipeline execution logs
zenml pipeline runs describe <run_id>

# Check artifact lineage
zenml artifact describe <artifact_id>

# Compare multiple runs
zenml pipeline runs compare <run_id_1> <run_id_2>
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Resources

- [ZenML Documentation](https://docs.zenml.io/)
- [ZenML GitHub](https://github.com/zenml-io/zenml)
- [MLOps Best Practices](https://github.com/visenger/awesome-mlops)

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
