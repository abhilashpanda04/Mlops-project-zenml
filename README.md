
# MLOPS with ZENml

This project we build machine learning model based on zenml frame work , which will help us to learn mlops.

## Project Description

"This project uses ZenML to build and orchestrate a machine learning pipeline for olist cutsomer oil dataset.

## Project Structure

* **configuration:**  contains configuration files for your ZenML stack, data sources, or other project settings.
* **data:**  This is where you'll store your datasets (or links to them) used for training and evaluation.
* **ignore:**  This folder might contain files or patterns to be ignored by version control (like a `.gitignore` file).
* **pipeline:**  Holds the core of your ZenML pipeline definition (`training_pipeline.py`).
* **src:**  Contains Python modules for specific pipeline steps:
    * `data_cleaning.py`:  Handles data preprocessing and cleaning.
    * `evaluation.py`:  Implements model evaluation metrics and logic.
    * `model_development.py`:  Defines your machine learning model architecture and training process.
* **steps:**  Might contain custom ZenML steps used in your pipeline.
* **.gitignore:** Specifies files or folders to be excluded from version control.
* **README.md:** This file! Provides an overview and instructions for the project
* **__init__.py:**  Indicates that `src` and potentially other folders are Python packages.
* **requirements.txt:** Lists the Python dependencies required to run your project.
* **run_pipeline.py:**  Script to execute your ZenML pipeline.

## Getting Started

1. **Clone the repository:** 
   ```bash
   git clone [https://github.com/](https://github.com/)abhilashpanda04/Mlops-project-zenml.git
    ```
Use code with caution.

Install dependencies:

```bash
pip install -r requirements.txt
```
Configure ZenML:

Follow the ZenML documentation to set up your ZenML stack and configure any necessary data connectors or artifact stores.
Run the pipeline:

```bash
python run_pipeline.py
```

