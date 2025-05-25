# wine-quality-mlops3
"MLOps pipeline for UCI Wine Quality prediction."

# Wine Quality MLOps Pipeline

This project demonstrates a robust MLOps pipeline for predicting wine quality based on chemical features using the UCI Wine Quality dataset.

## Project Overview

The goal of this project is to create a scalable and production-ready MLOps pipeline. This involves:

1.  **Model Training**: Training a machine learning model (Random Forest Classifier) on the UCI Wine Quality dataset.
2.  **Flask API**: Building a lightweight Flask API for real-time inference.
3.  **Automated Testing & Deployment**: Implementing Continuous Integration/Continuous Deployment (CI/CD) with GitHub Actions.
4.  **Deployment**: Hosting the application on a Google Cloud Virtual Machine (VM) for public accessibility.

## Flowchart of MLOps Pipeline



## Folder Structure


📦 wine-quality-mlops2/
├📁 data/                  # Dataset (winequality-red.csv)
├📁 model/                 # Trained model (wine_quality_model.pkl)
├📁 app/                   # Flask API (app.py)
│   └── 📄 init.py     # Module initializer
├📁 tests/                 # Unit tests (test_model.py)
├📁 .github/workflows/     # GitHub Actions (ci.yml)
├📁 notebooks/             # EDA and Model Training (eda.ipynb)
├📄 Dockerfile             # Containerization config
├📄 requirements.txt       # Python dependencies
├📄 .gitignore             # Git exclusions
└─📄 README.md              # Project overview


## Getting Started

Follow the steps in this README to set up and deploy your own MLOps pipeline.

---


