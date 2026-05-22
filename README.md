
APP LINK=

<img width="1900" height="793" alt="image" src="https://github.com/user-attachments/assets/77dd1822-fd28-425f-9cf6-cb426e06740d" />


# 🩺 Hybrid Skin Cancer Diagnostic System (XAI-Enabled)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-blue.svg)](https://www.docker.com/)

An end-to-end clinical decision support system for skin cancer classification. This project implements a **Hybrid AI Architecture** combining the spatial feature extraction of a fine-tuned **ResNet50** deep learning backbone with the downstream tabular classification power of **XGBoost**, backed by **Explainable AI (XAI)** frameworks for high-stakes clinical transparency.

---

## 🚀 Key Features

* **Hybrid Deep Learning & Tree Architecture:** ResNet50 (Feature Extractor) + XGBoost (Classification Head) pipeline for robust malignancy scoring.
* **Dual-Layer Explainable AI (XAI):** * **Grad-CAM:** Spatial pixel-attribution mapping to localize lesion boundaries.
  * **SHAP (SHapley Additive exPlanations):** Global and local feature-level impact profiling on the gradient-boosted head.
* **Production-Grade API:** High-performance asynchronous FastAPI backend handling multi-part image uploads and inference workloads.
* **Clinical Dashboard:** Interactive UI built with Streamlit providing real-time data submission, inference triggers, and visual explanations.
* **Automated Audit Reporting:** Generates side-by-side diagnostic PDF/HTML reports for physical clinical audits.

---

## 🏗️ Project Structure

```text
4-XAI/
├── .github/               # CI/CD Workflows
├── api/                   # FastAPI Backend Application
│   ├── static/            # Cached images and generated XAI reports
│   └── main.py            # API Gateway & Endpoints
├── app/                   # Web application configurations
│   └── app_gui.py         # Streamlit UI Frontend
├── best_model/            # Model serialization directory
│   └── ResNet50_Final.ipynb # Model training & evaluation notebook
├── data/                  # Local training and evaluation datasets
├── models/                # Production-ready weights (.pth, .json)
├── notebooks/             # R&D, experimentation, and prototype code
├── src/                   # Core Modular Engineering Package
│   ├── __init__.py        # Package initializer
│   ├── config.py          # Global hyperparameter & system configuration
│   ├── data_engine.py     # ETL, image preprocessing, and augmentation pipeline
│   ├── models.py          # Custom PyTorch-XGBoost hybrid module
│   ├── reporting.py       # PDF/HTML diagnostic generation engine
│   ├── xai_tabular.py     # SHAP execution & force plot serialization
│   └── xai_vision.py      # Custom Grad-CAM tensor hook implementation
├── tests/                 # Unit and integration testing suites
├── Dockerfile             # Multi-stage production build container logic
├── .dockerignore          # Docker build optimizations
├── .gitignore             # Version control exclusions
├── explain.py             # CLI script for local XAI generation
├── model_train.py         # CLI orchestration script for model training
└── README.md              # Documentation

```

---

## 🔬 System Architecture & Inference Flow

```
[Input Lesion Image] ──> [Data Engine: Preprocessing & Resizing]
                                  │
                                  ▼
                     [Fine-Tuned ResNet50 Backbone]
                                  │
                 (Extracted 2048-Dimensional Latent Vector)
                                  │
                                  ├───> [Grad-CAM Hook] ──> [Visual Heatmap]
                                  │
                                  ▼
                         [XGBoost Classifier]
                                  │
                                  ├───> [SHAP Explainer] ──> [Force Plots]
                                  ▼
                    [Malignancy Probabilities & Report]

```

### 1. The Hybrid Pipeline

The inference cycle breaks down structural processing into two optimized domains. The **ResNet50** CNN backbone processes raw dermoscopic images, transforming unstructured pixels into high-dimensional geometric embeddings. These embeddings are fed directly into **XGBoost**, utilizing gradient boosting over structured trees to optimize non-linear classification limits.

### 2. Clinical Interpretability (XAI)

To satisfy rigorous medical transparency criteria (e.g., NHS/NICE AI frameworks in the UK):

* **Grad-CAM** extracts gradients from the final convolutional layer of ResNet50, projecting spatial attention weights directly over the raw lesion to verify that the model isn't latching onto confounding artifacts (e.g., hair, marker ink).
* **SHAP** calculates Shapley values for the feature vector input, outputting a mathematical distribution of features forcing the decision boundary toward or away from a positive diagnosis.

---

## 🛠️ Local Installation & Orchestration

### Prerequisites

* Python 3.11+
* CUDA-supported GPU (Optional but recommended for rapid training)

### 1. Environment Configuration

```bash
# Clone Repository
git clone [https://github.com/YOUR_USERNAME/skin-cancer-xai.git](https://github.com/YOUR_USERNAME/skin-cancer-xai.git)
cd skin-cancer-xai

# Create and Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install Production Dependencies
pip install -r requirements.txt

```

### 2. Execution Pipeline

* **Train the Hybrid Network:**
```bash
python model_train.py

```


* **Spin Up the API Gateway Backend:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

```


* **Launch the Streamlit Medical UI:**
```bash
streamlit run app/app_gui.py

```



---

## 🐳 Containerized Deployment

The application is completely containerized to separate operational layers, optimizing it for production deployments (e.g., AWS ECS, Kubernetes).

```bash
# Build the container locally
docker build -t skin-cancer-xai-system:latest .

# Run the container mapping both the API and Streamlit interfaces
docker run -p 8000:8000 -p 8501:8501 skin-cancer-xai-system:latest

```

---

## 📊 Dashboard Visualizations

*Here, insert your 3 screenshots sequentially under clear headings like: **"Medical Practitioner Upload Interface"**, **"Grad-CAM Heatmap Localization"**, and **"SHAP Explainer Output"**.*

<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/fc742c06-9f0f-4e08-9c3e-a71342898b2a" />


<img width="1686" height="351" alt="image" src="https://github.com/user-attachments/assets/1d4e4cbb-1ff0-4853-8152-d43f0e1138b7" />


<img width="1047" height="400" alt="image" src="https://github.com/user-attachments/assets/1caf43fc-aee2-4a9f-a150-6293c6687bd5" />

```

---


