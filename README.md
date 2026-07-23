```markdown
# 🩺 Hybrid Skin Cancer Diagnostic System (XAI-Enabled)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-blue.svg)](https://www.docker.com/)

An end-to-end clinical decision support system for skin cancer classification. This project implements a **Hybrid AI Architecture** combining the spatial feature extraction of **ResNet50** with the tabular classification power of **XGBoost**, backed by **Explainable AI (XAI)** for clinical transparency.

---

## 🚀 Key Features

* **Hybrid Architecture:** ResNet50 (Backbone) + XGBoost (Head) for superior accuracy on dermoscopic images.
* **Dual-Layer XAI:** * **Grad-CAM:** Spatial localization of lesions.
    * **SHAP Force Plots:** Feature-level contribution analysis.
* **Production-Ready API:** High-performance FastAPI backend.
* **Clinical UI:** Streamlit-based dashboard for medical practitioners.
* **Automated Reporting:** Generates side-by-side diagnostic audit reports.

---

## 🏗️ Project Structure

```text
├── api/                   # FastAPI Backend & Static Asset Hosting
├── src/                   # Modular Source Code
│   ├── models.py          # Hybrid Model Architecture
│   ├── xai_vision.py      # Grad-CAM Implementation
│   ├── xai_tabular.py     # SHAP Logic
│   └── reporting.py       # Automated Clinical Reports
├── models/                # Serialized Model Weights (.pth & .json)
├── app_gui.py             # Streamlit Frontend
├── Dockerfile             # Containerization Logic
└── requirements.txt       # Dependency Management

```

---

## 🔬 How It Works

### 1. The Hybrid Pipeline

The system uses a two-stage inference process. First, the ResNet50 backbone (pre-trained on ImageNet and fine-tuned) extracts 2048 high-dimensional features from the lesion. These features are then passed to an XGBoost classifier, which provides the final malignancy probability.

### 2. Interpretability (XAI)

To meet clinical safety standards, every diagnosis is justified:

* **Grad-CAM** visualizes the specific pixels the CNN focused on.
* **SHAP Force Plots** show how the extracted features influenced the XGBoost decision, providing a "reasoning" path for the final classification.

---

## 🛠️ Installation & Setup

### Local Setup

1. **Clone the Repo:**
```bash
git clone [https://github.com/YOUR_USERNAME/skin-cancer-xai.git](https://github.com/YOUR_USERNAME/skin-cancer-xai.git)
cd skin-cancer-xai

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Launch the Stack:**
* **Terminal 1 (Backend):** `python -m uvicorn api.main:app --reload`
* **Terminal 2 (Frontend):** `streamlit run app_gui.py`



---

## 🐳 Docker Deployment

The entire stack is containerized for consistent deployment:

```bash
docker build -t skin-cancer-ai .
docker run -p 8000:8000 skin-cancer-ai

```
