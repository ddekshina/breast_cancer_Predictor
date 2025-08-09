# Breast Cancer Predictor

## Overview

Breast Cancer Predictor is a **machine learning-based application** that estimates breast cancer risk from patient data. It offers both a **web interface** and a **command-line tool** for predictions, model details, and personalized health recommendations.

## Key Features

* **Web App (Flask)** – Enter patient details and view predictions in your browser.
* **Command-Line Tool** – Run predictions and view model info from the terminal.
* **High Accuracy** – \~96-97% accuracy and \~98-99% AUC score.
* **Health Recommendations** – Automated guidance based on predicted risk.
* **Training Pipeline** – Script to train, evaluate, and save updated models.

## Tech Stack

* **Languages:** Python, HTML, JavaScript
* **Libraries:** Flask, scikit-learn, NumPy, Pandas, joblib, matplotlib, seaborn

## Quick Start

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Train the model** (once)

```bash
python breast_cancer_trainer.py
```

3. **Run predictions**

* CLI:

```bash
python breast_cancer_predictor.py --sample
python breast_cancer_predictor.py --info
```

* Web App:

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000)

## Performance

* Accuracy: \~96-97%
* AUC: \~98-99%
* Prediction time: <100ms

---

A modular, easy-to-deploy tool for breast cancer risk assessment, combining an intuitive UI with a robust ML backend.
