# Text Buzzword Analyzer

A machine learning project for detecting corporate language (corporate speak, buzzwords) and distinguishing it from plain English.

The goal is to develop and compare three approaches to binary text classification: traditional machine learning, recurrent neural networks, and transformer-based architectures.

## 1. Project Overview

The system classifies text into two categories:

| Label | Class | Description |
|---|---|---|
| 0 | Plain English | Clear, straightforward English |
| 1 | Corporate Speak | Corporate jargon, buzzwords, and unnecessarily complicated business language |

Three classification approaches are implemented:

**Stage 1 — TF-IDF + Logistic Regression**

A baseline machine learning model using TF-IDF for text vectorization and Logistic Regression for classification.

**Stage 2 — BiGRU**

A Bidirectional Gated Recurrent Unit network designed to capture word sequences and contextual information.

**Stage 3 — DistilBERT**

A transformer-based model using the pretrained DistilBERT architecture, fine-tuned for binary text classification.

## 2. Project Structure

```text
text-buzzword-analyzer/
│
├── data/
│   └── raw/
│       ├── buzzwords.txt
│       └── concrete.txt
│
├── docs/
│   └── report.md
│
├── reports/
│   ├── comparison.md
│   └── metrics_*.json
│
├── scripts/
│   ├── analyze_text.py
│   ├── diagnose_stage1.py
│   ├── download_dataset.py
│   ├── inspect_dataset.py
│   ├── inspect_samples.py
│   ├── prepare_data.py
│   ├── summarize.py
│   ├── train_baseline.py
│   ├── train_nn.py
│   └── train_transformer.py
│
├── src/
│   ├── config.py
│   ├── external.py
│   └── metrics.py
│
├── tests/
│   └── test_prepare_synthetic.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

Downloaded datasets, generated data splits, trained model weights, temporary files, and Python cache are excluded from version control.

## 3. Installation

### Requirements

- Python 3.10+
- pip
- Git

### Clone the repository

```bash
git clone https://github.com/USERNAME/text-buzzword-analyzer.git
cd text-buzzword-analyzer
```

### Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Data Preparation

The project uses source vocabulary files and a downloaded dataset for training and evaluation.

The source vocabulary files are located in:

```text
data/raw/
├── buzzwords.txt
└── concrete.txt
```

Generated datasets are stored locally and are not included in the repository.

### Download and inspect the dataset

```bash
python scripts/download_dataset.py
python scripts/inspect_dataset.py
```

### Prepare the dataset

```bash
python scripts/prepare_data.py
```

The resulting dataset splits are expected to be stored in:

```text
data/processed/
├── train.csv
├── val.csv
└── test.csv
```

## 5. Model Training

Each classification approach has a separate training script.

### Stage 1 — TF-IDF + Logistic Regression

```bash
python scripts/train_baseline.py
```

### Stage 2 — BiGRU

```bash
python scripts/train_nn.py
```

### Stage 3 — DistilBERT

```bash
python scripts/train_transformer.py
```

Trained model artifacts are stored locally and excluded from Git version control.

## 6. Evaluation and Analysis

The project includes utilities for inspecting predictions, diagnosing classification errors, and evaluating model performance.

Available scripts:

```bash
python scripts/analyze_text.py
python scripts/diagnose_stage1.py
python scripts/summarize.py
```

Evaluation metrics and comparison reports are stored in:

```text
reports/
```

Metrics are saved in JSON format, while comparative analysis is documented in Markdown.

## 7. Testing

The project includes tests for synthetic data preparation.

Run the tests using:

```bash
python -m pytest tests/
```

## 8. Reproducibility

The repository contains source code, scripts, configuration files, and evaluation reports.

The following artifacts are intentionally excluded:

- Original downloaded datasets and generated dataset splits.
- Trained model weights and checkpoints.
- Python virtual environments.
- Temporary files and caches.

To reproduce the experiments, install the dependencies, obtain the required source data, and execute the data preparation and training scripts.

## 9. Technologies

- Python
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Pandas
- NumPy
- TF-IDF
- Logistic Regression
- BiGRU
- DistilBERT

## 10. Project Status

The project implements three approaches to binary text classification and provides a framework for comparing their performance.

Potential future improvements include hyperparameter optimization, additional evaluation datasets, model interpretability, and classification performance improvements.

---

**Project:** Text Buzzword Analyzer  
**Task:** Binary Text Classification  
**Classes:** Plain English / Corporate Speak
