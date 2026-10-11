# 📰 News Headlines Multi-Class Classification Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) pipeline for collecting, preprocessing, benchmarking, and evaluating real-world news headlines across four distinct categories: **Politics**, **Finance**, **Weather**, and **Sport**. 

The system compares traditional machine learning classifiers utilizing **TF-IDF n-gram vectorization** against state-of-the-art **Transformer-based deep learning (DistilBERT)**.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Repository Structure](#-repository-structure)
3. [Datasets](#-datasets)
4. [Data Acquisition & Web Scraping (`scraper.py`)](#-data-acquisition--web-scraping-scraperpy)
5. [Data Preprocessing Pipeline (`preprocess.py`)](#-data-preprocessing-pipeline-preprocesspy)
6. [Training & Classification Models](#-training--classification-models)
   - [Classical Machine Learning (`train_models.py`)](#1-classical-machine-learning-train_modelspy)
   - [Deep Learning / DistilBERT (`train_bert.py`)](#2-deep-learning--distilbert-train_bertpy)
7. [Benchmark Results & Discussion (`compare_models.py`)](#-benchmark-results--discussion-compare_modelspy)
8. [Installation & Reproduction](#-installation--reproduction)
9. [Team Contributions](#-team-contributions)
10. [License](#-license)

---

## 📌 Project Overview

News classification is a foundational task in text processing and automated information indexing. This project delivers an automated, reproducible research workflow:
- **Corpus Collection:** Automated scraping of Google News RSS feeds to assemble a custom corpus of 4,000 headlines (exactly 1,000 balanced instances per category).
- **Dual Preprocessing:** Tailored preprocessing pipelines for bag-of-words/n-gram features versus contextual neural representations.
- **Model Benchmarking:** Comprehensive evaluation of 5 distinct models (Logistic Regression, Naive Bayes, Linear Support Vector Machine, Random Forest, and Fine-Tuned DistilBERT).
- **Academic Reporting:** Automatic generation of publication-ready confusion matrix grids, comparative bar plots, classification metrics (Precision, Recall, Macro F1, Accuracy), and structured evaluation logs.

---

## 📂 Repository Structure

```text
├── contributions/
│   └── KAAN_ERK_TORAMAN_220901564_kaan-erk.md   # Individual contribution report
├── classical_models_results/
│   ├── all_classical_models_confusion_matrices.png # 2x2 Grid confusion matrix plot
│   ├── all_models_report.txt                       # Detailed metrics per classical model
│   ├── model_comparison.csv                       # Classical models metric summary
│   └── confusion_matrix_*.png                      # Individual confusion matrix heatmaps
├── bert_model_results/
│   ├── bert_report.txt                             # DistilBERT classification report
│   ├── bert_summary.csv                            # DistilBERT test set metrics
│   └── confusion_matrix_bert.png                   # DistilBERT confusion matrix heatmap
├── overall_comparison/
│   ├── all_models_comparison.csv                   # Final ranking of all 5 models
│   ├── all_models_comparison_chart.png             # Performance bar chart
│   └── FINAL_EVALUATION_REPORT.txt                 # Academic summary & discussion report
├── news_dataset_4000.csv                           # Raw scraped balanced dataset (4,000 rows)
├── train_data.csv                                  # Stratified training split (80% / 3,184 rows)
├── test_data.csv                                   # Stratified testing split (20% / 796 rows)
├── scraper.py                                      # Google News RSS scraper & cleaner
├── preprocess.py                                   # Text normalization & stratified splitter
├── train_models.py                                 # TF-IDF + 4 Classical ML classifiers
├── train_bert.py                                   # DistilBERT fine-tuning pipeline
├── compare_models.py                               # Comprehensive benchmarking & visualization
├── requirements.txt                                # Python package dependencies
├── .gitignore                                      # Ignored checkpoints and cache
├── LICENSE                                         # MIT License
└── README.md                                       # Project documentation
```

---

## 📊 Datasets

The repository provides both raw and preprocessed datasets:

| File | Type | Samples | Description |
| :--- | :---: | :---: | :--- |
| [`news_dataset_4000.csv`](news_dataset_4000.csv) | **Raw** | 4,000 | Scraped headlines with columns `text` and `label`. Exactly 1,000 items per class. |
| [`train_data.csv`](train_data.csv) | **Cleaned (Train)** | 3,184 | 80% stratified split with columns `text`, `label`, and sanitized `clean_text`. |
| [`test_data.csv`](test_data.csv) | **Cleaned (Test)** | 796 | 20% stratified holdout split with identical class distribution. |

**Target Classes:**
- `Politics`
- `Finance`
- `Weather`
- `Sport`

---

## 🌐 Data Acquisition & Web Scraping (`scraper.py`)

The data collection engine is implemented in [`scraper.py`](scraper.py):
1. **Source:** Google News RSS feeds (`https://news.google.com/rss/search?q=...`).
2. **Query Diversity:** Uses 15 distinct, high-signal keyword queries per category (60 queries in total) to eliminate topic bias.
3. **Automated Source Stripping:** Real news headlines contain publisher tags (e.g., `" - BBC"`, `" - Reuters"`). A regex filter dynamically strips these suffixes (`re.sub(r'\s*-\s*[^-]+$', '', title)`) so classifiers learn semantic domain content rather than publisher signatures.
4. **Deduplication:** Utilizes real-time set tracking to drop duplicate headlines.
5. **Class Balancing:** Automatically limits headlines to **strictly 1,000 items per category**, preventing class imbalance issues from the ground up.

---

## 🧹 Data Preprocessing Pipeline (`preprocess.py`)

The preprocessing module in [`preprocess.py`](preprocess.py) handles text normalization and dataset splitting:

### 1. Dual-Representation Architecture
- **For Classical ML (`clean_text`):**
  - Case folding (lowercasing).
  - Punctuation and numeric character removal (`re.sub(r'[^a-z\s]', ' ', text)`).
  - Stopword filtering using NLTK English stopwords.
  - Length pruning (dropping artifacts with fewer than 3 tokens).
- **For Deep Learning / Transformers (`text`):**
  - The raw headline text is explicitly preserved. Pre-trained BERT/DistilBERT models leverage casing, punctuation, and linguistic syntax for bidirectional contextual self-attention.

### 2. Stratified Splitting
- Uses `train_test_split` with `stratify=df['label']` and `random_state=42`.
- **Train Split (80%):** 3,184 samples (~796 per class).
- **Test Split (20%):** 796 samples (~199 per class).

---

## 🧠 Training & Classification Models

### 1. Classical Machine Learning (`train_models.py`)
- **Feature Extraction:** `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))` capturing unigrams and bigrams.
- **Classifiers Evaluated:**
  - **Logistic Regression:** Multinomial solver, L2 regularization (`max_iter=1000`).
  - **Naive Bayes:** Multinomial Naive Bayes (`MultinomialNB`), ideal for sparse text matrices.
  - **Linear SVM:** Support Vector Classification with linear kernel (`LinearSVC`).
  - **Random Forest:** Ensemble of 100 decision trees (`RandomForestClassifier(n_estimators=100)`).
- **Artifacts Saved:** Individual heatmaps, a combined 2x2 grid confusion matrix ([`classical_models_results/all_classical_models_confusion_matrices.png`](classical_models_results/all_classical_models_confusion_matrices.png)), and evaluation reports.

### 2. Deep Learning / DistilBERT (`train_bert.py`)
- **Base Architecture:** `distilbert-base-uncased` (6 transformer layers, 66M parameters).
- **Tokenization:** Subword WordPiece tokenization with `max_length=64`, dynamic padding, and truncation.
- **Training Hyperparameters:**
  - Optimization: `AdamW` with learning rate $\eta = 3 \times 10^{-5}$
  - Batch size: $32$
  - Epochs: $2$
  - Loss function: Cross-Entropy Loss
- **Artifacts Saved:** Classification report, confusion matrix ([`bert_model_results/confusion_matrix_bert.png`](bert_model_results/confusion_matrix_bert.png)), and summary CSV.

---

## 📈 Benchmark Results & Discussion (`compare_models.py`)

Running [`compare_models.py`](compare_models.py) aggregates all evaluated models on the unseen holdout test set:

### Final Leaderboard

| Rank | Model | Representation / Method | Accuracy | Macro F1-Score |
| :---: | :--- | :--- | :---: | :---: |
| 🥇 **1** | **BERT (DistilBERT)** | **Contextual Embedding (Transformer)** | **98.99%** | **0.9899** |
| 🥈 **2** | **Linear SVM** | TF-IDF (1,2-gram) | **97.36%** | **0.9735** |
| 🥉 **3** | **Naive Bayes** | TF-IDF (1,2-gram) | **97.23%** | **0.9723** |
| 4 | **Logistic Regression** | TF-IDF (1,2-gram) | **96.86%** | **0.9685** |
| 5 | **Random Forest** | TF-IDF (1,2-gram) | **94.84%** | **0.9485** |

### Key Insights & Findings
1. **Transformer Superiority:** DistilBERT achieved the highest performance (**98.99% Accuracy / F1**). Self-attention mechanisms capture cross-word dependencies and nuanced context that bag-of-words methods cannot encode.
2. **High Efficiency of Linear SVM & Naive Bayes:** Linear SVM reached **97.36% Accuracy**, trailing the deep transformer by only ~1.6% while training in under a second. News headlines naturally exhibit high vocabulary specificity per domain (e.g., *dividend*, *typhoon*, *midterms*, *quarterback*), making TF-IDF weighted linear decision boundaries exceptionally strong baselines.
3. **Random Forest Behavior:** Random Forest performed lowest among tested models (94.84%) due to high sparsity and orthogonality in the 5,000-dimensional TF-IDF space.

---

## 🚀 Installation & Reproduction

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/kaan-erk/NLP.git
cd NLP

# (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Full Pipeline Step-by-Step

```bash
# Step 1: Scrape 4,000 balanced headlines from Google News
python scraper.py

# Step 2: Clean data and generate stratified 80/20 train/test splits
python preprocess.py

# Step 3: Train and evaluate classical machine learning models
python train_models.py

# Step 4: Fine-tune and evaluate DistilBERT
python train_bert.py

# Step 5: Generate benchmark charts and final evaluation report
python compare_models.py
```

---

## 👥 Team Contributions

Per the submission guidelines, individual contribution documents are available in the [`contributions/`](contributions/) directory:

- [**Kaan Erk Toraman** (`220901564`, `@kaan-erk`)](contributions/KAAN_ERK_TORAMAN_220901564_kaan-erk.md):
  - Architecture and implementation of the data acquisition pipeline ([`scraper.py`](scraper.py)).
  - RSS feed query engineering across all 4 categories (60 search queries).
  - Regex sanitization of news headlines (removing publisher suffix signatures).
  - Real-time deduplication and strict balancing of the 4,000 sample corpus ([`news_dataset_4000.csv`](news_dataset_4000.csv)).

---

## 📄 License

This project, its accompanying datasets, and evaluation scripts are released under the [MIT License](LICENSE). You are free to use, modify, and distribute this work with attribution.
