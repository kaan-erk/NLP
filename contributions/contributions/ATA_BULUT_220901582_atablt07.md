# Team Member Contribution Report

- **Name:** Ata Bulut
- **Student ID:** 220901582
- **GitHub Username:** atablt07
- **Primary Module / File:** `train_models.py`

---

## 1. Overview of Contributions

In this phase of the project, I was responsible for the **feature extraction pipeline, classical machine learning model benchmarking, and automated evaluation metrics logging**. 

Specifically, I developed and pushed `train_models.py`, which implements an end-to-end pipeline that loads preprocessed text datasets, transforms textual features using n-gram TF-IDF vectorization, trains multiple classical classifiers, and logs comprehensive performance comparisons and visualizations.

---

## 2. Detailed Technical Breakdown

### A. Data Handling & Preprocessing Verification
- Built data loading logic using `pandas` to ingest `train_data.csv` and `test_data.csv`.
- Handled potential missing values via `dropna(subset=['clean_text', 'label'])` to maintain training integrity across target classes (`Politics`, `Finance`, `Weather`, `Sport`).
- Configured automated extraction and sorting of unique target class labels for dynamic matrix plotting.

### B. Feature Extraction (TF-IDF Vectorization)
- Implemented `TfidfVectorizer` from `scikit-learn` combining unigrams and bigrams (`ngram_range=(1, 2)`).
- Restricted the vocabulary space to the top 5,000 features (`max_features=5000`) to prevent overfitting and optimize memory usage during matrix operations.
- Ensured strict separation between training and test sets by fitting the vectorizer solely on `X_train` and transforming `X_test`.

### C. Multi-Model Architecture & Training
- Configured and integrated four distinct classical classifiers to benchmark performance across different algorithmic families:
  - **Logistic Regression:** Configured with `max_iter=1000` to ensure convergence.
  - **Multinomial Naive Bayes (`MultinomialNB`):** Applied as a standard probabilistic baseline for bag-of-words text classification.
  - **Linear Support Vector Machine (`LinearSVC`):** Implemented for high-dimensional text margin maximization.
  - **Random Forest Classifier (`RandomForestClassifier`):** Configured with 100 estimators (`n_estimators=100`) as an ensemble baseline.
- Automated sequential model fitting, prediction generation, and metric aggregation.

### D. Evaluation Metrics, Visualizations & Reporting
- Integrated standard evaluation metrics: overall **Accuracy** and **Macro F1-Score** (`f1_score(average='macro')`) to account for any class distribution nuances.
- **Visualizations:**
  - Automated generation of individual confusion matrices per model using `seaborn.heatmap` and distinct color palettes, saved as high-resolution PNGs (300 DPI).
  - Designed a combined $2 \times 2$ grid subplot (`all_classical_models_confusion_matrices.png`) for direct side-by-side visual error comparison.
- **Automated Logging:**
  - Engineered tabular export to `model_comparison.csv`, sorted descending by F1-Score.
  - Generated `all_models_report.txt` containing full `classification_report` printouts (precision, recall, f1 per category) and raw confusion matrices, automatically flagging the best-performing model.

---

## 3. GitHub Commits & Diff Highlights

- **File Added:** `train_models.py`
- **Target Output Directory Managed:** `classical_models_results/`
- **Key Dependencies Utilized:** `scikit-learn`, `pandas`, `matplotlib`, `seaborn`
