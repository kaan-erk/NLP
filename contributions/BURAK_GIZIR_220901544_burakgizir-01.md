# Team Member Contribution Report

- **Name:** Burak Gizir
- **Student ID:** 220901544
- **GitHub Username:** burakgizir-01
- **Primary Module / File:** `train_bert.py`

---

## 1. Overview of Contributions

In this stage of the project, I was responsible for designing and implementing the **deep learning and transformer-based text classification architecture**[cite: 2].

I authored and pushed `train_bert.py`, which fine-tunes a pre-trained Transformer model (`distilbert-base-uncased`) on our news dataset[cite: 2]. The script handles tokenization, PyTorch dataset creation, end-to-end model fine-tuning with automatic checkpoint caching, test set evaluation, and metric logging (reports, summaries, and heatmaps)[cite: 2].

---

## 2. Detailed Technical Breakdown

### A. Data Ingestion & Label Mapping
- Ingested raw headline samples from `train_data.csv` and `test_data.csv` using `pandas`[cite: 2].
- Extracted unique target categories dynamically and constructed bidirectional label-to-index mappings (`label2id`, `id2label`) to interface seamlessly with Hugging Face classification heads[cite: 2].

### B. Tokenization & PyTorch Data Pipeline
- Integrated the Hugging Face `AutoTokenizer` using the `distilbert-base-uncased` checkpoint[cite: 2].
- Implemented a custom data preparation pipeline `prepare_data()` converting raw headline strings into PyTorch `TensorDataset` instances[cite: 2]:
  - Enforced sequence truncation and dynamic padding with `max_length=64`[cite: 2].
  - Extracted `input_ids` and `attention_mask` tensors, coupled with long-integer label targets[cite: 2].
- Configured PyTorch `DataLoader` pipelines with a batch size of 32, applying random shuffling to the training split[cite: 2].

### C. Model Fine-Tuning & Checkpoint Caching
- Configured dynamic hardware routing via `torch.device` to support GPU acceleration (`cuda`) with automatic CPU fallback[cite: 2].
- Initialized `AutoModelForSequenceClassification` loaded with pre-configured category dimensions (`num_labels=len(categories)`)[cite: 2].
- Implemented conditional checkpoint loading: the script checks for an existing saved checkpoint in `bert_model_results/model` to eliminate redundant re-training cycles during evaluation runs[cite: 2].
- Constructed the training loop:
  - Optimizer: `AdamW` with a learning rate of $3 \times 10^{-5}$[cite: 2].
  - Executed 2-epoch fine-tuning with gradient zeroing, backpropagation on sequence classification cross-entropy loss, and periodic progress/loss logging[cite: 2].
  - Persisted the fine-tuned model and tokenizer weights using `.save_pretrained()`[cite: 2].

### D. Model Evaluation, Visualization & Logging
- Evaluated model inference on the test split using `torch.no_grad()` to extract logits and computed class predictions via `torch.argmax(outputs.logits, dim=1)`[cite: 2].
- Benchmarked performance using **Macro F1-Score** and **Accuracy** via `scikit-learn`[cite: 2].
- Automated the export of multiple evaluation artifacts into `bert_model_results/`[cite: 2]:
  - **Confusion Matrix Heatmap:** High-resolution (300 DPI) Seaborn heatmap visualization saved as `confusion_matrix_bert.png`[cite: 2].
  - **Comprehensive Text Report:** Exported precision, recall, and F1-score breakdowns alongside the raw confusion matrix in `bert_report.txt`[cite: 2].
  - **Tabular Metric Summary:** Saved benchmark overview to `bert_summary.csv`[cite: 2].

---

## 3. GitHub Commits & Diff Highlights

- **File Added:** `train_bert.py`[cite: 2]
- **Target Output Directory Managed:** `bert_model_results/` (including saved weights in `bert_model_results/model`)[cite: 2]
- **Key Dependencies Utilized:** `torch`, `transformers` (Hugging Face), `scikit-learn`, `pandas`, `seaborn`, `matplotlib`[cite: 2]
