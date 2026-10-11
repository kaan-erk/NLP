# ==============================================================================
# STEP 3: TF-IDF VECTORIZATION AND CLASSICAL MODELS TRAINING
# ==============================================================================
# This script:
# 1. Loads train_data.csv and test_data.csv (Politics, Finance, Weather, Sport).
# 2. Vectorizes clean_text using TF-IDF (1,2-grams, 5000 max features).
# 3. Trains 4 classical Machine Learning classifiers:
#    - Logistic Regression
#    - Naive Bayes (MultinomialNB)
#    - Linear Support Vector Machine (LinearSVC)
#    - Random Forest
# 4. Generates individual and combined 2x2 grid Confusion Matrices.
# 5. Saves all evaluation outputs to the 'classical_models_results' directory.
# ==============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

OUTPUT_DIR = "classical_models_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. LOAD DATA
print("Loading data...")
train_df = pd.read_csv("train_data.csv").dropna(subset=['clean_text', 'label'])
test_df = pd.read_csv("test_data.csv").dropna(subset=['clean_text', 'label'])

X_train = train_df['clean_text']
y_train = train_df['label']
X_test = test_df['clean_text']
y_test = test_df['label']
classes = sorted(list(y_train.unique()))

print(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")
print(f"Classes ({len(classes)} Total): {classes}")

# 2. TF-IDF VECTORIZATION
print("\nPerforming TF-IDF Vectorization...")
vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# 3. DEFINE MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# 4. TRAIN AND EVALUATE MODELS
print("\n" + "="*50)
print("       CLASSICAL MODELS BENCHMARK RESULTS")
print("="*50)

results = []
report_blocks = []

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes_list = axes.flatten()
palettes = ['Blues', 'Greens', 'Purples', 'Oranges']

for idx, (name, model) in enumerate(models.items()):
    # Fit model
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)
    
    # Calculate metrics
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    
    results.append({
        "Model": name,
        "Method": "TF-IDF (1,2-gram)",
        "Accuracy": round(acc, 4),
        "F1_Score": round(f1, 4)
    })
    
    print(f"{name:<22} -> Acc: {acc*100:.2f}% | F1: {f1*100:.2f}%")
    
    cm = confusion_matrix(y_test, predictions, labels=classes)
    
    # 4a. Individual Confusion Matrix plot
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=palettes[idx], xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {name} (F1: {f1*100:.2f}%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    clean_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{clean_name}.png"), dpi=300)
    plt.close()

    # 4b. Subplot on combined 2x2 grid
    sns.heatmap(cm, annot=True, fmt='d', cmap=palettes[idx], xticklabels=classes, yticklabels=classes, ax=axes_list[idx])
    axes_list[idx].set_title(f"{name} (Acc: {acc*100:.1f}% | F1: {f1*100:.1f}%)", fontsize=12, fontweight='bold')
    axes_list[idx].set_xlabel("Predicted Label", fontsize=10)
    axes_list[idx].set_ylabel("True Label", fontsize=10)

    # Detailed report text
    report_blocks.append(f"""--------------------------------------------------
MODEL: {name.upper()}
--------------------------------------------------
Accuracy: {acc*100:.2f}%
F1-Score: {f1*100:.2f}%

Classification Report:
{classification_report(y_test, predictions, target_names=classes)}

Confusion Matrix (Numerical):
{cm}
""")

# Save combined 2x2 grid plot
fig.suptitle("Confusion Matrices of All Classical Models", fontsize=16, fontweight='bold')
plt.tight_layout()
combined_cm_path = os.path.join(OUTPUT_DIR, "all_classical_models_confusion_matrices.png")
fig.savefig(combined_cm_path, dpi=300)
plt.close()

# 5. SAVE CSV AND SUMMARY REPORTS
results_df = pd.DataFrame(results).sort_values(by="F1_Score", ascending=False).reset_index(drop=True)
csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

report_path = os.path.join(OUTPUT_DIR, "all_models_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("==================================================\n")
    f.write("CLASSICAL MODELS DETAILED EVALUATION REPORT\n")
    f.write("==================================================\n\n")
    f.write(f"BEST CLASSICAL MODEL: {results_df.iloc[0]['Model'].upper()} (F1: {results_df.iloc[0]['F1_Score']*100:.2f}%)\n\n")
    f.write("\n".join(report_blocks))

print("\n" + "="*50)
print(f" BEST CLASSICAL MODEL: {results_df.iloc[0]['Model'].upper()} (F1: {results_df.iloc[0]['F1_Score']*100:.2f}%)")
print(f" Outputs saved to '{OUTPUT_DIR}':")
print(f"  * {combined_cm_path}")
print(f"  * {report_path}")
print(f"  * {csv_path}")
print("="*50)
