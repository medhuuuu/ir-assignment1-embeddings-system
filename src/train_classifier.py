import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# ── 1. Load your processed CSVs ──────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv("./data/train.csv")
test_df  = pd.read_csv("./data/test.csv")

train_texts  = train_df["text"].tolist()
test_texts   = test_df["text"].tolist()
train_labels = train_df["label"].tolist()
test_labels  = test_df["label"].tolist()

# ── 2. Generate Embeddings ────────────────────────────────────────────────────
# all-MiniLM-L6-v2 converts each review into a 384-dimension vector
# Similar meaning = vectors close together in space
print("Loading embedding model (downloads ~80MB first time)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings for train set... (takes 1-2 min)")
X_train = embedder.encode(train_texts, show_progress_bar=True)

print("Generating embeddings for test set...")
X_test  = embedder.encode(test_texts,  show_progress_bar=True)

# Save embeddings so you don't have to regenerate them
np.save("./data/X_train.npy", X_train)
np.save("./data/X_test.npy",  X_test)
print("Embeddings saved!")

# ── 3. Train Multiple Classifiers ────────────────────────────────────────────
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM":          LinearSVC(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100),
}

results = {}
best_acc = 0
best_name = ""
best_model = None

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)
    acc = accuracy_score(test_labels, preds)
    results[name] = acc
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(test_labels, preds, target_names=["negative","positive"]))

    if acc > best_acc:
        best_acc  = acc
        best_name = name
        best_model = clf

# ── 4. Save the Best Model ───────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
with open("model/classifier.pkl", "wb") as f:
    pickle.dump(best_model, f)
print(f"\n✅ Best model: {best_name} ({best_acc:.4f}) saved to model/classifier.pkl")

# ── 5. Print Summary Table ───────────────────────────────────────────────────
print("\n─── Results Summary ───────────────────────────────")
for name, acc in results.items():
    marker = " ← BEST" if name == best_name else ""
    print(f"  {name:<25} Accuracy: {acc:.4f}{marker}")
print("────────────────────────────────────────────────────")
print("\nCopy this table into report!")


embedder.save("model/embedder")
print("✅ Embedder saved too!")