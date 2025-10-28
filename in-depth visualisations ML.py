import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


# Config / Paths
DATA_PATH = Path("datasets_process") / "Spam" / "spam_merged_clean.csv"
RANDOM_STATE = 42
FIGS = {
    "cm": "fig_cm_confusion.png",
    "roc": "fig_roc.png",
    "pr": "fig_pr.png",
    "nb_top": "fig_nb_top_tokens.png",
    "lc": "fig_learning_curve.png",
    "cal": "fig_calibration_curve.png",
    "elbow": "fig_kmeans_elbow.png",
    "sil": "fig_kmeans_silhouette.png",
    "pca_true": "fig_pca_true_labels.png",
    "pca_km": "fig_pca_kmeans_clusters.png",
    "km_terms": "fig_kmeans_top_terms.png",
}

# Load data
def load_data(path: Path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["spam", "text"]).copy()
    X = df["text"].astype(str).str.lower()
    y = (
        df["spam"]
        .astype(str).str.lower().str.strip()
        .map({"1": 1, "0": 0})
    )
    mask = y.notna()
    return X.loc[mask], y.loc[mask].astype(int)

# Train / split utilities
def split_and_vectorize(X, y):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    vec = TfidfVectorizer(stop_words="english")
    Xtr_vec = vec.fit_transform(Xtr)
    Xte_vec = vec.transform(Xte)
    return vec, Xtr_vec, Xte_vec, ytr, yte, Xtr, Xte

# Classification visuals
def plot_confusion_matrix(y_true, y_pred, out):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Ham (0)", "Spam (1)"])
    disp.plot(values_format="d")
    plt.title("Confusion Matrix – Multinomial Naive Bayes")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def plot_pr(y_true, y_score, out):
    ps, rs, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(rs, ps, lw=2, label=f"PR curve (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve – Multinomial Naive Bayes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def plot_learning_curve_pipe(X, y, out):
    # Pipeline for repeated vectorization during CV
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB())
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, cv=cv, n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5), scoring="f1"
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, marker="o", label="Train F1")
    plt.plot(train_sizes, val_mean, marker="o", label="Validation F1")
    plt.xlabel("Training set size")
    plt.ylabel("F1 score")
    plt.title("Learning Curve – Multinomial Naive Bayes (TF-IDF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def plot_calibration(y_true, y_prob, out, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--", lw=1, label="Perfectly calibrated")
    plt.plot(mean_pred, frac_pos, marker="o", label="Naive Bayes")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve – Multinomial Naive Bayes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

# Clustering visuals
def plot_kmeans_elbow(X_vec, out, k_min=2, k_max=10):
    inertias = []
    ks = list(range(k_min, k_max+1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        km.fit(X_vec)
        inertias.append(km.inertia_)
    plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Inertia (within-cluster SSE)")
    plt.title("KMeans Elbow Plot (TF-IDF)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def plot_kmeans_silhouette(X_vec, out, k=2, sample_cap=5000):
    # For large corpora, subsample to speed up silhouette_samples
    if X_vec.shape[0] > sample_cap:
        # simple uniform subsample of rows
        idx = np.random.RandomState(RANDOM_STATE).choice(X_vec.shape[0], sample_cap, replace=False)
        X_use = X_vec[idx]
    else:
        X_use = X_vec

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(X_use)
    sil_avg = silhouette_score(X_use, labels)
    sil_vals = silhouette_samples(X_use, labels)

    y_lower = 10
    plt.figure(figsize=(8, 6))
    for i in range(k):
        ith_cluster_sil_vals = sil_vals[labels == i]
        ith_cluster_sil_vals.sort()
        size_i = ith_cluster_sil_vals.shape[0]
        y_upper = y_lower + size_i
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, ith_cluster_sil_vals
        )
        plt.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10

    plt.axvline(x=sil_avg, color="k", linestyle="--")
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.title(f"Silhouette Plot (k={k}) – avg={sil_avg:.3f}")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    
# Main
def main():
    print(f"Loading: {DATA_PATH}")
    X, y = load_data(DATA_PATH)

    # Train/test split & vectorize
    vec, Xtr_vec, Xte_vec, ytr, yte, Xtr_raw, Xte_raw = split_and_vectorize(X, y)

    # ---- Naive Bayes Classification ----
    clf = MultinomialNB().fit(Xtr_vec, ytr)
    y_pred = clf.predict(Xte_vec)
    y_prob = clf.predict_proba(Xte_vec)[:, 1]

    plot_confusion_matrix(yte, y_pred, FIGS["cm"])
    plot_pr(yte, y_prob, FIGS["pr"])
    plot_learning_curve_pipe(X, y, FIGS["lc"])
    plot_calibration(yte, y_prob, FIGS["cal"])

    # ---- KMeans Clustering ----
    plot_kmeans_elbow(Xtr_vec, FIGS["elbow"], k_min=2, k_max=9)
    plot_kmeans_silhouette(Xtr_vec, FIGS["sil"], k=2, sample_cap=5000)

    print("\nSaved figures:")
    for k, v in FIGS.items():
        print("-", v)

if __name__ == "__main__":
    main()
