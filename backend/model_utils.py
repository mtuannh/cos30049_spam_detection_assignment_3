from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import silhouette_score

# === Config ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "datasets_process" / "Spam" / "spam_merged_clean.csv"
RANDOM_STATE = 42

class SpamModel:
    """
    Reuses A2 pipeline: TF-IDF + MultinomialNB for classification,
    KMeans for quick unsupervised diagnostics.
    """
    def __init__(self):
        self.vec = TfidfVectorizer(stop_words="english")
        self.clf = MultinomialNB()
        self.count_vec = CountVectorizer(stop_words="english")
        self._cache: Dict[str, Any] = {}
        self._top_words: List = []
        self._label_ratio: Dict[str, int] = {}
        self._lengths: List[int] = []

    # ---------- Data ----------
    def load_data(self):
        df = pd.read_csv(DATA_PATH)
        # expect columns: spam (0/1) and text (cleaned)
        df = df.dropna(subset=["spam", "text"]).copy()
        X = df["text"].astype(str).str.lower()
        # robust mapping from string/number to 0/1
        y = (
            df["spam"]
            .apply(lambda v: int(str(v).strip() in ("1", "true", "spam")))
            .astype(int)
        )
        return X, y

    # ---------- Train & caches ----------
    def train(self):
        X, y = self.load_data()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        Xtr_vec = self.vec.fit_transform(Xtr)
        Xte_vec = self.vec.transform(Xte)
        self.clf.fit(Xtr_vec, ytr)

        y_pred = self.clf.predict(Xte_vec)
        metrics = {
            "accuracy": float(accuracy_score(yte, y_pred)),
            "precision": float(precision_score(yte, y_pred)),
            "recall": float(recall_score(yte, y_pred)),
            "f1": float(f1_score(yte, y_pred)),
            "n_train": int(Xtr_vec.shape[0]),
            "n_test": int(Xte_vec.shape[0]),
        }

        # caches for later endpoints
        self._cache.update({
            "X_train_vec": Xtr_vec, "y_train": ytr.reset_index(drop=True),
            "X_test_vec": Xte_vec, "y_test": yte.reset_index(drop=True),
            "X_all": X.reset_index(drop=True), "y_all": y.reset_index(drop=True)
        })

        # Top spam words (CountVectorizer on spam texts)
        spam_texts = X[y == 1]
        if len(spam_texts) > 0:
            Xc = self.count_vec.fit_transform(spam_texts)
            freqs = np.asarray(Xc.sum(axis=0)).ravel()
            vocab = self.count_vec.get_feature_names_out()
            idx = np.argsort(freqs)[::-1][:20]
            self._top_words = [(vocab[i], int(freqs[i])) for i in idx]
        else:
            self._top_words = []

        self._label_ratio = {"ham": int((y == 0).sum()), "spam": int((y == 1).sum())}
        self._lengths = pd.Series(X.astype(str).apply(len)).tolist()
        return metrics

    # ---------- Predict ----------
    def predict_one(self, text: str) -> Dict[str, Any]:
        Xv = self.vec.transform([text.lower()])
        proba = float(self.clf.predict_proba(Xv)[:, 1][0])
        label = int(proba >= 0.5)
        # simple explanation: top TF-IDF tokens in this input
        tfidf = Xv.toarray()[0]
        top_idx = np.argsort(tfidf)[::-1]
        feature_names = self.vec.get_feature_names_out()
        explanations = [feature_names[i] for i in top_idx if tfidf[i] > 0][:5]
        return {"label": label, "probability": proba, "top_tokens": explanations}

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict_one(t) for t in texts]

    # ---------- Metrics ----------
    def metrics(self) -> Dict[str, Any]:
        Xte_vec = self._cache["X_test_vec"]
        yte = self._cache["y_test"]
        y_pred = self.clf.predict(Xte_vec)
        return {
            "accuracy": float(accuracy_score(yte, y_pred)),
            "precision": float(precision_score(yte, y_pred)),
            "recall": float(recall_score(yte, y_pred)),
            "f1": float(f1_score(yte, y_pred)),
            "n_test": int(Xte_vec.shape[0]),
        }

    # ---------- Charts payload (core 3) ----------
    def charts_payload(self) -> Dict[str, Any]:
        return {
            "label_distribution": self._label_ratio,
            "message_length_hist": self._lengths[:5000],  # FE will bin
            "top_spam_words": self._top_words,
        }

    # ---------- Advanced: PR curve ----------
    def pr_curve(self) -> Dict[str, Any]:
        Xte_vec = self._cache["X_test_vec"]
        yte = self._cache["y_test"]
        probs = self.clf.predict_proba(Xte_vec)[:, 1]
        precision, recall, thresholds = precision_recall_curve(yte, probs)
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        }

    # ---------- Advanced: Calibration ----------
    def calibration(self, n_bins: int = 10) -> Dict[str, Any]:
        Xte_vec = self._cache["X_test_vec"]
        yte = self._cache["y_test"]
        probs = self.clf.predict_proba(Xte_vec)[:, 1]
        prob_true, prob_pred = calibration_curve(yte, probs, n_bins=n_bins, strategy="uniform")
        return {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "bins": int(n_bins),
        }

    # ---------- KMeans: elbow ----------
    def kmeans_elbow(self, sample_cap: int = 4000) -> Dict[str, Any]:
        X_all = self._cache["X_all"]
        n = min(len(X_all), sample_cap)
        Xs = X_all.sample(n, random_state=RANDOM_STATE)
        Xv = self.vec.transform(Xs)
        ks = list(range(2, 7))
        inertias = []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
            km.fit(Xv)
            inertias.append(float(km.inertia()))
        return {"k_list": ks, "inertias": inertias}

    # ---------- KMeans: scores (silhouette & v-measure for k=2) ----------
    def kmeans_scores(self, sample_cap: int = 4000) -> Dict[str, Any]:
        X_all = self._cache["X_all"]
        y_all = self._cache["y_all"]
        n = min(len(X_all), sample_cap)
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_all), n, replace=False)
        Xs = X_all.iloc[idx]
        ys = y_all.iloc[idx]
        Xv = self.vec.transform(Xs)

        km = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(Xv)

        sil = float(silhouette_score(Xv, labels))
        vms = float(v_measure_score(ys, labels))
        return {"silhouette": sil, "v_measure": vms}
