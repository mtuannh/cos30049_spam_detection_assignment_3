import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUTS = {
    "fig1": "fig1_message_length_distribution.png",
    "fig2": "fig2_spam_vs_ham_pie.png",
    "fig3": "fig3_top20_words_spam.png",
    "fig4": "fig4_boxplot_message_length_vs_spam.png",
    "fig5": "fig5_label_distribution.png",
}

CANDIDATE_FILES = [
    "spam_merged_clean.csv",
    "spam_merged.csv",
    "spam_merged_clean.csv",         
    "spam_dataset_merged.csv",
    "spam_dataset_clean.csv",
    "spam_dataset.csv",
    "spam2_clean.csv",
    "spam2.csv",
    "emails_clean.csv",
    "emails.csv",
]

# In case sklearn isn't available, we fallback to a simple tokenizer
def safe_import_sklearn():
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        return CountVectorizer
    except Exception:
        return None

CountVectorizer = safe_import_sklearn()


def find_existing_file():
    for f in CANDIDATE_FILES:
        if os.path.exists(f):
            return f
        # also try absolute path locations used earlier
        if os.path.exists(os.path.join("/mnt/data", f)):
            return os.path.join("/mnt/data", f)
        return os.path.join("datasets_process", "Spam", "spam_merged_clean.csv")

    raise FileNotFoundError(
        "Could not find any dataset. Put your merged/clean CSV next to this script "
        "or update CANDIDATE_FILES."
    )


def load_dataset():
    path = find_existing_file()
    df = pd.read_csv(path)
    # normalize columns
    df.columns = [c.lower().strip() for c in df.columns]

    # try to detect text & label columns
    text_candidates = [c for c in df.columns if any(k in c for k in ["text", "message", "content", "body"])]
    label_candidates = [c for c in df.columns if any(k in c for k in ["spam", "label", "category", "class", "target"])]

    if not text_candidates:
        raise ValueError("Could not detect text column. Include 'text' or 'message' in a column name.")
    if not label_candidates:
        raise ValueError("Could not detect label column. Include 'spam', 'label', or 'category' in a column name.")

    text_col = text_candidates[0]
    label_col = label_candidates[0]

    # standardize labels to {0: ham, 1: spam}
    # if the column is already numeric 0/1, this will preserve it
    mapping = {"ham": 0, "spam": 1, "0": 0, "1": 1}
    if df[label_col].dtype == object:
        df[label_col] = df[label_col].str.strip().str.lower().map(mapping).fillna(df[label_col])
    df[label_col] = df[label_col].astype(str).str.strip().str.lower().map(mapping)
    # some files may already be numeric 0/1—coerce again
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    # drop rows with missing text
    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str)

    return df, text_col, label_col, path


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)               # remove URLs
    s = re.sub(r"\d+", " ", s)                   # remove numbers
    s = s.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def top_words_spam(df, text_col, label_col, topk=20):
    spam_texts = df.loc[df[label_col] == 1, text_col].astype(str).tolist()
    if not spam_texts:
        return []

    if CountVectorizer:
        vec = CountVectorizer(stop_words="english", max_features=None)
        X = vec.fit_transform(spam_texts)
        vocab = vec.get_feature_names_out()
        freqs = np.asarray(X.sum(axis=0)).ravel()
        # sort by frequency desc and take topk
        idx = np.argsort(freqs)[::-1][:topk]
        return [(vocab[i], int(freqs[i])) for i in idx]
    else:
        # Fallback simple tokenizer (no stopword list)
        from collections import Counter
        tokens = []
        for t in spam_texts:
            tokens.extend(clean_text(t).split())
        freqs = Counter(tokens).most_common(topk)
        return freqs


def make_fig1_message_length_hist(df, text_col, out_path):
    lengths = df[text_col].astype(str).apply(len).values
    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(lengths, bins=50, alpha=0.9)
    # crude KDE-like overlay (optional visual touch without seaborn)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(lengths)
        xs = np.linspace(min(lengths), max(lengths), 200)
        ys = kde(xs)
        # scale KDE to histogram peak
        ys_scaled = ys * (n.max() / ys.max())
        plt.plot(xs, ys_scaled, linewidth=2)
    except Exception:
        pass
    plt.title("Figure 1: Message Length Distribution")
    plt.xlabel("Message Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_fig2_spam_ham_pie(df, label_col, out_path):
    counts = df[label_col].value_counts().to_dict()
    ham = counts.get(0, 0)
    spam = counts.get(1, 0)
    plt.figure(figsize=(6, 6))
    plt.pie([ham, spam], labels=["Ham", "Spam"], autopct="%1.1f%%", startangle=90)
    plt.title("Figure 2: Spam vs Ham Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_fig3_top_words_bar(df, text_col, label_col, out_path, topk=20):
    pairs = top_words_spam(df, text_col, label_col, topk=topk)
    if not pairs:
        # make an empty figure to keep pipeline consistent
        plt.figure(figsize=(8, 5))
        plt.title("Figure 3: Top 20 Frequent Words in Spam Messages (no spam found)")
        plt.axis("off")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    words, freqs = zip(*pairs)
    y_pos = np.arange(len(words))[::-1]  # plot top at top

    plt.figure(figsize=(10, 6))
    plt.barh(y_pos, freqs)
    plt.yticks(y_pos, words)
    plt.xlabel("Frequency")
    plt.title("Figure 3: Top 20 Frequent Words in Spam Messages")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_fig4_boxplot_length_vs_label(df, text_col, label_col, out_path):
    # Prepare data for matplotlib boxplot: group lengths by label
    lengths_ham = df.loc[df[label_col] == 0, text_col].astype(str).apply(len).values
    lengths_spam = df.loc[df[label_col] == 1, text_col].astype(str).apply(len).values

    plt.figure(figsize=(8, 5))
    plt.boxplot([lengths_ham, lengths_spam], labels=["Ham (0)", "Spam (1)"], showfliers=False)
    plt.title("Figure 4: Message Length vs Spam Category")
    plt.xlabel("Category")
    plt.ylabel("Message Length")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_fig5_label_distribution(df, label_col, out_path):
    counts = df[label_col].value_counts().sort_index()
    labels = [str(k) for k in counts.index]
    vals = counts.values

    plt.figure(figsize=(6, 5))
    plt.bar(labels, vals)
    plt.title("Figure 5: Label Distribution (Before Balancing)")
    plt.xlabel("Category (0 = Ham, 1 = Spam)")
    plt.ylabel("Count")
    for i, v in enumerate(vals):
        plt.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    df, text_col, label_col, path = load_dataset()

    # Standard feature for reuse
    df["message_length"] = df[text_col].astype(str).apply(len)

    print(f"Loaded: {path}")
    print(f"Detected text column:  {text_col}")
    print(f"Detected label column: {label_col}")
    print(f"Rows: {len(df)} | Spam ratio: {df[label_col].mean():.3f}")

    # Generate figures
    make_fig1_message_length_hist(df, text_col, OUTPUTS["fig1"])
    make_fig2_spam_ham_pie(df, label_col, OUTPUTS["fig2"])
    make_fig3_top_words_bar(df, text_col, label_col, OUTPUTS["fig3"], topk=20)
    make_fig4_boxplot_length_vs_label(df, text_col, label_col, OUTPUTS["fig4"])
    make_fig5_label_distribution(df, label_col, OUTPUTS["fig5"])

    print("\nSaved figures:")
    for k, v in OUTPUTS.items():
        print(f"- {v}")


if __name__ == "__main__":
    main()