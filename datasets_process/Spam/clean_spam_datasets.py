import re
import csv
import pandas as pd
import numbers
import numpy as np
from pathlib import Path
from pandas.api.types import is_list_like
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

#input files' paths
DATA_DIR = Path("datasets_process/Spam")

IN_EMAILS = DATA_DIR / "emails.csv"
IN_SPAM_DS = DATA_DIR / "spam_dataset.csv"
IN_SPAM2   = DATA_DIR / "spam2.csv"

#output files' paths          
OUT_EMAILS  = DATA_DIR / "emails_clean.csv"
OUT_SPAM_DS = DATA_DIR / "spam_dataset_clean.csv"
OUT_SPAM2   = DATA_DIR / "spam2_clean.csv"
OUT_ALL = DATA_DIR / "spam_merged_clean.csv"


STOPWORDS = ENGLISH_STOP_WORDS  #stopwords from scikit-learn

#cleaning regex patterns
URL_RE   = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
HTML_RE  = re.compile(r"<.*?>")
NONALPH  = re.compile(r"[^a-z0-9\s]")

def normalize_label(labels):
    # create a pandas Series without splitting strings into characters
    if isinstance(labels, pd.Series):
        s = labels.copy()
    elif is_list_like(labels) and not isinstance(labels, (str, bytes)):
        s = pd.Series(list(labels))
    else:
        s = pd.Series([labels])

    def _map(x):
        if pd.isna(x):
            return pd.NA
        if isinstance(x, bool):
            return 1 if x else 0
        if isinstance(x, numbers.Number):
            try:
                iv = int(x)
            except Exception:
                return pd.NA
            if iv == 1:
                return 1
            if iv == 0:
                return 0
            return pd.NA
        xs = str(x).strip().lower()

        # common spam/ham tokens
        spam_tokens = {
            "1", "1.0", "true", "t", "yes", "y",
            "spam", "junk", "phish", "phishing", "unsolicited", "junk-mail"
        }
        ham_tokens = {
            "0", "0.0", "false", "f", "no", "n",
            "ham", "legit", "legitimate", "not spam", "not_spam", "non-spam", "normal"
        }

        if xs in spam_tokens:
            return 1
        if xs in ham_tokens:
            return 0

        # fallback heuristics: look for words inside the label
        if "spam" in xs:
            return 1
        if "ham" in xs or "not spam" in xs or "legit" in xs:
            return 0

        return pd.NA

    return s.map(_map)

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)
    s = HTML_RE.sub(" ", s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = NONALPH.sub(" ", s)
    tokens = [t for t in s.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df.dropna(subset=["text", "spam"])
    df = df[df["text"].str.len() > 0]

    #content cleaning
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    
    return df[["text", "spam"]]

def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)

#process each dataset
def process_emails():
    df = pd.read_csv(
        IN_EMAILS,
        sep=",",
        engine="python",
        quotechar='"',
        escapechar="\\",
        encoding="utf-8",
        on_bad_lines="skip",
    )
    #map columns ignoring case and spaces
    colmap = {c.lower().strip(): c for c in df.columns}
    text_col = colmap.get("text") or colmap.get("message") or colmap.get("message_content")
    spam_col = colmap.get("spam") or colmap.get("is_spam") or colmap.get("label") or colmap.get("category")
    if text_col is None or spam_col is None:
        raise RuntimeError(f"[emails.csv] text/spam columns cannot be found: {df.columns.tolist()}")
    df = df[[text_col, spam_col]].rename(columns={text_col: "text", spam_col: "spam"})
    #remove leading "Subject:" if exists
    df["text"] = df["text"].str.replace(r"^Subject:\s*", "", case=False, regex=True)

    df["spam"] = normalize_label(df["spam"])
    df = df.dropna(subset=["spam"])
    df["spam"] = df["spam"].astype(int)
    
    #convert spam to 0/1
    s = df["spam"].astype(str).str.lower().str.strip()
    df.loc[s.isin(["spam", "ham"]), "spam"] = s.map({"spam": 1, "ham": 0})
    df = finalize(df)
    
    return df

def process_spam_dataset():
    df = pd.read_csv(
        IN_SPAM_DS,
        sep=",",
        engine="python",
        quotechar='"',
        escapechar="\\",
        encoding="utf-8",
        on_bad_lines="skip",
    )
    colmap = {c.lower().strip(): c for c in df.columns}
    text_col = colmap.get("message_content") or colmap.get("text") or colmap.get("message")
    spam_col = colmap.get("is_spam") or colmap.get("spam") or colmap.get("label") or colmap.get("category")
    if text_col is None or spam_col is None:
        raise RuntimeError(f"[spam_dataset.csv] text/spam columns cannot be found: {df.columns.tolist()}")
    df = df[[text_col, spam_col]].rename(columns={text_col: "text", spam_col: "spam"})
    df["spam"] = normalize_label(df["spam"])
    df = df.dropna(subset=["spam"])
    df["spam"] = df["spam"].astype(int)
    
    df = finalize(df)

    return df

def process_spam2():
    df = pd.read_csv(
        IN_SPAM2,
        sep=",",
        engine="python",
        quotechar='"',
        escapechar="\\",
        encoding="utf-8",
        on_bad_lines="skip",
    )
    colmap = {c.lower().strip(): c for c in df.columns}
    text_col = colmap.get("message") or colmap.get("text") or colmap.get("message_content")
    cat_col  = colmap.get("category")
    if text_col is None or cat_col is None:
        raise RuntimeError(f"[spam2.csv] message/category columns cannot be found: {df.columns.tolist()}")
    df = df[[text_col, cat_col]].rename(columns={text_col: "text", cat_col: "spam"})
    df["spam"] = normalize_label(df["spam"])
    df = df.dropna(subset=["spam"])
    df["spam"] = df["spam"].astype(int)
    
    df = finalize(df)
    
    return df

if __name__ == "__main__":
    dfs = []

    if not IN_EMAILS.exists():
        print(f"Cannot found {IN_EMAILS.resolve()}")
    else:
        dfs.append(process_emails())

    if not IN_SPAM_DS.exists():
        print(f"Cannot found {IN_SPAM_DS.resolve()}")
    else:
        dfs.append(process_spam_dataset())

    if not IN_SPAM2.exists():
        print(f"Cannot found {IN_SPAM2.resolve()}")
    else:
        dfs.append(process_spam2())

    if len(dfs) == 0:
        print("No input datasets found.")
    else:
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.drop_duplicates(subset=["text"]).reset_index(drop=True)
        save_csv(merged, OUT_ALL)
        print(f"{OUT_ALL} | rows={len(merged)}")
        print("All done.")
