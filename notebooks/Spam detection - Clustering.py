import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, v_measure_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# read file csv using pandas
df = pd.DataFrame(pd.read_csv("datasets_process/Spam/spam_merged_clean.csv"))
df = df.dropna(subset=["spam", "text"]).copy()

# Df X
X = df["text"].astype(str).str.lower()

# Df and cleaning y
Y = df["spam"].astype(str).str.lower().str.strip().map({"1":1, "0":0})
# keep new df
df = df.loc[Y.notna()].copy()
Y = Y.loc[Y.notna()].astype(int)

# split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

#TF- IDF (Term Frequency - Inverse Document Frequency)
vec = TfidfVectorizer(stop_words="english")
X_train_vec = vec.fit_transform(X_train)
X_test_vec  = vec.transform(X_test)

# USING CLUSTERING (KMEANS)
# Create model
kmeans = KMeans(n_clusters = 2, random_state = 42)
# Train model
km = kmeans.fit(X_train_vec)
Y_pred_cluster = km.predict(X_test_vec)
# Output
# measures the “coherence” and “separation” of KMeans
print("Silhouette:", silhouette_score(X_test_vec, Y_pred_cluster))
# measure the similarity between cluster labels and the real labels (spam/ham)
print("V_measure:", v_measure_score(Y_test, Y_pred_cluster))