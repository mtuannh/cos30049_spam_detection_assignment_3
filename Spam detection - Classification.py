import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
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

# USING CLASSIFICATION (NAIVE BAYES)
#train model
clf = MultinomialNB().fit(X_train_vec, Y_train)
Y_pred = clf.predict(X_test_vec)

# Evaluate
print("Predictions:", Y_pred.tolist()[:5])
print("Accuracy:", accuracy_score(Y_test, Y_pred))
# The propotion of how many True Spam which the model predicts
print("Precision: ", precision_score(Y_test,Y_pred))
# The propotion of how many True spam in reality
print("Recall: ", recall_score(Y_test,Y_pred))
# Average harmonic of precision and recall
print("F1: ", f1_score(Y_test,Y_pred))