**Spam Email Detection Project - README**
#Environment setup
#Use this command to create a new environment
conda create -n spam_detection python=3.12 -y
#This command is for activating the environment
conda activate spam_detection
#Install required libraries
conda install pandas numpy matplotlib scikit-learn scipy -y

**Model training**
Classification-Naive Bayes (Spam detection - Classification.py)
1. Load cleaned dataset.
2. Split data into training (80%) and testing (20%).
3. Convert text into TF-IDF.
4. Trains a Multinomial Naive Bayes classifier.
5. Evaluates performance using: accuracy, recall, precision, f1 score
Run:
python "Spam detection - Classification.py"
Expected sample output:
Predictions: [0, 1, 1, 0, 0]
Accuracy: 0.98
Precision: 0.97
Recall: 0.95
F1: 0.96

Clustering-KMeans (Spam detection - Clustering.py)
1. Load cleaned dataset.
2. Split data into training (80%) and testing (20%).
3. Convert text into TF-IDF vectors.
4. Train KMeans model.
5. Evaluates using: silhouette score, v measure
Run:
python "Spam detection - Clustering.py"
Expected sample output:
Silhouette: 0.43
V_measure: 0.78
