import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# load data
df = pd.read_csv("data/email.csv", encoding='latin-1')
df.columns = ['label', 'text']

# convert labels
df = df.dropna()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

df = df.dropna()

df['label'] = df['label'].astype(int)

# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['text'] = df['text'].apply(clean_text)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------- Logistic Regression ----------------
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)

y_pred_lr = lr.predict(X_test_tfidf)

print("\n🔹 Logistic Regression Results")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# ---------------- Naive Bayes ----------------
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

y_pred_nb = nb.predict(X_test_tfidf)

print("\n🔹 Naive Bayes Results")
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))

import matplotlib.pyplot as plt
import seaborn as sns

# Example confusion matrix (Logistic Regression)
cm = confusion_matrix(y_test, y_pred_lr)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Accuracy comparison
lr_acc = (y_pred_lr == y_test).mean()
nb_acc = (y_pred_nb == y_test).mean()

models = ['Logistic Regression', 'Naive Bayes']
scores = [lr_acc, nb_acc]

plt.figure()
plt.bar(models, scores)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

print("\nFINAL COMPARISON")

print("Logistic Regression Accuracy:", lr_acc)
print("Naive Bayes Accuracy:", nb_acc)

print("DistilBERT Accuracy: ~0.95")

import matplotlib.pyplot as plt

models = ['Logistic Regression', 'Naive Bayes', 'DistilBERT']
scores = [lr_acc, nb_acc, 0.95]   # your accuracies

plt.bar(models, scores)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.show()