📧 Spam Email Classification Project

🔍 Overview

This project classifies emails into:

- Spam (1)
- Ham (0)

It uses both Machine Learning and Deep Learning (BERT) models.

---

⚙️ Technologies Used

- Python
- Pandas
- Scikit-learn
- PyTorch
- Transformers (DistilBERT)

---

📂 Dataset

- Dataset: Spam Email Dataset (Kaggle)
- File: "email.csv"
- Columns:
  - "label" → spam / ham
  - "text" → email content

---

🧠 Models Used

1. Logistic Regression

- Simple and fast model
- Good baseline performance

2. Naive Bayes

- Works well for text classification

3. DistilBERT (Deep Learning)

- Transformer-based model
- Gives high accuracy

---

📊 Results

Model| Accuracy
Logistic Regression| ~97%
Naive Bayes| ~96%
DistilBERT| ~95%

---

📈 Visualization

- Accuracy comparison graph created using matplotlib

---

🚀 How to Run

pip install -r requirements.txt
python test.py
python bert_train.py

---

📌 Features

- Data cleaning and preprocessing
- TF-IDF vectorization
- Model comparison
- Confusion matrix & classification report

---

👨‍💻 Author

Mohd Arsalan
