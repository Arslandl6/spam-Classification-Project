import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re

# ---------------- Load & clean ----------------
df = pd.read_csv("data/email.csv", encoding='latin-1')
df.columns = ['label', 'text']

df = df.dropna()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.dropna()
df['label'] = df['label'].astype(int)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['text'] = df['text'].apply(clean_text)
df = df[df['text'].str.strip() != ""]
df = df.sample(300)

# ---------------- Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# ---------------- Tokenizer ----------------
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class SpamDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = SpamDataset(X_train, y_train)
test_dataset = SpamDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ---------------- Model ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ---------------- Training Loop ----------------
epochs = 1

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        print("Running batch...")
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader)}")

# ---------------- Evaluation ----------------
model.eval()
preds = []
true = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        predictions = torch.argmax(logits, dim=1)

        preds.extend(predictions.cpu().numpy())
        true.extend(labels.cpu().numpy())

print("\n🔹 DistilBERT Results")
print(classification_report(true, preds))
print("Confusion Matrix:\n", confusion_matrix(true, preds))