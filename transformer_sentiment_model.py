import torch
from transformers import AutoTokenizer, \
    DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from data_loader import SentimentTreeBank

# Constants
MODEL_NAME = "distilroberta-base"
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0
N_EPOCHS = 2
SEQ_LEN = 128  # Max sequence length for tokenization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset
dataset = SentimentTreeBank(path="stanfordSentimentTreebank", split_words=True)

# Split Data
train_data = dataset.get_train_set()  # Full sentences for training
val_data = dataset.get_validation_set()  # Validation set

# Extract Sentences and Labels
train_sentences = [" ".join(sent.text) for sent in train_data]
train_labels = [int(sent.sentiment_class) for sent in train_data]
val_sentences = [" ".join(sent.text) for sent in val_data]
val_labels = [int(sent.sentiment_class) for sent in val_data]


# Custom Dataset for Transformer
class TransformerDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# Initialize Tokenizer and Dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = TransformerDataset(train_sentences, train_labels, tokenizer,
                                   max_length=SEQ_LEN)
val_dataset = TransformerDataset(val_sentences, val_labels, tokenizer,
                                 max_length=SEQ_LEN)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize Transformer Model
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME,
                                                            num_labels=2)
model.to(DEVICE)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY)


# Training Function
def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()

    accuracy = correct / len(data_loader.dataset)
    return total_loss / len(data_loader), accuracy


# Evaluation Function
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()

    accuracy = correct / len(data_loader.dataset)
    return total_loss / len(data_loader), accuracy


# Training Loop
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                        criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    print(f"Epoch {epoch + 1}/{N_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# Save the Model
model.save_pretrained("distilroberta_sentiment_model")
tokenizer.save_pretrained("distilroberta_sentiment_model")
