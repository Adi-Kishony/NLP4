def transformer_sentiment_analysis():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader
    import evaluate
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from data_loader import SentimentTreeBank

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset for loading data for transformer models
        """
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    def train_epoch(model, data_loader, optimizer, dev):
        """
        Perform an epoch of training of the model with the optimizer
        """
        model.train()
        total_loss = 0.0

        for batch in tqdm(data_loader, desc="Training"):
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    def evaluate_model(model, data_loader, dev, metric=None):
        """
        Evaluate the model and compute accuracy and loss
        """
        model.eval()
        total_loss = 0
        total_samples = 0

        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
                loss = outputs.loss  # Get the loss from the model
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                # Accumulate the loss
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Add predictions and references to metric
                if metric is not None:
                    metric.add_batch(predictions=predictions.cpu().numpy(),
                                     references=labels.cpu().numpy())

        # Compute the average loss
        avg_loss = total_loss / total_samples

        # Compute other metrics if provided
        other_metrics = metric.compute() if metric is not None else None

        return {"loss": avg_loss, "metrics": other_metrics}

    # Load the dataset
    dataset = SentimentTreeBank(path="stanfordSentimentTreebank", split_words=True)

    # Split data into train and validation sets
    train_data = dataset.get_train_set()
    val_data = dataset.get_validation_set()

    # Prepare sentences and labels
    train_sentences = [" ".join(sent.text) for sent in train_data]
    train_labels = [int(sent.sentiment_class) for sent in train_data]
    val_sentences = [" ".join(sent.text) for sent in val_data]
    val_labels = [int(sent.sentiment_class) for sent in val_data]

    # Parameters
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = 2  # Binary classification
    epochs = 2
    batch_size = 64
    learning_rate = 1e-5

    # Model, tokenizer, and metric
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=num_labels).to(dev)
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    metric = evaluate.load("accuracy")

    # Tokenization
    train_encodings = tokenizer(train_sentences, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_sentences, truncation=True, padding=True, max_length=128)

    # Datasets and DataLoaders
    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Store metrics
    train_losses = []
    val_accuracies = []
    train_accuracies = []
    val_losses = []


    # Training and evaluation loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, dev)
        val_loss = evaluate_model(model, val_loader, dev, metric)['loss']
        val_accuracy = evaluate_model(model, val_loader, dev, metric)['metrics']['accuracy']
        train_accuracy = evaluate_model(model, train_loader, dev, metric)['metrics']['accuracy']
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        print(f"Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Plot training loss and validation accuracy
    epochs_range = list(range(1, epochs + 1))
    plt.figure(figsize=(10, 5))

    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Transformer Training Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'loss_transformer.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Transformer Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f'acc_transformer.png')
    plt.show()


if __name__ == '__main__':
    print("transformer new")
    transformer_sentiment_analysis()

