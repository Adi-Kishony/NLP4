import torch
import torch.nn as nn
import numpy as np
import os

import tqdm
from torch.utils.data import DataLoader, Dataset
import data_loader
import pickle
import matplotlib.pyplot as plt
from data_loader import get_negated_polarity_examples, get_rare_words_examples


# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------
device = get_available_device()
print("Using device: ", device)

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    embeddings = np.zeros((embedding_dim, len(sent.text)))
    for i, word in enumerate(sent.text):
        if word in word_to_vec:
            embeddings[:, i] = word_to_vec[word]

    return np.mean(embeddings, axis=1)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size)
    one_hot[ind] = 1
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    indices = [word_to_ind[word] for word in sent.text if word in word_to_ind]
    if not indices:
        return np.zeros(len(word_to_ind))
    one_hots = [get_one_hot(len(word_to_ind), idx) for idx in indices]
    return np.mean(one_hots, axis=0)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    word_ind_dict = {}
    i = 0
    for word in words_list:
        if word in word_ind_dict:
            continue
        else:
            word_ind_dict[word] = i
            i += 1
    return word_ind_dict


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    embeddings = [word_to_vec[word] if word in word_to_vec else np.zeros(embedding_dim) for word in sent.text]
    if len(embeddings) < seq_len:
        embeddings.extend([np.zeros(embedding_dim)] * (seq_len - len(embeddings)))
    return np.array(embeddings[:seq_len])


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape

    def get_dataset(self):
        return self.sentiment_dataset

    def get_sentences(self, data_subset=TRAIN):
        return self.sentences[data_subset]



# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        _, (hidden, _) = self.lstm(text)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

    def predict(self, text):
        logits = self.forward(text)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        return pred.astype(np.int32)


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        logit = self.forward(x)
        return torch.sigmoid(logit)


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """

    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """

    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for x_batch, y_batch in data_iterator:
        optimizer.zero_grad()
        x_batch = x_batch.float()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        predictions = model(x_batch).squeeze(1)
        loss = criterion(predictions, y_batch)
        acc = binary_accuracy(torch.sigmoid(predictions), y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(data_iterator), epoch_acc / len(data_iterator)


def evaluate(model, data_iterator, criterion, data_manager=None):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []
    all_labels = []

    if data_manager:
        dataset = data_manager.get_dataset()
        sentences = data_manager.get_sentences(TEST)
        rare_words_examples_indices = get_rare_words_examples(sentences, dataset)
        negated_polarity_examples_indices = get_negated_polarity_examples(sentences)

    with torch.no_grad():
        for x_batch, y_batch in data_iterator:
            x_batch = x_batch.float()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(x_batch).squeeze(1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            loss = criterion(predictions, y_batch)
            acc = binary_accuracy(torch.sigmoid(predictions), y_batch)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    if data_manager:
        pred_rare = [all_predictions[i] for i in rare_words_examples_indices]
        pred_negated = [all_predictions[i] for i in
                        negated_polarity_examples_indices]
        true_rare = [all_labels[i] for i in rare_words_examples_indices]
        true_negated = [all_labels[i] for i in
                        negated_polarity_examples_indices]
        acc_rare = binary_accuracy(torch.sigmoid(torch.tensor(pred_rare)), torch.tensor(true_rare))
        acc_negated = binary_accuracy(torch.sigmoid( torch.tensor(pred_negated)), torch.tensor(true_negated))
        return epoch_loss / len(data_iterator), epoch_acc / len(data_iterator), acc_rare.item(), acc_negated.item()
    return epoch_loss / len(data_iterator), epoch_acc / len(data_iterator), None, None


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for x_batch, _ in data_iter:
            batch_predictions = model.predict(x_batch).squeeze(1)
            predictions.extend(batch_predictions.cpu().numpy())

    return predictions


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_iterator = data_manager.get_torch_iterator(TRAIN)
    val_iterator = data_manager.get_torch_iterator(VAL)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm.tqdm(range(n_epochs)):
        train_loss, train_acc = train_epoch(model, train_iterator, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, val_iterator, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    n_epochs = 20
    data_manager = DataManager(data_type=ONEHOT_AVERAGE, batch_size=64)
    input_dim = data_manager.get_input_shape()[0]
    model = LogLinear(input_dim)
    model.to(device)
    history = train_model(model, data_manager, n_epochs=n_epochs, lr=0.01, weight_decay=0.001)

    plot_acc_loss(model, history, data_manager, "log_lin_one_hot", n_epochs)



def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    n_epochs = 20
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=64, embedding_dim=W2V_EMBEDDING_DIM)
    input_dim = data_manager.get_input_shape()[0]
    model = LogLinear(input_dim)
    model.to(device)
    history = train_model(model, data_manager, n_epochs=n_epochs, lr=0.01, weight_decay=0.001)

    plot_acc_loss(model, history, data_manager, "log linear w2v", n_epochs)

def plot_acc_loss(model, history, data_manager, title, n_epochs=20):
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epochs + 1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, n_epochs + 1), history['val_loss'], label='Validation Loss')
    plt.title(title + " Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'loss_{title}.png')
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epochs + 1), history['train_acc'], label='Train Accuracy')
    plt.plot(range(1, n_epochs + 1), history['val_acc'], label='Validation Accuracy')
    plt.title(title + " Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f'accuracy_{title}.png')
    plt.show()

    test_iterator = data_manager.get_torch_iterator("test")
    test_loss, test_acc, rare_acc, negated_acc = evaluate(model, test_iterator, nn.BCEWithLogitsLoss(), data_manager)
    print(
        f"Rare words accuracy for model {title}: {rare_acc}")
    print(
        f"Negated polarity accuracy for model {title}: {negated_acc}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")



def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    n_epochs = 4
    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=64, embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(embedding_dim=W2V_EMBEDDING_DIM, hidden_dim=100, n_layers=1, dropout=0.5)
    model.to(device)
    history = train_model(model, data_manager, n_epochs=n_epochs, lr=0.001, weight_decay=0.0001)
    plot_acc_loss(model, history, data_manager, "lstm", n_epochs)


if __name__ == '__main__':
    print("running all models")
    train_log_linear_with_one_hot()
    train_log_linear_with_w2v()
    train_lstm_with_w2v()
