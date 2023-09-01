import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext 
from torch.nn.utils.rnn import pad_sequence
from settings import *
from model_visual import *


def load_text_model(model_path):
    loaded_model = torch.load(model_path)
    return loaded_model


def load_test_dataset(jsonl_file):
    test_texts = []
    test_labels = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            test_texts.append(data['text'])
            test_labels.append(data['label'])
    return test_texts, test_labels

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(lstm_units * 2, vocab_size)  # Assuming bidirectional LSTM
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.linear(lstm_out)
        return self.softmax(output)

class ReportGenerator:
    def __init__(self, jsonl_file, lr):
        self.jsonl_file = jsonl_file
        self.word_index = {}
        self.max_length = 0
        self.model = None
        self.lr = lr
        
    def build_vocab(self, texts):
        vocab = set()
        for words in texts:
            vocab.update(words)
            
        vocab_list = []
        vocab_list.append('PAD')
        vocab_list.append('UNK')
        vocab_list = vocab_list + list(vocab)
        
        word_to_index = {word: index for index, word in enumerate(vocab_list)}
        
        return word_to_index


    def load_dataset(self):
        texts = []
        labels = []
        with open(self.jsonl_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                texts.append(data['text'])
                labels.append(data['label'])

        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        texts = [tokenizer(text) for text in texts]

        self.word_index = self.build_vocab(texts)

        sequences = [torch.tensor([self.word_index[word] for word in seq]) for seq in texts]
        self.max_length = max(len(seq) for seq in sequences)

        padded_sequences = pad_sequence(sequences, batch_first=True)

        return padded_sequences, labels

    def build_model(self, embedding_dim, lstm_units):
        self.model = TextGenerator(len(self.word_index)+1, embedding_dim, lstm_units)
        self.model.loss_fn = nn.CrossEntropyLoss()
        self.model.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_model(self, X, y, epochs, batch_size):
        print('Training text model...')
        self.model.train()
        y = self.one_hot_encode_labels(y)
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                self.model.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.model.loss_fn(predictions, batch_y.long())
                loss.backward()
                self.model.optimizer.step()

    def generate_report(self, data):
        self.model.eval()
        predicted_sequence = self.model(data.unsqueeze(0)).squeeze()
        predicted_labels = [self.index_to_word(idx) for idx in np.argmax(predicted_sequence.detach().numpy(), axis=1)]
        print(predicted_labels)
        return ' '.join(predicted_labels)

    def index_to_word(self, index):
        for word, idx in self.word_index.items():
            if idx == index:
                return word
        return None

    def one_hot_encode_labels(self, labels):
        encoded_labels = []
        for label in labels:
            encoded_label = torch.zeros(len(self.word_index) + 1)
            for word in label.split(','):
                word = word.strip("'")
                if word in self.word_index:
                    encoded_label[self.word_index[word]] = 1
            encoded_labels.append(encoded_label)
        return torch.stack(encoded_labels)


def train_report_generator(train_dataset, embedding_dim=100, lstm_units=128, epochs=10, batch_size=32, lr=0.1):
    report_generator = ReportGenerator(train_dataset, lr = lr)
    X, y = report_generator.load_dataset()
    report_generator.build_model(embedding_dim, lstm_units)
    report_generator.train_model(X, y, epochs, batch_size)
    return report_generator


def generate_reports_and_save(trained_report_generator, test_jsonl_file, output_jsonl_file):
    texts= []
    labels = []
    test_data = []
    with open(test_jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            test_data.append(data)
            texts.append(data['text'])
            labels.append(data['label'])
    
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    texts = [tokenizer(text) for text in texts]

    sequences = [torch.tensor([trained_report_generator.word_index[word] if word in trained_report_generator.word_index.keys() else trained_report_generator.word_index['UNK'] for word in seq]) for seq in texts]

    texts = pad_sequence(sequences, batch_first=True)

    with open(output_jsonl_file, 'w') as output_file:
        for i, data in enumerate(test_data):
            report = trained_report_generator.generate_report(texts[i])
            data['generated_report'] = report

            output_file.write(json.dumps(data) + '\n')
            print(f"Generated report for ID: {i}, Text: {texts[i]}")
            print(f"Generated Report: {report}")
            print("-------------------------------------------------")
