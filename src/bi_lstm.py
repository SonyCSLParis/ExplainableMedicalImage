# Importing necessary libraries and modules
import json
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Embedding
from settings import *  # Importing settings and model_visual from custom modules

# Function to load a saved text generation model
def load_text_model(model_path):
    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_model

# Function to load the test dataset from a JSONL file
def load_test_dataset(jsonl_file):
    test_texts = []
    test_labels = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            test_texts.append(data['text'])
            test_labels.append(data['label'])
    return test_texts, test_labels

# Class for generating reports from text
class ReportGenerator:
    def __init__(self, jsonl_file):
        self.jsonl_file = jsonl_file
        self.word_index = {}
        self.max_length = 0
        self.model = None

    # Function to load the dataset from a JSONL file and preprocess it
    def load_dataset(self):
        texts = []
        labels = []
        with open(self.jsonl_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                texts.append(data['text'])
                labels.append(data['label'])

        # Tokenize and pad the sequences
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(texts)
        self.word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(texts)
        self.max_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')

        return padded_sequences, labels

    # Function to build the text generation model
    def build_model(self, embedding_dim, lstm_units):
        self.model = Sequential()
        self.model.add(Embedding(len(self.word_index) + 1, embedding_dim, input_length=self.max_length))
        self.model.add(Bidirectional(LSTM(lstm_units)))
        self.model.add(Dense(len(self.word_index) + 1, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Function to train the text generation model
    def train_model(self, X, y, epochs, batch_size):
        y = self.one_hot_encode_labels(y)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    # Function to generate a report from input text
    def generate_report(self, text):
        sequence = self.text_to_sequence(text)
        padded_sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')
        predicted_sequence = self.model.predict(padded_sequence)[0]
        predicted_labels = [self.index_to_word(idx) for idx in np.argmax(predicted_sequence, axis=1)]
        return ' '.join(predicted_labels)

    # Function to convert text to a sequence of word indices
    def text_to_sequence(self, text):
        sequence = []
        for word in text.split():
            if word in self.word_index:
                sequence.append(self.word_index[word])
            else:
                sequence.append(0)  # 0 for unknown words
        return sequence

    # Function to convert word index back to word
    def index_to_word(self, index):
        for word, idx in self.word_index.items():
            if idx == index:
                return word
        return None

    # Function to one-hot encode the labels
    def one_hot_encode_labels(self, labels):
        encoded_labels = []
        for label in labels:
            encoded_label = np.zeros(len(self.word_index) + 1)
            for word in label.split(','):
                word = word.strip("'")
                if word in self.word_index:
                    encoded_label[self.word_index[word]] = 1
            encoded_labels.append(encoded_label)
        return np.array(encoded_labels)

# Function to train a report generator
def train_report_generator(train_dataset, embedding_dim=100, lstm_units=128, epochs=10, batch_size=32):
    report_generator = ReportGenerator(train_dataset)
    X, y = report_generator.load_dataset()
    report_generator.build_model(embedding_dim, lstm_units)
    report_generator.train_model(X, y, epochs, batch_size)
    return report_generator

# Function to generate reports and save them to an output JSONL file
def generate_reports_and_save(trained_report_generator, test_jsonl_file, output_jsonl_file):
    with open(test_jsonl_file, 'r') as file:
        test_data = [json.loads(line) for line in file]

    with open(output_jsonl_file, 'w') as output_file:
        for i, data in enumerate(test_data):
            text = data['text']
            report = trained_report_generator.generate_report(text)
            data['generated_report'] = report

            output_file.write(json.dumps(data) + '\n')
            print(f"Generated report for ID: {i}, Text: {text}")
            print(f"Generated Report: {report}")
            print("-------------------------------------------------")
