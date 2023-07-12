import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding

class ReportGenerator:
    def __init__(self, jsonl_file):
        self.jsonl_file = jsonl_file
        self.word_index = {}
        self.max_length = 0
        self.model = None

    def load_dataset(self):
        texts = []
        labels = []
        with open(self.jsonl_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                texts.append(data['text'])
                labels.append(data['label'])

        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(texts)
        self.word_index = tokenizer.word_index

        sequences = tokenizer.texts_to_sequences(texts)
        self.max_length = max(len(seq) for seq in sequences)

        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')

        return padded_sequences, labels

    def build_model(self, embedding_dim, lstm_units):
        self.model = Sequential()
        self.model.add(Embedding(len(self.word_index) + 1, embedding_dim, input_length=self.max_length))
        self.model.add(Bidirectional(LSTM(lstm_units)))
        self.model.add(Dense(len(self.word_index) + 1, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def train_model(self, X, y, epochs, batch_size):
        y = self.one_hot_encode_labels(y)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def generate_report(self, text):
        sequence = self.text_to_sequence(text)
        padded_sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')
        predicted_sequence = self.model.predict(padded_sequence)[0]
        predicted_labels = [self.index_to_word(idx) for idx in np.argmax(predicted_sequence, axis=1)]
        return ' '.join(predicted_labels)

    def text_to_sequence(self, text):
        sequence = []
        for word in text.split():
            if word in self.word_index:
                sequence.append(self.word_index[word])
            else:
                sequence.append(0)  # 0 for unknown words
        return sequence

    def index_to_word(self, index):
        for word, idx in self.word_index.items():
            if idx == index:
                return word
        return None

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
