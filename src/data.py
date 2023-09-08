import os
import re
import json
import torch
import torchtext
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn.utils.rnn import pad_sequence

label_dict = {'Atelectasis':0, 'Cardiomegaly':1, 'Consolidation':2, 'Edema':3, 'EnlargedCardiomediastinum':4, 'Fracture':5,
              'LungLesion':6, 'LungOpacity':7, 'PleuralEffusion':8, 'Pneumonia':9, 'Pneumothorax':10, 'PleuralOther':11,
              'SupportDevices':12, 'NoFinding':13}
n_labels = len(label_dict.keys())

def collate_fn(batch):
    imgs = [item[0].unsqueeze(0) for item in batch]
    imgs = torch.cat(imgs, dim=0)
    
    reports = [item[1] for item in batch]
    reports = pad_sequence(reports, batch_first=True, padding_value=0)
    
    targets = [item[2].unsqueeze(0) for item in batch]
    targets = torch.cat(targets, dim=0)    

    return imgs, reports, targets
     

class MIMIC_CXR(Dataset):
    
    def __init__(self, data_path, image_path, train_flag=False, word_idx=None):
        self.data_path = data_path
        self.image_path = image_path
        self.train_flag = train_flag
        
        # Vocabulary is built from the training set, so if we are building the
        # validation or test set, we take word_idx in input
        if train_flag:
            self.word_idx = self.word_to_index()
            self.idx_word = self.index_to_word(self.word_idx)
        else:
            self.word_idx = word_idx
            self.idx_word = self.index_to_word(word_idx)
        
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(l) for l in open(data_path)]
        
    def __len__(self):
        return len(self.data)
    
    def word_to_index(self):
        texts = []
        with open(self.data_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                texts.append(data['text'])

        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        texts = [tokenizer(text) for text in texts]
        
        vocab = set()
        for words in texts:
            vocab.update(words)            
            
        vocab_list = []
        
        # Adding PAD, UNK, START, and END tokens
        vocab_list.append('PAD')
        vocab_list.append('UNK')
        vocab_list.append('START')
        vocab_list.append('END')
        vocab_list = vocab_list + list(vocab)

        word_to_index = {word: index for index, word in enumerate(vocab_list)}

        return word_to_index
    
    def index_to_word(self, vocab):
        idx_to_word = {}
        for k in vocab.keys():
            idx_to_word[vocab[k]] = k
            
        return idx_to_word
    
    def process_image(self, image):
        if self.train_flag:
            transform = transforms.Compose(
                [transforms.Resize((256,256)),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        return transform(image)
    
    def __getitem__(self, idx):
        
        sample = self.data[idx]
        
        # Preprocess image
        img_path = sample['img']
        image = Image.open(os.path.join(img_path)).convert("RGB")
        processed_image = self.process_image(image)
        
        # Preprocess report
        report = sample['text']
        
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        tokens = tokenizer(report)

        # Substitute words with corresponding indices in vocabulary, adding START and END tokens
        # at the beginning and end of the report.
        sequences = [self.word_idx['START']]
        for word in tokens:
            if word in self.word_idx.keys():
                sequences.append(self.word_idx[word])
            else:
                sequences.append(self.word_idx['UNK'])
        sequences.append(self.word_idx['END'])
        sequences = torch.tensor(sequences)
        
        # Preprocess label
        labels = re.sub("\'|\ ", "", re.sub('\"', '', sample['label'])).split(',')
        
        # Dealing with missing labels (DISCUSS WITH MARTINA)
        if labels == ['']:
            labels = ['NoFinding']
            
        y = torch.zeros(n_labels)

        for i in range(len(labels)):
            y[label_dict[labels[i]]] = 1.0
            
        return processed_image, sequences, y