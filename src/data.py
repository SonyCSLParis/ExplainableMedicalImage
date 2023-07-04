import os
import re
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

label_dict = {'Atelectasis':0, 'Cardiomegaly':1, 'Consolidation':2, 'Edema':3, 'EnlargedCardiomediastinum':4, 'Fracture':5,
              'LungLesion':6, 'LungOpacity':7, 'PleuralEffusion':8, 'Pneumonia':9, 'Pneumothorax':10, 'PleuralOther':11,
              'SupportDevices':12, 'NoFinding':13}
n_labels = len(label_dict.keys())

class MIMIC_CXR(Dataset):
    
    def __init__(self, data_path, image_path):
        self.data_path = data_path
        self.image_path = image_path
        
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(l) for l in open(data_path)]
        
    def __len__(self):
        return len(self.data)
    
    def process_image(self, image):
        transform = transforms.Compose([transforms.RandomResizedCrop(512, scale=(0.8, 1.1), ratio=(3/4, 4/3)),   
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return transform(image)
    
    def __getitem__(self, idx):
        
        sample = self.data[idx]
        
        # Preprocess image
        img_path = sample['img']
        image = Image.open(os.path.join(img_path)).convert("RGB")
        processed_image = self.process_image(image)
        
        # Preprocess report
        report = sample['text']
        # TODO Tokenize + encode
        
        # Preprocess label
        labels = re.sub("\'|\ ", "", re.sub('\"', '', sample['label'])).split(',')
        y = torch.zeros(n_labels)
        
        for i in range(len(labels)):
            y[label_dict[labels[i]]] = 1.0
        
            
        return image, report, y