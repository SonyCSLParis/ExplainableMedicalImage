import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

class ResNet50(nn.Module):
    def __init__(self, hid_dim, out_dim, dropout):
        super(ResNet50, self).__init__()
        
        resnet = torchvision.models.resnet50(pretrained=True)
        self.model_wo_fc = nn.Sequential(*(list(resnet.children())[:-1]))
                
        for param in self.model_wo_fc.parameters():
            param.requires_grad = True
        
        self.fc = nn.Sequential(nn.Linear(2048, hid_dim),
                                      nn.ReLU(),
                                      nn.Dropout(dropout))
        self.out_fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # Getting ResNet50 features
        features = self.model_wo_fc(x)
        features = torch.flatten(features, 1)
        
        # Reshaping to 1024
        features = self.fc(features)
        
        # Getting output probabilities
        out = self.out_fc(features)
        
        return features, out

def find_most_salient_patch(saliency_map, patch_size=64, stride=1):

    pooled_saliency = F.avg_pool2d(saliency_map.unsqueeze(0), kernel_size=patch_size, stride=stride)
    max_value, indices = torch.max(pooled_saliency.view(-1), dim=0)
    index_1d = indices.item()

    height, width = saliency_map.shape
    ycord = index_1d // width
    xcord = index_1d % width

    top_left_x = xcord * stride
    top_left_y = ycord * stride

    return top_left_x, top_left_y
    
def extract_patch_features(image_model, images, labels, patch_size, feat_size):
    # Get batch predictions
    features_image, logits = image_model(images.to(device))
    preds = torch.round(torch.sigmoid(logits))
    
    batch_features = torch.zeros((images.shape[0], feat_size))

    # For each image in the batch, compute the sum of the features of relevant regions for correctly predicted classes
    for i, image in enumerate(images):
        label = labels[i]
        pred = preds[i]
        
        # Get explanations just for correctly predicted classes
        features = torch.zeros((1, feat_size))
        for j, label_j in enumerate(label):
            label_j = label_j.item()
            pred_j = pred[j].item()
            
            if label_j == pred_j and label_j == 1.:
                # Get the model's predictions
                _, logits = image_model(image.unsqueeze(0).to(device))
                
                # Get saliency
                cam = cam_extractor(j, logits)[0]
                resize = torchvision.transforms.Resize((image.shape[1], image.shape[2]), interpolation=PIL.Image.BICUBIC)
                mask = resize(cam).squeeze(0)
    
                # Extract most activated patch_size x patch_size patch in the original image
                patch = image.numpy().transpose(1, 2, 0)
                topx, topy = find_most_salient_patch(mask)
                patch = patch[y:y+patch_size, x:x+patch_size]
                patch = patch / np.max(patch)
                
                # Extract features from j_th patch
                features_j, _ = image_model(torch.tensor(patch.transpose(2,0,1)).unsqueeze(0).to(device))
                
                # Sum features extracted from single patches
                features = features + features_j

        batch_features[i] = features
        return torch.cat((batch_features, features_image), dim=1) # shape [batch_size, 2048]
