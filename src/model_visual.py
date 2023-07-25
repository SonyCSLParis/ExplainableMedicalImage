import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

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