import os
import random
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as t
from tqdm import tqdm
import torch


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50FeatureExtractor, self).__init__()
        # Load the pretrained ResNet-50 model, removing the last fully connected layer
        self.model = models.resnet50(pretrained=pretrained)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # Flatten the output for embedding
        return x


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Use ResNet-50 as the feature extractor
        self.feature_extractor = ResNet50FeatureExtractor(pretrained=True)
        
        # Add a fully connected layer to create the final embedding
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # Embedding size
        )

    def forward_once(self, x):
        # Extract features and pass through FC layers for final embedding
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

    def forward(self, img1, img2):
        # Obtain embeddings for both images
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        return output1, output2



class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss


def get_embedding(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        embedding = model.forward_once(image.unsqueeze(0))
    return embedding


def main():
    model = SiameseNetwork()
    inp1 = torch.rand(32, 3, 224, 224)
    inp2 = torch.rand(32, 3, 224, 224)
    lbl = torch.rand(32)
    lbl = (lbl > 0.5).int()
    
    print (inp1.shape, inp2.shape, lbl.shape)
    
    out1, out2 = model(inp1, inp2)
    print(out1.shape, out2.shape)
    
    lossfn = ContrastiveLoss()
    loss = lossfn(out1, out2, lbl)
    
    print (loss)
    

if __name__=="__main__":
    main()
