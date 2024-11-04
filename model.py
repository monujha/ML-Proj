import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as t
from tqdm import tqdm
<<<<<<< HEAD
import torch
from torchvision.models import ResNet50_Weights  # Add this import
=======
from sklearn.model_selection import train_test_split
>>>>>>> 800e308c45a1a6dd8c00c5af101cebc4cd17da39

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50FeatureExtractor, self).__init__()
<<<<<<< HEAD
        # Load the pretrained ResNet-50 model, removing the last fully connected layer
        if pretrained:
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # Update here
        else:
            self.model = models.resnet50(weights=None)
=======
        self.model = models.resnet50(pretrained=pretrained)
>>>>>>> 800e308c45a1a6dd8c00c5af101cebc4cd17da39
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x

<<<<<<< HEAD



=======
>>>>>>> 800e308c45a1a6dd8c00c5af101cebc4cd17da39
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = ResNet50FeatureExtractor(pretrained=True)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

    def forward(self, img1, img2):
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
