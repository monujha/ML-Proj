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
from sklearn.model_selection import train_test_split

datapath = r'./OfficeHomeDataset_10072016/'
csv_file = os.path.join(datapath, 'datanew.csv')


df = pd.read_csv(csv_file)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


train_csv = os.path.join(datapath, 'train_pairs.csv')
test_csv = os.path.join(datapath, 'test_pairs.csv')
train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)
from sklearn.model_selection import train_test_split

datapath = r'./OfficeHomeDataset_10072016/'
csv_file = os.path.join(datapath, 'datanew.csv')


df = pd.read_csv(csv_file)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


train_csv = os.path.join(datapath, 'train_pairs.csv')
test_csv = os.path.join(datapath, 'test_pairs.csv')
train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)



class SiameseImageDataset(Dataset):
    def __init__(self, datapath, csv_file):
        super(SiameseImageDataset, self).__init__()
        self.datapath = datapath
        self.df = pd.read_csv(csv_file, index_col=None, header=0)
        
        self.label_to_images = {}
        for _, row in self.df.iterrows():
            label = row['name'].split('/')[2]
            if label not in self.label_to_images:
                self.label_to_images[label] = []
            self.label_to_images[label].append(row['name'])
        
        self.train_transform = t.Compose([
            t.Resize([224, 224]),
            t.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img1_path = self.df['name'][index]
        label1 = img1_path.split('/')[2]

        is_positive_pair = random.choice([True, False])

        if is_positive_pair:
            img2_path = random.choice(self.label_to_images[label1])
            label = 1
        else:
            label2 = random.choice(list(self.label_to_images.keys()))
            while label2 == label1:
                label2 = random.choice(list(self.label_to_images.keys()))
            img2_path = random.choice(self.label_to_images[label2])
            label = 0

        img1 = Image.open(os.path.join(self.datapath, img1_path)).convert('RGB')
        img2 = Image.open(os.path.join(self.datapath, img2_path)).convert('RGB')
        img1 = self.train_transform(img1)
        img2 = self.train_transform(img2)

        return (img1, img2), torch.tensor(label, dtype=torch.float32)



def main():
    data = SiameseImageDataset("./OfficeHomeDataset_10072016/", "./OfficeHomeDataset_10072016/datanew.csv")
    dataloader = DataLoader(data, batch_size=32, shuffle=True, num_workers=0)
    print(len(dataloader))
    c = 0
    for (img1, img2), lbl in dataloader:
        if (c>10):
            break
        print (img1.shape, img2.shape, lbl.shape)
        c = c + 1
        
        
if __name__=="__main__":
    main()