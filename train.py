import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader import SiameseImageDataset
from model import *
from utils import *

def train_siamese(model, dataloader, criterion, optimizer, device, epochs=10, checkpoint_path="checkpoint.pth"):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        step = 0
        for (img1, img2), labels in dataloader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 20 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step:{step}, Loss: {loss.item():.4f}")
            step += 1
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    model.eval()

def retrieve_similar_images(model, dataset, device, top_n=5):
    model.eval()
    all_retrieved_images = []

    for label in dataset.label_to_images.keys():
        image_paths = dataset.get_images_by_label(label, count=top_n)
        for img_path in image_paths:
            img = Image.open(os.path.join(dataset.datapath, img_path)).convert('RGB')
            img_tensor = dataset.train_transform(img).unsqueeze(0).to(device)
            embedding = get_embedding(model, img_tensor, device)
            all_retrieved_images.append((embedding, img_path, label))

    return all_retrieved_images

def main(load_checkpoint_flag=False, checkpoint_path="checkpoint.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data = SiameseImageDataset(
        r'C:/Users/monuk/Downloads/OfficeHomeDataset_10072016/project/OfficeHomeDataset_10072016',
        r'C:/Users/monuk/Downloads/OfficeHomeDataset_10072016/project/OfficeHomeDataset_10072016/datanew.csv'
    )

    dataloader = DataLoader(data, batch_size=32, shuffle=True, num_workers=0)

    start_epoch = 0
    if load_checkpoint_flag:
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

    train_siamese(model, dataloader, criterion, optimizer, device, epochs=10, checkpoint_path=checkpoint_path)

    # Inference: Retrieve similar images from each class
    retrieved_images = retrieve_similar_images(model, data, device, top_n=5)

    # Display the retrieved images and their labels
    for embedding, img_path, label in retrieved_images:
        print(f"Label: {label}, Image Path: {img_path}")

if __name__ == "__main__":
    main(load_checkpoint_flag=False)
