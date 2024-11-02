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

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    model.eval()

def retrieve_similar_images(model, query_image, dataset, device, top_n=5):
    query_embedding = get_embedding(model, query_image, device)
    distances = []
    
    for (img, _) in dataset:
        img = img.to(device)
        embedding = get_embedding(model, img, device)
        distance = F.pairwise_distance(query_embedding, embedding)
        distances.append((distance.item(), img))
    
    distances.sort(key=lambda x: x[0])
    return [img for _, img in distances[:top_n]]

def main(load_checkpoint_flag=False, checkpoint_path="checkpoint.pth"):
    print("DATA LOADER FILE STARTED...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Model, criterion, and optimizer set")

    data = SiameseImageDataset(
        r'C:/Users/monuk/Downloads/OfficeHomeDataset_10072016/project/OfficeHomeDataset_10072016',
        r'C:/Users/monuk/Downloads/OfficeHomeDataset_10072016/project/OfficeHomeDataset_10072016/datanew.csv'
    )

    print("Data loaded...")
    dataloader = DataLoader(data, batch_size=32, shuffle=True, num_workers=0)

    # Load checkpoint if flag is set
    start_epoch = 0
    if load_checkpoint_flag:
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

    # Train model
    train_siamese(model, dataloader, criterion, optimizer, device, epochs=10, checkpoint_path=checkpoint_path)

    # Retrieve similar images
    query_image, _ = data[0]  
    similar_images = retrieve_similar_images(model, query_image, data, device, top_n=5)
    print("Retrieved similar images:", similar_images)

if __name__ == "__main__":
    main(load_checkpoint_flag=False)  # Set to False if you don't want to load the checkpoint
