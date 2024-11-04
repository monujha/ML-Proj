import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, fbeta_score
from data_loader import SiameseImageDataset
from model import *
from utils import *
import matplotlib.pyplot as plt

def calculate_f1_score(model, dataloader, device, threshold=0.5, beta=1.0):
    """Calculate F1/F2 score for the model."""
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for (img1, img2), labels in dataloader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # Get model outputs and calculate distances
            output1, output2 = model(img1, img2)
            distances = F.pairwise_distance(output1, output2)
            
            # Predict labels based on threshold
            predictions = (distances < threshold).float()
            
            # Store predictions and labels for F1 score calculation
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate F-beta score using all predictions and labels
    f_score = fbeta_score(all_labels, all_predictions, beta=beta)
    return f_score

def train_siamese(model, dataloader, criterion, optimizer, device, epochs=10, checkpoint_path="checkpoint.pth"):
    train_losses = []
    f1_scores = []

    for epoch in range(epochs):
        model.train()
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
        
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}")

        # Calculate F1/F2 score for the epoch
        f1 = calculate_f1_score(model, dataloader, device)
        f1_scores.append(f1)
        print(f"Epoch [{epoch + 1}/{epochs}], F1 Score: {f1:.4f}")

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    # Plot and save training loss and F1 score
    plot_metrics(train_losses, f1_scores)

def plot_metrics(train_losses, f1_scores):
    """Plot and save train losses and F1 scores over epochs."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()

    # Plot F1 score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, f1_scores, 'r-', label='F1 Score')
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("F1 Score Over Epochs")
    plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

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
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    print("Model, criterion, and optimizer set")

    data = SiameseImageDataset(
        r"./OfficeHomeDataset_10072016/",
        r"./OfficeHomeDataset_10072016/datanew.csv"
    )

    print("Data loaded...")
    dataloader = DataLoader(data, batch_size=64, shuffle=True, num_workers=2)

    # Load checkpoint if flag is set
    start_epoch = 0
    if load_checkpoint_flag:
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

    # Train model
    train_siamese(model, dataloader, criterion, optimizer, device, epochs=10, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    main(load_checkpoint_flag=False)  # Set to False if you don't want to load the checkpoint
