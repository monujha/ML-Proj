import os
import random
import os
import torch
from torch.utils.data import DataLoader
from data_loader import SiameseImageDataset
from model import SiameseNetwork
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, filepath="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch + 1}")

def load_checkpoint(filepath, model, optimizer):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded, starting from epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found at the specified path.")
        return 0

def retrieve_images_from_class(dataset, class_name, num_images=5):
    """Retrieve a specified number of images from a given class."""
    class_images = dataset.df[dataset.df['name'].str.contains(class_name)]['name'].tolist()
    # print(class_images)
    class_images = ["OfficeHomeDataset_10072016"+cl for cl in class_images]
    return class_images[:num_images]

def display_images(images, class_name):
    """Display a list of images with the class name."""
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img.permute(1, 2, 0))  # Change from (C, H, W) to (H, W, C)
        plt.title(class_name)
        plt.axis('off')
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    
    # Load the trained model (ensure the checkpoint path is correct)
    checkpoint_path = "checkpoint.pth"  # Change to your checkpoint path if needed
    load_checkpoint(checkpoint_path, model)

    # Prepare the dataset
    data_path = r"./OfficeHomeDataset_10072016/"
    csv_file = r"./OfficeHomeDataset_10072016/datanew.csv"
    dataset = SiameseImageDataset(data_path, csv_file)

    # Get unique class labels
    unique_classes = dataset.df['name'].apply(lambda x: x.split('/')[2]).unique()

    # Loop through each class and retrieve images
    for class_name in unique_classes:
        print(f"Retrieving images for class: {class_name}")
        images = retrieve_images_from_class(dataset, class_name, num_images=5)
        
        # Transform images to tensors
        tensor_images = []
        for img_path in images:
            print(f"Loading image: {img_path}")  # Debug print
            tensor_images.append(transforms.ToTensor()(Image.open(img_path).convert("RGB")))
        
        # Display images with their class label
        display_images(tensor_images, class_name)

if __name__ == "_main_":
    main()