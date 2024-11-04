import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader import SiameseImageDataset
from model import SiameseNetwork
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def load_checkpoint(filepath, model):
    """Load the model checkpoint."""
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {filepath}")
    else:
        print("Checkpoint file is not present")

def retrieve_images_from_class(dataset, class_name, num_images=5):
    """Retrieve a specified number of images from a given class."""
    class_images = dataset.df[dataset.df['name'].str.contains(class_name)]['name'].tolist()
    class_images = ["./OfficeHomeDataset_10072016"+cl for cl in class_images]
    return class_images[:num_images]

def calculate_average_embedding(model, images, device):
    """Calculate the average embedding for a list of images."""
    embeddings = []
    transform = transforms.ToTensor()
    with torch.no_grad():
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
            embedding = model.forward_once(img_tensor)
            embeddings.append(embedding)
    avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
    return avg_embedding

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

    # Calculate average embedding for each class
    class_embeddings = {}
    for class_name in unique_classes:
        print(f"Processing class: {class_name}")
        images = retrieve_images_from_class(dataset, class_name, num_images=5)
        avg_embedding = calculate_average_embedding(model, images, device).squeeze(0)
        class_embeddings[class_name] = avg_embedding

    # Process the query image
    query_image_path = input("Enter the path of the query image: ")
    query_image = Image.open(query_image_path).convert("RGB")
    query_tensor = transforms.ToTensor()(query_image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        query_embedding = model.forward_once(query_tensor)

    # Find the class with the closest embedding to the query image
    min_distance = float('inf')
    closest_class = None
    for class_name, avg_embedding in class_embeddings.items():
        distance = F.pairwise_distance(query_embedding, avg_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_class = class_name

    print(f"Query image is closest to the class: {closest_class}")

    # Retrieve and display images from the closest class
    closest_class_images = retrieve_images_from_class(dataset, closest_class, num_images=5)
    tensor_images = [transforms.ToTensor()(Image.open(img).convert("RGB")) for img in closest_class_images]
    display_images(tensor_images, closest_class)

if __name__ == "__main__":
    main()
