import os
import torch

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

    
