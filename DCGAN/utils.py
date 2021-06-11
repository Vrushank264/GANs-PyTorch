import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader 

IMG_SIZE = 64

def get_data(root):
    
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]
        )

    dataset = datasets.ImageFolder(root = root, transform = transform)
    
    loader = DataLoader(dataset, batch_size = 64, shuffle = True)
    
    return loader
    