# datasets/datasets.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(dataset_name, batch_size, num_workers=16, shuffle_train=True):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
            
    if dataset_name.lower() == "tiny-imagenet":

        train_dir = "datasets/tiny-imagenet-200/train"
        val_dir = "datasets/tiny-imagenet-200/val/image"
        assert os.path.isdir(train_dir), f"Training directory not found: {train_dir}"
        assert os.path.isdir(val_dir), f"Validation directory not found: {val_dir}"

        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

    elif dataset_name.lower() == 'rp2k':

        train_dir = "datasets/rp2k/train"
        val_dir = "datasets/rp2k/val/images"

        assert os.path.isdir(train_dir), f"Training directory not found: {train_dir}"
        assert os.path.isdir(val_dir), f"Validation directory not found: {val_dir}"

        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not supported.")
    
    print(f"[{dataset_name}] Found {len(train_dataset.classes)} classes")
    print(f"[{dataset_name}] Sample classes: {train_dataset.classes[:5]}")      
    return train_loader, val_loader