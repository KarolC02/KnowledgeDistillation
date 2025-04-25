# datasets/datasets.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(dataset_name, batch_size, num_workers=16):
    if dataset_name.lower() != "tiny-imagenet":
        raise NotImplementedError(f"Dataset '{dataset_name}' is not supported.")

    train_dir = "datasets/tiny-imagenet-200/train"
    val_dir = "datasets/tiny-imagenet-200/val/images"

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
