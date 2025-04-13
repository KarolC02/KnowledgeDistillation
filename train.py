# Import dependencies
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import models, datasets
from torchvision import transforms 
from torch.utils.tensorboard import SummaryWriter
from random import randint
from models.models import model_dict
from utils.arg_utils import get_args
import random
import numpy as np
from datetime import datetime

def main():
    args = get_args()
    set_seed()
    model_name = args.model
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    parallel = args.parallel
    train(model_name, batch_size, num_epochs, lr, parallel)
    


def train(model_name, batch_size, num_epochs, lr, parallel):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = model_dict[model_name]()

    if(parallel):
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    train_dataset = datasets.ImageFolder("datasets/tiny-imagenet-200/train", transform=transform)
    val_dataset = datasets.ImageFolder("datasets/tiny-imagenet-200/val/images", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size = batch_size , shuffle=True, num_workers= 16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size , shuffle=False, num_workers = 16, pin_memory=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"logs/tiny_image_net_{model_name}_lr={lr}_epochs={num_epochs}_batch_size={batch_size}_{timestamp}")


    validate_model(model, val_loader, device, 0, writer)

    for epoch in range(num_epochs):
        train_one_epoch(train_loader, optimizer, device, epoch, num_epochs, model, criterion, writer)

    validate_model(model, val_loader, device, num_epochs, writer)

    torch.save(model.state_dict(), f"logs/tiny_image_net_{model_name}_lr={lr}_epochs={num_epochs}_batch_size={batch_size}/checkpoint.pth")
    writer.close()
    

def train_one_epoch(train_loader, optimizer, device, epoch, num_epochs, model, criterion, writer):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs,1)
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)

        running_loss += loss.item()
    accuracy = 100 * running_correct / running_total
    writer.add_scalar('Training Loss', running_loss / running_total, epoch + 1 )
    writer.add_scalar('Training Accuracy', accuracy, epoch + 1)

def validate_model(model, val_loader, device, curr_epoch, writer):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    writer.add_scalar("Validation Accuracy", val_accuracy, curr_epoch)
    

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    main()