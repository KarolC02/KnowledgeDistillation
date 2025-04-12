# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from tqdm import tqdm
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import models, datasets
from torchvision import transforms 
import torchvision
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import LRScheduler, ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import ProgressBar, TensorboardLogger
import ignite.contrib.engines.common as common
from torch.utils.tensorboard import SummaryWriter
import opendatasets as od
import os
import sys
from random import randint
import urllib
import zipfile

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 64

train_dataset = datasets.ImageFolder("datasets/tiny-imagenet-200/train", transform=transform)
val_dataset = datasets.ImageFolder("datasets/tiny-imagenet-200/val/images", transform=transform)

train_loader = DataLoader(train_dataset, batch_size = batch_size , shuffle=True, num_workers= 16, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size = batch_size , shuffle=False, num_workers = 16, pin_memory=True)

import torch.nn.functional as F

class myCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 500)
        self.fc2 = nn.Linear(500, 340)
        self.fc3 = nn.Linear(340,200)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = nn.Dropout(0.5)(F.relu(self.fc1(x)))
        x = nn.Dropout(0.5)(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


lr  = 0.001

from torchvision.models import resnet18, ResNet18_Weights

model_name = "resnet18_pretrained"
if(model_name == "resnet18_pretrained" ):
    model = models.resnet18(weights = ResNet18_Weights.DEFAULT)
elif(model_name == "myCNN"):
    model = myCNN()
# model = nn.DataParallel(model)  # Use all available GPUs
model = model.to(device)  # Move the model to the device (GPU)

optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
num_epochs = 10

examples = iter(val_loader)
examample_data, example_labels = next(examples)

writer = SummaryWriter("logs/tiny_image_net")
img_grid = torchvision.utils.make_grid()
writer.add_image('tiny_image_net_images',img_grid)
writer.close()
sys.exit()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    batches = 0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs,1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        running_loss += loss.item()
        if (batches + 1) % 100 == 0:
            accuracy = 100 * correct / total
            writer.add_scalar('Training Accuracy', accuracy, epoch * len(train_loader) + batches)
        batches += 1
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    writer.add_scalar('Epoch Accuracy (Training)', 100 * correct / total, epoch)
    if( epoch % 1 == 0 ):
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

        print(f"Validation Accuracy in epoch {epoch}: {100 * correct / total:.2f}%")
        writer.add_scalar('Validation Accuracy', 100 * correct / total, epoch)

writer.close()
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

print(f"Validation Accuracy: {100 * correct / total:.2f}%")