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
from models.models import model_dict
import urllib
import zipfile
from utils.arg_utils import get_args

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


args = get_args()
model_name = args.model
model = model_dict[args.model]()
lr = args.lr
num_epochs = args.num_epochs
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()


examples = iter(val_loader)
example_data, example_labels = next(examples)

writer = SummaryWriter(log_dir="logs/tiny_image_net", comment=f"_{model_name}_lr={lr}_epochs={num_epochs}")

# img_grid = torchvision.utils.make_grid(example_data)
# writer.add_image('tiny_image_net_images',img_grid)
model.eval()
example_data = example_data.to(device)
with torch.no_grad():
    writer.add_graph(model, example_data)

running_loss = 0.0
running_correct = 0
running_total = 0

for epoch in range(num_epochs):
    model.train()
    batch = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Currently processing batch 
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels) # Single batch loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs,1)
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)

        running_loss += loss.item()
        # -------------------------------
        # UNCOMMENT IF YOU WANT TO UPDATE EVERY X BATCHES
        # -------------------------------

        # if (batch) % x == 0:
        #     accuracy = 100 * running_correct / running_total
        #     # len(train_loader) -> number of batches in the training loader
        #     writer.add_scalar('Training Loss', running_loss / x, epoch * len(train_loader) )
        #     writer.add_scalar('Training Accuracy', accuracy, epoch * len(train_loader) + batch)
        #     running_loss = 0.0
        #     running_correct = 0
        #     running_total = 0

    accuracy = 100 * running_correct / running_total
    writer.add_scalar('Training Loss', running_loss / running_total, epoch + 1 )
    writer.add_scalar('Training Accuracy', accuracy, epoch + 1)
    running_loss = 0.0
    running_correct = 0
    running_total = 0


    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    # writer.add_scalar('Epoch Accuracy (Training)', 100 * correct / total, epoch * len(tr))
    # if( epoch % 1 == 0 ):
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for inputs, labels in val_loader:
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             outputs = model(inputs)
    #             _, predicted = torch.max(outputs, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #     print(f"Validation Accuracy in epoch {epoch}: {100 * correct / total:.2f}%")
    #     writer.add_scalar('Validation Accuracy', 100 * correct / total, epoch)

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

val_accuracy = 100 * correct / total
print(f"Validation Accuracy: {val_accuracy:.2f}%")
writer.add_scalar("Validation Accuracy", val_accuracy)
writer.close()