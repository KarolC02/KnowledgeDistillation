# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import models, datasets
from torchvision import transforms as T

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import LRScheduler, ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import ProgressBar, TensorboardLogger
import ignite.contrib.engines.common as common

import opendatasets as od
import os
from random import randint
import urllib
import zipfile
 
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

DATA_DIR = 'datasets/tiny-imagenet-200'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')

batch_size = 32
def generate_dataloader(data, name, transform, batch_size):
    if data is None:
        return None
    
    if transform is None:
        dataset = datasets.ImageFolder(data, transform = T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform = transform)

    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle= (name=="train"), **kwargs) # ** = unpack
    
    return dataloader

dataset = pd.read_csv("datasets/tiny-imagenet-200/val/val_annotations.txt", sep = "\t", header = None, names = ['File', "Class", "X", "Y", "H", "w"])

val_img_dir = os.path.join(VAL_DIR, "images")
with open(os.path.join(VAL_DIR, 'val_annotations.txt'), 'r') as fp:
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]


for img, folder in val_img_dict.items():
    new_path =  (os.path.join(val_img_dir, folder))
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    if(os.path.exists(os.path.join(val_img_dir, img)) ):
        os.rename( os.path.join(val_img_dir, img), os.path.join(new_path, img))

class_to_name_dict = {}
with open(os.path.join(DATA_DIR, "words.txt"), "r") as f:
    lines = f.readlines()
    for line in lines:
        words = line.strip('\n').split("\t")
        class_to_name_dict[words[0]] = words[1]

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def show_batch(dataloader):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    imshow(make_grid(images)) # Using Torchvision.utils make_grid function
    
def show_image(dataloader):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    random_num = randint(0, len(images)-1)
    imshow(images[random_num])
    label = labels[random_num]
    print(f'Label: {label}, Shape: {images[random_num].shape}')

preprocess_transform = T.Compose([
                T.Resize(256), # Resize images to 256 x 256
                T.CenterCrop(224), # Center crop image
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 
])

preprocess_transform_pretrain = T.Compose([
                T.Resize(256), # Resize images to 256 x 256
                T.CenterCrop(224), # Center crop image
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])


train_loader = generate_dataloader(TRAIN_DIR, "train", transform = preprocess_transform, batch_size = batch_size)

train_loader_pretrain = generate_dataloader(TRAIN_DIR, "train",
                                  transform=preprocess_transform_pretrain, batch_size = batch_size)

val_loader = generate_dataloader(val_img_dir, "val",
                                 transform=preprocess_transform, batch_size = batch_size)

val_loader_pretrain = generate_dataloader(val_img_dir, "val",
                                 transform=preprocess_transform_pretrain, batch_size = batch_size)

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=200)

model = model.to(device)

lr = 0.001  
num_epochs = 10  
log_interval = 300  

loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=lr)

trainer = create_supervised_trainer(model, optimizer, loss_func, device=device)

ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"Batch Loss": x})

metrics = {
    "accuracy": Accuracy(), 
    "loss": Loss(loss_func),
}


train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

@trainer.on(Events.STARTED)
def start_message():
    print("Begin training")

@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_batch(trainer):
    batch = (trainer.state.iteration - 1) % trainer.state.epoch_length + 1
    print(
        f"Epoch {trainer.state.epoch} / {num_epochs}, "
        f"Batch {batch} / {trainer.state.epoch_length}: "
        f"Loss: {trainer.state.output:.3f}"
    )

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(trainer):
    print(f"Epoch [{trainer.state.epoch}] - Loss: {trainer.state.output:.2f}")
    train_evaluator.run(train_loader_pretrain)
    epoch = trainer.state.epoch
    metrics = train_evaluator.state.metrics
    print(f"Train - Loss: {metrics['loss']:.3f}, "
          f"Accuracy: {metrics['accuracy']:.3f} "
          )

common.save_best_model_by_val_score(
          output_path="best_models",
          evaluator=evaluator,
          model=model,
          metric_name="accuracy",
          n_saved=1,
          trainer=trainer,
          tag="val"
)

tb_logger = TensorboardLogger(log_dir="logs")


tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=log_interval),
    tag="training",
    output_transform=lambda loss: {"Batch Loss": loss},
)


for tag, evaluator in [("training", train_evaluator), ("validation", evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )


trainer.run(train_loader_pretrain, max_epochs=num_epochs)

tb_logger.close()
