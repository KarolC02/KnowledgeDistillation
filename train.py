import os
import random
import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from models.models import model_dict
from utils.arg_utils import get_args
from datasets import get_dataloaders
from utils.seed_utils import set_seed
from trainer.train_loop import train_one_epoch
from trainer.val_loop import validate_model


def main():
    args = get_args()
    set_seed(args.seed)
    train(args)


def train(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model_dict[args.model]()
    if args.parallel:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = get_dataloaders(args.dataset, args.batch_size, args.num_workers)

    save_dir = os.path.join(
        args.logdir,
        args.dataset,
        args.model,
        f"lr={args.lr:.0e}_bs={args.batch_size}_epochs={args.num_epochs}_parallel={args.parallel}"
    )
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))

    start_epoch = 0

    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from checkpoint at epoch {start_epoch}")

    for epoch in range(start_epoch, args.num_epochs):
        train_one_epoch(train_loader, optimizer, device, epoch, args.num_epochs, model, criterion, writer, val_loader)

        if (epoch + 1) % args.save_checkpoint_every == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, checkpoint_path)

    validate_model(model, val_loader, device, args.num_epochs, writer)

    final_checkpoint_path = os.path.join(save_dir, "final_checkpoint.pth")
    torch.save({
        "epoch": args.num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, final_checkpoint_path)

    writer.close()

if __name__ == "__main__":
    main()
