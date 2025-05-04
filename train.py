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
from trainer.val_loop import validate_single_batch
from utils.adapt_model import adapt_model_to_classes

def main():
    args = get_args()
    set_seed(args.seed)
    train(args)


def train(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model_dict[args.model]()

    if args.adapt_model:
        print("Before adaptation:")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"  {name}: {module}")
        model = adapt_model_to_classes(model, args.num_classes)
        print("After adaptation:")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"  {name}: {module}")
    else:
        print("Not adapting the model to", args.num_classes, "classes")


    if args.parallel:
        model = nn.DataParallel(model)
        print("Parallelizing model")
    else:
        print("Not parallelizing model")
    model.to(device)

    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    

    scheduler = None
    if args.lr_decay_every > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_factor
        )
        
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = get_dataloaders(args.dataset, args.batch_size, args.num_workers)

    exp_name = (
        f"lr={args.lr:.0e}_"
        f"bs={args.batch_size}_"
        f"epochs={args.num_epochs}_"
        f"wd={args.weight_decay}_"
        f"do={args.dropout}"
    )

    if args.lr_decay_every > 0:
        exp_name += f"_decayEvery={args.lr_decay_every}_gamma={args.lr_decay_factor}"

    save_dir = os.path.join(args.logdir, args.dataset, args.model, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))



    start_epoch = 0

    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from checkpoint at epoch {start_epoch}")

    print("Quick sanity check on validation set...")
    validate_single_batch(model, val_loader, device)

    for epoch in range(start_epoch, args.num_epochs):
        train_one_epoch(train_loader, optimizer, device, epoch, args.num_epochs, model, criterion, writer, val_loader)
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("lr", current_lr, epoch)
            print(f"[Epoch {epoch+1}] Learning rate adjusted to: {current_lr:.2e}")


        if (epoch + 1) % args.save_checkpoint_every == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, checkpoint_path)

    final_checkpoint_path = os.path.join(save_dir, "final_checkpoint.pth")
    torch.save({
        "epoch": args.num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, final_checkpoint_path)

    writer.close()

if __name__ == "__main__":
    main()
