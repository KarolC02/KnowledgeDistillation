import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.distill_args import get_args
from utils.seed_utils import set_seed
from utils.transforms import get_standard_imagenet_transform
from models.models import model_dict
from train import train, validate_model
from datasets import get_dataloaders


def main():
    args = get_args()
    set_seed()

    dataset_root = f"datasets/{args.dataset}" 
    transform = get_standard_imagenet_transform()

    # Train teacher if not found
    teacher_ckpt_dir = os.path.join(
        args.logdir,
        args.dataset,
        args.teacher_model,
        f"lr={args.teacher_lr:.0e}_bs={args.teacher_batch_size}_epochs={args.teacher_num_epochs}_parallel={args.parallel}"
    )
    teacher_ckpt = os.path.join(teacher_ckpt_dir, args.teacher_checkpoint_name)
    print(teacher_ckpt)
    # Clean up any accidental bash escape issues    
    teacher_ckpt = teacher_ckpt.replace('\\=', '=')

    if not os.path.isfile(teacher_ckpt):
        print("[INFO] Teacher checkpoint not found. Training teacher...")
        train_args = args  
        train_args.model = args.teacher_model
        train_args.batch_size = args.teacher_batch_size
        train_args.lr = args.teacher_lr
        train_args.num_epochs = args.teacher_num_epochs
        train_args.parallel = args.parallel
        train_args.dataset = args.dataset
        train(train_args)
    else:
        print("[INFO] Found teacher checkpoint. Skipping training.")

    # Generate logits if needed
    logits_filename = f"logits_bs={args.teacher_batch_size}_lr={args.teacher_lr:.0e}_epochs={args.teacher_num_epochs}_adapted={args.adapt_model}_ckpt={args.teacher_checkpoint_name.replace('.pth', '')}.pth"
    logits_path = os.path.join(args.logits_dir, args.dataset, args.teacher_model, logits_filename)

    if not os.path.isfile(logits_path):
        print("[INFO] Logits not found. Generating with teacher model...")
        model = model_dict[args.teacher_model]()
        state_dict = torch.load(teacher_ckpt)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()

        train_loader = DataLoader(
            datasets.ImageFolder(os.path.join(dataset_root, "train"), transform=transform),
            batch_size=args.teacher_batch_size,
            shuffle=False, num_workers=16, pin_memory=True
        )
        save_logits(model, train_loader, args.parallel, logits_path, args)
        print("[INFO] Logits saved.")

    # Load distillation dataset
    train_dataset = DistillationDataset(
        os.path.join(dataset_root, "train"), logits_path, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    val_loader = DataLoader(
        datasets.ImageFolder(os.path.join(dataset_root, "val/images"), transform=transform),
        batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True
    )

    student = model_dict[args.student_model]()
    if args.parallel:
        student = nn.DataParallel(student)
    student.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    log_path = os.path.join(
        args.logdir,
        "distill",
        args.dataset,
        f"{args.teacher_model}_to_{args.student_model}_T={args.temperature}_alpha={args.alpha}"
    )
    writer = SummaryWriter(log_dir=log_path)

    for epoch in range(args.num_epochs):
        train_one_epoch_distillation(train_loader, optimizer, student, epoch, args.num_epochs, writer, args.temperature, args.alpha, val_loader)

    validate_model(student, val_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"), args.num_epochs, writer)

    out_path = os.path.join(args.modeldir, f"distilled_{args.teacher_model}_to_{args.student_model}.pth")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(student.state_dict(), out_path)
    print(f"[INFO] Saved distilled model to {out_path}")


def save_logits(model, loader, parallel, path, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if parallel:
        model = nn.DataParallel(model)
    model.to(device)

    all_logits = []
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Forward pass for logits"):
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "logits": torch.cat(all_logits),
        "batch_size": args.teacher_batch_size,
        "lr": args.teacher_lr,
        "num_epochs": args.teacher_num_epochs,
        "adapted": args.adapt_model,
        "teacher_checkpoint_used": args.teacher_checkpoint_name,
        "teacher_model": args.teacher_model,
        "dataset": args.dataset,
    }, path)



class DistillationDataset(Dataset):
    def __init__(self, root, logits_path, transform=None):
        self.dataset = datasets.ImageFolder(root, transform=transform)
        self.logits = torch.load(logits_path)['logits']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, self.logits[idx]


def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    return alpha * hard + (1 - alpha) * soft


def train_one_epoch_distillation(loader, optimizer, model, epoch, total_epochs, writer, T, alpha, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels, teacher_logits in tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}"):
        inputs, labels, teacher_logits = inputs.to(device), labels.to(device), teacher_logits.to(device)
        student_logits = model(inputs)
        loss = distillation_loss(student_logits, teacher_logits, labels, T, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (student_logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / total
    writer.add_scalar("Distill/TrainLoss", total_loss / total, epoch+1)
    writer.add_scalar("Distill/TrainAcc", acc, epoch+1)

    if (epoch + 1) % 5 == 0:
        validate_model(model, val_loader, device, epoch, writer)


if __name__ == "__main__":
    main()
