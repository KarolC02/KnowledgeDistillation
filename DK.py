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
from trainer.val_loop import validate_single_batch
from utils.adapt_model import adapt_model_to_classes

def main():
    args = get_args()
    set_seed()

    transform = get_standard_imagenet_transform()

    if not args.teacher_checkpoint_path or not os.path.isfile(args.teacher_checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found at: {args.teacher_checkpoint_path}")
    print(f"[INFO] Using teacher checkpoint: {args.teacher_checkpoint_path}")

    logits_path = infer_logits_path(args)

    if not os.path.isfile(logits_path):
        print("[INFO] Logits not found. Generating with teacher model...")

        teacher_model_name = extract_teacher_model_name(args.teacher_checkpoint_path, args.dataset)
        model = model_dict[teacher_model_name]()
        if args.adapt_model:
            model = adapt_model_to_classes(model, num_classes=args.num_classes)
        checkpoint = torch.load(args.teacher_checkpoint_path)
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        dummy_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
        output = model(dummy_input)
        assert output.shape[1] == args.num_classes, f"Teacher model output size {output.shape[1]} does not match {args.num_classes} classes."

        model.eval()
        train_loader, _ = get_dataloaders(args.dataset, args.batch_size, shuffle_train=False)
        save_logits(model, train_loader, args.parallel, logits_path, args)
        print("[INFO] Logits saved.")
    else:
        print("Found logits")
 
    train_dataset = DistillationDataset(
        root=f"datasets/{args.dataset}/train" if args.dataset != 'tiny-imagenet' else "datasets/tiny-imagenet-200/train",
        logits_path=logits_path,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    _, val_loader = get_dataloaders(args.dataset, args.batch_size, shuffle_train=False)

    student = model_dict[args.student_model]()
    if args.adapt_model:
        student = adapt_model_to_classes(student, num_classes=args.num_classes)

    if args.parallel:
        student = nn.DataParallel(student)
    student.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_factor) if args.lr_decay_every > 0 else None

    log_path = os.path.join(
        args.logdir,
        args.dataset,
        "distill",
        f"{args.teacher_checkpoint_path}_to_{args.student_model}_T={args.temperature}_alpha={args.alpha}"
    )
    writer = SummaryWriter(log_dir=log_path)

    for epoch in range(args.num_epochs):
        train_one_epoch_distillation(train_loader, optimizer, student, epoch, args.num_epochs, writer, args.temperature, args.alpha, val_loader)
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("lr", current_lr, epoch)
            print(f"[Epoch {epoch+1}] Learning rate adjusted to: {current_lr:.2e}")


    validate_model(student, val_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"), args.num_epochs - 1, writer)

    out_path = build_distilled_model_path(args)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(student.state_dict(), out_path)
    print(f"[INFO] Saved distilled model to {out_path}")

def save_logits(model, loader, parallel, path, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if parallel:
        model = nn.DataParallel(model)
    model.to(device)

    all_logits = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Forward pass for logits"):
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            assert logits.shape[1] == args.num_classes, f"Expected logits to have shape (_, {args.num_classes}), got {logits.shape}"

    all_paths = [os.path.abspath(path) for (path, _) in loader.dataset.samples]
    assert len(all_paths) == sum([b.shape[0] for b in all_logits]), "Mismatch between number of logits and image paths"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "logits": torch.cat(all_logits),
        "paths": all_paths,
        "adapted": args.adapt_model,
        "teacher_checkpoint_used": args.teacher_checkpoint_path,
        "dataset": args.dataset,
    }, path)

class DistillationDataset(Dataset):
    def __init__(self, root, logits_path, transform=None):
        self.transform = transform
        self.imagefolder = datasets.ImageFolder(root)
        saved = torch.load(logits_path)
        self.logits = saved['logits']
        self.paths = saved['paths']

        path_to_idx = {os.path.abspath(path): (i, label) for i, (path, label) in enumerate(self.imagefolder.samples)}
        self.samples = []
        self.labels = []
        for path in self.paths:
            abspath = os.path.abspath(path)
            if abspath not in path_to_idx:
                raise ValueError(f"Saved path {abspath} not found in dataset.")
            i, label = path_to_idx[abspath]
            self.samples.append(self.imagefolder.samples[i])
            self.labels.append(label)

        assert len(self.logits) == len(self.samples), "Mismatch between number of logits and number of samples"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.imagefolder.loader(path)
        if self.transform:
            img = self.transform(img)

        true_path, true_label = self.imagefolder.samples[idx]
        assert label == true_label, f"Label mismatch: expected {true_label}, got {label}"

        return img, label, self.logits[idx]

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    hard = F.cross_entropy(student_logits, labels)
    if alpha == 1:
        return hard
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    return alpha * hard + (1 - alpha) * soft

def train_one_epoch_distillation(loader, optimizer, model, epoch, total_epochs, writer, T, alpha, val_loader, log_interval=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{total_epochs}")
    for batch_idx, (inputs, labels, teacher_logits) in pbar:
        assert teacher_logits.shape[0] == inputs.shape[0], "Mismatch in batch sizes between inputs and teacher logits"

        inputs, labels, teacher_logits = inputs.to(device), labels.to(device), teacher_logits.to(device)
        student_logits = model(inputs)

        assert student_logits.shape == teacher_logits.shape, "Mismatch in shape between student and teacher logits"

        loss = distillation_loss(student_logits, teacher_logits, labels, T, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (student_logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        avg_loss = total_loss / total
        acc = 100.0 * correct / total
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{acc:.2f}%"})

        if (batch_idx + 1) % log_interval == 0:
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(loader) + batch_idx)
            writer.add_scalar("Train/Accuracy", acc, epoch * len(loader) + batch_idx)
            validate_single_batch(model, val_loader, device)
            model.train()

    writer.add_scalar("Train/EpochLoss", total_loss / total, epoch + 1)
    writer.add_scalar("Train/EpochAccuracy", acc, epoch + 1)
    validate_model(model, val_loader, device, epoch, writer)

def infer_logits_path(args):
    if args.logits_path != "-":
        return args.logits_path

    checkpoint_dir = os.path.dirname(args.teacher_checkpoint_path)
    relative_ckpt_path = os.path.relpath(checkpoint_dir, start=os.path.commonpath([checkpoint_dir, args.logdir]))
    logits_folder = os.path.join(args.logits_dir, relative_ckpt_path)
    return os.path.join(logits_folder, "logits.pth")

def build_distilled_model_path(args):
    teacher_exp_folder = os.path.basename(os.path.dirname(args.teacher_checkpoint_path))
    student_exp = (
        f"TO_{args.student_model}_"
        f"lr={args.lr:.0e}_"
        f"bs={args.batch_size}_"
        f"epochs={args.num_epochs}_"
        f"wd={args.weight_decay}_"
        f"do={args.dropout}"
    )
    if args.lr_decay_every > 0:
        student_exp += f"_decayEvery={args.lr_decay_every}_gamma={args.lr_decay_factor}"

    filename = f"{student_exp}_FROM_{teacher_exp_folder}.pth"
    return os.path.join(args.modeldir, args.dataset, "distill", filename)

def extract_teacher_model_name(checkpoint_path, dataset_name):
    parts = os.path.normpath(checkpoint_path).split(os.sep)
    try:
        dataset_index = parts.index(dataset_name)
        return parts[dataset_index + 1]
    except (ValueError, IndexError):
        raise ValueError(f"Could not extract teacher model name from path: {checkpoint_path}")

if __name__ == "__main__":
    main()
