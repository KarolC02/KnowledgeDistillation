from utils.distill_args import get_args
import os.path
from train import train
from models.models import model_dict
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms 
from torch.utils.data import DataLoader
from train import set_seed
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from train import validate_model
from datetime import datetime

def main():
    # get args
    args = get_args()
    set_seed()
    teacher_model_name = args.teacher_model
    student_model_name = args.student_model
    T = args.temperature
    alpha = args.alpha
    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    logdir = args.logdir
    parallel = args.parallel
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Check if we have this model saved
    full_model_name = f"tiny_image_net_{teacher_model_name}_lr=0.001_epochs=10_batch_size=128"
    model_path = os.path.join("logs", full_model_name, "checkpoint.pth")
    if os.path.isfile(model_path):
        print(f"Model {full_model_name} found in the logs directory. No need for training the teacher.")
    else:
        print(f"Model {full_model_name} has not been found in the logs directory. initializing training")
        train(teacher_model_name, batch_size, num_epochs, lr, parallel)
        print(f"Training successful, proceeding with distillation {teacher_model_name} -> {student_model_name}")
    # Check if we have logits of this model
    logits_file = os.path.join(
        "saved_logits",
        full_model_name,
        "logits.pth"
    )


    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder("datasets/tiny-imagenet-200/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size = batch_size , shuffle=False, num_workers= 16, pin_memory=True)
    val_dataset = datasets.ImageFolder("datasets/tiny-imagenet-200/val/images", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers = 16, pin_memory=True)

    if os.path.isfile(logits_file):
        print(f"Found logits from the trained model: {full_model_name}, no need to forward pass")
    else:
        print(f"Logits from model {full_model_name} not found. Performing forward pass...")
        model = model_dict[teacher_model_name]()
        state_dict = torch.load(model_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')  # Remove 'module.' if present
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)
        save_logits(model, device, train_loader, parallel, batch_size, logits_file)
        print(f"Forward pass performed successfully. Logits saved to {os.path.join('saved_logits', full_model_name, 'logits.pth')}")

    train_dataset = DistillationDataset(
        root="datasets/tiny-imagenet-200/train",
        logits_path=logits_file,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,       
        shuffle=True,                
        num_workers=16,
        pin_memory=True
    )

    student_model = model_dict[student_model_name]()
    optimizer = torch.optim.Adam(student_model.parameters(), lr = lr)
    if(parallel):
        student_model = nn.DataParallel(student_model)
    student_model.to(device)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"logs/distillation_{teacher_model_name}_to_{student_model_name}_T={T}_alpha={alpha}_{timestamp}")

    # validate_model(student_model, val_loader, device, curr_epoch=0, writer=writer)

    for epoch in range(num_epochs):
        train_one_epoch_distillation(train_loader, optimizer, device, epoch, num_epochs, student_model, writer, T, alpha, val_loader)



    validate_model(student_model, val_loader, device, num_epochs, writer=writer)

    
    save_path = os.path.join("saved_models", f"distilled_{teacher_model_name}_to_{student_model_name}_batch_size={batch_size}_lr={lr}/checkpoint.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(student_model.state_dict(), save_path)
    print(f"Saved distilled student model at {save_path}")


def save_logits(model, device, train_loader, parallel, batch_size, save_path):

    if(parallel):
        model = nn.DataParallel(model)
    model.to(device)

    model.eval()
    all_logits = []
    all_indices = []
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_logits.append(outputs.cpu())
            all_indices.extend(range(idx * batch_size, idx * batch_size + inputs.size(0)))
    all_logits = torch.cat(all_logits, dim=0)
    all_indices = torch.tensor(all_indices)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'logits': all_logits, 'indices': all_indices}, save_path)

class DistillationDataset(Dataset):
    def __init__(self, root, logits_path, transform=None):
        self.dataset = datasets.ImageFolder(root, transform=transform)
        logits_data = torch.load(logits_path)
        self.logits = logits_data['logits']    # (N, C)
        self.indices = logits_data['indices']  # (N,)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        teacher_logits = self.logits[idx]
        return img, label, teacher_logits

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)  
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
    return alpha * hard_loss + (1 - alpha) * soft_loss


def train_one_epoch_distillation(train_loader, optimizer, device, epoch, num_epochs, model, writer, T, alpha, val_loader):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    num_batches = 0
    total_batches = len(train_loader)

    for batch_idx, (inputs, labels, teacher_logits) in enumerate(tqdm(train_loader, desc=f"Distill Epoch {epoch+1}/{num_epochs}")):
        inputs = inputs.to(device)
        labels = labels.to(device)
        teacher_logits = teacher_logits.to(device)

        student_logits = model(inputs)

        loss = distillation_loss(student_logits, teacher_logits, labels, T, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(student_logits, 1)
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)

        running_loss += loss.item()
        num_batches += 1

        # if (batch_idx + 1) % 100 == 0:
        #     batch_accuracy = 100 * running_correct / running_total
        #     print(f"Batch {batch_idx+1}/{total_batches}: Loss = {loss.item():.4f}, Accuracy = {batch_accuracy:.2f}%")

    if( (epoch) % 5 == 0 ):
        validate_model(model, val_loader, device, epoch, writer=writer)

    training_loss = running_loss / running_total
    accuracy = 100 * running_correct / running_total

    writer.add_scalar('Training Loss', training_loss, epoch + 1)
    writer.add_scalar('Training Accuracy', accuracy, epoch + 1)




if __name__ == "__main__":
    main()