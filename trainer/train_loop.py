import torch
from tqdm import tqdm
from trainer.val_loop import validate_single_batch
from trainer.val_loop import validate_model

def train_one_epoch(train_loader, optimizer, device, epoch, num_epochs, model, criterion, writer=None, val_loader=None, log_interval=50):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        if batch_idx % log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            acc = correct / total
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{acc:.2%}"})

            if writer:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/Accuracy", acc, global_step)
            
            validate_single_batch(model, val_loader, device)

    avg_loss = running_loss / len(train_loader)
    acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_loss:.4f} | Train Acc: {acc:.2%}")
    if writer:
        writer.add_scalar("Train/EpochLoss", avg_loss, epoch)
        writer.add_scalar("Train/EpochAccuracy", acc, epoch)

    if val_loader:
        validate_model(model, val_loader, device, curr_epoch=epoch, writer=writer)
