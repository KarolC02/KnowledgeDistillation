import torch
from tqdm import tqdm
from trainer.val_loop import validate_model

def train_one_epoch(train_loader, optimizer, device, epoch, num_epochs, model, criterion, writer, val_loader, validate_every=5):
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

        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)

        running_loss += loss.item()

    accuracy = 100 * running_correct / running_total
    writer.add_scalar('Training Loss', running_loss / running_total, epoch + 1)
    writer.add_scalar('Training Accuracy', accuracy, epoch + 1)

    if epoch in {0, 1, 2} or (epoch + 1) % validate_every == 0:
        validate_model(model, val_loader, device, epoch, writer)
