import torch

def validate_model(model, val_loader, device, curr_epoch, writer):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if curr_epoch == 0 and batch_idx == 0:
                print("Sanity check — logits shape:", outputs.shape)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_accuracy = 100 * correct / total
    avg_loss = running_loss / len(val_loader)

    print(f"Validation — Epoch {curr_epoch+1}: Accuracy = {val_accuracy:.2f}%, Loss = {avg_loss:.4f}")
    
    if writer:
        writer.add_scalar("Validation Accuracy", val_accuracy, curr_epoch + 1)
        writer.add_scalar("Validation Loss", avg_loss, curr_epoch + 1)

def validate_single_batch(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        inputs, labels = next(iter(val_loader))
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    val_accuracy = 100 * correct / total
    print(f"Sanity check: Validation Accuracy on 1 batch: {val_accuracy:.2f}%")
