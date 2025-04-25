import torch

def validate_model(model, val_loader, device, curr_epoch, writer):
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
    print(f"Validation Accuracy at epoch {curr_epoch+1}: {val_accuracy:.2f}%")
    writer.add_scalar("Validation Accuracy", val_accuracy, curr_epoch + 1)
