import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description="Train a flower image classifier")
    parser.add_argument("data_dir", type=str,
                        help="Path to dataset (must contain train/, valid/, test/ subdirs)")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Directory to save checkpoint (default: current dir)")
    parser.add_argument("--arch", type=str, default="vgg16",
                        choices=["vgg16", "densenet121"],
                        help="Pretrained model architecture (default: vgg16)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--hidden_units", type=int, default=512,
                        help="Units in hidden layer (default: 512)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training if available")
    return parser.parse_args()


def build_dataloaders(data_dir):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms),
        "valid": datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=eval_transforms),
        "test":  datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=eval_transforms),
    }
    dataloaders = {
        split: DataLoader(ds, batch_size=64, shuffle=(split == "train"))
        for split, ds in image_datasets.items()
    }
    return image_datasets, dataloaders


def build_model(arch, hidden_units, num_classes=102):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        in_features = 25088
    else:
        model = models.densenet121(pretrained=True)
        in_features = 1024

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, num_classes),
        nn.LogSoftmax(dim=1),
    )
    model.classifier = classifier
    return model


def train(model, dataloaders, criterion, optimizer, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = correct = total = 0

        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for inputs, labels in dataloaders["valid"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        print(
            f"Epoch {epoch+1}/{epochs}  "
            f"Train loss: {running_loss/len(dataloaders['train']):.4f}  "
            f"Train acc: {correct/total:.4f}  "
            f"Val loss: {val_loss/len(dataloaders['valid']):.4f}  "
            f"Val acc: {val_correct/val_total:.4f}"
        )


def save_checkpoint(model, optimizer, arch, hidden_units, epochs, class_to_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "arch": arch,
        "hidden_units": hidden_units,
        "epochs": epochs,
        "class_to_idx": class_to_idx,
        "classifier": model.classifier,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    path = os.path.join(save_dir, "flower_classifier.pth")
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def main():
    args = get_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_datasets, dataloaders = build_dataloaders(args.data_dir)
    model = build_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train(model, dataloaders, criterion, optimizer, device, args.epochs)

    model.class_to_idx = image_datasets["train"].class_to_idx
    save_checkpoint(model, optimizer, args.arch, args.hidden_units, args.epochs,
                    model.class_to_idx, args.save_dir)


if __name__ == "__main__":
    main()
