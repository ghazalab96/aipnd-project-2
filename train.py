# train.py - Script to train a new network on a dataset and save the model as a checkpoint
import torch
import argparse
from torchvision import datasets, transforms, models
from torch import nn, optim
import json
import os

def get_input_args():
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('data_dir', type=str, help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Pretrained model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden layer units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def train_model(args):
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    # Data transformations
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Load model
    model = models.vgg16(pretrained=True) if args.arch == 'vgg16' else models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    # Define classifier
    classifier = nn.Sequential(
        nn.Linear(25088, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        running_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {running_loss/len(dataloader):.4f}")
    
    # Save checkpoint
    model.class_to_idx = dataset.class_to_idx
    checkpoint = {
        'arch': args.arch,
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier
    }
    torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))
    print("Checkpoint saved!")

if __name__ == '__main__':
    args = get_input_args()
    train_model(args)



# python train.py flowers --save_dir ./ --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu