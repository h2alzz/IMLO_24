#train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import time
import os
import torch.nn.functional as F
# ======================
# Simplified CNN Model
# ======================

class MyCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MyCNN, self).__init__()
        
        # First convolutional block with slightly more filters
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Added second conv layer
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second convolutional block with more filters
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Added second conv layer
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Third convolutional block with more filters
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Added second conv layer
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling for better feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Improved fully connected layers with skip connection
        self.fc1 = nn.Linear(256, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Global average pooling instead of flattening
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers with residual connection
        identity = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        # Final classification layer
        x = self.fc3(x)
        
        return x

    def init_weights(self):
        """Initialize model weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
# ======================
# Training function
# ======================
def train(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    if scheduler:
        scheduler.step()
        
    train_loss = running_loss / len(loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# ======================
# Validation function
# ======================
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def main():
    start_time = time.time()
    
    # Set device - prioritize GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Define transforms - basic transform for faster processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Augmentation only for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Load datasets with different transforms
    train_val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    
    # Use less validation data for faster evaluation
    val_size = 5000  # Fixed validation size
    train_size = len(train_val_dataset) - val_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    # Increase batch size for faster training
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, 
                             num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, 
                           num_workers=2, pin_memory=True if torch.cuda.is_available() else False)

    # Initialize model
    # model = MyCNN().to(device)
    model = MyCNN(num_classes=10, dropout_rate=0.5).to(device)
    model.init_weights()
    
    # Use mixed precision training if available (for newer GPUs)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # Learning rate scheduler - reduce learning rate when plateauing
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Reduce number of epochs
    num_epochs = 18
    best_val_acc = 0
    patience = 3
    counter = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved Best Model!')
            counter = 0
        else:
            counter += 1
            
        # Early stopping
        if counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
            
        # Check if total training time is approaching 4 hours
        total_time = time.time() - start_time
        hours = total_time / 3600
        if hours > 3.5:  # Stop if approaching 4 hours
            print(f"Training time limit approaching ({hours:.2f} hours). Stopping training.")
            break
            
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    main()