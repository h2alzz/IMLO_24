#test.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# ======================
# Simplified CNN Model - Same as in train.py
# ======================
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        
        # First convolutional block - reduced filters
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second convolutional block - reduced filters
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Third convolutional block - reduced filters
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Simplified fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# ======================
# Test function
# ======================
def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    # Time tracking
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    test_acc = 100. * correct / total
    
    # Compute inference time
    inference_time = time.time() - start_time
    
    return test_acc, class_correct, class_total, inference_time

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Define transforms - same normalization as training but no augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Load test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    # Class names for reporting
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # DataLoader with larger batch size for faster inference
    test_loader = DataLoader(
        test_dataset, 
        batch_size=512, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model
    model = MyCNN().to(device)
    
    # Load best model
    try:
        model.load_state_dict(torch.load('/kaggle/input/model-v4-1/best_model_v4_1.pth'))
        print("Successfully loaded model from 'best_model.pth'")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run test
    test_acc, class_correct, class_total, inference_time = test(model, test_loader, device)
    
    # Print overall accuracy
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Inference Time: {inference_time:.2f} seconds")
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f'{classes[i]}: {class_acc:.2f}%')

if __name__ == "__main__":
    main()