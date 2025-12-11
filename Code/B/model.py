import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
from torchvision import transforms

class AddGaussianNoise(object):
    """
    Custom Transform: Add Gaussian noise to the Tensor
    """
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class BreastMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        images: numpy array (N, 28, 28)
        labels: numpy array (N, 1)
        """
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=1)
            
        self.images = torch.tensor(images, dtype=torch.float32) / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long).squeeze() 
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolution Layer 1: Input 1 channel (grayscale), Output 32 channels, Kernel size 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        
        # Convolution Layer 2: Input 32 channels, Output 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Fully Connected Layers
        # Input dimensions: 64 channels × 7 × 7 pixels
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2) # 输出 2 类 (Benign, Malignant)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        # x shape: (Batch, 1, 28, 28)
        
        x = self.pool(F.relu(self.conv1(x))) # Batch, 32, 14, 14
        x = self.pool(F.relu(self.conv2(x))) # Batch, 64, 7, 7
        
        #Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), 100 * correct / total

def evaluate(model, loader, criterion, device, return_details=False):
    """
    Evaluate on the validation set or test set
    
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if return_details:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
    avg_loss = running_loss / len(loader)
    avg_acc = 100 * correct / total

    if return_details:
        return avg_loss, avg_acc, all_labels, all_preds
    else:
        return avg_loss, avg_acc

def run_experiment(train_images, train_labels, val_images, val_labels, test_images, test_labels, 
                   aug_mode, device):
    """ 
    General function for running a single random forest experiment
    
    """
    
    print(f"\nRunning Experiment: Augmentation Mode: {aug_mode}")
    
    if aug_mode == 'geometric':
        # Strategy A: Geometric Transformation (Rotation + Reflection) 
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
        ])
        EPOCHS = 50
        
    elif aug_mode == 'noise':
        # Strategy B: Pixel Transformation (Gaussian Noise)
        train_transform = transforms.Compose([
            AddGaussianNoise(mean=0., std=0.05),
        ])
        EPOCHS = 50
        
    else:
        # Baseline
        train_transform = None
        EPOCHS = 50

    eval_transform = None
        
    print(f"Transform Strategy: {train_transform}")
    print(f"Training Budget: {EPOCHS} Epochs")
    
    # DataLoader
    train_dataset = BreastMNISTDataset(train_images, train_labels, transform=train_transform)
    val_dataset = BreastMNISTDataset(val_images, val_labels, transform=eval_transform)
    test_dataset = BreastMNISTDataset(test_images, test_labels, transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialise the model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        
        # Early Stopping: Saving the Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict() 
            
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{EPOCHS}] Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    print(f"Best validation set accuracy: {best_val_acc:.2f}%")
    
    # test
    model.load_state_dict(best_model_state)
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, return_details=True)
    print(f"{aug_mode} Test Accuracy: {test_acc:.2f}%")

    print(classification_report(y_true, y_pred, digits=4))
    
    return test_acc

def run_model_B(train_images, train_labels, val_images, val_labels, test_images, test_labels):
    print("Initiating Model B CNN: Comparative testing of multiple data augmentation techniques")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # test 1: Baseline
    acc_baseline = run_experiment(train_images, train_labels, val_images, val_labels, test_images, test_labels, 
                   aug_mode='none', device=device)
    # test 2: Geometric
    acc_Geometric = run_experiment(train_images, train_labels, val_images, val_labels, test_images, test_labels, 
                   aug_mode='geometric', device=device)
    # test 3: Noise 
    acc_noise = run_experiment(train_images, train_labels, val_images, val_labels, test_images, test_labels, 
                   aug_mode='noise', device=device)
    
    print("\nModel B final comparison results:")
    print(f"Baseline Accuracy: {acc_baseline:.2f}%")
    print(f"Geometric Accuracy: {acc_Geometric:.2f}%")
    print(f"noise Accuracy: {acc_noise:.2f}%")
    
    
    if acc_Geometric > acc_baseline:
        print("Conclusion: Data augmentation (Geometric Transformation) has successfully enhanced the model's generalisation capability.")
    else:
        print("Conclusion: Data augmentation (Geometric Transformation) yields negligible results.")

    if acc_noise > acc_baseline:
        print("Conclusion: Data augmentation (Pixel Transformation) has successfully enhanced the model's generalisation capability.")
    else:
        print("Conclusion: Data augmentation (Pixel Transformation) yields negligible results.")

    if acc_Geometric > acc_noise:
        print("Conclusion: Data augmentation (Geometric transformations) is more effective at improving the model's generalisation capability.")
    else:
        print("Conclusion: Data augmentation (Pixel Transformation) is more effective at improving the model's generalisation capability.")

    return {"baseline_acc": acc_baseline, "Geometric_acc": acc_Geometric, "noise_acc": acc_noise}