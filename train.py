import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import get_model

def train_model(dataset_dir, epochs=10, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Define Directories
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    
    # 2. Data Augmentation and Normalization
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 3. Load Datasets
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")
    
    # Save class names to classes.txt
    with open("classes.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print("Saved classes.txt")
    
    # 4. Initialize Model, Loss, Optimizer, Scheduler
    model = get_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Fine-tuning: smaller learning rate for pretrained backbone, slightly larger for fc is handled
    # but a flat small learning rate of 1e-4 works beautifully with Adam for transfer learning.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    # For tracking history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_acc = 0.0
    
    # 5. Training Loop
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # Update progress bar
            train_bar.set_postfix(loss=loss.item(), acc=(running_corrects.double() / total_samples).item())
            
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = (running_corrects.double() / total_samples).item()
        
        # Evaluation Phase
        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0
        total_val_samples = 0
        
        test_bar = tqdm(test_loader, desc=f"Test Epoch {epoch+1}")
        with torch.no_grad():
            for inputs, labels in test_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data)
                total_val_samples += inputs.size(0)
                
                test_bar.set_postfix(loss=loss.item(), acc=(running_val_corrects.double() / total_val_samples).item())
                
        epoch_val_loss = running_val_loss / total_val_samples
        epoch_val_acc = (running_val_corrects.double() / total_val_samples).item()
        
        print(f"Epoch {epoch+1} Results:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
        print(f"  Test Loss:  {epoch_val_loss:.4f} | Test Acc:  {epoch_val_acc*100:.2f}%")
        
        # Append history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_loss'].append(epoch_val_loss)
        history['test_acc'].append(epoch_val_acc)
        
        # Step Scheduler
        scheduler.step(epoch_val_acc)
        
        # Save Best Model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  => Saved new best model with test accuracy: {best_acc*100:.2f}%")
            
    print(f"\nTraining Complete! Best Test Accuracy: {best_acc*100:.2f}%")
    
    # 6. Plot & Save Curves
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), history['test_loss'], label='Test Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Acc', marker='o')
    plt.plot(range(1, epochs + 1), history['test_acc'], label='Test Acc', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Saved training_curves.png")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'Dataset')
    
    # 10 epochs is usually plenty for transfer learning on MobileNetV3
    train_model(dataset_dir, epochs=10, batch_size=32, lr=1e-4)
