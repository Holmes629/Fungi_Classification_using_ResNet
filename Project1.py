import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve


class MyClassifier:
    def __init__(self):
        self.class_labels = ['edible_1', 'edible_2', 'edible_3', 'edible_4', 'edible_5', 
                             'poisonous_1', 'poisonous_2', 'poisonous_3', 'poisonous_4', 'poisonous_5']
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup(self):
        ''' This function initializes the ResNet50 model and loads custom weights '''
        self.model = models.resnet50(weights=None)  # Initialize without any weights
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        
        # Modify the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, len(self.class_labels))
        )
        
        # Load the custom weights
        weights_path = r"resnet50_fungi_model1.pth"
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        # Move the model to the specified device
        self.model = self.model.to(self.device)
        
        # Use a more sophisticated fine-tuning strategy
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last two layers
        for layer in [self.model.layer4, self.model.fc]:
            for param in layer.parameters():
                param.requires_grad = True

    def test_image(self, image):
        ''' This function will be given a PIL image and should return the predicted class label '''
        self.model.eval()  # Set to evaluation mode
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(self.device)  # Preprocess and add batch dimension

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
        
        predicted_class = self.class_labels[predicted.item()]
        return predicted_class
    
    
if __name__=='__main__': 
    # Data Pre Processing-- Transformations for training and validation (No augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainval_dataset = torchvision.datasets.ImageFolder('trainval', transform = transform)
    
    # Plotting class distribution
    class_counts = [trainval_dataset.targets.count(i) for i in range(len(trainval_dataset.classes))]

    plt.bar(trainval_dataset.classes, class_counts)
    plt.xlabel('Classes')
    plt.ylabel('Number of images')
    plt.title('Class Distribution in Fungi Dataset')
    plt.xticks(rotation=45)
    plt.show()
    
    # spliting the data into training and validation sets
    train_portion = 0.95
    val_portion = 0.05

    all_idxes = np.arange(len(trainval_dataset))  # Get all indice
    all_targets = trainval_dataset.targets         # Get all labels

    train_idx, val_idx = train_test_split(all_idxes, train_size=train_portion, stratify=all_targets, random_state=0) 

    # Creating train and validation subsets
    train_dataset = torch.utils.data.Subset(trainval_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(trainval_dataset, val_idx)

    print(f'Size of train dataset: {len(train_dataset)}')
    print(f'Size of val dataset: {len(val_dataset)}')
    
    # defining data loaders
    batch_size = 16
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers = 2)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers = 2)

    # Data Visualisation
    # Visualizing the class distribution
    train_classes = [trainval_dataset.targets[idx] for idx in train_idx]
    val_classes = [trainval_dataset.targets[idx] for idx in val_idx]

    plt.figure(figsize=(10, 6))
    train_class_counts = np.bincount(train_classes)
    val_class_counts = np.bincount(val_classes)

    # Bar chart for training and validation set class distribution
    plt.bar(range(len(train_class_counts)), train_class_counts, label='Training Set', alpha=0.7, color='blue')
    plt.bar(range(len(val_class_counts)), val_class_counts, label='Validation Set', alpha=0.7, color='green', bottom=train_class_counts)

    plt.xlabel('Classes')
    plt.ylabel('Number of images')
    plt.title('Class Distribution: Training vs Validation')
    plt.legend()
    plt.show()
    
    model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
    print(model)
    # Initialize ResNet50 pre-trained model- 
    in_features = model.fc.in_features
    model.fc = nn.Linear(model.fc.in_features, 10)
    print(model.fc)
    
    # moving the model to GPU if available else uses CPU 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Correct check for GPU availability
    model = model.to(device)
    
    def train_model(model, criterion, optimizer, trainloader, valloader, num_epochs=10):
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
        best_val_acc= 0
        
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            correct, total = 0, 0

            for inputs, labels in trainloader:
                # Move inputs and labels to the same device as the model
                inputs, labels = inputs.to(classifier.device), labels.to(classifier.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader)}, Accuracy: {train_acc}")
            val_loss, val_acc= validate_model(model, valloader, criterion)
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'resnet50_fungi_model.pth')
            
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
            
    def validate_model(model, valloader, criterion):
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(classifier.device), labels.to(classifier.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return val_loss/len(valloader), correct/total
    
    classifier = MyClassifier() # Creating an instance of MyClassifier
    classifier.setup() # setting up the ResNet50 model inside MyClassifier
        
    criterion = nn.CrossEntropyLoss() # Loss function and optimizer (using CrossEntropyLoss)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.model.parameters()), lr=0.001)

    # next step is to train the model
    train_model(classifier.model, criterion, optimizer, trainloader, valloader, num_epochs=25)


