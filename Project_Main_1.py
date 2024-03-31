# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:22:19 2024

@author: UTENTE
"""
# This code is meant to classify six different types of medical scans: It does so by employing a CNN modeled on the TinyVGG architecture
# It then plots the loss function gradient during training and testing, and saves the model parameters.
# Additionally, it plots a confusion matrix, which shows that the two classes to get more commonly misclassified are Hand X-Ray and CXR
# To this end I tried to add a feature attribution function but could not work with open cv or captum, the final GAP layer is there for 
# auto-Grad-CAM with cv2
# The dataset used is Medical MNIST from Kaggle https://www.kaggle.com/datasets/andrewmvd/medical-mnist/data
# It is made of 60000 64x64 jpeg images divided into six labels, which are HeadCT, Hand(X-Ray), ChestCT, CXR(Chest X-Ray), BreastMRI, AbdomenCT
# The number of epochs is at line 249, to be changed at will, it currently is at five
# Author is Tommaso Monicelli, the purpose of this file is coding an image classification algorithm using a convolutional neural network


# Import Modules
import os
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torchmetrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics, mlxtend
import random
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix


# Define paths to the dataset
dataset_dir = r"C:\Users\tomma\Downloads\archive_MedMnist"  #Replace this with the folder on your pc

# Define path of the model parameters
saved_model_path = r"C:\Users\UTENTE\Downloads\PyTorch_Project\trained_TinyVGG.pth" #Replace this with the folder on your pc

# Define transformations
transform = transforms.Compose([transforms.ToTensor()]) #Since the size is of 64x64 I didn't find reshaping to be necessary

# Load dataset
full_dataset = datasets.ImageFolder(dataset_dir, transform=transform)


# Extract class names (label names) from the dataset
label_names = full_dataset.classes #This takes as names the names of the folders

# Split dataset into train and test sets
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Define data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



#Define device in a device-agnostic mode
device='cuda' if torch.cuda.is_available() else 'cpu'


# Create a convolutional neural network, inspired by TinyVGG architecture 
# This module is ready for an analysis involving feature maps heatmap, to run with cv2
    
class TinyVGGModified(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        # Convolutional blocks as before
        self.features = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.classifier = nn.Linear(hidden_units, output_shape)

    def forward(self, x):
        x = self.features(x)
        self.feature_maps = x  # Store feature maps for CAM
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x



#Instantiate model
model = TinyVGGModified(input_shape=3, 
    hidden_units=10, 
    output_shape=len(label_names)).to(device)

#Pick Loss Function, Optimizer and Accuracy metric

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)
accuracy_fn = 100*(torchmetrics.classification.MulticlassAccuracy(num_classes = len(label_names))).to(device)
model.to(device)

# Define training and testing functions
# Training functions are adapted from Daniel Bourke's one from
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/03_pytorch_computer_vision.ipynb

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc



def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    all_pred_labels = []
    all_true_labels = []
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
            # Collect predicted labels and true labels
            all_pred_labels.extend(test_pred_labels.cpu().numpy())
            all_true_labels.extend(y.cpu().numpy())
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc, all_pred_labels,all_true_labels


from tqdm.auto import tqdm

#  Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    #  Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    #  Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc, all_pred_labels,all_true_labels = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        #  Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        #  Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    #  Return the filled results at the end of the epochs
    return results, all_pred_labels,all_true_labels


# Start the timer

from timeit import default_timer as timer 
start_time = timer()

# Pick number of epochs
epochs = 5

# Train model
model_results,all_pred_labels, all_true_labels = train(model=model, 
                        train_dataloader=train_loader,
                        test_dataloader=test_loader,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=epochs)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")


# Save the trained model's parameters
torch.save(model.state_dict(),saved_model_path)


# Create the confusion matrix
confmat = confusion_matrix(y_true=all_true_labels, y_pred=all_pred_labels)

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(10, 7))
cm = ax.matshow(confmat, cmap=plt.cm.Blues)

# Add color bar
plt.colorbar(cm)

# Add annotations for cell values
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

# Set labels for axes
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(ticks=np.arange(len(label_names)), labels=label_names, rotation=45)
plt.yticks(ticks=np.arange(len(label_names)), labels=label_names)

# Show plot
plt.show()
# Define a function to compute loss curve for training and testing
def plot_loss_curve(train_loss, test_loss):
    epochs = len(train_loss)
    plt.plot(range(1, epochs + 1), train_loss, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_loss, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting it
plot_loss_curve(model_results["train_loss"], model_results["test_loss"])


#Visualizing some misclassified examples
def visualize_misclassified_samples(model, test_loader, label_names, num_samples=5):
    misclassified_samples = []
    true_labels = []
    predicted_labels = []
    
    # Set the model to evaluation mode
    model.eval()
    
    # Iterate through the test dataset
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Collect misclassified samples
            misclassified_idx = (predicted != labels).nonzero().squeeze()
            if len(misclassified_idx.shape) == 0:
                misclassified_idx = misclassified_idx.unsqueeze(0)  # Convert scalar tensor to 1D tensor
            if len(misclassified_idx) == 0:
                continue  # Skip if no misclassified samples found in this batch
            
            misclassified_samples.extend(images[misclassified_idx][:num_samples].cpu())
            true_labels.extend(labels[misclassified_idx][:num_samples].cpu().numpy())
            predicted_labels.extend(predicted[misclassified_idx][:num_samples].cpu().numpy())
            
            # Break the loop if enough misclassified samples are collected
            if len(misclassified_samples) >= num_samples:
                break
    
    if len(misclassified_samples) == 0:
        print("No misclassified samples found.")
        return
    
    # Plot misclassified samples
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    fig.suptitle('Misclassified Samples', fontsize=16)
    
    for i, (sample, true_label, predicted_label) in enumerate(zip(misclassified_samples, true_labels, predicted_labels)):
        sample = sample.permute(1, 2, 0)  # Reshape (C, H, W) to (H, W, C) for visualization
        axes[i].imshow(sample)
        axes[i].set_title(f'True: {label_names[true_label]}\nPredicted: {label_names[predicted_label]}')
        axes[i].axis('off')
    
    plt.show()
    

visualize_misclassified_samples(model=model, test_loader=test_loader, label_names=label_names, num_samples=5)