""" 
Created by Veronica Obodozie
Project: Fashion MINST

This project atempts to apply quantum machine learning to the fashion minst dataset.
"""

#-----------------Importing Important Functions and Modules---------------------------#
#Functions
from utils import *
import torch 
import torchvision
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
import time
from PIL import Image
from datetime import datetime

import matplotlib.pyplot as plt
import os

# Pennylane
import pennylane as qml
from pennylane import numpy as np

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
#--------------------------------------------#

##-----------------------------------------------------------------------------------------------------------##
# set paths to retrieve data
# TRAIN_PATH = r""
# VAL_PATH = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Val"
# TEST_PATH = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test"

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

##-----------------------------------------------------------------------------------------------------------##
# Set the hyperparameters
print('------- Setting Hyperparameters-----------')
# Quantum
n_qubits = 2 #10
n_layers= 6 # # of variational layers, depth of circut
# General
batch_size = 16 # Change Batch Size o
learning_rate = 1e-3 #4
num_workers = 0 #2
nepochs = 1 #"Use it to change iterations"
weight_decay = 1e-4
best_loss = 1e+20 # number gotten from initial resnet18 run
stop_count = 6
print(f'batch_size = {batch_size}, learning_rate = {learning_rate} num_workers = {num_workers} , nepochs = {nepochs} , best_loss = {best_loss}, weight_decay={weight_decay}')
##-----------------------------------------------------------------------------------------------------------##


#--------------------- Data Loading and Pre-processing  -----------------------#
# Processing to match pre-trained networks
print('------- DATA PROCESSING --------')
# ResNet50
data_transforms = transforms.Compose([
    #transforms.Resize((28, 28)),
    transforms.ToTensor(), #]) #,
    transforms.Normalize((0.5,), (0.5,))])

# Load develpoment dataset
devset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform = data_transforms)
#devset = devset.reshape(devset.shape[0], 28, 28, 1)
train_set_size = int(len(devset) * 0.8)
val_set_size = len(devset) - train_set_size

# Load the development set again using the proper pre-processing transforms

# Split the development set into train and validation
trainset, valset = torch.utils.data.random_split(devset, [train_set_size, val_set_size], generator=torch.Generator().manual_seed(42))

# Get the data loader for the train set
train_loader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

# Get the data loader for the test set
val_loader = DataLoader(valset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

# Get the test set
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=data_transforms)
#testset = testset.reshape(testset.shape[0], 28, 28, 1)
# Get the data loader for the test set
test_loader = DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)


# Define the class labels for the Fashion MNIST dataset.
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')


##-----------------------------------------------------------------------------------------------------------##

# # Visualizing a sample image from dataset
# # Create a subplot with 4x4 grid
# fig, axs = plt.subplots(4, 4, figsize=(8, 8))

# # Loop through each subplot and plot an image
# for i in range(4):
#     for j in range(4):
#         image, label = trainset[i * 4 + j]  # Get image and label
#         image_numpy = image.numpy().squeeze()    # Convert image tensor to numpy array
#         axs[i, j].imshow(image_numpy, cmap='gray')  # Plot the image
#         axs[i, j].axis('off')  # Turn off axis
#         axs[i, j].set_title(f"Label: {label}")  # Set title with label

# plt.tight_layout()  # Adjust layout
# plt.show()  # Show plot

#--------------------------------------------MODEL-------------------------------------------#
# Quantum
n_layers = 6
weight_shapes = {"weights": (n_layers, n_qubits)}
# layer shape is (layers x qubits) so this example is 6 x 2

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Converting to layer
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# Sequential model
clayer_1 = torch.nn.Linear(28, n_qubits)
clayer_2 = torch.nn.Linear(n_qubits, 10)
gelu = nn.GELU()
softmax = torch.nn.Softmax(dim=1)
layers = [clayer_1, qlayer, clayer_2, gelu, softmax]
fmqModel = torch.nn.Sequential(*layers)

#Classic
# Sequential model
clayer_1 = torch.nn.Linear(28, n_qubits)
clayer_q = torch.nn.Linear(n_qubits, n_qubits)
clayer_2 = torch.nn.Linear(n_qubits, 10)
gelu = nn.GELU()
softmax = torch.nn.Softmax(dim=1)
layers = [clayer_1, clayer_q, clayer_2, gelu]#, softmax]
net = torch.nn.Sequential(*layers) #fmModel

#--------------------------------------------#
#------------------ Training Parameters --------------------------#
print('-------------------Pretrained Model Type: RESNET 50------------------------')
# resnet_input = (3, 224, 224)
PATH = './fm_resNet.pth' # Path to save the best model

net.to(device)

criterion = nn.functional.cross_entropy #nn.L1Loss()  # Loss function
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)  # all params trained
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.9)

#--------------------------------------------#

#---------------------- Training and Validation ----------------------#
e = []
trainL= []
valL =[]
counter = 0
print('Traning and Validation \n')
# Development time
training_start_time = time.time()
for epoch in range(nepochs):  # loop over the dataset multiple times
    # Training Loop
    net.train()
    train_loss = 0.0
    for images, labels in train_loader:
        # Move the images and labels to the computational device (CPU or GPU)
        images, labels = images.to(device), labels.to(device).float()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss = train_loss/len(train_loader)
    print(f'{epoch + 1},  train loss: {train_loss :.3f},', end = ' ')

    scheduler.step()

#---------------Validation----------------------------#
    net.eval()
    val_loss = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for images, labels in val_loader:
            # Move the images and labels to the computational device (CPU or GPU)
            images, labels = images.to(device), labels.to(device)

            outputs = net(images, 'val')
            loss = criterion(outputs, labels)

            val_loss += loss.item()
        val_loss = val_loss/len(val_loader)
        print(f'val loss: {val_loss :.3f}')
        
        
    valL.append(val_loss)
    trainL.append(train_loss)
    e.append(epoch)

    # Save best model
    if val_loss < best_loss:
        print("Saving model")
        torch.save(net.state_dict(), PATH)
        best_loss = val_loss
        counter = 0
        # Early stopping
    elif val_loss > best_loss:
        counter += 1
        if counter >= stop_count:
            print("Early stopping")
            break
    else:
        counter = 0

print('Training finished, took {:.4f}s'.format(time.time() - training_start_time))

# Visualize training and Loss functions
plt.figure()
plt.plot(e, valL, label = "Val loss")
plt.plot(e, trainL, label = "Train loss")
plt.xlabel("Epoch (iteration)")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("lossfunction.png") 
plt.show()
print('Finished Training\n')

#--------------------------------------------#

#---------------------- Testing ----------------------#
print('Testing \n')
net.load_state_dict(torch.load(PATH))
print('--------------- TESTING----------------')
metrics_eval(net, test_loader, device)
#--------------------------------------------#
#---------------------References-----------------------#
#Challenge: https://kelvins.esa.int/pose-estimation-2021/challenge/
#Code: 