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
from torchvision.datasets import ImageFolder

from torchvision.models import resnet18, ResNet18_Weights

import numpy as np
import sys
import time
from PIL import Image

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
data_PATH = r"Projects\dataSat"

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

##-----------------------------------------------------------------------------------------------------------##
# Set the hyperparameters
print('------- Setting Hyperparameters-----------')
# Quantum
n_qubits = 4
n_layers= 6 # # of variational layers, depth of circut
# General
batch_size = 16 # Change Batch Size o
learning_rate = 1e-3 #4
num_workers = 0
nepochs = 3 #"Use it to change iterations"
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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dataset = ImageFolder(data_PATH, transform= data_transforms)

# total dataset 5631
# train_set, val_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.7), int(len(dataset) * 0.15), int(len(dataset) * 0.15)])
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [3941, 845, 845])

# # Get the data loader for the train set
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

##-----------------------------------------------------------------------------------------------------------##

# # Visualizing a sample image from dataset
# # Create a subplot with 4x4 grid
# fig, axs = plt.subplots(4, 4, figsize=(8, 8))

# # Loop through each subplot and plot an image
# for i in range(4):
#     for j in range(4):
#         image, label = train_set[i * 4 + j]  # Get image and label
#         image_numpy = image.numpy().squeeze()    # Convert image tensor to numpy array
#         axs[i, j].imshow(image_numpy, cmap='gray')  # Plot the image
#         axs[i, j].axis('off')  # Turn off axis
#         axs[i, j].set_title(f"Label: {label}")  # Set title with label

# plt.tight_layout()  # Adjust layout
# # plt.show()  # Show plot

# #--------------------------------------------#
# #------------------ Training Parameters --------------------------#
# print('-------------------Pretrained Model Type: RESNET 50------------------------')
# resnet_input = (3, 224, 224)
# net = satImgqModel(4, resnet_input, n_qubits)
# PATH = './satImq_resNet.pth' # Path to save the best model

# net.to(device)

# criterion = nn.CrossEntropyLoss()  # Loss function
# optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)  # all params trained
# #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9)
# scheduler = ExponentialLR(optimizer, gamma=0.9)

# #--------------------------------------------#

# #---------------------- Training and Validation ----------------------#
# e = []
# trainL= []
# valL =[]
# counter = 0
# print('Traning and Validation \n')
# # Development time
# training_start_time = time.time()
# for epoch in range(nepochs):  # loop over the dataset multiple times
#     # Training Loop
#     net.train()
#     train_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data[0].to(device), data[1].to(device)
#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#     train_loss = train_loss/i
#     print(f'{epoch + 1},  train loss: {train_loss :.3f},', end = ' ')

#     scheduler.step()

# #---------------Validation----------------------------#
#     net.eval()
#     val_loss = 0.0
#     # since we're not training, we don't need to calculate the gradients for our outputs
#     with torch.no_grad():
#         for i, data in enumerate(valloader, 0):
#             # get the inputs; data is a list of [inputs, labels] 
#             inputs, labels = data[0].to(device), data[1].to(device)

#             outputs = net(inputs)
#             loss = criterion(outputs, labels)

#             val_loss += loss.item()
#         val_loss = val_loss/i
#         print(f'val loss: {val_loss :.3f}')
        
        
#     valL.append(val_loss)
#     trainL.append(train_loss)
#     e.append(epoch)

#     # Save best model
#     if val_loss < best_loss:
#         print("Saving model")
#         torch.save(net.state_dict(), PATH)
#         best_loss = val_loss
#         counter = 0
#         # Early stopping
#     elif val_loss > best_loss:
#         counter += 1
#         if counter >= stop_count:
#             print("Early stopping")
#             break
#     else:
#         counter = 0

# print('Training finished, took {:.4f}s'.format(time.time() - training_start_time))

# # Visualize training and Loss functions
# plt.figure()
# plt.plot(e, valL, label = "Val loss")
# plt.plot(e, trainL, label = "Train loss")
# plt.xlabel("Epoch (iteration)")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid()
# plt.savefig("lossfunction.png") 
# plt.show()
# print('Finished Training\n')

# #--------------------------------------------#

#---------------------- Testing ----------------------#
print('Testing \n')
resnet_input = (3, 224, 224)
PATH = './satIm_resNet.pth' # Path to save the best model
net = satImgModel(4, resnet_input, n_qubits)
net.load_state_dict(torch.load(PATH))
print('--------------- TESTING----------------')
metrics_eval(net, testloader, device)
#--------------------------------------------#
#---------------------References-----------------------#
