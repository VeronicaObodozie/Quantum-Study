# Model and Data
"""
Created on Thu Oct 24 14:58:00 2024
@author: Veron
"""

# useful packages
#---------- Importing useful packages --------------#
import torch # pytorch main library
import glob
import torchvision # computer vision utilities
import torchvision.transforms as transforms # transforms used in the pre-processing of the data
from torchvision import *

from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import os
import re

import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns

# Pennylane
import pennylane as qml
from pennylane import numpy as np
# Detailing the Customized dataset, Garbage classiication model and the function to extract text, image path and labels
##-----------------------------------------------------------------------------------------------------------##

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Making the Quantum layer
@qml.qnode(dev)
def qnode(inputs, weights):
    #qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with= 0, normalize= True)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


##-----------------------------------------------------------------------------------------------------------##
#----------- Defining Model. -------------#
# Quantum Model
class satImgqModel(nn.Module):
    def __init__(self,  num_classes, input_shape, n_qubits):
        super().__init__()

        self.num_classes = num_classes
        self.input_shape = input_shape

        # Image feature extraction layer
        self.feature_extractor = resnet18(weights=ResNet18_Weights)
        # layers are frozen by using eval()
        self.feature_extractor.eval()
        # freeze params
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        n_features = self._get_conv_output(self.input_shape)
        self.image_features = nn.Linear(n_features, 16)
        self.bnorm = nn.BatchNorm1d(16)
        # self.classifier = nn.Linear(n_features, num_classes)
        # 24 layers 10 qubits
        # Quantum Classifier
        # Create weight shape
        n_layers = 6
        self.weight_shapes = {"weights": (n_layers, n_qubits)}
        # layer shape is (layers x qubits) so this example is 10 x 2

        self.classifier = qml.qnn.TorchLayer(qnode, self.weight_shapes)

# this gets the number of filters from the feature extractor output
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.feature_extractor(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
    
    # will be used during inference
    def forward(self, image):
       
    # Image feature layers
       # Image feature layers
       image_features = self.feature_extractor(image) # RessNet50 transfer learning
       image_features = image_features.view(image_features.size(0), -1)
       image_features = self.image_features(image_features)
       image_features = F.gelu(self.bnorm(image_features))
       x = self.classifier(image_features)
       x= F.log_softmax(x, dim=1)
       return x


# Classical Model
class satImgModel(nn.Module):
    def __init__(self,  num_classes, input_shape, n_qubits):
        super().__init__()

        self.num_classes = num_classes
        self.input_shape = input_shape

        # Image feature extraction layer
        self.feature_extractor = resnet18(weights=ResNet18_Weights)
        # layers are frozen by using eval()
        self.feature_extractor.eval()
        # freeze params
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        n_features = self._get_conv_output(self.input_shape)
        self.image_features = nn.Linear(n_features, 28)
        self.bnorm = nn.BatchNorm1d(28)
        self.classifier = nn.Linear(28, num_classes)

# this gets the number of filters from the feature extractor output
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.feature_extractor(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # will be used during inference
    def forward(self, image):
       
    # Image feature layers
       # Image feature layers
       image_features = self.feature_extractor(image) # RessNet50 transfer learning
       image_features = image_features.view(image_features.size(0), -1)
       image_features = self.image_features(image_features)
       image_features = F.gelu(self.bnorm(image_features))
       x = self.classifier(image_features)
       x= F.log_softmax(x, dim=1)
       return x


# Metrics
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:57:42 2024

@author: Veron
"""

# Shows the Accuracy, f1-score and confusion matrix
##-----------------------------------------------------------------------------------------------------------##
#-------------- Metrics -------------#

# F1- SCORE
def metrics_eval(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels =  data[0], data[1]

            outputs = model(inputs)
           
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # append true and predicted labels
            y_true.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())

                # calculate macro F1 score
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')

        # calculate micro F1 score 
        f1_micro = f1_score(y_true, y_pred, average='micro')
        print(f'F1-SCORE of the netwwork is given ass micro: {f1_micro}, macro: {f1_macro}')
        
        # ACCURACY
        accuracy = 100*accuracy_score(y_true, y_pred, normalize=True)
        print(f'Accuracy of the network on the test images: {accuracy} %')
        
        # #CONFUSION MATRIX
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
##-----------------------------------------------------------------------------------------------------------##

