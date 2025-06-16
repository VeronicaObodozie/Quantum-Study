""" 
Created 23/05/2025
by Veronica 

This code important quantum circuits and layer types:
> Dataset class from dataset authors/ ESA challenge
> Quantum Model
> Classical modifications
"""
# Important Modules
from matplotlib import pyplot as plt
from random import randint
import numpy as np
import json
from PIL import Image
from datetime import datetime
import csv
import torch 
from torchvision import transforms, models 
from torch.utils.data import DataLoader 

from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
import os

# Pennylane
import pennylane as qml
from pennylane import numpy as np

""" For amplitude embedding
With 7 qubits you get 128
With 4 you get 16
6 qubits 64 """
n_qubits = 4# 7
dev = qml.device("default.qubit", wires=n_qubits)

#---------------- TEMPLATE BASED ----------------------#
# fULLY CONNECTED LAYER WITH NASIC ENTANGLEMENT

""" Angle embedding features must be length 4 or less
Amplitude embedding input size must be length 128 or less """


# Making the Quantum layer
@qml.qnode(dev)
def qnode(inputs, weights):
    #qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with= 0, normalize= True)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


# RANDOM LAYERS
@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    #qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with= 0, normalize= True)
    #qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    qml.RandomLayers(weights=weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]



# QUANVOLUTIONAL LAYERS


#----------------- CUSTOM CIRCUITS --------------------#
