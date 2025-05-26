""" 
Created 23/05/2025
by Veronica 

This code contains important utility classes:
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

#------------------- Utility Class and Functions -------------------------#
# All are pytorch dunctions frm the speedplus utils
# https://gitlab.com/EuropeanSpaceAgency/speedplus-utils/-/blob/main/utils.py?ref_type=heads
# dataset
class PyTorchSatellitePoseEstimationDataset(Dataset):
    """ SPEED dataset that can be used with DataLoader for PyTorch training. """
    def __init__(self, split, speed_root, transform=None):
        if split in {'train', 'validation'}:
            self.image_root = os.path.join(speed_root, 'synthetic', 'images')
            with open(os.path.join(speed_root, "synthetic", split + '.json'), 'r') as f:
                label_list = json.load(f)
        else:
            self.image_root = os.path.join(speed_root, split, 'images')
            with open(os.path.join(speed_root, split, 'test.json'), 'r') as f:
                label_list = json.load(f)
        self.sample_ids = [label['filename'] for label in label_list]
        self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']}
            for label in label_list}
        self.split = split
        self.transform = transform
    def __len__(self):
        return len(self.sample_ids)
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_name = os.path.join(self.image_root, sample_id)
        # note: despite grayscale images, we are converting to 3 channels here,
        # since most pre-trained networks expect 3 channel input
        pil_image = Image.open(img_name).convert('RGB')
        q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
        y = np.concatenate([q, r])
        if self.transform is not None:
            torch_image = self.transform(pil_image)
        else:
            torch_image = pil_image
        return torch_image, y
#--------------------------------------------#

#------------------- Model -------------------------#
# Define the model
class PoseResNetModel(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.input_shape = shape
        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        # n_features = self._get_conv_output(self.input_shape)
        self.feature_extractor.fc = nn.Identity()#nn.Linear(n_features, 128)
        # self.bnorm = nn.BatchNorm1d(128)
        self.pose = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 7))


    # def _get_conv_output(self, shape):
    #     batch_size = 1
    #     tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

    #     output_feat = self.feature_extractor(tmp_input)
    #     n_size = output_feat.data.view(batch_size, -1).size(1)
    #     return n_size

    def forward(self, image):
       # Image feature layers
       image_features = self.feature_extractor(image) # RessNet50 transfer learning
       # image_features = image_features.view(image_features.size(0), -1)
       # image_features = self.bnorm(image_features)
       pose = self.pose(image_features)
       return pose



#--------------------------------- Quantum Model --------------------------------------#
n_qubits = 7
dev = qml.device("default.qubit", wires=n_qubits)

# Making the Quantum layer
@qml.qnode(dev)
def qnode(inputs, weights):
    #qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with= 0, normalize= True)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Quantum Resnet
class PoseQuanModel(nn.Module):
    def __init__(self, shape, n_qubits):
        super().__init__()
        self.input_shape = shape
        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        # self.feature_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
        # n_features = self._get_conv_output(self.input_shape)
        n_layers = 6
        self.weight_shapes = {"weights": (n_layers, n_qubits)}
        self.feature_extractor.fc = nn.Linear(512, 128) # nn.Identity()#nn.Linear(n_features, 128)
        # self.bnorm = nn.BatchNorm1d(128)
        self.pose = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(2048, 128),#(512, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            # layer shape is (layers x qubits) so this example is 10 x 2

            qml.qnn.TorchLayer(qnode, self.weight_shapes))


    # def _get_conv_output(self, shape):
    #     batch_size = 1
    #     tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

    #     output_feat = self.feature_extractor(tmp_input)
    #     n_size = output_feat.data.view(batch_size, -1).size(1)
    #     return n_size

    def forward(self, image):
       # Image feature layers
       image_features = self.feature_extractor(image) # RessNet50 transfer learning
       # image_features = image_features.view(image_features.size(0), -1)
       # image_features = self.bnorm(image_features)
       pose = self.pose(image_features)
       return pose

# # Quantum Resnet
# def PoseQuanModel(shape, n_qubits):
#     feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
#     n_layers = 6
#     weight_shapes = {"weights": (n_layers, n_qubits)}
#     # self.feature_extractor.fc = nn.Identity()#nn.Linear(n_features, 128)
#     feature_extractor.fc =qml.qnn.TorchLayer(qnode, weight_shapes)
#     return feature_extractor


# # New Version
# class PoseNew(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         self.input_shape = shape
#         self.feature_extractor = efficientnet_b7(weights='DEFAULT')
#         self.bnorm = nn.BatchNorm1d(1000)
#         self.pose = nn.Linear(1000, 7)

#     def forward(self, image, val):
#        # Image feature layers
#        image_features = self.feature_extractor(image) # Efficientnet transfer learning
#        image_features = self.bnorm(image_features)
#        pose = self.pose(F.gelu(image_features))
#        return pose

# New Version
