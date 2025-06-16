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
# class PoseResNetModel(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         self.input_shape = shape
#         self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
#         # n_features = self._get_conv_output(self.input_shape)
#         self.feature_extractor.fc = nn.Identity()#nn.Linear(n_features, 128)
#         # self.bnorm = nn.BatchNorm1d(128)
#         self.pose = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 7))


#     # def _get_conv_output(self, shape):
#     #     batch_size = 1
#     #     tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

#     #     output_feat = self.feature_extractor(tmp_input)
#     #     n_size = output_feat.data.view(batch_size, -1).size(1)
#     #     return n_size

#     def forward(self, image):
#        # Image feature layers
#        image_features = self.feature_extractor(image) # RessNet50 transfer learning
#        # image_features = image_features.view(image_features.size(0), -1)
#        # image_features = self.bnorm(image_features)
#        pose = self.pose(image_features)
#        return pose



class PoseResNetModel(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.input_shape = shape
        self.initialized_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.initialized_model.fc.in_features
        self.initialized_model.fc = torch.nn.Linear(num_ftrs, 7)

    def forward(self, image):
       # Image feature layers
       pose = self.initialized_model(image) # RessNet50 transfer learning
       return pose


#--------------------------------- Quantum Model --------------------------------------#
n_qubits = 7
dev = qml.device("default.qubit", wires=n_qubits)

# Making the Quantum layer
@qml.qnode(dev)
# def qnode(inputs, weights):
#     #qml.AngleEmbedding(inputs, wires=range(n_qubits))
#     qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with= 0, normalize= True)
#     qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
#     return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Try random layer and angle emedding
# This is for Quantum 3
def qnode(inputs, weights):
    # qml.AngleEmbedding(inputs, wires=range(n_qubits)) # Angle requires only 4 inputs
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with= 0, normalize= True)
    #qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    qml.RandomLayers(weights=weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Quantum circuit 4
class PoseQuantumNew(nn.Module):
    # Quantum -> Convolutional - > Bnorm, relu, Avg pool -> FC
    def __init__(self, shape):
        super().__init__()
        self.input_shape = shape
        self.conv1 = qml.qnn.TorchLayer(qnode, self.weight_shapes) # Quantum Layer
        self.bnorm2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 128, 3, 2) # input, output, kernel, stride
        self.bnorm2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pose = nn.Linear(128, 7)

    def forward(self, image):
       # Image feature layers
       x = self.conv1(image) 
       x = self.relu(self.bnorm1(x))
       x = self.conv2(image) 
       x = self.relu(self.bnorm2(x))
       x = self.avgpool(x)
       x = torch.flatten(x, 1)
       pose = self.pose(x)
       return pose

# Quantum 3
class PoseQuanModel(nn.Module):
    def __init__(self, shape, n_qubits):
        super().__init__()
        n_layers = 3
        self.weight_shapes = {"weights": (n_layers, n_qubits)}
        self.input_shape = shape
        self.pose =qml.qnn.TorchLayer(qnode, self.weight_shapes)
        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = torch.nn.Linear(num_ftrs, 7)

        # self.feature_extractor.fc = nn.Linear(num_ftrs, 128)
        # # self.bnorm = nn.BatchNorm1d(128)
        # self.pose =qml.qnn.TorchLayer(qnode, self.weight_shapes)

    def forward(self, image):
       # Image feature layers
       image_features = self.pose(image) # RessNet18 transfer learning
       pose = self.feature_extractor(image_features)
    #    image_features = self.feature_extractor(image) # RessNet18 transfer learning
    #    image_features = image_features.view(image_features.size(0), -1)
    #    # image_features = self.bnorm(image_features)
    #    pose = self.pose(image_features)
       return pose

# Quantum circuit 1
# # Quantum Resnet
# class PoseQuanModel(nn.Module):
#     def __init__(self, shape, n_qubits):
#         super().__init__()
#         self.input_shape = shape
#         self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
#         # self.feature_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
#         # n_features = self._get_conv_output(self.input_shape)
#         n_layers = 6
#         self.weight_shapes = {"weights": (n_layers, n_qubits)}
#         self.feature_extractor.fc = nn.Linear(512, 128) # nn.Identity()#nn.Linear(n_features, 128)
#         # self.bnorm = nn.BatchNorm1d(128)
#         self.pose = nn.Sequential(
#             nn.Flatten(),
#             # nn.Linear(2048, 128),#(512, 128),
#             # nn.BatchNorm1d(128),
#             nn.ReLU(),
#             # layer shape is (layers x qubits) so this example is 10 x 2

#             qml.qnn.TorchLayer(qnode, self.weight_shapes))


#     # def _get_conv_output(self, shape):
#     #     batch_size = 1
#     #     tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

#     #     output_feat = self.feature_extractor(tmp_input)
#     #     n_size = output_feat.data.view(batch_size, -1).size(1)
#     #     return n_size

#     def forward(self, image):
#        # Image feature layers
#        image_features = self.feature_extractor(image) # RessNet50 transfer learning
#        # image_features = image_features.view(image_features.size(0), -1)
#        # image_features = self.bnorm(image_features)
#        pose = self.pose(image_features)
#        return pose


# DIDNT Work
# # Quantum Resnet
# def PoseQuanModel(shape, n_qubits):
#     feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
#     n_layers = 6
#     weight_shapes = {"weights": (n_layers, n_qubits)}
#     # self.feature_extractor.fc = nn.Identity()#nn.Linear(n_features, 128)
#     feature_extractor.fc =qml.qnn.TorchLayer(qnode, weight_shapes)
#     return feature_extractor


# New Version
# class PoseNew(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         self.input_shape = shape
#         self.conv1 = nn.Conv2d(3, 64, 3, 2) # input, output, kernel, stride
#         self.bnorm1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(64, 128, 3, 2) # input, output, kernel, stride
#         self.bnorm2 = nn.BatchNorm2d(128)
#         self.relu = nn.ReLU()
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.pose = nn.Linear(128, 7)

#     def forward(self, image):
#        # Image feature layers
#        x = self.conv1(image) 
#        x = self.relu(self.bnorm1(x))
#        x = self.conv2(x) 
#        x = self.relu(self.bnorm2(x))
#        x = self.avgpool(x)
#        x = torch.flatten(x, 1)
#        pose = self.pose(x)
#        return pose


class PoseNew(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.input_shape = shape
        # self.conv1 = nn.Conv2d(3, 64, 3, 2) # input, output, kernel, stride
        # self.bnorm1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 128, 3, 2) # input, output, kernel, stride
        self.bnorm2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.pose = nn.Linear(128, 7)
        self.weight_shapes = {"weights": (2, 7)}

        self.pose = qml.qnn.TorchLayer(qnode, self.weight_shapes)

    def forward(self, image):
       # Image feature layers
    #    x = self.conv1(image) 
    #    x = self.relu(self.bnorm1(x))
       x = self.conv2(image) 
       x = self.relu(self.bnorm2(x))
       x = self.avgpool(x)
       x = torch.flatten(x, 1)
       pose = self.pose(x)
       return pose

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # Input: [batch_size, 1, 28, 28]
        self.conv1 = nn.Conv2d(1, 32, 3)  # Output: [batch_size, 32, 26, 26]
        
        # Input: [batch_size, 32, 26, 26]
        self.conv2 = nn.Conv2d(32, 64, 3) # Output: [batch_size, 64, 11, 11]
        
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Flattening: [batch_size, 64*5*5]
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Input: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))
        # Shape: [batch_size, 32, 26, 26]
        x = F.max_pool2d(x, 2)
        # Shape: [batch_size, 32, 13, 13]
        
        x = F.relu(self.conv2(x))
        # Shape: [batch_size, 64, 11, 11]
        x = F.max_pool2d(x, 2)
        # Shape: [batch_size, 64, 5, 5]
        
        x = x.view(-1, 64 * 5 * 5) # Flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
# New Version

# Try Unet
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class unet(nn.Module):
    def __init__(self, n_classes, n_channels=13, bilinear=True):
        super(unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits