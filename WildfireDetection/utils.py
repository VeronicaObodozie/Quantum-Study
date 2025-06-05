""" This shows important utility files
> The dataset class based on the baseline code from dataset authours
> The model
 """
import torch
from torch import nn
import torch.nn.functional as F
import os.path as osp
import numpy as np
from torch.utils import data
from tqdm import tqdm

#-------------------------------------------------------------------------------#
# Mmodel

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

#-------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------#
# Dataset


modename = ['all_bands',                        #0
            'all_bands_aerosol',                #1
            'rgb',                              #2
            'rgb_aerosol',                      #3
            'swir',                             #4
            'swir_aerosol',                     #5
            'nbr',                              #6
            'nbr_aerosol',                      #7   
            'ndvi',                             #8
            'ndvi_aerosol',                     #9 
            'rgb_swir_nbr_ndvi',                #10
            'rgb_swir_nbr_ndvi_aerosol',]       #11

Max_values = np.array([4.58500000e+03, 8.96800000e+03, 1.03440000e+04, 1.01840000e+04,
                      1.67280000e+04, 1.65260000e+04, 1.63650000e+04, 1.31360000e+04,
                      8.61900000e+03, 6.21700000e+03, 1.56140000e+04, 1.55500000e+04,
                      7.12807226e+00])
                      
def create_rgb_composite(patch_data):
    rgb_patch = patch_data[[3, 2, 1], :, :]
    return rgb_patch

def create_rgb_aerosol_composite(patch_data):
    rgb_aerosol_patch = patch_data[[3, 2, 1,-1], :, :]
    return rgb_aerosol_patch

def create_swir_composite(patch_data):
    swir_composite = patch_data[[11, 7, 3], :, :]
    return swir_composite

def create_swir_aerosol_composite(patch_data):
    swir_aerosol_composite = patch_data[[11, 7, 3,-1], :, :]
    return swir_aerosol_composite

def create_nbr_composite(patch_data):
    nbr = (patch_data[7, :, :] - patch_data[11, :, :]) / (patch_data[7, :, :] + patch_data[11, :, :])
    nbr_composite = np.stack((nbr, patch_data[3, :, :], patch_data[2, :, :]), axis=0)
    return nbr_composite

def create_nbr_aerosol_composite(patch_data):
    nbr = (patch_data[7, :, :] - patch_data[11, :, :]) / (patch_data[7, :, :] + patch_data[11, :, :])
    nbr_aerosol_composite = np.stack((nbr, patch_data[3, :, :], patch_data[2, :, :], patch_data[-1, :, :]), axis=0)
    return nbr_aerosol_composite

def create_ndvi_composite(patch_data):
    ndvi = (patch_data[7, :, :] - patch_data[3, :, :]) / (patch_data[7, :, :] + patch_data[3, :, :])
    ndvi_composite = np.stack((ndvi, patch_data[3, :, :], patch_data[2, :, :]), axis=0)
    return ndvi_composite

def create_ndvi_aerosol_composite(patch_data):
    ndvi = (patch_data[7, :, :] - patch_data[3, :, :]) / (patch_data[7, :, :] + patch_data[3, :, :])
    ndvi_aerosol_composite = np.stack((ndvi, patch_data[3, :, :], patch_data[2, :, :], patch_data[-1, :, :]), axis=0)
    return ndvi_aerosol_composite

def create_rgb_swir_nbr_ndvi_composite(patch_data):
    nbr = (patch_data[7, :, :] - patch_data[11, :, :]) / (patch_data[7, :, :] + patch_data[11, :, :])
    ndvi = (patch_data[7, :, :] - patch_data[3, :, :]) / (patch_data[7, :, :] + patch_data[3, :, :])
    rgb_swir_nbr_ndvi_composite = np.stack((patch_data[3, :, :], patch_data[2, :, :], patch_data[1, :, :], patch_data[11, :, :], nbr, ndvi), axis=0)
    return rgb_swir_nbr_ndvi_composite

def create_rgb_swir_nbr_ndvi_aerosol_composite(patch_data):
    nbr = (patch_data[7, :, :] - patch_data[11, :, :]) / (patch_data[7, :, :] + patch_data[11, :, :])
    ndvi = (patch_data[7, :, :] - patch_data[3, :, :]) / (patch_data[7, :, :] + patch_data[3, :, :])
    rgb_swir_nbr_ndvi_aerosol_composite = np.stack((patch_data[3, :, :], patch_data[2, :, :], patch_data[1, :, :], patch_data[11, :, :], nbr, ndvi, patch_data[-1, :, :]), axis=0)
    return rgb_swir_nbr_ndvi_aerosol_composite

class Sen2FireDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, mode=1):
        self.root = root
        self.list_path = list_path
        self.mode = mode
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []

        for name in self.img_ids:
            patch = osp.join(self.root, name)
            self.files.append({
                "patch": patch,
                "name": name
            })
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
    
        image = np.load(datafiles["patch"])['image'].astype(np.float32) / 10000.
        aerosol = np.load(datafiles["patch"])['aerosol']
        label = np.load(datafiles["patch"])['label']
        name = datafiles["name"]
        
        if self.mode == 0:            
            return image, label, np.array(image.shape), name
        else:
            data = np.concatenate([image, aerosol[np.newaxis,...]], axis=0)
            # Normalize the data using the provided Max_values
            # data /= Max_values[:, np.newaxis, np.newaxis]
            if self.mode == 2:             
                data = create_rgb_composite(data)
            elif self.mode == 3:      
                data = create_rgb_aerosol_composite(data)
            elif self.mode == 4:      
                data = create_swir_composite(data)
            elif self.mode == 5:      
                data = create_swir_aerosol_composite(data)
            elif self.mode == 6:      
                data = create_nbr_composite(data)
            elif self.mode == 7:      
                data = create_nbr_aerosol_composite(data)
            elif self.mode == 8:      
                data = create_ndvi_composite(data)
            elif self.mode == 9:      
                data = create_ndvi_aerosol_composite(data)
            elif self.mode == 10:      
                data = create_rgb_swir_nbr_ndvi_composite(data)
            elif self.mode == 11:      
                data = create_rgb_swir_nbr_ndvi_aerosol_composite(data)
        return data, label, np.array(data.shape), name
    
    def compute_max_values(self):
        max_values = np.zeros((13,))  # 12 bands + 1 aerosol band

        for index in tqdm(range(len(self))):
            datafiles = self.files[index]
            image = np.load(datafiles["patch"])['image'].astype(np.float32)
            aerosol = np.load(datafiles["patch"])['aerosol']

            # Update the max values for each band
            max_values[:12] = np.maximum(max_values[:12], image.max(axis=(1, 2)))
            max_values[12] = max_values[12] if aerosol.max() < max_values[12] else aerosol.max()

        return max_values
    
if __name__ == '__main__':
        
    root_path = "/Sen2Fire/"
    list_file_path = "train.txt"
    train_dataset = Sen2FireDataSet(root=root_path, list_path=list_file_path, max_iters=None, mode=1)
    max_values = train_dataset.compute_max_values()

    # Print the computed max values
    print("Max Values:")
    print(max_values)

#-------------------------------------------------------------------------------#