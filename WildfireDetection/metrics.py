from utils import *
from tools import *
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils import data

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from torchvision import *

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, confusion_matrix

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns

def basseTest(test_dataloader, net, data_dir, interp, snapshot_dir):
        
    tbar = tqdm(test_dataloader)
    for _, batch in enumerate(tbar):  
        image, _,_,name = batch        
        image = image.float().cuda()
        
        with torch.no_grad():
            pred = net(image)
        _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy().astype('uint8')         

        patch_name = name[0].split('/')[1]
        patch_path = os.path.join(snapshot_dir, patch_name)
        np.savez_compressed(patch_path, label=pred)

    # Path to the directory containing test image patches
    image_dir = os.path.join(data_dir,"scene4")
    n_row = 21
    n_col = 24
    # Image size and patch information
    patch_size = (512, 512)
    overlap = 128
    original_image_size = (3, n_row * (patch_size[0] - overlap) + overlap, n_col * (patch_size[1] - overlap) + overlap)

    # Initialize an empty array to store the reconstructed image
    reconstructed_rgb = np.zeros(original_image_size)
    reconstructed_label = np.zeros(original_image_size[1:])
    reconstructed_pred = np.zeros(original_image_size[1:])

    # Iterate through rows and columns of patches
    for row in tqdm(range(1, n_row + 1)):  
        for col in range(1, n_col + 1): 
            # Load the patch
            patch_name = f"scene_4_patch_{row}_{col}.npz"
            patch_path = os.path.join(image_dir, patch_name)
            patch_data = np.load(patch_path)['image']
            patch_gt = np.load(patch_path)['label']
            pred_path = os.path.join(snapshot_dir, patch_name)
            patch_pred = np.load(pred_path)['label']

            # Calculate the starting indices for this patch in the original image
            start_row = (row - 1) * (patch_size[0] - overlap)
            start_col = (col - 1) * (patch_size[1] - overlap)

            # Update the reconstructed image with the patch
            reconstructed_rgb[:, start_row:start_row + patch_size[0], start_col:start_col + patch_size[1]] = patch_data[[3,2,1],:,:]
            reconstructed_label[start_row:start_row + patch_size[0], start_col:start_col + patch_size[1]] = patch_gt

            if (col==0)and(row==0):            
                reconstructed_pred[start_row:start_row + patch_size[0], start_col:start_col + patch_size[1]] = patch_pred
            elif (col==0)and(row!=0):
                reconstructed_pred[start_row+int(overlap/2):start_row + patch_size[0], start_col:start_col + patch_size[1]] = patch_pred[int(overlap/2):,:]
            elif (col!=0)and(row==0):
                reconstructed_pred[start_row:start_row + patch_size[0], start_col+int(overlap/2):start_col + patch_size[1]] = patch_pred[:,int(overlap/2):]
            elif (col!=0)and(row!=0):
                reconstructed_pred[start_row+int(overlap/2):start_row + patch_size[0], start_col+int(overlap/2):start_col + patch_size[1]] = patch_pred[int(overlap/2):,int(overlap/2):]

    # Apply ggplot style
    plt.style.use('ggplot')
    cmap = ListedColormap(['white', 'red'])

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].imshow(reconstructed_rgb.transpose(1, 2, 0)/1500.)
    axs[0].axis('off')
    axs[0].set_title('RGB image', fontsize=12)

    axs[1].imshow(reconstructed_rgb.transpose(1, 2, 0)/1500., alpha=0.6)
    axs[1].imshow(reconstructed_pred, cmap=cmap, alpha=0.7)
    axs[1].axis('off')
    axs[1].set_title('Detection', fontsize=12)

    axs[2].imshow(reconstructed_rgb.transpose(1, 2, 0)/1500., alpha=0.6)
    axs[2].imshow(reconstructed_label, cmap=cmap, alpha=0.7)
    axs[2].axis('off')
    axs[2].set_title('Label', fontsize=12)
    legend_labels = ['Non-fire', 'Fire']
    plt.legend(handles=[plt.Rectangle((0,0),1,1,facecolor=cmap(i),edgecolor='black') for i in range(2)], labels=legend_labels, fontsize=12, frameon=False, bbox_to_anchor=(1.04, 0), loc="lower left")
    plt.tight_layout()

    save_path = os.path.join(snapshot_dir, "Fig_5_map.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)


#------------------- Template Metrics ------------------------#
def metrics_eval(model, loader, interp):
    model.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            image, label,_,_ = batch
            labels = label.squeeze().numpy()
            images = image.float().cuda()
            outputs = model(images)
           
            _, predicted = torch.max(outputs.data, 1)

            _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
            pred = pred.squeeze().data.cpu().numpy()  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # append true and predicted labels
            y_true.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())

                # calculate macro F1 score
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')

        # F1 score, Precision, Recall
        print(classification_report(y_true,y_pred))

        # TP, FP, TN, FN
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        iou = tp/(tp+fp+fn) # intersection over union
        print(f'Intersection over Union is: {iou}')
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