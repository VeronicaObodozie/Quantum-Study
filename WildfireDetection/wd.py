""" 
Created by Veronica Obodozie
Base AI Dev/Test Code
"""
#-----------------Importing Important Functions and Modules---------------------------#
#Functions
from utils import *
from tools import *
from metrics import *
import torch 
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
import numpy as np
import sys
import time
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import csv

from sklearn.metrics import mean_squared_error, accuracy_score
import argparse
import os
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm


# Pennylane
import pennylane as qml
from pennylane import numpy as np
#-------------------------------------------------------------------------------#


#---------------------------------Data Preprocessing and loading----------------------------------------------#
def data_load(batch_size, num_workers, mode, num_steps_stop):
    # Processing to match networks
    print('------- DATA PROCESSING --------')
    data_transforms = v2.Compose([
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
        v2.Resize(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Important directories
    data_dir = './Sen2Fire/'
    train_list = './train.txt'
    val_list = './val.txt'
    test_list = './test.txt'

    # Loading training set, using 20% for validation
    train_set = Sen2FireDataSet(data_dir, train_list, max_iters=num_steps_stop*batch_size,mode=mode) #(data_dir, train_list,data_transforms, mode=mode)
    # data_dir, train_list, max_iters=num_steps_stop*batch_size,mode=mode
    val_set = Sen2FireDataSet(data_dir, val_list, max_iters=num_steps_stop*batch_size,mode=mode) #(data_dir, val_list,data_transforms, mode=mode)
    test_set = Sen2FireDataSet(data_dir, train_list, max_iters=num_steps_stop*batch_size,mode=mode) #(data_dir, test_list,data_transforms, mode=mode)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_set,batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_set, train_dataloader, val_dataloader, test_dataloader
#-------------------------------------------------------------------------------#

#--------------------------------- DEVELOPMENT ----------------------------------------------#
def devTemplate(nepochs, net, device, optimizer, criterion, scheduler, interp, train_dataloader, stop_count,  val_dataloader, PATH):
    #---------------------- Training and Validation ----------------------#
    e = []
    trainL= []
    valL =[]
    counter = 0
    F1_best = 0. 
    
    print('Traning and Validation \n')
    # Development time
    training_start_time = time.time()
    for epoch in range(nepochs):  # loop over the dataset multiple times
        # Training Loop
        
        train_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            net.train()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device).long()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)

            #------------------Specific-------------------#
            out_interp = interp(outputs)
            loss = criterion(out_interp, labels) # criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/i
        print(f'{epoch + 1},  train loss: {train_loss :.3f},', end = ' ')
        scheduler.step()
    #---------------Validation----------------------------#
        net.eval()
        val_loss = 0.0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels] 
                inputs, labels = data[0].to(device), data[1].to(device).long()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_loss = val_loss/i
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
    # Total number of parameters
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f'Total Number of Parameters: {pytorch_total_params}')
    # Visualize training and Loss functions
    plt.figure()
    plt.plot(e, valL, label = "Val loss")
    plt.plot(e, trainL, label = "Train loss")
    plt.xlabel("Epoch (iteration)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("lossfunctionunet.png") 
    plt.show()
    print('Finished Training\n')
    #--------------------------------------------#

#-------------------------------------------------------------------------------#


#--------------------------------- DEVELOPMENT and Testingbase code----------------------------------------------#
def base(net, weight, f, device, optimizer, learning_rate, interp, num_classes, train_dataloader, epsilon, name_classes, num_steps_stop, snapshot_dir,num_steps, test_dataloader, val_dataloader):
    #---------------------- Training and Validation ----------------------#
    hist = np.zeros((num_steps_stop,4))
    F1_best = 0.    
    #L_seg = nn.CrossEntropyLoss()
    class_weights = [1, weight]
    L_seg = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device) )
    for batch_index, train_data in enumerate(train_dataloader):
        if batch_index==num_steps_stop:
            break
        tem_time = time.time()
        adjust_learning_rate(optimizer,learning_rate,batch_index,num_steps)
        net.train()
        optimizer.zero_grad()
        
        patches, labels, _, _ = train_data
        patches = patches.to(device)      
        labels = labels.to(device).long()

        pred = net(patches)           
        pred_interp = interp(pred)
              
        # Segmentation Loss
        L_seg_value = L_seg(pred_interp, labels)
        _, predict_labels = torch.max(pred_interp, 1)
        lbl_pred = predict_labels.detach().cpu().numpy()
        lbl_true = labels.detach().cpu().numpy()
        metrics_batch = []
        for lt, lp in zip(lbl_true, lbl_pred):
            _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=num_classes)
            metrics_batch.append(mean_iu)                
        batch_miou = np.nanmean(metrics_batch, axis=0)  
        batch_oa = np.sum(lbl_pred==lbl_true)*1./len(lbl_true.reshape(-1))
                    
        hist[batch_index,0] = L_seg_value.item()
        hist[batch_index,1] = batch_oa
        hist[batch_index,2] = batch_miou
        
        L_seg_value.backward()
        optimizer.step()

        hist[batch_index,-1] = time.time() - tem_time

        if (batch_index+1) % 10 == 0: 
            print('Iter %d/%d Time: %.2f Batch_OA = %.2f Batch_mIoU = %.2f CE_loss = %.3f'%(batch_index+1,num_steps,10*np.mean(hist[batch_index-9:batch_index+1,-1]),np.mean(hist[batch_index-9:batch_index+1,1])*100,np.mean(hist[batch_index-9:batch_index+1,2])*100,np.mean(hist[batch_index-9:batch_index+1,0])))
            f.write('Iter %d/%d Time: %.2f Batch_OA = %.2f Batch_mIoU = %.2f CE_loss = %.3f\n'%(batch_index+1,num_steps,10*np.mean(hist[batch_index-9:batch_index+1,-1]),np.mean(hist[batch_index-9:batch_index+1,1])*100,np.mean(hist[batch_index-9:batch_index+1,2])*100,np.mean(hist[batch_index-9:batch_index+1,0])))
            f.flush() 

        # evaluation per 500 iterations
        if (batch_index+1) % 500 == 0:            
            print('Validating..........')  
            f.write('Validating..........\n')  

            net.eval()
            TP_all = np.zeros((num_classes, 1))
            FP_all = np.zeros((num_classes, 1))
            TN_all = np.zeros((num_classes, 1))
            FN_all = np.zeros((num_classes, 1))
            n_valid_sample_all = 0
            F1 = np.zeros((num_classes, 1))
            IoU = np.zeros((num_classes, 1))
            
            tbar = tqdm(val_dataloader)
            for _, batch in enumerate(tbar):  
                image, label,_,_ = batch
                label = label.squeeze().numpy()
                image = image.float().cuda()
                
                with torch.no_grad():
                    pred = net(image)

                _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
                pred = pred.squeeze().data.cpu().numpy()                       
                               
                TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1),label.reshape(-1),num_classes)
                TP_all += TP
                FP_all += FP
                TN_all += TN
                FN_all += FN
                n_valid_sample_all += n_valid_sample

            OA = np.sum(TP_all)*1.0 / n_valid_sample_all
            for i in range(num_classes):
                P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
                R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
                F1[i] = 2.0*P*R / (P + R + epsilon)
                IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)
            
                if i==1:
                    print('===>' + name_classes[i] + ' Precision: %.2f'%(P * 100))
                    print('===>' + name_classes[i] + ' Recall: %.2f'%(R * 100))            
                    print('===>' + name_classes[i] + ' IoU: %.2f'%(IoU[i] * 100))              
                    print('===>' + name_classes[i] + ' F1: %.2f'%(F1[i] * 100))   
                    f.write('===>' + name_classes[i] + ' Precision: %.2f\n'%(P * 100))
                    f.write('===>' + name_classes[i] + ' Recall: %.2f\n'%(R * 100))            
                    f.write('===>' + name_classes[i] + ' IoU: %.2f\n'%(IoU[i] * 100))              
                    f.write('===>' + name_classes[i] + ' F1: %.2f\n'%(F1[i] * 100))   
                
            mF1 = np.mean(F1)   
            mIoU = np.mean(F1)           
            print('===> mIoU: %.2f mean F1: %.2f OA: %.2f'%(mIoU*100,mF1*100,OA*100))
            f.write('===> mIoU: %.2f mean F1: %.2f OA: %.2f\n'%(mIoU*100,mF1*100,OA*100))
                
            if F1[1]>F1_best:
                F1_best = F1[1]
                # save the models        
                print('Save Model')   
                f.write('Save Model\n')   
                model_name = 'best_model.pth'
                torch.save(net.state_dict(), os.path.join(snapshot_dir, model_name))
    
    saved_state_dict = torch.load(os.path.join(snapshot_dir, model_name))  
    net.load_state_dict(saved_state_dict)

    print('Testing..........')  
    f.write('Testing..........\n')  

    net.eval()
    TP_all = np.zeros((num_classes, 1))
    FP_all = np.zeros((num_classes, 1))
    TN_all = np.zeros((num_classes, 1))
    FN_all = np.zeros((num_classes, 1))
    n_valid_sample_all = 0
    F1 = np.zeros((num_classes, 1))
    IoU = np.zeros((num_classes, 1))
    
    tbar = tqdm(test_dataloader)
    for _, batch in enumerate(tbar):  
        image, label,_,_ = batch
        label = label.squeeze().numpy()
        image = image.float().cuda()
        
        with torch.no_grad():
            pred = net(image)

        _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy()                       
                        
        TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1),label.reshape(-1),num_classes)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample

    OA = np.sum(TP_all)*1.0 / n_valid_sample_all
    for i in range(num_classes):
        P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
        R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
        F1[i] = 2.0*P*R / (P + R + epsilon)
        IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)
    
        if i==1:
            print('===>' + name_classes[i] + ' Precision: %.2f'%(P * 100))
            print('===>' + name_classes[i] + ' Recall: %.2f'%(R * 100))            
            print('===>' + name_classes[i] + ' IoU: %.2f'%(IoU[i] * 100))              
            print('===>' + name_classes[i] + ' F1: %.2f'%(F1[i] * 100))   
            f.write('===>' + name_classes[i] + ' Precision: %.2f\n'%(P * 100))
            f.write('===>' + name_classes[i] + ' Recall: %.2f\n'%(R * 100))            
            f.write('===>' + name_classes[i] + ' IoU: %.2f\n'%(IoU[i] * 100))              
            f.write('===>' + name_classes[i] + ' F1: %.2f\n'%(F1[i] * 100))   
        
    mF1 = np.mean(F1)   
    mIoU = np.mean(F1)           
    print('===> mIoU: %.2f mean F1: %.2f OA: %.2f'%(mIoU*100,mF1*100,OA*100))
    f.write('===> mIoU: %.2f mean F1: %.2f OA: %.2f\n'%(mIoU*100,mF1*100,OA*100))        
    f.close()
    saved_state_dict = torch.load(os.path.join(snapshot_dir, model_name))  
    np.savez(snapshot_dir+'Precision_'+str(int(P * 10000))+'Recall_'+str(int(R * 10000))+'F1_'+str(int(F1[1] * 10000))+'_hist.npz',hist=hist) 
    
    #--------------------------------------------#

#-------------------------------------------------------------------------------#

#-------------------------------- MAIN FUNCTION -----------------------------------------------#
def main():
    # Check if GPU is available
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    print(dev)
    #--------------------------------------------#
    # Set the hyperparameters
    print('------- Setting Hyperparameters-----------')
    batch_size = 4 # Change Batch Size o
    learning_rate = 1e-3 #4
    num_workers =2#4
    nepochs = 5 #"Use it to change iterations"
    weight_decay = 5e-4 #1e-4
    best_loss = 1e+20 # number gotten from initial resnet18 run
    stop_count = 7
    print(f'batch_size = {batch_size}, learning_rate = {learning_rate} num_workers = {num_workers} , nepochs = {nepochs} , best_loss = {best_loss}, weight_decay={weight_decay}')

    # From Baseline
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
    data_dir = './Sen2Fire/'
    num_classes = 2
    mode = 5
    num_steps =5000
    num_steps_stop= 5000
    epsilon = 1e-14
    weight = 10
    snapshot_dir ='./Exp/'
    snapshot_dir = snapshot_dir+'input_'+modename[mode]+'/weight_'+str(weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)
    f = open(snapshot_dir+'Training_log.txt', 'w')

    # Call Data Loader
    train_set, train_dataloader, val_dataloader, test_dataloader = data_load(batch_size, num_workers, mode, num_steps_stop)
    name_classes = np.array(['non-fire','fire'], dtype=str)

    #------------------ Training Parameters --------------------------#
    print('-------------------Pretrained Model Type: RESNET 50------------------------')
    input_size_train = (512, 512)
    
    # Create network
    if mode == 0:
        net = unet(n_classes=num_classes, n_channels=12)
    elif mode == 1:
        net = unet(n_classes=num_classes, n_channels=13)
    elif mode == 2 or mode == 4 or mode == 6 or mode == 8:
        net = unet(n_classes=num_classes, n_channels=3)
    elif mode == 3 or mode == 5 or mode == 7 or mode == 9:       
        net = unet(n_classes=num_classes, n_channels=4)
    elif mode == 10:       
        net = unet(n_classes=num_classes, n_channels=6)
    elif mode == 11:       
        net = unet(n_classes=num_classes, n_channels=7)
    # net = unet(input_size_train)
    PATH = './wd_unet.pth' # Path to save the best model
    net.to(device)

    # # visualize class distribution
    # class_labels, class_counts = np.unique(train_set.labels, return_counts=True)
    # # Calculating Class weights
    # train_set_size = int(len(train_set))
    # class_weights = []
    # for count in class_counts:
    #     weight = 1 / (count / train_set_size)
    #     class_weights.append(weight)

    class_weights = [1, weight]
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9)

    # interpolation for the probability maps and labels 
    interp = nn.Upsample(size=(input_size_train[1], input_size_train[0]), mode='bilinear')
    # L_seg = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))

    # Template
    # Training
    devTemplate(nepochs, net, device, optimizer, criterion, scheduler, interp, train_dataloader, stop_count,  val_dataloader, PATH)
    print('Testing \n')
    net.load_state_dict(torch.load(PATH))
    metrics_eval(net, test_dataloader, interp)

    # Base Code
    # Training
        # Create network
    if mode == 0:
        net = unet(n_classes=num_classes, n_channels=12)
    elif mode == 1:
        net = unet(n_classes=num_classes, n_channels=13)
    elif mode == 2 or mode == 4 or mode == 6 or mode == 8:
        net = unet(n_classes=num_classes, n_channels=3)
    elif mode == 3 or mode == 5 or mode == 7 or mode == 9:       
        net = unet(n_classes=num_classes, n_channels=4)
    elif mode == 10:       
        net = unet(n_classes=num_classes, n_channels=6)
    elif mode == 11:       
        net = unet(n_classes=num_classes, n_channels=7)
    net.to(device)
    base(net, weight, f, device, optimizer, learning_rate, interp, num_classes, train_dataloader, epsilon, name_classes, num_steps_stop, snapshot_dir,num_steps, test_dataloader, val_dataloader)
    #---------------------- Testing ----------------------#
    print('Testing \n')
    saved_state_dict = torch.load(os.path.join(snapshot_dir, 'best_model.pth'))  
    net.load_state_dict(saved_state_dict)
    basseTest(test_dataloader, net, data_dir, interp, snapshot_dir)

    # Base
#-------------------------------------------------------------------------------#

def init_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

if __name__ == '__main__':
    main()
#----------------------------------------- REFERENCES ---------------------------------#

#-------------------------------------------------------------------------------#