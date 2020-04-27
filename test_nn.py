# %load run_ben_net_down.py
import subprocess
import os
import random
from PIL import Image
import numpy as np
import pandas as pd
#for image transform
import cv2

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import random
import time

from model_loader_CP2 import *
#from CP_helper import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from data_helper import UnlabeledDataset, LabeledDataset
from data_helper_triangle_down import TriangleLabeledDataset,image_names, get_mask_name, load_mask
from helper import collate_fn, draw_box
import argparse
import datetime

def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def gen_train_val_index(labeled_scene_index):
    breakpt = len(labeled_scene_index)//3
    labeled_scene_index_shuf = labeled_scene_index
    random.shuffle(labeled_scene_index_shuf)

    train_labeled_scene_index = labeled_scene_index_shuf[:-breakpt]
    val_labeled_scene_index = labeled_scene_index_shuf[-breakpt: ]
    return train_labeled_scene_index, val_labeled_scene_index

def save_file_to_cloud(filename):
    current_dir = os.getcwd()
    path = os.path.join(current_dir,"bucket_upload.sh")
    if cloud:
        subprocess.run(f"{path} {filename}",shell=True)
    return filename

def download_file(filename,cloud_filename):
    current_dir = os.getcwd()
    path = os.path.join(current_dir,"download_bucket.sh")
    command = f"{path} {cloud_filename} {filename}"
    print(f"running {command}")
    if cloud_filename:
        subprocess.run(command,shell=True)
    print(f"loading {filename}")
    return filename

def save_cam_model(item,filename,cloud = True):
    torch.save(item, filename)
    if cloud:
        save_file_to_cloud(filename)
    return filename



    
def load_model(filename,cloud_filename=None):
    if cloud_filename:
        download_file(filename,cloud_filename)
    return torch.load(filename)

def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
     
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i ,(sample, target, road_image, extra, road_image_mod) in enumerate(loader):
             
            sample_ = torch.stack(sample,0).cuda() #should be [batch size,3, h,w]
            
            labels = torch.stack(road_image_mod, 0).cuda()
            
            
            outputs = model(sample_)
            predicted = (outputs>0.5).int() ## convert to bineary
            
            total += (labels.size(0)*labels.size(1))
            correct += predicted.eq(labels.int()).sum().item()
        
    return (100 * correct / total)




def train(feat_extractor, **train_kwargs):
    for cam in image_names: #let's try just front camera
        print("training {}".format(cam))
        #make camera specific train loader
        labeled_trainset = training_tools[cam][1]
        train_loader = torch.utils.data.DataLoader(labeled_trainset , batch_size=train_kwargs["batch"], 
                                                  shuffle=True, num_workers=2, collate_fn=collate_fn)
        labeled_valset = training_tools[cam][2]
        val_loader = torch.utils.data.DataLoader(labeled_valset , batch_size=train_kwargs["batch"], 
                                                  shuffle=True, num_workers=2, collate_fn=collate_fn)



        output_layer = training_tools[cam][0] #output the layer

    
        #make camera spcific model
        model = nn.Sequential(feat_extractor, output_layer, nn.Sigmoid()).cuda()

        criterion = torch.nn.BCELoss(reduction = 'sum') #trying summation
        param_list = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(param_list, lr=train_kwargs["lr"], eps=train_kwargs["eps"])
        train_losses = []
        val_accs = []

        model.train()
        for e in range(train_kwargs["epochs"]):
            t = time.process_time()

            for i ,(sample, target, road_image, extra, road_image_mod) in enumerate(train_loader):
                sample_ = torch.stack(sample,0).cuda() #should be [batch size,3, h,w]
                labels = torch.stack(road_image_mod, 0).cuda() #should be [batch size, cropsize]

                optimizer.zero_grad()
                outputs = model(sample_) 

                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                # validate every 200 iterations
                if i > 0 and i % 100== 0:
                    val_acc = test_model(val_loader, model) #calls model.eval()
                    val_accs.append(val_acc)
                    #do some stuff
                    elapsed_time = time.process_time() - t
                    print('Epoch: [{}], Step: [{}], Train Loss {:.4f}, Validation Acc: {:.4f}, time {:.4f}'.format( 
                               e+1, i+1, loss,  val_acc, elapsed_time))
                    model.train() #go back to training
                    t = time.process_time()
        #save model
        print("save camera model") 
        item = {

            'model_state_dict': model.state_dict(),
            'feat_extractor_state_dict':  feat_extractor.state_dict(),
            'output_layer_state_dict': output_layer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_accs': val_accs
            }
        filename = "./models/resnet_1"+cam[:-5]+".pt"
    
    
