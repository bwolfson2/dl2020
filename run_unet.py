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
from Unet import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box



def gen_train_val_loader(labeled_scene_index, **kwargs):
    labeled_scene_index_shuf = labeled_scene_index
    random.shuffle(labeled_scene_index_shuf)
    train_labeled_scene_index = labeled_scene_index_shuf[:-10] #hard code we know there are only 28 scenes that are labeled
    val_labeled_scene_index = labeled_scene_index_shuf[-10:]
    
    print(len(train_labeled_scene_index), len(val_labeled_scene_index))
    print(train_labeled_scene_index[0], val_labeled_scene_index[0])
        
    loadkwargs = {'batch_size': 2,
    'shuffle': True,
    'collate_fn':collate_fn,
    'num_workers':2,
    
    }
    
    labeled_trainset = LabeledDataset(scene_index=train_labeled_scene_index, **kwargs)
    print(len(labeled_trainset))
    trainloader = torch.utils.data.DataLoader(labeled_trainset, **loadkwargs)
    
    labeled_valset = LabeledDataset(scene_index=val_labeled_scene_index, **kwargs)  
    print(len(labeled_valset))
    valloader = torch.utils.data.DataLoader(labeled_valset, **loadkwargs)
    
    result={"train" : trainloader,
           "val": valloader
        
    }
    return result

def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
     
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i ,(sample, target, road_image, extra) in enumerate(loader):
             
            sample_ = ModelLoader.get_binary_road_map(sample).cuda() #should be [batch size, 800, 800]
            labels = torch.stack(road_image, 0).cuda() #should be [batch size, 800, 800]
            
            
            outputs = model(sample_.unsqueeze(1))
            outputs = outputs.squeeze(1)
            predicted = (outputs>0.5).int() ## convert to bineary
            
            total += (labels.size(0)*labels.size(1)*labels.size(2))
            correct += predicted.eq(labels.int()).sum().item()
        
    return (100 * correct / total)
     


def train(train_val_loader, **train_kwargs):
    #initialize stuff...
    train_loader = train_val_loader["train"]
    val_loader = train_val_loader["val"]
    
    unet = UNet(in_channel=1,out_channel=1).cuda()
    criterion = torch.nn.BCELoss()
    param_list = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(param_list, lr = train_kwargs["lr"], momentum=train_kwargs["momentum"])
    train_losses = []
    val_accs = []
    
    unet.train()
    for e in range(train_kwargs["epochs"]):
        t = time.process_time()

        for i ,(sample, target, road_image, extra) in enumerate(train_loader):
            sample_ = ModelLoader.get_binary_road_map(sample).cuda() #should be [batch size, 800, 800]
            labels = torch.stack(road_image, 0).cuda() #should be [batch size, 800, 800]
            
            optimizer.zero_grad()
            outputs = unet(sample_.unsqueeze(1)) #unet needs the channels dimension
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            # validate every 200 iterations
            if i > 0 and i % 100== 0:
                val_acc = test_model(val_loader, unet) #calls model.eval()
                val_accs.append(val_acc)
                #do some stuff
                elapsed_time = time.process_time() - t
                print('Epoch: [{}], Step: [{}], Train Loss {:.4f}, Validation Acc: {:.4f}, time {:.4f}'.format( 
                           e+1, i+1, loss,  val_acc, elapsed_time))
                unet.train() #go back to training
                t = time.process_time()
    #save model
    torch.save(unet.state_dict(), "./models/unet_2.pt")
    
    

if __name__ == "__main__":
    print("this is a new test")
    
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0);


    image_folder = 'data'
    annotation_csv = 'data/annotation.csv'

    
    unlabeled_scene_index = np.arange(106)
    labeled_scene_index = np.arange(106, 134)
    
    
    kwargs = {
    #'first_dim': 'sample',
    'transform': transforms.ToTensor(),
    'image_folder': image_folder,
    'annotation_file': annotation_csv,
    'extra_info': True}
    
    
    train_val_loader = gen_train_val_loader(labeled_scene_index, **kwargs)
    
    
    train_kwargs={
    'epochs':1,
    "lr": 0.01,
    'momentum': 0.99
    }
    
    
    train(train_val_loader, **train_kwargs)
    print('finished')