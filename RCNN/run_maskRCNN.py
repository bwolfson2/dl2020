import os
import sys
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

sys.path.insert(1, '//workspace/dl2020')
#from model_loader_CP2 import *
from CP_helper_RCNN import *
from data_helper_RCNN import *

#from Unet import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data


 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate


from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
import argparse



def train(model, optimizer, train_data_loader, device, epoch, print_freq=100):
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch_combModel(model, optimizer, train_data_loader, device, epoch, print_freq=100)
        # update the learning rate
        lr_scheduler.step()
     
    #save model
    print("saving model") 
    torch.save({

            'model_state_dict': model.state_dict(),
            #'feat_extractor_state_dict':  feat_extractor.state_dict(),
            #'output_layer_state_dict': output_layer.state_dict(),
            #'optimizer_state_dict': optimizer.state_dict(),
            #'train_losses': train_losses,
            #'val_accs': val_accs
            }, "../models/maskRCNN1.pt")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run maskRCNN, first argument is num epochs')
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", help="number of epochs",
                        type=int)
    args = parser.parse_args()
    epoch = args.num_epochs
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    image_folder = '../data'
    annotation_csv = '../data/annotation.csv'
    
    
    
    unlabeled_scene_index = np.arange(106)
    # The scenes from 106 - 133 are labeled
    # You should devide the labeled_scene_index into two subsets (training and validation)
    labeled_scene_index = np.arange(106, 134)
    
    ## make train and validation sets ##
    train_labeled_scene_index, val_labeled_scene_index  = gen_train_val_index(labeled_scene_index)
    
    kwargs = {
    #'first_dim': 'sample',
    'transform': transforms.ToTensor(),
    'image_folder': image_folder,
    'annotation_file': annotation_csv,
    'extra_info': True}
 
    print('gen data loaders')
    dataset_train = LabeledDataset(scene_index=train_labeled_scene_index, **kwargs)
    dataset_val = LabeledDataset(scene_index=val_labeled_scene_index, **kwargs)

    train_data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=3, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    val_data_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=3, shuffle=False, num_workers=4,
        collate_fn=collate_fn)
    
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and car
    num_classes = 2  
    # get the model using our helper function
    #model = get_instance_segmentation_model(num_classes)
    model = CombModel()

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, #reduce from 0.005 to help with the classifer loss = nan issue
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    print('training')
    train(model, optimizer, train_data_loader, device, epoch, print_freq=100)
    
    print('finished')