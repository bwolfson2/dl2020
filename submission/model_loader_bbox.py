import torch
import torch.nn as nn
#import torch.nn.functional as F
import torchvision

 
import torchvision.transforms as transforms
from auto_encoder import *


# import your model class
# import ...

from bbox import *

'''
#### NEED TO DO to save both models ###
checkpt = torch.load("../models/fastRCNN_autoencode14epoch.pt") #or from my dropbox
bbox_model = fr50_Model()
bbox_model.load_state_dict(checkpt["model_state_dict"]

checkpt2 = torch.load("where ben saved his model")  
road_model = Ben_Model()
road_model.load_state_dict(ccheckpt2["model_state_dict"]

troch.save({
        "bbox_state_dict": bbox_model.state_dict(),
        "road_state_dict": road_model.state_dict(),
)

'''
# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    #load_autoencode
   
    return torchvision.transforms.ToTensor()

# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'Musketeers'
    team_number = 1
    round_number = 1
    team_member = ["Ben Wolfson", "Calliea Pan", "Arushi Himatsingka"]
    contact_email = 'calliea.pan@nyu.edu'

    def __init__(self, model_file='musketeer.pt'):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        checkpoint = torch.load(model_file)
        self.AE = get_autoencoder().to(self.device) #from auto_encoder.py
        bbox_model = fr50_Model() #from bbox.py
        bbox_model.load_state_dict(checkpoint['bbox_state_dict'])
        bbox_model.eval()
        self.bbox_model = bbox_model.to(self.device)
        
        ### load ben's model ###
        
        
        
        ## end load ben's model##
        self.normalize = transforms.Normalize(mean=[0.6394939, 0.6755114, 0.7049375],
                                     std=[0.31936955, 0.3117349 , 0.2953726 ])
        

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        
        samp_pan = sew_images_panorm(samples) #convert to panoramic tensor
        samp_pan = [self.normalize(i) for i in samp_pan]
        samp_pan_t = torch.stack(samp_pan, dim = 0)
        images = self.AE.return_image_tensor(samp_pan_t.to(self.device))
        
        pred = self.bbox_model(images)
        box_list = []
        for p in pred:
            boxes = convert_boxes(p, self.device)
            box_list.append(boxes)
        result = tuple(box_list)
        return result
    
        #return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        
        return torch.rand(1, 800, 800) > 0.5