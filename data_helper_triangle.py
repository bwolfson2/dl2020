import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from helper import convert_map_to_lane_map, convert_map_to_road_map

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT.jpeg',
    'CAM_FRONT_LEFT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    'CAM_FRONT_RIGHT.jpeg'
    ]



def load_mask(camera):
    mask = np.load(camera.replace(".jpeg",".npy"))
    mask = mask.reshape(800,800).transpose()
    return mask
    
    
    
#BASICALLY YOU JUST HAVE TO SPECIFY THE IMAGE TO LPAD IN THE __GET_ITEM__ on init
# The dataset class for labeled data.
class TriangleLabeledDataset(torch.utils.data.Dataset):    
    def __init__(self, image_folder, annotation_file, scene_index, transform,extra_info=True,camera='CAM_FRONT.jpeg'):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data 
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """
        
        assert(camera in image_names)
        self.camera = camera
        self.mask = load_mask(camera)
        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info
    
    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}') 
        
        #get camera and only get that image
        image_path = os.path.join(sample_path, self.camera)
        image = Image.open(image_path)
        image = self.transform(image)

        data_entries = self.annotation_dataframe[(self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']].to_numpy()
        categories = data_entries.category_id.to_numpy()
        
        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)
        ##Preprocess road image
        road_image = torch.Tensor(road_image.numpy()*self.mask)
        road_image_mod = road_image[self.mask]
        
        target = {}
        target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 4)
        target['category'] = torch.as_tensor(categories)

        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with 
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)
            
            extra = {}
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image

            return image, target, road_image, extra, road_image_mod
        
        else:
            return image, target, road_image, road_image_mod
        

def test_loader(camera='CAM_BACK.jpeg'):
    transform = torchvision.transforms.ToTensor()
    # The labeled dataset can only be retrieved by sample.
    # And all the returned data are tuple of tensors, since bounding boxes may have different size
    # You can choose whether the loader returns the extra_info. It is optional. You don't have to use it.
    labeled_trainset = TriangleLabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index,
                                      transform=transform,
                                      extra_info=True,
                                    camera = camera
                                     )
    trainloader = torch.utils.data.DataLoader(labeled_trainset , batch_size=2, \
                                              shuffle=False, num_workers=2, collate_fn=collate_fn)

    sample, target, road_image, extra = iter(trainloader).next()
    plt.imshow(sample[0].numpy().transpose(1, 2, 0))
    plt.axis('off');
    fig, ax = plt.subplots()

    ax.imshow(road_image[0], cmap='binary');
    plot_mask(labeled_trainset.mask)
    return road_image[0],labeled_trainset.mask
    
if __name__ == "__main__":   
    ri,m=test_loader()