"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# import your model class
# import ...

# Put your transform function here, we will use it for our dataloader
def get_transform(): 
    # return torchvision.transforms.Compose([
    # 
    # 
    # ])
    pass

'''
##or resnet18:


transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) 

transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
model = models.resnet18(pretrained = True) #load resnet18 with pretrained, or set to False 
'''

class ModelLoader():
    # Fill the information for your team
    team_name = 'Musketeers'
    team_member = [3]
    contact_email = 'cp2530@nyu.edu'

    def __init__(model_file):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        pass

    def get_bounding_boxes(samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        pass
    def test_func():
        print('test func')

    def sew_images(sing_samp):
        # sing_samp is [6, 3, 256, 306], one item is batch
        # output is the image object of all 6 pictures 'sown' together
        #############
        # A | B | C #
        # D | E | F #
        #############
        
        A1 = sing_samp[0][0]
        A2 = sing_samp[0][1]
        A3 = sing_samp[0][2]

        B1 = sing_samp[1][0]
        B2 = sing_samp[1][1]
        B3 = sing_samp[1][2]

        C1 = sing_samp[2][0]
        C2 = sing_samp[2][1]
        C3 = sing_samp[2][1]

        D1 = sing_samp[3][0]
        D2 = sing_samp[3][1]
        D3 = sing_samp[3][2]

        E1 = sing_samp[4][0]
        E2 = sing_samp[4][1]
        E3 = sing_samp[4][2]

        F1 = sing_samp[5][0]
        F2 = sing_samp[5][1]
        F3 = sing_samp[5][2]

        #print("F shape {}".format(F1.shape))

        T1 = torch.cat([A1, B1, C1], 0)
        T2 = torch.cat([A2, B2, C2], 0)
        T3 = torch.cat([A3, B3, C3], 0)

        B1 = torch.cat([D1, E1, F1], 0)
        B2 = torch.cat([D2, E2, F2], 0)
        B3 = torch.cat([D3, E3, F3], 0)
        #print("T1 shape {}".format(T1.shape))

        comb1 = torch.cat([T1,B1], 1)
        comb2 = torch.cat([T2,B2], 1)
        comb3 = torch.cat([T3,B3], 1)

        #print("comb1 shape {}".format(comb1.shape)) #should be 768, 612
        comb = torch.stack([comb1, comb2, comb3])
        toImg = transforms.ToPILImage()
        result = toImg(comb)
        return result
        
    
    def get_binary_road_map(samples):
        #samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        
        
        #sew together the 6 images (across 3 channels) to size (768, 612) then resize to (880, 880) 
        
        transTot = transforms.Compose([transforms.Resize((880,880)), transforms.ToTensor()])        
        bstack_list = []

        for i in range(len(samples)): #for each of the 2 in the batch
            #print(i)
            sing_samp = samples[i] #get each sample
            comb_img = ModelLoader.sew_images(sing_samp) 

            comb =transTot(comb_img)
            comb = comb.sum(0)/3 #sum across 3 channels and then normalize by 3 to get all values between [0,1] 
            bstack_list.append(comb)
        
        result = torch.stack(bstack_list, 0)
        return result
        