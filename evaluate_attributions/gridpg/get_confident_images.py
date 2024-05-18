import os
import pdb
from functools import partial

import torch
import torch.nn as nn

import cv2
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.models import resnet50
from torch.autograd import Variable

# other imports
from zennit.image import imsave
import mmpretrain
from mmpretrain import get_model

# import external modules
from image_dataloader_from_file import *

# define and load the models here
my_model = mmpretrain.get_model('/path/to/model_config.py', pretrained='/path/to/pretrained_checkpoint.pth')

model_dict = {
        'my_model': my_model,
        }

model_ids = {
        'my_model': 0,
        }

model_acc = {
        'my_model': 0,
        }

print("MODELS DEFINED AND LOADED!")

# some extras like transforms
invTrans = Compose([Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                    Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])


class BatchNormalize:
    def __init__(self, mean, std, device=None):
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

# arguments as variables, should probably make an args parser.
data_file = "/path/to/imagenet_val.txt"
data_root = "/path/to/imagenet/val/"
output_dir = None
batch_size = 1
cpu = False
shuffle = False
seed = 0xDEADBEEF
cut_off = 0.95
max_class_count = 100

confident_images_dir = str(cut_off)+"_confident_images_subset_dir/"
os.makedirs(confident_images_dir, exist_ok = True)


def main(
    #dataset_root=dataset_root,
    batch_size=batch_size,
    n_outputs=200,
    cpu=cpu,
    shuffle=shuffle,
    seed=seed,
    output_dir=output_dir,
    cut_off = cut_off
):

    # use the gpu if requested and available, else use the cpu                            
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')   
                                                                                          
    # mean and std of ILSVRC2012 as computed for the torchvision models                   
    norm_fn = BatchNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), device=device) 
                                                                                          
    # transforms as used for torchvision model evaluation                                 
    transform = Compose([                                                                 
        Resize(256),                                                                      
        CenterCrop(224),                                                                  
        ToTensor(),                                                                       
    ])

    # define dataloader and dataset
    dataset = image_loader(data_file, data_root, input_transforms=transform)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    for model_name in model_dict:
        model = model_dict[model_name]
        model.to(device)
        model.eval()

        # disable requires_grad for all parameters, we do not need their modified gradients
        for param in model.parameters():
            param.requires_grad = False
        
    print("MODELS SETUP")

    # the current sample index for creating file names                                    
    sample_index = 0                                                                      
                                                                                          
    class_counts_dict = {}                                                                
    for cls in range(n_outputs):                                                          
        class_counts_dict[cls] = 0                                                        
                                                                                          
    for loader_idx, (data, target) in enumerate(loader):                                  
                                                                                          
        # we use data without the normalization applied for visualization, and with the normalization applied as
        # the model input                                                                 
        data_norm = norm_fn(data.clone().to(device))
        target = target.to(device)                                                        
                                                                                          
        if class_counts_dict[int(target[0])] >= max_class_count:                      
            continue                                                                      
                                                                                          
        #print("Started with IDX:{}".format(sample_index), loader_idx)
        #sup_target = -1
        
        # check confidence and correct classes only
        flag = True
        for model_name in model_dict:
            model = model_dict[model_name]

            if 'std' in model_name:
                output = model(data_norm)
                output_probs = nn.Softmax(dim=1)(output)
            elif 'bcos' in model_name:
                bcos_data = torch.cat([data, 1-data], dim=1)
                output = model(bcos_data.to(device))
                output_probs = nn.Sigmoid()(output)

            #print(model_name, output_probs.shape)

            model_acc[model_name] += (output_probs.argmax(1) == target).sum().detach().cpu().item()

            if (output_probs.argmax(1) == target).sum().detach().cpu().item() == 0 or float(output_probs.max()) < cut_off:
                #print(model_name, (output_probs.argmax(1) == target).sum().detach().cpu().item(), float(output_probs.max()))
                flag = False
                #break
        
        if (loader_idx+1)%100 == 0:
            for model_name in model_acc:
                print(model_name, model_acc[model_name]/(loader_idx+1))

        # condition breaks here so no further processing
        if flag == False:
            #print(model_name, "BROKEN")
            continue
        
        # save the input image
        fname_ssl = str(sample_index)+"_"+str(target[0].cpu().item())

        imsave(confident_images_dir+fname_ssl+".png", data[0])
        print("DONE {} ...".format(sample_index), loader_idx)
        sample_index += len(data)
        
        class_counts_dict[int(target[0])] += 1

    for model_name in model_acc:
        print(model_name, model_acc[model_name]/len(dataset))
        
                                                                                          
                                                                                          
if __name__ == '__main__':                                                                
    main()                                                                                
    exit(0)
