# standard python imports
import os
import pdb
import pickle
from functools import partial
import math

# more standard tools
import cv2
import numpy as np

# pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, GaussianBlur
from torch.autograd import Variable
from torchvision.models import resnet50

# zennit imports
from zennit.image import imsave, CMAPS

# mmpretrain imports
import mmpretrain
from mmpretrain import get_model

# import external modules
from grid_pg_image_dataloader import *
from interpretability.utils import * # for bcos visualizations
from lime_attribution import Lime

# define and load the models here                                                             
my_model = mmpretrain.get_model('/path/to/model_config.py', pretrained='/path/to/pretrained_checkpoint.pth')
                                                                                              
model_dict = {                                                                                
        'my_model': my_model,                                                                 
        } 

for name in model_dict:
    model = model_dict[name]
    model.data_preprocessor = None

attr_raw = {
        'lime'
        }

map_size=3
attr_methods = {}
attr_mean_map = {}
attr_all_scores = {}
for attr in attr_raw:
    for model_name in model_dict:
        if ('_resnet' in model_name and attr == 'bcos') \
                or ('_bcosresnet' in model_name and attr == 'lrp'):
            continue
        attr_methods[attr+'_'+model_name] = []
        attr_mean_map[attr+'_'+model_name] = np.zeros((map_size*map_size, 224*map_size, 224*map_size))
        attr_all_scores[attr+'_'+model_name] = []

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
data_file='/path/to/grid_pg_images_3x3_list.txt'                                          
data_root='/path/to/grid_pg_images_3x3'                                                   
output_dir = '/path/to/output_dir'; os.makedirs(output_dir, exist_ok = True) 
batch_size = 1
n_outputs = 1000 # number of classes
cpu = False
shuffle = False
cmap = "bwr"
seed = 0xDEADBEEF
map_size=3

def get_pg_val(contribution_map, index, map_size=3, k=None):

    h, w = contribution_map.shape[:2]
    assert h == w, "height should be equal to width"
    
    contribution_map = torch.from_numpy(contribution_map)
    if k is None:
        contribution_map = F.avg_pool2d(contribution_map.unsqueeze(0), 5, padding=2, stride=1)[0]
    else:
        contribution_map = GaussianBlur(k, sigma=k/4)(contribution_map.unsqueeze(0))[0]
    contribution_map = contribution_map.numpy()
    # remove -ve contributions
    contribution_map[contribution_map<0] = 0
    # apply smoothening here??
    i, j = index//map_size, index%map_size
    mult_fact = h//map_size
    pg_val = np.sum(contribution_map[i*mult_fact:(i+1)*mult_fact, j*mult_fact:(j+1)*mult_fact]) / (np.sum(contribution_map)+1e-12)
    return pg_val

def main(
    #dataset_root=dataset_root,
    batch_size=batch_size,
    n_outputs=n_outputs,
    cpu=cpu,
    shuffle=shuffle,
    cmap=cmap,
    seed=seed,
    output_dir=output_dir,
    map_size=map_size
):

    # use the gpu if requested and available, else use the cpu                            
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')   
                                                                                          
    # mean and std of ILSVRC2012 as computed for the torchvision models                   
    norm_fn = BatchNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), device=device) 
                                                                                          
    # transforms as used for torchvision model evaluation                                 
    transform = Compose([                                                                 
        ToTensor(),                                                                       
    ])

    # define dataloader and dataset
    dataset = image_loader(data_root, data_file, input_transforms=transform)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    for model_name in model_dict:
        model = model_dict[model_name]
        model.to(device)
        model.eval()

        # disable requires_grad for all parameters, we do not need their modified gradients
        for param in model.parameters():
            param.requires_grad = False
        
    print("MODELS EVAL MODE SETUP AND REQUIRES GRAD MADE FALSE")

    # the current sample index for creating file names                                    
    sample_index = 0                                                                      
                                                                                          
    for loader_idx, (data, target) in enumerate(loader):                                  
                                                                                          
        # we use data without the normalization applied for visualization, and with the normalization applied as
        # the model input                                                                 
        data_norm = norm_fn(data.clone().to(device))
        target = target.to(device)                                                        
        
        # save the input image
        #fname = output_dir+"/"+str(sample_index)+"_"+str(target[0].cpu().item())
        fname = output_dir+"/"+str(sample_index)
        imsave(fname+"_"+"input.png", data[0])
        """
        # read the input image for plotting
        img = cv2.imread(fname+"_"+"input.png")
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # setup figure and axes for plotting
        fig, ax = plt.subplots(nrows=len(model_dict)//2, ncols=4, figsize=(40, 40))
        """

        # lime attributions

        for model_name in model_dict:                                                     
            model = model_dict[model_name]                                                

            lime_explainer = None
            loss_type = 'ce' if '_ce' in model_name else 'bce'
                                                                                          
            inp = data.clone().to(device)                                                 
            if '_resnet' in model_name:                                                   
                curr_data = data_norm.clone().to(device)                                  
                lime_explainer = Lime(model, num_classes=n_outputs, model_type='std', full_map=True, loss_type=loss_type)
            else:                                                                         
                curr_data = torch.cat([inp, 1-inp], dim=1)                                
                lime_explainer = Lime(model, num_classes=n_outputs, model_type='bcos', full_map=True, loss_type=loss_type)
                                                                                          
            for target_idx in range(target.shape[1]):                                     

                contribution_map = lime_explainer.attribute(curr_data, torch.tensor([int(target[0, target_idx])]))[0, 0].numpy()

                                                                                          
                pg_score = get_pg_val(contribution_map, target_idx, map_size=map_size)
                #print(sample_index, model_name, target_idx, pg_score)
                attr_methods['lime_'+model_name].append(pg_score)
                attr_mean_map['lime_'+model_name][target_idx] += contribution_map
                                                                                          
                # first normalize                                                         
                amax = np.abs(contribution_map).max((0, 1), keepdims=True)+1e-12
                contribution_map = (contribution_map + amax) / 2 / amax                   
                # saving image                                                            
                fname = output_dir+'/'+str(sample_index)+'_'+str(target_idx)+'_lime_'+model_name+'.png'
                imsave(fname, contribution_map, vmin=0., vmax=1., level=1.0, cmap=cmap)   
                attr_all_scores['lime_'+model_name].append((pg_score, fname))
                                                                                          
            del inp, curr_data, contribution_map, lime_explainer
            torch.cuda.empty_cache()                                                      
                                                                                          
        for attr in attr_methods:
            print(attr, np.nanmean(attr_methods[attr]), np.nanstd(attr_methods[attr]))
            # without 0.0s
            nonzero_list = []
            for xitem in attr_methods[attr]:
                if math.isnan(xitem) or xitem == 0.0:
                    continue
                nonzero_list.append(xitem)
            print("without zeros", attr, np.nanmean(nonzero_list), np.nanstd(nonzero_list))

        print("DONE {} ...".format(sample_index))

        with open(output_dir+'/'+experiment_name+'_'+'attr_methods.pkl', 'wb') as f:
            pickle.dump(attr_methods, f)

        sample_index += len(data)
        if sample_index >= 500:
            break

    for item in attr_mean_map:
        curr_map = attr_mean_map[item]
        print(item, curr_map.shape)
        for target_idx in range(curr_map.shape[0]):
            fname = output_dir+'/'+experiment_name+'_'+str(target_idx)+'_meanmap_'+item
            contribution_map = curr_map[target_idx]
            np.save(fname+'.npy', contribution_map)
            amax = np.abs(contribution_map).max((0, 1), keepdims=True)+1e-12
            contribution_map = (contribution_map + amax) / 2 / amax
            imsave(fname+'.png', contribution_map, vmin=0., vmax=1., level=1.0, cmap=cmap)
            
    with open(output_dir+'/'+experiment_name+'_'+'attr_all_scores.pkl', 'wb') as f:
        pickle.dump(attr_all_scores, f)
                                                                                          
if __name__ == '__main__':                                                                

    main()                                                                                
    exit(0)
