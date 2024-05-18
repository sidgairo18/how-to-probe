# standard python imports
import os
import pdb
import pickle
from functools import partial

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
from zennit.attribution import Gradient, SmoothGrad, IntegratedGradients, Occlusion
from zennit.composites import COMPOSITES
from zennit.image import imsave, CMAPS
from zennit.torchvision import VGGCanonizer, ResNetCanonizer

# captum imports
from captum.attr import LayerGradCam, InputXGradient, IntegratedGradients, GuidedBackprop
from captum.attr import LayerAttribution

# mmpretrain imports
import mmpretrain
from mmpretrain import get_model

# import external modules
from grid_pg_image_dataloader import *
from interpretability.utils import * # for bcos visualizations

# function to compute bcos attributions
def get_bcos_attributions(image, grad, ax, row, col, smooth=False, percentile=99.5):
    curr_prod = (grad*image[0]).sum(0).detach().cpu().numpy()
    coloured_exp = grad_to_img(image[0], grad)
    return curr_prod, coloured_exp

def get_att_maps(image_variable, output, ax, x, y, k=1, model_name='', label=None):
    # attribution stuff here
    # if label is given then use this
    if label is not None:
        image_variable.grad = None
        output[0,label].backward(retain_graph=True)
        w = image_variable.grad[0]
        curr_prod, coloured_exp = get_bcos_attributions(image_variable, w, ax, x, 1, smooth=True)
        return curr_prod, coloured_exp
    
    # else get attributions for the top k values
    topk, c_idcs = output[0].topk(200)
    for idx in range(k):
        image_variable.grad = None
        topk[idx].backward(retain_graph=True)
        w = image_variable.grad[0]
        curr_prod, coloured_exp = get_bcos_attributions(image_variable, w, ax, x, 1, smooth=True)
    return curr_prod, coloured_exp

# define and load the models here                                                             
my_model = mmpretrain.get_model('/path/to/model_config.py', pretrained='/path/to/pretrained_checkpoint.pth')
                                                                                              
model_dict = {                                                                                
        'my_model': my_model,                                                                 
        }
for name in model_dict:
    model = model_dict[name]
    model.data_preprocessor = None

MODELS = {
    'resnet50': (resnet50, ResNetCanonizer),
}

ATTRIBUTORS = {
        'gradient': Gradient,
        }

attr_raw = {
        'lrp',
        'gradcam',
        'bcos',
        'ixg',
        #'intgrad',
        'gbp'
        }

map_size=3
attr_methods = {}
attr_sparsity = {}
attr_entropy = {}
attr_mean_map = {}
attr_all_scores = {}
for attr in attr_raw:
    for model_name in model_dict:
        if ('_resnet' in model_name and attr == 'bcos') \
                or ('_bcosresnet' in model_name and attr == 'lrp'):
            continue
        attr_methods[attr+'_'+model_name] = []
        attr_sparsity[attr+'_'+model_name] = []
        attr_entropy[attr+'_'+model_name] = []
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
attributor_name = ""
composite_name = ""
batch_size = 1
n_outputs = 1000 # number of classes in imagenet, update for your own dataset
cpu = False
shuffle = False
with_bias = True
relevance_norm = "symmetric"
cmap = "bwr"
level = 1.0
seed = 0xDEADBEEF
map_size=3

# compute GridPG score
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

# compute Shannon Entropy
def get_shannon_entropy(contribution_map, index, map_size=3, k=None):
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
    # normalize contribution_map
    contribution_map = contribution_map / (np.sum(contribution_map)+1e-10)
    # compute entropy
    entropy = -np.sum(contribution_map * np.log2(contribution_map + 1e-10))

    return entropy

def get_sparsity(contribution_map, index, map_size=3, k=None):

    return compute_gini_index(contribution_map)

# compute Gini Index
# thanks to https://github.com/oliviaguest/gini
# and https://github.com/jfc43/advex/blob/61380b3053f2300ba97f510e7344312c6852644a/One-layer-Experiments/advtrain/model_metrics.py#L10, paper: https://arxiv.org/pdf/1810.06583
def compute_gini_index(array):
  """Calculate the Gini coefficient of a numpy array."""
  # based on bottom eq:
  # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
  # from:
  # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
  # All values are treated equally, arrays must be 1d:
  array[array < 0] = 0
  array = np.array(array, dtype=np.float64)
  array = np.abs(array.flatten())
  if np.amin(array) < 0:
    # Values cannot be negative:
    array -= np.amin(array)
  # Values cannot be 0:
  array += 0.0000001
  # Values must be sorted:
  array = np.sort(array)
  # Index per array element:
  index = np.arange(1, array.shape[0] + 1)
  # Number of array elements:
  n = array.shape[0]
  # Gini coefficient:
  return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def generate_map(
    inputs=None,
    attributor_name=None,
    composite_name=None,
    model=None,
    model_name=None,
    n_outputs=None,
    cpu=cpu,
    with_bias=with_bias,
    cmap=cmap,
    level=level,
    relevance_norm=relevance_norm,
    seed=seed,
    sample_id=0,
    output_dir=output_dir,
    norm_fn=None,
    bbox=None,
    row_idx=None,
    ax = None,
    percentile=99.5,
    map_size=3,
    target_idx=-1
):
    '''Generate heatmaps of an image folder at DATASET_ROOT to files RELEVANCE_FORMAT.
    RELEVANCE_FORMAT is a format string, for which {sample} is replaced with the sample index.
    '''
    # set a manual seed for the RNG
    torch.manual_seed(seed)

    # use the gpu if requested and available, else use the cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')

    # convenience identity matrix to produce one-hot encodings
    eye = torch.eye(n_outputs, device=device)

    # function to compute output relevance given the function output and a target
    def attr_output_fn(output, target):
        # output times one-hot encoding of the target labels of size (len(target), 1000)
        return output * eye[target]

    # create a composite if composite_name was set, otherwise we do not use a composite
    if attributor_name == "gradient":
        composite_name = "epsilon_gamma_box"
    composite = None
    if composite_name is not None:
        composite_kwargs = {}
        if composite_name == 'epsilon_gamma_box':
            # the maximal input shape, needed for the ZBox rule
            shape = (batch_size, 3, 224*map_size, 224*map_size)

            # the highest and lowest pixel values for the ZBox rule
            composite_kwargs['low'] = norm_fn(torch.zeros(*shape, device=device))
            composite_kwargs['high'] = norm_fn(torch.ones(*shape, device=device))
        # provide the name 'bias' in zero_params if no bias should be used to compute the relevance
        if not with_bias and composite_name in [                                          
            'epsilon_gamma_box',                                                          
            'epsilon_plus',                                                               
            'epsilon_alpha2_beta1',                                                       
            'epsilon_plus_flat',                                                          
            'epsilon_alpha2_beta1_flat',                                                  
            'excitation_backprop',                                                        
        ]:                                                                                
            composite_kwargs['zero_params'] = ['bias']                                    
                                                                                          
        # use torchvision specific canonizers, as supplied in the MODELS dict             
        composite_kwargs['canonizers'] = [MODELS["resnet50"][1]()]                        
                                                                                          
        # create a composite specified by a name; the COMPOSITES dict includes all preset composites provided by zennit.
        composite = COMPOSITES[composite_name](**composite_kwargs)                        
                                                                                          
    # specify some attributor-specific arguments                                          
    attributor_kwargs = {                                                                 
        'smoothgrad': {'noise_level': 0.1, 'n_iter': 20},                                 
        'integrads': {'n_iter': 20},                                                      
        'inputxgrad': {'n_iter': 1},                                                      
        'occlusion': {'window': (56, 56), 'stride': (28, 28)},                            
    }.get(attributor_name, {})                                                            
                                                                                          
    # create an attributor, given the ATTRIBUTORS dict given above. If composite is None, the gradient will not be
    # modified for the attribution                                                        
    attributor = ATTRIBUTORS[attributor_name](model, composite, **attributor_kwargs)      
                                                                                          
    # enter the attributor context outside the data loader loop, such that its canonizers and hooks do not need to be
    # registered and removed for each step. This registers the composite (and applies the canonizer) to the model
    # within the with-statement
    with attributor:                                                                      
        data_norm, target = inputs                                                    
        target = target.unsqueeze(0)
        # we use data without the normalization applied for visualization, and with the normalization applied as
        # create output relevance function of output with fixed target                
        output_relevance = partial(attr_output_fn, target=target)                     
                                                                                      
        # this will compute the modified gradient of model, where the output relevance is chosen by the as the
        # model's output for the ground-truth label index                             
        output, relevance = attributor(data_norm, output_relevance)                   
                                                                                      
        # sum over the color channel for visualization                                
        relevance = np.array(relevance.sum(1).detach().cpu())                         
        relevance_raw = relevance.copy()
        relevance_positive = relevance.copy()                                         
        relevance_positive[relevance_positive < 0] = 0.0                              
        #epg[model_name][attributor_name].append(compute_epg(relevance_positive[0], bbox))
                                                                                      
        # normalize between 0. and 1. given the specified strategy                    
        if relevance_norm == 'symmetric':                                             
            # 0-aligned symmetric relevance, negative and positive can be compared, the original 0. becomes 0.5
            amax = np.abs(relevance).max((1, 2), keepdims=True)+1e-12
            relevance = (relevance + amax) / 2 / amax                                 
        elif relevance_norm == 'absolute':                                            
            # 0-aligned absolute relevance, only the amplitude of relevance matters, the original 0. becomes 0.
            relevance = np.abs(relevance)                                             
            relevance /= relevance.max((1, 2), keepdims=True)                         
        elif relevance_norm == 'unaligned':                                           
            # do not align, the original minimum value becomes 0., the original maximum becomes 1.
            rmin = relevance.min((1, 2), keepdims=True)                               
            rmax = relevance.max((1, 2), keepdims=True)                               
            relevance = (relevance - rmin) / (rmax - rmin)                            
        
        #relevance_map = relevance[0]
        #fname = output_dir+"/"+attributor_name+"_"+model_name+"_"+sample_id+"_"+str(target[0].cpu().item())+".png"
        fname = output_dir+'/'+sample_id+'_'+str(target_idx)+'_'+attributor_name+'_'+model_name+'.png'
        imsave(fname, relevance[0], vmin=0., vmax=1., level=level, cmap=cmap)        
        return relevance_raw[0], fname
        #return relevance_positive[0]



def main(
    #dataset_root=dataset_root,
    batch_size=batch_size,
    n_outputs=n_outputs,
    cpu=cpu,
    shuffle=shuffle,
    with_bias=with_bias,
    cmap=cmap,
    level=level,
    relevance_norm=relevance_norm,
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

        # non-bcos attributions
        for model_name in model_dict:
            if "bcos" in model_name:
                continue
            model = model_dict[model_name]
            for attributor_name in ATTRIBUTORS:
                for target_idx in range(target.shape[1]):
                    contribution_map, fname = generate_map(
                            inputs=(data_norm, target[0,target_idx]),
                            n_outputs = n_outputs,
                            attributor_name=attributor_name,
                            model=model,
                            model_name=model_name,
                            sample_id=str(sample_index),
                            norm_fn=norm_fn,
                            bbox=None,
                            #row_idx=row_idx,
                            ax=None,
                            map_size=map_size,
                            target_idx=target_idx)
                    pg_score = get_pg_val(contribution_map.copy(), target_idx, map_size=map_size)
                    entropy_score = get_shannon_entropy(contribution_map.copy(), target_idx, map_size=map_size)
                    sparsity_score = get_sparsity(contribution_map.copy(), target_idx, map_size=map_size)
                    attr_methods['lrp_'+model_name].append(pg_score)
                    attr_entropy['lrp_'+model_name].append(entropy_score)
                    attr_sparsity['lrp_'+model_name].append(sparsity_score)
                    attr_all_scores['lrp_'+model_name].append((pg_score, fname))
                    attr_mean_map['lrp_'+model_name][target_idx] += contribution_map

        # gradcam attributions
        for model_name in model_dict:
            model = model_dict[model_name]
            layer_gc = LayerGradCam(model, model.backbone.layer4[-1].conv3)

            inp = data.clone().to(device)
            if '_resnet' in model_name:
                curr_data = data_norm.clone().to(device)
            else:
                curr_data = torch.cat([inp, 1-inp], dim=1)

            for target_idx in range(target.shape[1]):
                contribution_map = layer_gc.attribute(curr_data, int(target[0, target_idx]))

                contribution_map = LayerAttribution.interpolate(contribution_map, (224*map_size, 224*map_size), interpolate_mode='bilinear')
                contribution_map = contribution_map[0, 0].detach().cpu().numpy()

                pg_score = get_pg_val(contribution_map.copy(), target_idx, map_size=map_size)
                entropy_score = get_shannon_entropy(contribution_map.copy(), target_idx, map_size=map_size)
                sparsity_score = get_sparsity(contribution_map.copy(), target_idx, map_size=map_size)
                attr_methods['gradcam_'+model_name].append(pg_score)
                attr_entropy['gradcam_'+model_name].append(entropy_score)
                attr_sparsity['gradcam_'+model_name].append(sparsity_score)
                attr_mean_map['gradcam_'+model_name][target_idx] += contribution_map

                # first normalize
                amax = np.abs(contribution_map).max((0, 1), keepdims=True)+1e-12
                contribution_map = (contribution_map + amax) / 2 / amax
                # saving image
                fname = output_dir+'/'+str(sample_index)+'_'+str(target_idx)+'_gradcam_'+model_name+'.png'
                imsave(fname, contribution_map, vmin=0., vmax=1., level=1.0, cmap=cmap)
                attr_all_scores['gradcam_'+model_name].append((pg_score, fname))

        # inputxgradient attributions                                                     
        for model_name in model_dict:                                                     
            model = model_dict[model_name]                                                
            ixg = InputXGradient(model)                                                   
                                                                                          
            inp = data.clone().to(device)                                                 
            if '_resnet' in model_name:                                                   
                curr_data = data_norm.clone().to(device)                                  
            else:                                                                         
                curr_data = torch.cat([inp, 1-inp], dim=1)                                
                                                                                          
            for target_idx in range(target.shape[1]):                                     
                contribution_map = ixg.attribute(curr_data, target=int(target[0, target_idx]))
                contribution_map = contribution_map[0].sum(0).detach().cpu().numpy()      
                                                                                          
                pg_score = get_pg_val(contribution_map.copy(), target_idx, map_size=map_size, k=129)
                entropy_score = get_shannon_entropy(contribution_map.copy(), target_idx, map_size=map_size)
                sparsity_score = get_sparsity(contribution_map.copy(), target_idx, map_size=map_size)
                attr_methods['ixg_'+model_name].append(pg_score)
                attr_entropy['ixg_'+model_name].append(entropy_score)
                attr_sparsity['ixg_'+model_name].append(sparsity_score)
                attr_mean_map['ixg_'+model_name][target_idx] += contribution_map
                                                                                          
                # first normalize                                                         
                amax = np.abs(contribution_map).max((0, 1), keepdims=True)+1e-12
                contribution_map = (contribution_map + amax) / 2 / amax                   
                # saving image                                                            
                fname = output_dir+'/'+str(sample_index)+'_'+str(target_idx)+'_ixg_'+model_name+'.png'
                imsave(fname, contribution_map, vmin=0., vmax=1., level=1.0, cmap=cmap)   
                attr_all_scores['ixg_'+model_name].append((pg_score, fname))
                                                                                          
            del inp, curr_data, contribution_map                                          
            torch.cuda.empty_cache()                                                      
                                                                                          
        # intgrad attributions                                                            
        """                                                                               
        for model_name in model_dict:                                                     
            model = model_dict[model_name]                                                
            intgrad = IntegratedGradients(model)                                          
                                                                                          
            inp = data.clone().to(device)                                                 
            if '_resnet' in model_name:                                                   
                curr_data = data_norm.clone().to(device)                                  
            else:                                                                         
                curr_data = torch.cat([inp, 1-inp], dim=1)                                
                                                                                          
            for target_idx in range(target.shape[1]):                                     
                contribution_map = intgrad.attribute(curr_data, target=int(target[0, target_idx]))
                print('intgrad', contribution_map.shape)                                  
                                                                                          
            del inp, curr_data, contribution_map                                          
            torch.cuda.empty_cache()                                                      
        """                                                                               
                                                                                          
        # guidedbackpropagation attributions                                              
        for model_name in model_dict:                                                     
            model = model_dict[model_name]                                                
            gbp = GuidedBackprop(model)                                                   
                                                                                          
            inp = data.clone().to(device)                                                 
            if '_resnet' in model_name:                                                   
                curr_data = data_norm.clone().to(device)                                  
            else:                                                                         
                curr_data = torch.cat([inp, 1-inp], dim=1)                                
                                                                                          
            for target_idx in range(target.shape[1]):                                     
                contribution_map = gbp.attribute(curr_data, target=int(target[0, target_idx]))
                contribution_map = contribution_map[0].sum(0).detach().cpu().numpy()      
                                                                                          
                pg_score = get_pg_val(contribution_map.copy(), target_idx, map_size=map_size)    
                entropy_score = get_shannon_entropy(contribution_map.copy(), target_idx, map_size=map_size)
                sparsity_score = get_sparsity(contribution_map.copy(), target_idx, map_size=map_size)
                attr_methods['gbp_'+model_name].append(pg_score)
                attr_entropy['gbp_'+model_name].append(entropy_score)
                attr_sparsity['gbp_'+model_name].append(sparsity_score)
                attr_mean_map['gbp_'+model_name][target_idx] += contribution_map
                                                                                          
                # first normalize                                                         
                amax = np.abs(contribution_map).max((0, 1), keepdims=True)+1e-12
                contribution_map = (contribution_map + amax) / 2 / amax                   
                # saving image                                                            
                fname = output_dir+'/'+str(sample_index)+'_'+str(target_idx)+'_gbp_'+model_name+'.png'
                imsave(fname, contribution_map, vmin=0., vmax=1., level=1.0, cmap=cmap)   
                attr_all_scores['gbp_'+model_name].append((pg_score, fname))
                                                                                          
            del inp, curr_data, contribution_map                                          
            torch.cuda.empty_cache()


        # bcos stuff here                                                                 
        for model_name in model_dict:                                                     
            if "bcosresnet" in model_name:
                model = model_dict[model_name]
                row_idx = 0
                for target_idx in range(target.shape[1]):

                    if model.neck is not None:
                        for mod in model.neck.modules():
                            if hasattr(mod, "explanation_mode"):
                                mod.explanation_mode(True)
                            if hasattr(mod, "set_explanation_mode"):
                                mod.set_explanation_mode(True)
                            if hasattr(mod, "detach"):
                                mod.detach = True
                            if hasattr(mod, "detach_var"):
                                mod.detach_var = True
                    for mod in model.head.modules():
                        if hasattr(mod, "explanation_mode"):
                            mod.explanation_mode(True)
                        if hasattr(mod, "set_explanation_mode"):
                            mod.set_explanation_mode(True)
                        if hasattr(mod, "detach"):
                            mod.detach = True
                        if hasattr(mod, "detach_var"):
                            mod.detach_var = True

                    with model.backbone.explanation_mode():
                        inp = data.clone().to(device)                                         
                        inp = torch.cat([inp, 1-inp], dim=1)                                  
                        image_variable = Variable(inp, requires_grad = True)                  
                        output = model(image_variable)                                        
                        #output = nn.Sigmoid()(output)                                    
                        #ax[row_idx, 0].imshow(img.astype(np.uint8))                           
                        contribution_map, coloured_exp = get_att_maps(image_variable, output, None, row_idx, 1, k=1, model_name=model_name, label=int(target[0,target_idx]))

                    #fname = output_dir+"/"+"bcos_"+model_name+"_"+str(sample_index)+"_"+str(target[0,target_idx].cpu().item())+".png"
                    pg_score = get_pg_val(contribution_map.copy(), target_idx, map_size=map_size)
                    entropy_score = get_shannon_entropy(contribution_map.copy(), target_idx, map_size=map_size)
                    sparsity_score = get_sparsity(contribution_map.copy(), target_idx, map_size=map_size)
                    attr_methods['bcos_'+model_name].append(pg_score)
                    attr_entropy['bcos_'+model_name].append(entropy_score)
                    attr_sparsity['bcos_'+model_name].append(sparsity_score)
                    attr_mean_map['bcos_'+model_name][target_idx] += contribution_map
                    #print(coloured_exp.shape, "coloured exp shape")

                    # first normalize
                    amax = np.abs(contribution_map).max((0, 1), keepdims=True)+1e-12
                    contribution_map = (contribution_map + amax) / 2 / amax
                    # saving image
                    fname = output_dir+'/'+str(sample_index)+'_'+str(target_idx)+'_bcos-att_'+model_name+'.png'
                    imsave(fname, contribution_map, vmin=0., vmax=1., level=1.0, cmap=cmap)
                    attr_all_scores['bcos_'+model_name].append((pg_score, fname))
                    #fname = output_dir+'/'+str(sample_index)+'_'+str(target_idx)+'_bcos-col_'+model_name+'.png'
                    #imsave(fname, coloured_exp)

                    #save coloured image
                    fname = output_dir+'/'+str(sample_index)+'_'+str(target_idx)+'_bcos-col_'+model_name+'.png'
                    plt.imshow(coloured_exp)
                    plt.axis('off')
                    plt.savefig(fname, bbox_inches='tight')
                    plt.close()
        
        for attr in attr_methods:
            #print(attr, np.nanmean(attr_methods[attr]), np.nanstd(attr_methods[attr]))
            print(attr, np.nanmean(attr_methods[attr]), np.nanmean(attr_entropy[attr]), np.nanmean(attr_sparsity[attr]))
        print("DONE {} ...".format(sample_index))
        sample_index += len(data)
        if sample_index >= 500:
            break
    for item in attr_mean_map:
        curr_map = attr_mean_map[item]
        print(item, curr_map.shape)
        for target_idx in range(curr_map.shape[0]):
            fname = output_dir+'/'+str(target_idx)+'_meanmap_'+item
            contribution_map = curr_map[target_idx]
            np.save(fname+'.npy', contribution_map)
            amax = np.abs(contribution_map).max((0, 1), keepdims=True)+1e-12
            contribution_map = (contribution_map + amax) / 2 / amax
            imsave(fname+'.png', contribution_map, vmin=0., vmax=1., level=1.0, cmap=cmap)
            
    with open(output_dir+'/attr_all_scores.pkl', 'wb') as f:
        pickle.dump(attr_all_scores, f)
                                                                                          
if __name__ == '__main__':                                                                
    main()                                                                                
    exit(0)
