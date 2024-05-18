# standard python imports
import os
import pdb
from functools import partial

# more standard tools
import cv2
import numpy as np
from sklearn.metrics import f1_score

# pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, GaussianBlur
from torchvision.models import resnet50
from torch.autograd import Variable

# other imports
from zennit.attribution import Gradient, SmoothGrad, IntegratedGradients, Occlusion
from zennit.composites import COMPOSITES
from zennit.image import imsave, CMAPS
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
import mmpretrain
from mmpretrain import get_model

# captum imports
from captum.attr import LayerGradCam, InputXGradient, IntegratedGradients, GuidedBackprop
from captum.attr import LayerAttribution

# import external modules
from interpretability.utils import * # for bcos visualizations


# extra utils
object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

def encode_labels_raw(target):
    """
    Encode multiple labels using 1/0 encoding

    Args:
        target: xml tree file
    Returns:
        torch tensor encoding labels as 1/0 vector
    """

    size = target['annotation']['size']
    orig_dims = (int(size['height'][0]), int(size['width'][0])) # (height, width)

    ls = target['annotation']['object']
    idx_to_bbox = {}

    j = []
    if type(ls) == dict:
        if int(ls['difficult'][0]) == 0:
            j.append(object_categories.index(ls['name'][0]))
            if j[-1] not in idx_to_bbox:
                idx_to_bbox[j[-1]] = []
            
            bbox = ls['bndbox']
            idx_to_bbox[j[-1]].append((int(bbox['xmin'][0]), int(bbox['ymin'][0]),
                int(bbox['xmax'][0]), int(bbox['ymax'][0])))

    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult'][0]) == 0:
                j.append(object_categories.index(ls[i]['name'][0]))
                if j[-1] not in idx_to_bbox:
                    idx_to_bbox[j[-1]] = []

                bbox = ls[i]['bndbox']
                idx_to_bbox[j[-1]].append((int(bbox['xmin'][0]), int(bbox['ymin'][0]),
                    int(bbox['xmax'][0]), int(bbox['ymax'][0])))


    k = np.zeros(len(object_categories))
    k[j] = 1

    return torch.from_numpy(k), idx_to_bbox, orig_dims

def encode_labels(target):
    """
    Encode multiple labels using 1/0 encoding

    Args:
        target: xml tree file
    Returns:
        torch tensor encoding labels as 1/0 vector
    """

    ls = target['annotation']['object']

    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))
    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))

    k = np.zeros(len(object_categories))
    k[j] = 1

    return torch.from_numpy(k)


# function to compute bcos attributions
def get_bcos_attributions(image, grad, ax, row, col, smooth=False, percentile=99.5):
    curr_prod = (grad*image[0]).sum(0).detach().cpu().numpy()
    coloured_exp = grad_to_img(image[0], grad)
    # can plot the contribution map here
    # plot_contribution_map(F.avg_pool2d((curr_prod.unsqueeze(0)).sum(1, keepdim=True), 5, padding=2, stride=1)[0, 0], ax=ax[row, col], percentile=percentile)
    # returning raw bcos attributions without any -ve suppression
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
        """
        ax[x, 1].set_title(model_name+'_Bcos', fontsize=40)
        ax[x, 1].set_xlabel('Confidence {}'.format(round(float(output[0, c_idcs[idx]]), 3)), fontsize=40)
        """
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
        'intgrad',
        'gbp'
        }

attr_methods = {}                                                                         
for attr in attr_raw:                                                                     
    for model_name in model_dict:                                                         
        if ('_resnet' in model_name and attr == 'bcos') \
                or ('_bcosresnet' in model_name and attr == 'lrp'):                       
            continue                                                                      
        attr_methods[attr+'_'+model_name] = {}                                            
        attr_methods[attr+'_'+model_name]['epg'] = []                                     
        attr_methods[attr+'_'+model_name]['bbox_miou'] = []                               
        attr_methods[attr+'_'+model_name]['seg_miou'] = []

model_perf = {}
for model_name in model_dict:
    model_perf[model_name] = [[], []]

print("MODELS DEFINED AND LOADED!")

# setup config stuff here
# arguments as variables, should probably make an args parser.
data_root='/path/to/VOCdevkit'
voc_year='2007'
data_split='test'

thresh = 0.50
output_dir = '/path/to/output'; os.makedirs(output_dir, exist_ok=True)
attributor_name = ""
composite_name = ""
batch_size = 1
n_outputs = 20
cpu = False
shuffle = False
with_bias = True
relevance_norm = "symmetric"
cmap = "bwr"
level = 1.0
seed = 0xDEADBEEF
IMAGE_SIZE=224
#confidence = 0.95

# some extras like transforms
invTrans = Compose([Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.26862954, 1/0.26130258, 1/0.27577711 ]),
                        Normalize(mean = [ -0.48145466, -0.4578275, -0.40821073 ], std = [ 1., 1., 1. ])])


class BatchNormalize:
    def __init__(self, mean, std, device=None):
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

def binaryMaskIOU(mask1, mask2):   # From the question.
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and( mask1==1,  mask2==1 ))
    iou = intersection/(mask1_area+mask2_area-intersection+1e-10)
    return iou

def get_thresholded_mask(contribution_map, threshold=0.60, k=None):
    # dimensions
    h, w = contribution_map.shape[:2]
    # convert to torch
    contribution_map = torch.from_numpy(contribution_map)
    if k is None:
        contribution_map = F.avg_pool2d(contribution_map.unsqueeze(0), 5, padding=2, stride=1)[0]
    else:
        contribution_map = GaussianBlur(k, sigma=k/4)(contribution_map.unsqueeze(0))[0]
    # remove -ve contributions
    contribution_map[contribution_map<0] = 0
    th_attn = (contribution_map - contribution_map.min()) / (1e-12 + contribution_map.max() - contribution_map.min())
    th_attn[th_attn < threshold] = 0
    th_attn[th_attn >= threshold] = 1

    """
    contribution_map = contribution_map.unsqueeze(0)

    # linearize
    attentions = contribution_map.reshape(1, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    th_attn[0] = th_attn[0][idx2[0]]
    th_attn = th_attn.reshape(1, w, h).squeeze(0).float()
    """

    # interpolate
    #th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    return th_attn.numpy()

def get_seg_iou_val(contribution_map, mask, threshold=0.60, k=None):
    th_attn = get_thresholded_mask(contribution_map, threshold=threshold, k=k)
    iou_val = binaryMaskIOU(mask, th_attn)
    return round(iou_val, 2)


def get_iou_val(contribution_map, bboxes, base_w=IMAGE_SIZE, base_h=IMAGE_SIZE, threshold=0.60, k=None):

    #th_attn = get_thresholded_mask(contribution_map, threshold=0.20, k=k)
    th_attn = get_thresholded_mask(contribution_map, threshold=threshold, k=k)

    h, w  = th_attn.shape[:2]
    th_attn_3d = np.zeros((h, w, 3))
    th_attn_3d[:, :, 0] = th_attn
    th_attn_3d[:, :, 1] = th_attn
    th_attn_3d[:, :, 2] = th_attn
    th_attn_3d = th_attn_3d.astype(np.uint8)*255

    # find bbox around th_attn
    pred_bbox_mask = np.zeros((h,w))
    segmentation = np.where(th_attn == 1.0)
    try:
        xb_min = int(np.min(segmentation[1]))
        xb_max = int(np.max(segmentation[1]))
        yb_min = int(np.min(segmentation[0]))
        yb_max = int(np.max(segmentation[0]))
    except:
        xb_min, xb_max, yb_min, yb_max = 0, 1, 0, 1

    pred_bbox_mask[yb_min:yb_max, xb_min:xb_max] = 1
    th_attn_3d = cv2.rectangle(th_attn_3d, (xb_min, yb_min), (xb_max, yb_max), (255,0,0), 2)

    mask = np.zeros((h,w))
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max = (w*x_min)//base_w, (w*x_max)//base_w
        y_min, y_max = (h*y_min)//base_h, (h*y_max)//base_h
        th_attn_3d = cv2.rectangle(th_attn_3d, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
        mask[y_min:y_max, x_min:x_max] = 1.0

    iou_val = binaryMaskIOU(mask, pred_bbox_mask)
    return round(iou_val, 2), th_attn_3d


def get_epg_val(contribution_map, bboxes, base_w=IMAGE_SIZE, base_h=IMAGE_SIZE, k=None):                              
                                                                                          
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
    mask = np.zeros((h, w))
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max = (w*x_min)//base_w, (w*x_max)//base_w
        y_min, y_max = (h*y_min)//base_h, (h*y_max)//base_h
        mask[y_min:y_max, x_min:x_max] = 1.0

    epg_val = np.sum(contribution_map*mask) / (np.sum(contribution_map)+1e-12)

    return round(epg_val, 2)

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
            shape = (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)

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

        relevance[relevance < 0] = 0.0
        #epg[model_name][attributor_name].append(compute_epg(relevance_positive[0], bbox))

        # normalize between 0. and 1. given the specified strategy
        if relevance_norm == 'symmetric':
            # 0-aligned symmetric relevance, negative and positive can be compared, the original 0. becomes 0.5
            amax = np.abs(relevance).max((1, 2), keepdims=True)
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

        #fname = output_dir+'/'+sample_id+'_'+str(int(target))+'_'+attributor_name+'_'+model_name+'.png'
        #imsave(fname, relevance[0], vmin=0., vmax=1., level=level, cmap=cmap)

        return relevance_raw[0], relevance
        #return relevance_positive[0]


def main(
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
):

    # use the gpu if requested and available, else use the cpu                            
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')   
                                                                                          
    # mean and std of ILSVRC2012 as computed for the torchvision models                   
    norm_fn = BatchNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711), device=device)
                                                                                          
    # transforms as used for torchvision model evaluation                                 
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),                                                                          
        #CenterCrop(224),  
        ToTensor(),                                                                       
    ])                                                                                    
                                                                                          

    # define dataloader and dataset                                                       
    dataset = VOCDetection(                                                                       
        data_root,
        voc_year,
        data_split,
        download=False,                                                                           
        transform=transform,                                                                      
        #target_transform=encode_labels,                                                          
    )                                                                                             
    loader = DataLoader(                                                                      
        dataset,                                                                                  
        batch_size=1,                                                                             
        shuffle=shuffle)

    for model_name in model_dict:                                                         
        model = model_dict[model_name]                                                    
        model.to(device)                                                                  
        model.eval()                                                                      
                                                                                          
        # disable requires_grad for all parameters, we do not need their modified gradients
        for param in model.parameters():                                                  
            param.requires_grad = False                                                   
                                                                                          
    print("MODELS EVAL MODE SETUP AND REQUIRES GRAD MADE FALSE")
    epg_f = open(output_dir+'.txt', 'w')


    sample_index = 0
    for loader_idx, (data, target) in enumerate(loader):

        data_norm = norm_fn(data.clone().to(device))
        target, idx_to_bbox, orig_dims = encode_labels_raw(target)
        target = target.to(device)

        #print("DATA SHAPE", data.shape)

        # save input image
        fname = output_dir+"/"+str(sample_index)                                              
        imsave(fname+"_"+"input.png", data[0])

        flag = True
        
        # compute metrics
        for model_name in model_dict:
            model = model_dict[model_name]

            if '_resnet' in model_name:                                                   
                curr_data = data_norm.clone().to(device)                                  
            else:
                inp = data.clone().to(device)                                                 
                curr_data = torch.cat([inp, 1-inp], dim=1)  

            output = model(curr_data)
            output_probs = nn.Sigmoid()(output)
            output_probs[output_probs>=thresh] = 1.0
            output_probs[output_probs<thresh] = 0.0

            model_perf[model_name][0].append(target.cpu().numpy())
            model_perf[model_name][1].append(output_probs.cpu().numpy()[0])
            
            #if (output_probs[0]-target).sum() != 0:
            #    flag = False
            #    break
        
        """
        for model_name in model_dict:
            print(sample_index, model_name, f1_score(model_perf[model_name][0], model_perf[model_name][1], average='micro'))
        print("\n\n\n\n")
        continue
        """
        if flag == False:
            continue

        target = target.nonzero(as_tuple=True)[0].unsqueeze(0)

        # non-bcos attributions                                                           
        for model_name in model_dict:                                                     
            if "bcos" in model_name:                                                      
                continue                                                                  
            model = model_dict[model_name]                                                
            for attributor_name in ATTRIBUTORS:                                           
                for target_idx in range(target.shape[1]):                                 
                    contribution_map, relevance = generate_map(                                      
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
                            target_idx=target_idx)                                        
                    pg_score = get_epg_val(contribution_map, idx_to_bbox[int(target[0, target_idx].cpu())], base_w=orig_dims[1], base_h=orig_dims[0])


                    fname = output_dir+'/'+str(sample_index)+'_'+str(int(target[0,target_idx]))+'_'+'lrp'+'_'+model_name+'.png'
                    imsave(fname, relevance[0], vmin=0., vmax=1., level=level, cmap=cmap)  
                    attr_methods['lrp_'+model_name]['epg'].append(pg_score)

                    # iou computation                                                         
                    iou_val, th_attn = get_iou_val(contribution_map.copy(), idx_to_bbox[int(target[0, target_idx].cpu())], base_w=orig_dims[1], base_h=orig_dims[0], threshold=0.3, k=None)
                    attr_methods['lrp_'+model_name]['bbox_miou'].append(iou_val)          
                    epg_f.write(fname+' '+str(pg_score)+' '+str(iou_val)+'\n')            
                    fname = output_dir+'/'+str(sample_index)+'_'+str(int(target[0, target_idx]))+'_lrp_'+model_name+'_mask.png'
                    cv2.imwrite(fname, th_attn)                                           
                                                                                          
                    # seg miou coputation                                                 
                    #curr_mask = mask_numpy_array == int(target[0, target_idx])            
                    #seg_iou_val = get_seg_iou_val(contribution_map.copy(), curr_mask, threshold=0.60, k=None)
                    #attr_methods['lrp_'+model_name]['seg_miou'].append(seg_iou_val)

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
                                                                                              
                contribution_map = LayerAttribution.interpolate(contribution_map, (IMAGE_SIZE, IMAGE_SIZE), interpolate_mode='bilinear')
                contribution_map = contribution_map[0, 0].detach().cpu().numpy()              
                                                                                              
                pg_score = get_epg_val(contribution_map, idx_to_bbox[int(target[0, target_idx].cpu())], base_w=orig_dims[1], base_h=orig_dims[0])
                attr_methods['gradcam_'+model_name]['epg'].append(pg_score)                          
                # iou computation                                                         
                iou_val, th_attn = get_iou_val(contribution_map.copy(), idx_to_bbox[int(target[0, target_idx].cpu())], base_w=orig_dims[1], base_h=orig_dims[0], threshold=0.20, k=None)
                attr_methods['gradcam_'+model_name]['bbox_miou'].append(iou_val)          
                # seg miou coputation                                                     
                #curr_mask = mask_numpy_array == int(target[0, target_idx])                
                #seg_iou_val = get_seg_iou_val(contribution_map.copy(), curr_mask, threshold=0.20, k=None)
                #attr_methods['gradcam_'+model_name]['seg_miou'].append(seg_iou_val)
                                                                                              
                # first normalize                                                             
                contribution_map[contribution_map < 0] = 0.0                                  
                amax = np.abs(contribution_map).max((0, 1), keepdims=True)                    
                contribution_map = (contribution_map + amax) / 2 / (amax+1e-12)
                # saving image                                                                
                fname = output_dir+'/'+str(sample_index)+'_'+str(int(target[0,target_idx]))+'_gradcam_'+model_name+'.png'
                imsave(fname, contribution_map, vmin=0., vmax=1., level=1.0, cmap=cmap)

                # save bbox iou image
                epg_f.write(fname+' '+str(pg_score)+' '+str(iou_val)+'\n')
                fname = output_dir+'/'+str(sample_index)+'_'+str(int(target[0, target_idx]))+'_gradcam_'+model_name+'_mask.png'
                cv2.imwrite(fname, th_attn)

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
                                                                                          
                pg_score = get_epg_val(contribution_map, idx_to_bbox[int(target[0, target_idx].cpu())], base_w=orig_dims[1], base_h=orig_dims[0], k=129)
                attr_methods['ixg_'+model_name]['epg'].append(pg_score)                          
                # iou computation                                                         
                iou_val, th_attn = get_iou_val(contribution_map.copy(), idx_to_bbox[int(target[0, target_idx].cpu())], base_w=orig_dims[1], base_h=orig_dims[0], threshold=0.50, k=129)
                attr_methods['ixg_'+model_name]['bbox_miou'].append(iou_val)              
                # seg miou coputation                                                     
                #curr_mask = mask_numpy_array == int(target[0, target_idx])                
                #seg_iou_val = get_seg_iou_val(contribution_map.copy(), curr_mask, threshold=0.50, k=129)
                #attr_methods['ixg_'+model_name]['seg_miou'].append(seg_iou_val) 
                                                                                          
                #contribution_map = torch.from_numpy(contribution_map)                    
                #k = 129                                                                  
                #contribution_map = GaussianBlur(k, sigma=k/4)(contribution_map.unsqueeze(0))[0]
                #contribution_map = contribution_map.numpy()                              
                                                                                          
                # first normalize                                                         
                contribution_map[contribution_map < 0.0] = 0.0                            
                amax = np.abs(contribution_map).max((0, 1), keepdims=True)                
                contribution_map = (contribution_map + amax) / 2 / amax                   
                # saving image                                                            
                fname = output_dir+'/'+str(sample_index)+'_'+str(int(target[0, target_idx]))+'_ixg_'+model_name+'.png'
                imsave(fname, contribution_map, vmin=0., vmax=1., level=1.0, cmap=cmap)   

                # iou computation                                                         
                epg_f.write(fname+' '+str(pg_score)+' '+str(iou_val)+'\n')                
                fname = output_dir+'/'+str(sample_index)+'_'+str(int(target[0, target_idx]))+'_ixg_'+model_name+'_mask.png'
                cv2.imwrite(fname, th_attn)
                                                                                          
            del inp, curr_data, contribution_map                                          
            torch.cuda.empty_cache()


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

                pg_score = get_epg_val(contribution_map, idx_to_bbox[int(target[0, target_idx].cpu())], base_w=orig_dims[1], base_h=orig_dims[0])
                attr_methods['gbp_'+model_name]['epg'].append(pg_score)
                # iou computation                                                         
                iou_val, th_attn = get_iou_val(contribution_map.copy(), idx_to_bbox[int(target[0, target_idx].cpu())], base_w=orig_dims[1], base_h=orig_dims[0], threshold=0.30, k=129)
                attr_methods['gbp_'+model_name]['bbox_miou'].append(iou_val)              
                # seg miou coputation                                                     
                #curr_mask = mask_numpy_array == int(target[0, target_idx])                
                #seg_iou_val = get_seg_iou_val(contribution_map.copy(), curr_mask, threshold=0.30, k=129)
                #attr_methods['gbp_'+model_name]['seg_miou'].append(seg_iou_val)

                # first normalize
                contribution_map[contribution_map < 0.0] = 0.0
                amax = np.abs(contribution_map).max((0, 1), keepdims=True)
                contribution_map = (contribution_map + amax) / 2 / amax
                # saving image
                fname = output_dir+'/'+str(sample_index)+'_'+str(int(target[0, target_idx]))+'_gbp_'+model_name+'.png'
                imsave(fname, contribution_map, vmin=0., vmax=1., level=1.0, cmap=cmap)

                # iou computation                                                         
                epg_f.write(fname+' '+str(pg_score)+' '+str(iou_val)+'\n')                
                fname = output_dir+'/'+str(sample_index)+'_'+str(int(target[0, target_idx]))+'_gbp_'+model_name+'_mask.png'
                cv2.imwrite(fname, th_attn)

            del inp, curr_data, contribution_map
            torch.cuda.empty_cache()

        # intgrad attributions                                                                
        for model_name in model_dict:                                                         
            model = model_dict[model_name]                                                    
            intgrad = IntegratedGradients(model)                                              
                                                                                              
            inp = data.clone().to(device)                                                     
            if '_resnet' in model_name:                                                       
                curr_data = data_norm.clone().to(device)                                      
            else:                                                                             
                curr_data = torch.cat([inp, 1-inp], dim=1)                                    
                                                                                              
            for target_idx in range(target.shape[1]):                                         
                contribution_map = intgrad.attribute(curr_data, target=int(target[0, target_idx]), internal_batch_size=16)
                contribution_map = contribution_map[0].sum(0).detach().cpu().numpy()          
                                                                                              
                pg_score = get_epg_val(contribution_map, idx_to_bbox[int(target[0, target_idx].cpu())], base_w=orig_dims[1], base_h=orig_dims[0], k=129)
                attr_methods['intgrad_'+model_name]['epg'].append(pg_score)                          
                                                                                              
                # first normalize                                                         
                amax = np.abs(contribution_map).max((0, 1), keepdims=True)+1e-12              
                contribution_map = (contribution_map + amax) / 2 / amax                       
                # saving image                                                            
                #fname = output_dir+'/'+str(sample_index)+'_'+str(target_idx)+'_intgrad_'+model_name+'.png'
                #imsave(fname, contribution_map, vmin=0., vmax=1., level=1.0, cmap=cmap)   
                #attr_all_scores['intgrad_'+model_name].append((pg_score, fname))              
                                                                                              
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
                    pg_score = get_epg_val(contribution_map, idx_to_bbox[int(target[0, target_idx].cpu())], base_w=orig_dims[1], base_h=orig_dims[0])
                    attr_methods['bcos_'+model_name]['epg'].append(pg_score)                     
                    # iou computation                                                     
                    iou_val, th_attn = get_iou_val(contribution_map.copy(), idx_to_bbox[int(target[0, target_idx].cpu())], base_w=orig_dims[1], base_h=orig_dims[0], threshold=0.30, k=None)
                    attr_methods['bcos_'+model_name]['bbox_miou'].append(iou_val)         
                    #print(coloured_exp.shape, "coloured exp shape")                      
                    # seg miou coputation                                                     
                    #curr_mask = mask_numpy_array == int(target[0, target_idx])            
                    #seg_iou_val = get_seg_iou_val(contribution_map.copy(), curr_mask, threshold=0.30, k=None)
                    #attr_methods['bcos_'+model_name]['seg_miou'].append(seg_iou_val)
                    #print(coloured_exp.shape, "coloured exp shape")                      
                                                                                          
                    # first normalize                                                     
                    contribution_map[contribution_map < 0.0] = 0.0                        
                    amax = np.abs(contribution_map).max((0, 1), keepdims=True)            
                    contribution_map = (contribution_map + amax) / 2 / amax               
                    # saving image                                                        
                    fname = output_dir+'/'+str(sample_index)+'_'+str(int(target[0, target_idx]))+'_bcos-att_'+model_name+'.png'
                    imsave(fname, contribution_map, vmin=0., vmax=1., level=1.0, cmap=cmap)
                    epg_f.write(fname+' '+str(pg_score)+' '+str(iou_val)+'\n')            
                    fname = output_dir+'/'+str(sample_index)+'_'+str(int(target[0, target_idx]))+'_bcos-att_'+model_name+'_mask.png'
                    cv2.imwrite(fname, th_attn) 
                    # save coloured image                                                 
                    fname = output_dir+'/'+str(sample_index)+'_'+str(int(target[0, target_idx]))+'_bcos-col_'+model_name+'.png'
                    epg_f.write(fname+' '+str(pg_score)+'\n')
                    plt.imshow(coloured_exp)                                              
                    plt.axis('off')                                                       
                    plt.savefig(fname, bbox_inches='tight')                               
                    plt.close()                                                           
                    #fname = output_dir+'/'+str(sample_index)+'_'+str(target_idx)+'_bcos-col_'+model_name+'.png'
                    #imsave(fname, coloured_exp)                                          
                    # save curr_mask                                                      
                    #cv2.imwrite(output_dir+'/'+str(sample_index)+'_'+str(int(target[0, target_idx]))+'_classmask.png', curr_mask.astype(np.uint8)*255)
                                                                                          
        for metric in ['epg', 'bbox_miou']:                                   
            for attr in attr_methods:                                                     
                print(attr, metric,  np.nanmean(attr_methods[attr][metric]), np.nanstd(attr_methods[attr][metric]))
            print("\n\n")


        image_names_raw = os.listdir(output_dir)
        for target_idx in range(target.shape[1]):
            filt_patt = str(sample_index)+'_'+str(int(target[0, target_idx]))
            image_names = [x for x in image_names_raw if filt_patt in x]
            bboxes = idx_to_bbox[int(target[0, target_idx])]
            base_h, base_w = orig_dims
            for image_name in image_names:
                img = cv2.imread(output_dir+'/'+image_name)
                h,w = img.shape[:2]
                for bbox in bboxes:                                                                       
                    x_min, y_min, x_max, y_max = bbox                                                     
                    x_min, x_max = (w*x_min)//base_w, (w*x_max)//base_w
                    y_min, y_max = (h*y_min)//base_h, (h*y_max)//base_h
                    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

                cv2.imwrite(output_dir+'/'+image_name, img)


        """
        for model_name in model_dict:
            #print(model_name, model_perf[model_name][0], model_perf[model_name][1])
            print(sample_index, model_name, f1_score(model_perf[model_name][0], model_perf[model_name][1], average='micro'))
        """

        print(sample_index, "DONE!")
        print("\n\n\n\n")
        sample_index += 1
        if sample_index > 1000:
            break

    epg_f.close()

        

if __name__ == '__main__':                                                                    
    main()                                                                                    
    exit(0)
