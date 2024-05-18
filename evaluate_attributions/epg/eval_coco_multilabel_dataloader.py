import os                                                                                     
import random                                                                                 
import numpy as np                                                                            
import cv2                                                                                    
                                                                                              
import torch                                                                                  
from torch.utils.data import Dataset                                                          
import torchvision.transforms.functional as F                                                 
import torch.nn.functional as nn_F                                                            
from PIL import Image                                                                         

from pycocotools.coco import COCO
                                                                                              
class image_loader(Dataset):
    metainfo = {                                                                          
        'classes':                                                                        
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',            
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',                   
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',                
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',           
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',                    
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',           
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',             
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',               
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',              
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',                  
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',              
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',              
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),                           
        # palette is a list of color tuples, which is used for visualization.             
        'palette':                                                                        
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),           
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),             
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),                    
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),                   
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),                       
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),                      
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),                   
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),                       
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),                
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),                  
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),                       
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),                 
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),                       
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),                    
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),                      
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),                   
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),                  
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),                   
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),                       
         (246, 0, 122), (191, 162, 208)]                                                  
    }
                                                                                              
    def __init__(self, data_dir, txt_file, input_transforms=None):                            

        self.root = data_dir
        self.coco = COCO(txt_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.input_transforms = input_transforms                                              
        self.cat_ids = self.get_cat_ids(                                                 
            cat_names=self.metainfo['classes'])                                               
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):                            
        return self.coco.getCatIds(cat_names, sup_names, cat_ids)

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        idx_to_bbox = {}
        width, height = img.size
        orig_dims = [height, width]
        mask_dict = {}
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.cat2label[coco_annotation[i]['category_id']])

            if labels[-1] not in idx_to_bbox:
                idx_to_bbox[labels[-1]] = []
            idx_to_bbox[labels[-1]].append([int(xmin), int(ymin), int(xmax), int(ymax)])

            # get mask
            if int(labels[-1]) not in mask_dict:
                mask_dict[int(labels[-1])] = np.zeros((height, width))

            curr_mask = coco.annToMask(coco_annotation[i])
            mask_dict[int(labels[-1])][curr_mask > 0] = 1
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        labels_vector = np.zeros(len(self.metainfo['classes']))
        labels_vector[list(labels)] = 1
        labels_vector = torch.from_numpy(labels_vector)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels_vector
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        my_annotation["idx_to_bbox"] = idx_to_bbox
        my_annotation["orig_dims"] = orig_dims

        if self.input_transforms is not None:
            img = self.input_transforms(img)
        
        for lbl in mask_dict:
            mask_dict[lbl] = cv2.resize(mask_dict[lbl], (img.shape[2], img.shape[1]), interpolation=cv2.INTER_NEAREST)
        my_annotation["mask"] = mask_dict

        return img, my_annotation
                                                                                              
    def __len__(self):                                                                        
        return len(self.ids)
