import os                                                                                     
import random                                                                                 
import numpy as np                                                                            
import cv2                                                                                    
                                                                                              
import torch                                                                                  
from torch.utils.data import Dataset                                                          
import torchvision.transforms.functional as F                                                 
import torch.nn.functional as nn_F                                                            
from PIL import Image                                                                         
                                                                                              
class image_loader(Dataset):                                                                  
                                                                                              
    def __init__(self, data_dir, txt_file, input_transforms=None):                            
        self.data_dir = data_dir                                                              
        f = open(txt_file)                                                                    
        # load file_names                                                                     
        self.filenames = [x.strip() for x in f.readlines()]                                   
        self.input_transforms = input_transforms                                              
                                                                                              
    def __getitem__(self, index):                                                             
        image_name = os.path.join(self.data_dir, self.filenames[index])                       
                                                                                              
        with open(image_name, "rb") as f:                                                     
            image = Image.open(f).convert("RGB")                                              
                                                                                              
        if self.input_transforms is not None:                                                 
            image = self.input_transforms(image)                                              
                                                                                              
        labels = np.array([int(x) for x in self.filenames[index].split('.')[0].split('_')])[1:]
        labels = torch.from_numpy(labels)                                                     
                                                                                              
        return image, labels                                                                  
                                                                                              
    def __len__(self):                                                                        
        return len(self.filenames)
