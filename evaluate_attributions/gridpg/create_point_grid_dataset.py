import os
import random
import cv2
import numpy as np

def create_grid_sample_2x2(ord_images, no_of_classes, base_dir):
    count = 0
    sel_images = []
    sel_classes = []
    while True:
        curr_class = random.randint(0, no_of_classes-1)
        if curr_class not in sel_classes and curr_class in ord_images and len(ord_images[curr_class]) > 0:
            sel_classes.append(curr_class)
            curr_idx = random.randint(0, len(ord_images[curr_class])-1)
            img = cv2.imread(base_dir+ord_images[curr_class][curr_idx])
            sel_images.append(img)

        if len(sel_classes) == 4:
            break

    sel_classes = [str(x) for x in sel_classes]
    sel_classes = '_'.join(sel_classes)

    # making a grid (L->R)
    row1 = np.concatenate([sel_images[0], sel_images[1]], axis=1)
    row2 = np.concatenate([sel_images[2], sel_images[3]], axis=1)
    final_image = np.concatenate([row1, row2], axis=0)

    return final_image, sel_classes

def create_grid_sample_3x3(ord_images, no_of_classes, base_dir):
    count = 0
    sel_images = []
    sel_classes = []
    while True:
        curr_class = random.randint(0, no_of_classes-1)
        if curr_class not in sel_classes and curr_class in ord_images and len(ord_images[curr_class]) > 0:
            sel_classes.append(curr_class)
            curr_idx = random.randint(0, len(ord_images[curr_class])-1)
            img = cv2.imread(base_dir+ord_images[curr_class][curr_idx])
            sel_images.append(img)

        if len(sel_classes) == 9:
            break

    sel_classes = [str(x) for x in sel_classes]
    sel_classes = '_'.join(sel_classes)

    # making a grid (L->R)
    row1 = np.concatenate([sel_images[0], sel_images[1], sel_images[2]], axis=1)
    row2 = np.concatenate([sel_images[3], sel_images[4], sel_images[5]], axis=1)
    row3 = np.concatenate([sel_images[6], sel_images[7], sel_images[8]], axis=1)
    final_image = np.concatenate([row1, row2, row3], axis=0)

    return final_image, sel_classes




base_dir = '/path/to/confident_images_subset_dir/'
raw_images = os.listdir(base_dir)
#raw_images = [x for x in raw_images if 'sup' not in x]
number_of_classes = -1
class_to_image_dict = {}
for image in raw_images:
    image_class = int(image.split('_')[1].split('.')[0])
    number_of_classes = max(number_of_classes, int(image_class))
    if image_class not in class_to_image_dict:
        class_to_image_dict[image_class] = []
    class_to_image_dict[image_class].append(image)

"""
ordered_images = []
for idx in range(number_of_classes+1):
    ordered_images.append([])

for image in raw_images:
    image_class = int(image.split('_')[0])
    ordered_images[image_class].append(image)
"""

output_dir_2x2 = 'grid_pg_images_2x2'
output_dir_3x3 = 'grid_pg_images_3x3'
dataset_size = 500
for idx in range(dataset_size):
    img, name = create_grid_sample_2x2(class_to_image_dict, number_of_classes+1, base_dir) 
    cv2.imwrite(output_dir_2x2+'/'+str(idx)+'_'+name+'.png', img)
    img, name = create_grid_sample_3x3(class_to_image_dict, number_of_classes+1, base_dir) 
    cv2.imwrite(output_dir_3x3+'/'+str(idx)+'_'+name+'.png', img)
    print("IDX {} Done!".format(idx))
