import ue_to_coco as coco
import utils_coco as utils
import main_metadata_generator as custom_metadata
import os
import json
from PIL import Image
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from shutil import copy
import subprocess

def check_object_number(image_path: str):
    json_image_path = image_path[:-3] + 'json'
    
    with open(json_image_path) as f:
        object_data = json.load(f)
    
    return(object_data["numberOfTargetsInFrame"] > 0)

# Opening JSON file
f = open('config.json')
# returns JSON object as 
config = json.load(f)
        


path_to_segmentation = config['path_to_segmentation']
images_folder = os.path.join(path_to_segmentation, "Images")

destination_folder = os.path.join(config['output_folder'], "Images")
os.makedirs(destination_folder, exist_ok= True)

destination_folder_empty_images = os.path.join(config['output_folder'], "Empty_Images")
os.makedirs(destination_folder_empty_images, exist_ok= True)


if config['imagery']['bit_depth']['change']:
    print (f"changing bit depth resolution to {config['imagery']['bit_depth']['value']}")
    subprocess.run(['python', 'multiprocces_images.py'])

else:
    pass
    """
    #we just copy the images to the output folder
    path_to_segmentation = config['path_to_segmentation']
    images_folder = os.path.join(path_to_segmentation, "Images")

    for image_path in glob(os.path.join(images_folder, '*.png')):
        if (check_object_number(image_path) > 0):
            copy(image_path, os.path.join(destination_folder, os.path.basename(image_path)))
        else:
            copy(image_path, os.path.join(destination_folder_empty_images, os.path.basename(image_path)))
    """



print ('Exporting from fornax segmentation of objects and bboxes')
ue_dict = utils.get_segmentation() #can be multiproccesed
json_file_path = 'savedata.json'

# Save the data to a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(ue_dict, json_file)

if config['export']['coco']:
    #crate and export data in coco format (There is no support for rotated bounding boxes, the format itself does not support it)
    coco.generate_dada(ue_dict)

if config['export']['custom']:
    #create and export data in metadata format
    custom_metadata.generate_data(ue_dict)








