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


# Opening JSON file
f = open('config.json')
# returns JSON object as 
config = json.load(f)

def check_object_number(image_path: str):
    json_image_path = image_path[:-3] + 'json'
    
    with open(json_image_path) as f:
        object_data = json.load(f)
    
    return(object_data["numberOfTargetsInFrame"] > 0)


def change_bit_depth(path_image):
    if (check_object_number(path_image)):
        rgba_image = cv2.imread(path_image, cv2.IMREAD_UNCHANGED)
        # Load the RGBA image using OpenCV

        gray_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2GRAY)
        # Convert the RGBA image to grayscale (one band)

        gray_image_16bit = (gray_image.astype(np.uint16) << 8)
        # The line above left-shifts the 8-bit grayscale values by 8 bits to convert it to 16-bit.
        # This effectively fills the additional 8 bits with zeros, expanding the bit depth.
        
        gray_image_16bit_pil = Image.fromarray(gray_image_16bit)
        # Create a 16-bit grayscale image using PIL

        destination_filepath = os.path.join(destination_folder, os.path.splitext(os.path.basename(path_image))[0] + '.png')
        #creating file path in the 'destination folder'

        #gray_image_16bit_pil.save(destination_filepath, "JPEG2000" , bits = 16)
        gray_image_16bit_pil.save(destination_filepath, bits = 16) 
        # This line saves the 16-bit grayscale image as a png file with 16 bits per sample.
        # The "bits=16" argument ensures that the image is saved with a 16-bit depth.
        


    
path_to_segmentation = config['path_to_segmentation']
images_folder = os.path.join(path_to_segmentation, "Images")

destination_folder = os.path.join(config['output_folder'], "Images")
os.makedirs(destination_folder, exist_ok= True)

if config['imagery']['bit_depth']['change']:
    print (f"changing bit depth resolution to {config['imagery']['bit_depth']['value']}")
    for path_image in tqdm(glob(images_folder + '\\*.png')):
        change_bit_depth(path_image)



print ('Exporting from fornax segmentation of objects and bboxes')
ue_dict = utils.get_segmentation()
print (ue_dict)
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








