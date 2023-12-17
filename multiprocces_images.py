import multiprocessing
import cv2
import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from glob import glob
import pickle

# Opening JSON file
f = open('config.json')
# returns JSON object as 
config = json.load(f)

def is_there_object_number(image_path: str):
    json_image_path = image_path[:-3] + 'json'
    
    with open(json_image_path) as f:
        object_data = json.load(f)
    
    return(object_data["numberOfTargetsInFrame"] > 0)


def change_bit_depth(path_image, destination_folder):
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
        

def worker_function(target):
    path_to_segmentation = config['path_to_segmentation']
    images_folder = os.path.join(path_to_segmentation, "Images")

    destination_folder = os.path.join(config['output_folder'], "Images")
    destination_folder_empty_images = os.path.join(config['output_folder'], "Empty_Images")

    os.makedirs(destination_folder, exist_ok= True)
    os.makedirs(destination_folder_empty_images, exist_ok= True)

    if config['imagery']['bit_depth']['change']:
        print (f"changing bit depth resolution to {config['imagery']['bit_depth']['value']}")
        for path_image in tqdm(glob(images_folder + '\\*.png')[target[0]: target[1]]):    
            if is_there_object_number(path_image) > 0:#if image has objects
                change_bit_depth(path_image, destination_folder)
            else:
                change_bit_depth(path_image, destination_folder_empty_images)   


if __name__ == '__main__':
    # Get the number of CPU cores available on the system
    num_cores = multiprocessing.cpu_count() - 1
    
    path_to_segmentation = config['path_to_segmentation']
    images_folder = os.path.join(path_to_segmentation, "Images")
    list_ =  glob(images_folder + '\\*.png')

    destination_folder = os.path.join(config['output_folder'], "Images")
    os.makedirs(destination_folder, exist_ok= True)

    # Calculate the size of each sub-range of data for parallel processing
    divisor = len(list_) // num_cores
    list_groups = []
    start = 0

    # Divide the data into sub-ranges based on the number of CPU cores
    for i in range(1, num_cores, 1):
        list_groups.append([start, divisor * i])
        start = divisor * i
    list_groups.append([start, len(list_)])

    # Print the list of sub-ranges and the number of CPU cores
    print(list_groups)
    print(num_cores)

    # Create a list to store the process instances
    processes = []

    # Start a separate process for each sub-range of data
    for number in range(num_cores):
        process = multiprocessing.Process(target=worker_function, args=(list_groups[number],))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()