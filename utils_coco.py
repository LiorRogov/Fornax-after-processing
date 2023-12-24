import csv
import math
from tkinter.tix import Tree
import cv2
import numpy as np
import glob
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import json
import shutil 
import os

# Opening JSON file
f = open('config.json')
# returns JSON object as 
config = json.load(f)

is_debug = False
path_to_segmentation = config['path_to_segmentation']
metadata_folder = os.path.join(path_to_segmentation, "Metadata")
images_folder = os.path.join(path_to_segmentation, "Images")
masks_folder =  os.path.join(path_to_segmentation, "Masks")
metadata_name = config['metadata_name']
metadata_delimiter = config['metadata_delimiter']
TARGET_DATA_PATH = os.path.join(path_to_segmentation, "TargetsData")
OUTPUT_FOLDER =  config['output_folder']

def extract_segmentation_mask(mask, contour):
    # Read the image
    img = mask

    # Create an empty mask
    mask = np.zeros(img.shape, dtype=np.uint8)

    # Draw the contour on the visualization image
    cv2.drawContours(mask, [contour], -1, (255,255,255), thickness= cv2.FILLED)
    """
    cv2.imshow('countur mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    returned_mask = np.where(mask > 0, img, mask)

    """
    cv2.imshow('returned_mask', returned_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    return returned_mask

def get_object_color(mask ,color_array, countour):
    max_color_segmentation_pixel = 0
    max_color = []
    segmentation_mask = extract_segmentation_mask(mask, countour)

    for color in list(color_array):
        #debug
        """
        cv2.imshow('segmentation_mask', segmentation_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        
        R, G, B = color
        BGR = np.array([B, G, R])

        intensity_range = 0.05
        lower = np.clip(BGR - intensity_range * 255, 0, 255)
        upper = np.clip(BGR + intensity_range * 255, 0, 255)

        object_mask = cv2.inRange(segmentation_mask, lower, upper)

        count = cv2.countNonZero(object_mask)
        #debug
        """
        cv2.imshow('object_mask', object_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        
        if count > max_color_segmentation_pixel:
            max_color = color
            max_color_segmentation_pixel = count
    
    if max_color == []:
        return ([])
    
    return tuple(max_color)

    

def get_colours():
    """
    Open the Json file created by UE and extract the available data

    :return dict_colours: dictionary with colours, tags and object ids
    :rtype: dictionary
        key: coulour tuple (R,G,B) as int
        value: tuple of object id and category ('tag' in UE)
    """
    dict_colours = {}

    for object_json in glob.glob(f'{TARGET_DATA_PATH}/*.json'):
        #open metadata file 
        with open(object_json) as f:
            object_data = json.load(f)
        
        R = int(np.round(object_data["MaskRed"] * 255))
        G = int(np.round(object_data["MaskGreen"] * 255))
        B = int(np.round(object_data["MaskBlue"] * 255))
        
        dict_colours[(R,G,B)] = (str(object_data['MaskStencilCode']), object_data['tagName'])
           
    return dict_colours

"""
def get_polygons(image_name, dict_colours):
   
    Process flat colour images from UE and associated metadata
    and return polygon points and bounding boxes

    :param str image_name: path to the image
    :param dictionary dict_colours: colour dictionary
    :return list_poly: list of polygon points for each object on the image as well as
        it's bounding box, object id, category and contour total area, if the the object is
        split into multiple areas, each of them has a new list with polygon points
    :rtype: list
        structure: [[(x1, y1), (x2, y2),...], [(x1, y1), (x2, y2),...], (bbox), (id, category, area)]
  
    list_poly = []

    # load the flat coloured mask image
    image_base_name, image_ext = os.path.splitext(os.path.basename(image_name))
    mask_name = os.path.join(masks_folder, image_base_name + image_ext)
    
    mask = cv2.imread(mask_name)
    mask_grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)



    # process all colours to find the contours and bounding boxes
    for colour in dict_colours.keys():
        # define color boundaries
        # turn RGB into BGR (since OpenCV represents images as NumPy arrays in reverse order)
        R, G, B = colour
        BGR = np.array([B, G, R])
        
        intensity_range = 0.05
        lower = np.clip(BGR - intensity_range * 255, 0, 255)
        upper = np.clip(BGR + intensity_range * 255, 0, 255)
    
        # convert boundaries to NumPy arrays
        #lower = np.array(lower, dtype="uint8")
        #upper = np.array(upper, dtype="uint8")
        # find the colours within the specified boundaries and apply the mask
        object_mask = cv2.inRange(mask, lower, upper)
        # isolated_object = cv2.bitwise_and(image_flat, image_flat, mask = object_mask)

        # get contours using point approximation
        #countours = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        countours, _  = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        
        # add countours to segmentation structure
        for contour in countours:
            # Extract polygon pixels
            polygon_pixels = contour.squeeze()
            #if len(polygon_pixels) > 5:
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate rotated bounding box
            rotated_rect = cv2.minAreaRect(contour)
            rotated_box = cv2.boxPoints(rotated_rect).astype(int)

            # Calculate contour area
            contour_area = cv2.contourArea(contour)

            #debug
        
            cv2.drawContours(object_mask, [rotated_box], 0, (0, 255, 0), 2)  # Draw contours in blue
            cv2.imshow(f'{(R, G, B)}', object_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


            list_poly.append([polygon_pixels.tolist(), #segmentation pixels
                            (x, y, w, h), #bounding box
                            dict_colours[colour] + (contour_area,), #mask color and total area in pixels
                            rotated_box.tolist() # rotated bbox
                            ])

    return list_poly
"""
def get_polygons(image_name, dict_colours):
    """
    Process flat colour images from UE and associated metadata
    and return polygon points and bounding boxes

    :param str image_name: path to the image
    :param dictionary dict_colours: colour dictionary
    :return list_poly: list of polygon points for each object on the image as well as
        it's bounding box, object id, category and contour total area, if the the object is
        split into multiple areas, each of them has a new list with polygon points
    :rtype: list
        structure: [[(x1, y1), (x2, y2),...], [(x1, y1), (x2, y2),...], (bbox), (id, category, area)]
    """
    list_poly = []
    counter = 0
    # load the flat coloured mask image
    image_base_name, image_ext = os.path.splitext(os.path.basename(image_name))
    mask_name = os.path.join(masks_folder, image_base_name + image_ext)
    
    mask = cv2.imread(mask_name)
    mask_grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # get contours using point approximation
    #countours = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    countours, _  = cv2.findContours(mask_grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # add countours to segmentation structure
    for contour in countours:
        # Extract polygon pixels
        polygon_pixels = contour.squeeze()
       
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate rotated bounding box
        rotated_rect = cv2.minAreaRect(contour)
        rotated_box = cv2.boxPoints(rotated_rect).astype(int)

        # Calculate contour area
        contour_area = cv2.contourArea(contour)

        #lets try to understand what object it is
        colour = get_object_color(mask, dict_colours.keys(), contour)

        if colour != []:
            list_poly.append([polygon_pixels.tolist(), #segmentation pixels
                            (x, y, w, h), #bounding box
                            dict_colours[colour] + (contour_area,), #mask color and total area in pixels
                            rotated_box.tolist() # rotated bbox
                            ])
        else:
            counter+= 1
            print (counter)
            

        """
        cv2.drawContours(object_mask, [rotated_box], 0, (0, 255, 0), 2)  # Draw contours in blue
        cv2.imshow(f'{(R, G, B)}', object_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

    return list_poly


def check_object_number(image_path: str):
    json_image_path = image_path[:-3] + 'json'
    
    with open(json_image_path) as f:
        object_data = json.load(f)
    
    return(object_data["numberOfTargetsInFrame"] > 0)
    
def get_segmentation():
    failed_folder = os.path.join(OUTPUT_FOLDER, 'Failed_Images')
    os.makedirs(failed_folder, exist_ok= True)
    """
    Get segmentation data for all images in a folder

    :return dict_segmentation: dictionary of polygon points for each object on the image as well as
        it's bounding box, object id, category and contour total area, if the the object is
        split into multiple areas, each of them has a new list with polygon points
    :rtype: dictionary
        key: image path
        value: [[(x1, y1), (x2, y2),...], [(x1, y1), (x2, y2),...], (bbox), (id, category, area)]
    """
    #print("get_segmentation -> Working on segmentation...")

    dict_segmentation = {}
    # open csv file from UE and retrieve the data
    # key = colour; values = id, category
    dict_colours = get_colours()

    # get all image file names excluding the the segmentation ones ending wih "_s"
    images = glob.glob(images_folder + "/*.png")
    print("Total images: " + str(len(images))) #total number of images

    list_failed = []
    for image in tqdm(images):
        if check_object_number(image) > 0:
            list_poly = get_polygons(image, dict_colours)
            if list_poly:
                dict_segmentation[image] = list_poly

            else: #although the image has objects, it failed to recognize them...
                list_failed.append(image)
                print("No Segmentation found in image " + str(image))
                shutil.copy(image, os.path.join(failed_folder, os.path.basename(image)))
                #remove failed images from output images folder
                #remove from folder:
                folder = os.path.join(OUTPUT_FOLDER, 'Images')
                try:
                    os.remove(os.path.join(folder, os.path.basename(image))) #removing failed images from output
                except:
                    pass
        
    table = {'path_list': list_failed}
    with open(os.path.join(OUTPUT_FOLDER, 'failed_images.json'), 'w') as outfile:
        json.dump(table, outfile, indent=4)

    return dict_segmentation
