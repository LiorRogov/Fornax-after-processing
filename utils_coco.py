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
TARGET_DATA_PATH = config['dir_of_objects_info']
OUTPUT_FOLDER =  config['output_folder']

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

def find_closest_color(target_color, color_array):
    """
    Find the closest color in an array to a target color.

    Parameters:
    - target_color: The target RGB color as a tuple (R, G, B).
    - color_array: An array of RGB colors, each represented as a tuple (R, G, B).

    Returns:
    - The closest color from the array.
    """
    target_color = np.array(target_color)
    color_array = np.array(color_array)

    # Calculate the Euclidean distance between the target color and each color in the array
    distances = np.linalg.norm(color_array - target_color, axis=1)

    # Find the index of the closest color
    closest_index = np.argmin(distances)

    # Return the closest color
    closest_color = color_array[closest_index]
    return tuple(closest_color.tolist())

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

    # load the flat coloured mask image
    image_base_name, image_ext = os.path.splitext(os.path.basename(image_name))
    mask_name = os.path.join(masks_folder, image_base_name + image_ext)
    
    mask = cv2.imread(mask_name)
    kernel = np.ones((3,3), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations = 1)
    mask = dilation


    # process all colours to find the contours and bounding boxes
    for colour in dict_colours.keys():
        # define color boundaries
        # turn RGB into BGR (since OpenCV represents images as NumPy arrays in reverse order)
        R, G, B = colour
        BGR = np.array([B, G, R])
        
        intensity_range = 0.1 # 25% intensity range
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
            """
            cv2.drawContours(object_mask, [rotated_box], 0, (0, 255, 0), 2)  # Draw contours in blue
            cv2.imshow(f'{(R, G, B)}', object_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

            list_poly.append([polygon_pixels.tolist(), 
                            (x, y, w, h), 
                            dict_colours[colour] + (contour_area,), 
                            rotated_box.tolist()
                            ])

    return list_poly



    
def get_segmentation():
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
        #object_in_images = check_object_number(image)
        list_poly = get_polygons(image, dict_colours)
        if list_poly:
            dict_segmentation[image] = list_poly

        else:
            #dict_segmentation[image] = [[[0, 0], [1, 1]], (0, 0, 1, 1), (-1, 'None', 0.0), [[0,0], [0,1], [1,0], [1,1]]]
            list_failed.append(image)
            print("No Segmentation found in image " + str(image))
        
    table = {'path_list': list_failed}
    with open(os.path.join(OUTPUT_FOLDER, 'failed_images.json'), 'w') as outfile:
        json.dump(table, outfile, indent=4)

    return dict_segmentation
