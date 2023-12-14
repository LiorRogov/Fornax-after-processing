from datetime import datetime
from importlib import metadata
from math import fabs
import metadata_utiles as ue
import os
import cv2
import json
import pandas
from tqdm import tqdm 
import glob

# Opening JSON file
f = open('config.json')
# returns JSON object as 
config = json.load(f)

# file save name
json_name = "metadata.json"
dataset_version_number = "1.1"

def count_frames():
    num_of_frames = 0
    for file in os.listdir(ue.images_folder):
        if file.endswith(".png"):
            num_of_frames += 1
    return num_of_frames

def get_empty_json_format(annotation : bool = True ):
    # structure of the json
    if annotation:
        json_dict = {
            "info":
            {
                "description": "Vehicles Dataset",
                "version": dataset_version_number,
                "year": datetime.now().year,
                "generated_by": "Simlat",
                "date_created": datetime.today().strftime("%Y/%m/%d"),
                "number_of_frames": count_frames()
            },
            "models_list": [],
            "images": [],
            "annotations": [],
            "objects": []
        }
    
    else:
        json_dict = {
        "info":
        {
            "description": "Vehicles Dataset",
            "version": dataset_version_number,
            "year": datetime.now().year,
            "generated_by": "Simlat",
            "date_created": datetime.today().strftime("%Y/%m/%d"),
            "number_of_frames": count_frames()
        },
        "models_list": [],
        "images": [],
        }
    return json_dict




def make_objects(ue_dict):
    """
    Create categories from processed UE data

    :param dictionary ue_dict: segmentation data
    :return list_objects: list of dictionaries containing categories
    :rtype: list
        structure: [{supercategory, id, name}, {},...]
    """
    list_objects = []
    list_cat_appeared = []
    counter_cat = 0
    for data in tqdm(ue_dict.values()):
        for contours in data:
            object = contours[-2][1]
            if object not in list_cat_appeared:
                print (object)
                dict_category = {}
                list_cat_appeared.append(object)
                counter_cat += 1
                #dict_category["supercategory"] = "object"
                #dict_category["id"] = counter_cat
                #dict_category["object_class"] = object
                for file in os.listdir(ue.metadata_folder):
                    if file.startswith("target_" + object) and file.endswith('json'):
                        """
                        metadata_file = pandas.read_csv(ue.metadata_folder + "\\" + file)
                        dict_category["length"] = metadata_file["width"][0]
                        dict_category["width"] = metadata_file["height"][0]
                        dict_category["height"] = metadata_file["depth"][0]
                        """
                        #data is read now from json:
                        """
                        {
                            "tagName": "ambulance",
                            "categoryName": "vehicles",
                            "targetWidth": 1.9340488910675049,
                            "targetLenght": 5.1030879020690918,
                            "targetHeight": 2.237922191619873
                        }
                        """
                        f = open(ue.metadata_folder + "\\" + file)
                        data = json.load(f)

                        table = {
                            "id" : counter_cat,
                            "object_class" : object,
                            #"tagName" : data['tagName'],
                            "categoryName": data['categoryName'],
                            "length": data['targetLenght'],
                            "width": data["targetWidth"],
                            "height": data['targetHeight']
                        }
                list_objects.append(table)
    return list_objects

def find_index_of_path_in_list(image_path: str):
    path_to_segmentation = config['path_to_segmentation']
    images_folder = os.path.join(path_to_segmentation, "Images")
    image_list = glob.glob(os.path.join(images_folder, "*.png"))

    return image_list.index(image_path) + 1

def make_images(ue_dict):
    """
    Create image data from processed UE data

    :param dictionary ue_dict: segmentation data
    :return list_images: list of dictionaries containing image data
    :rtype: list
        structure: [{}, {},...]
    """
    list_images = []
    for image_id, image in tqdm(enumerate(ue_dict.keys())):
        json_file_path = image.replace("png","json")
        # Opening JSON file
        f = open(json_file_path)
        # returns JSON object as a dictionary
        dict_image = json.load(f)
        if dict_image["numberOfTargetsInFrame"] != 0:
            table = {
                "filename" : image.split("\\")[-1],
                "payloadName": dict_image['payloadName'],
                "sensorType": dict_image["sensorType"],
                "id" : image_id + 1,
                #"list_id": find_index_of_path_in_list(image_path= image),
                "height" : dict_image["frameHeight"],
                "width" : dict_image["frameWidth"],
                "hFov" : dict_image["hFov"],
                "vFov" : dict_image["vFov"],
                "gsd": dict_image["gSD"],
                "polarity": dict_image["polarity"],
                "cam_pitch" : dict_image["gimbalPitch"],
                "cam_roll" : dict_image["gimbalRoll"],
                "cam_yaw" : dict_image["gimbalYaw"],
                "payloadAGL" : dict_image["payloadAGL"],
                #dict_image["payloadLatitude"]: 0, #TODO: Add them to the data on each image
                #dict_image["payloadLongitude"]: 0,
                "landform" : dict_image["landform"],
                "season" : dict_image["season"],
                "lightCondition" : dict_image["lightCondition"],
                "weatherCondition" : dict_image["weatherCondition"],
                "fogLevel" : dict_image["fogLevel"],
                "timeOfDay" : dict_image["timeOfDay"],
                "levelOfBlurriness" : dict_image["levelOfBlurriness"]
            }
            list_images.append(table)


        
    print("Image Count " + str(len(list_images)))
    return list_images

def get_image_data_with_no_objects(ue_dict):
    path_to_segmentation = config['path_to_segmentation']
    images_folder = os.path.join(path_to_segmentation, "Images")

    index = len(ue_dict.keys())
    list_images = []
    for image in glob.glob(images_folder + '//*.png'):   
        json_file_path = image.replace("png","json")
        # Opening JSON file
        f = open(json_file_path)
        # returns JSON object as a dictionary
        dict_image = json.load(f)
        if dict_image["numberOfTargetsInFrame"] == 0:
            table = {
                "filename" : image.split("\\")[-1],
                "payloadName": dict_image['payloadName'],
                "sensorType": dict_image["sensorType"],
                "id" : index + 1,
                #"list_id": image_id + 1,
                "height" : dict_image["frameHeight"],
                "width" : dict_image["frameWidth"],
                "hFov" : dict_image["hFov"],
                "vFov" : dict_image["vFov"],
                "gsd": dict_image["gSD"],
                "polarity": dict_image["polarity"],
                "cam_pitch" : dict_image["gimbalPitch"],
                "cam_roll" : dict_image["gimbalRoll"],
                "cam_yaw" : dict_image["gimbalYaw"],
                "payloadAGL" : dict_image["payloadAGL"],
                #dict_image["payloadLatitude"]: 0, #TODO: Add them to the data on each image
                #dict_image["payloadLongitude"]: 0,
                "landform" : dict_image["landform"],
                "season" : dict_image["season"],
                "lightCondition" : dict_image["lightCondition"],
                "weatherCondition" : dict_image["weatherCondition"],
                "fogLevel" : dict_image["fogLevel"],
                "timeOfDay" : dict_image["timeOfDay"],
                "levelOfBlurriness" : dict_image["levelOfBlurriness"]
            }
            list_images.append(table)
            index += 1
    
    return list_images

def make_model_list(ue_dict):
    """
    Create models metadata

    :param dictionary ue_dict: segmentation data
    :return list_models: list of dictionaries containing models data
    :rtype: list
        structure: [{}, {},...]
    """
    list_models = []
    for file in tqdm(os.listdir(ue.metadata_folder)):
        if file.startswith("target") and file.endswith('json'):
            #metadata_file = pandas.read_csv(ue.metadata_folder + "\\" + file)  
            
            f = open(file)
            metadata_file = json.load(f)
            """
            {
                "tagName": "ambulance",
                "categoryName": "vehicles",
                "targetWidth": 1.9340488910675049,
                "targetLenght": 5.1030879020690918,
                "targetHeight": 2.237922191619873
            }
            """
            dict_target = {}
            dict_target["object_class"] = metadata_file["tagName"]
            #dict_target["categoryName"] = metadata_file["categoryName"]
            dict_target["width"] = metadata_file["targetWidth"]
            dict_target["height"] = metadata_file["targetLenght"]
            dict_target["depth"] = metadata_file["targetHeight"]

            list_models.append([dict_target])

    return list_models


def make_annotations(json_dict, ue_dict):
    """
    Create annotation data from processed UE data

    :param dictionary ue_dict: segmentation data
    :return list_annotations: list of dictionaries containing annotation data
    :rtype: list
        structure: [{}, {},...]
    """
    list_annotations = []
    for image_id, objects in tqdm(enumerate(ue_dict.values())):
        for contours in objects:
            dict_contours = {}
            dict_contours["segmentation"] = []
            # convert a list of tuples of ints into a list of ints
            for cont_list in contours[:-3]:
                print(len(cont_list))
                if len(contours) > 3 and len(cont_list) <= 2:
                    continue
                dict_contours["segmentation"].append([coord for pair in cont_list for coord in pair])
            dict_contours["area"] = contours[-2][2]
            dict_contours["image_id"] = image_id + 1
            dict_contours["bbox"] = list(contours[-3])
            dict_contours["object_id"] = next((item.get("id") for item in json_dict["objects"] if item["object_class"] == contours[-2][1]), 0)
            dict_contours["id"] = contours[-2][0]
            dict_contours["rotated_bbox"] = contours[-1]

            list_annotations.append(dict_contours)
    print("Annotation List Count " + str(len(list_annotations)))
    return list_annotations


def generate_data(ue_dict):
    """
    Create json in Custom format for all images
    """
    json_dict = get_empty_json_format()
    json_dict["objects"] = make_objects(ue_dict)
    json_dict["images"] = make_images(ue_dict)
    json_dict["annotations"] = make_annotations(json_dict, ue_dict)
    #json_dict["models_list"] = make_model_list(ue_dict)

    #create json for images with no objects
    json_no_annotation_images =  get_empty_json_format(annotation = False)
    
    json_no_annotation_images["images"] = get_image_data_with_no_objects(ue_dict)
    
    directory = config['output_folder']
    metadata_folder = "Metadata"
    final_directory= os.path.join(directory, metadata_folder)
    
    os.makedirs(final_directory, exist_ok= True)

    # print(json.dumps(json_dict))
    with open(os.path.join(final_directory, json_name), 'w') as outfile:
        json.dump(json_dict, outfile, indent=4)
    
    with open(os.path.join(final_directory, 'images_with_no_objects.json'), 'w') as outfile:
        json.dump(json_no_annotation_images, outfile, indent=4)

    print(f"Exported Metadata json file in Simlat Fornax format to {os.path.join(final_directory, json_name)}")

if __name__ == "__main__":
   pass
