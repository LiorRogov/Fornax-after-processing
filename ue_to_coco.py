from datetime import datetime
from tqdm import tqdm
import utils_coco as ue
import os
import cv2
import json
import pickle

# Opening JSON file
f = open('config.json')
# returns JSON object as 
config = json.load(f)

# file save name
json_name = "coco.json"

def get_format_stracture():
    # structure of the json
    json_dict = {
        "info":
        {
            "description": "Vehicles Dataset",
            "url": "N/A",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Simlat",
            "date_created": datetime.today().strftime("%Y/%m/%d")
        },
        "licenses": [
            {
                "url": "N/A",
                "id": 1,
                "name": "N/A"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    return json_dict

def make_categories(ue_dict):
    """
    Create categories from processed UE data

    :param dictionary ue_dict: segmentation data
    :return list_categories: list of dictionaries containing categories
    :rtype: list
        structure: [{supercategory, id, name}, {},...]
    """
    list_categories = []
    list_cat_appeared = []
    counter_cat = 0
    print ("Creating categories from processed UE data")
    for data in tqdm(ue_dict.values()):
        for contours in data:
            category = contours[-2][1]
            if category not in list_cat_appeared:
                dict_category = {}
                list_cat_appeared.append(category)
                counter_cat += 1
                dict_category["supercategory"] = "object"
                dict_category["id"] = counter_cat
                dict_category["name"] = category

                list_categories.append(dict_category)

    return list_categories


def make_images(ue_dict):
    """
    Create image data from processed UE data

    :param dictionary ue_dict: segmentation data
    :return list_images: list of dictionaries containing image data
    :rtype: list
        structure: [{}, {},...]
    """
    print ("Writing Frame settings to each frame")
    list_images = []
    for image_id, image in tqdm(enumerate(ue_dict.keys())):
        img = cv2.imread(image)
        height, width, _ = img.shape
        dict_image = {}
        modified_date = datetime.utcfromtimestamp(os.path.getmtime(image)).strftime('%Y-%m-%d %H:%M:%S')
        dict_image["licence"] = 1
        dict_image["file_name"] = image.split("\\")[-1]
        dict_image["coco_url"] = ""
        dict_image["height"] = height
        dict_image["width"] = width
        dict_image["date_captured"] = modified_date
        dict_image["flickr_url"] = ""
        dict_image["id"] = image_id + 1

        list_images.append(dict_image)
    print("Image Count " + str(len(list_images)))
    return list_images


def make_annotations(json_dict, ue_dict):
    """
    Create annotation data from processed UE data

    :param dictionary ue_dict: segmentation data
    :return list_annotations: list of dictionaries containing annotation data
    :rtype: list
        structure: [{}, {},...]
    """
    print ("Creating annotation data from processed UE data")
    list_annotations = []
    for image_id, objects in tqdm(enumerate(ue_dict.values())):
        
        for contours in objects:
            dict_contours = {}
            dict_contours["segmentation"] = []
            # convert a list of tuples of ints into a list of ints
            for cont_list in contours[:-3]:
                #print(len(cont_list))
                if len(contours) > 3 and len(cont_list) <= 2:
                    continue
                dict_contours["segmentation"].append([coord for pair in cont_list for coord in pair])
            dict_contours["area"] = contours[-2][2]
            dict_contours["iscrowd"] = 0
            dict_contours["image_id"] = image_id + 1
            dict_contours["bbox"] = list(contours[-3])
            #dict_contours["rotaion"] = 20
            dict_contours["category_id"] = next((item.get("id") for item in json_dict["categories"] if item["name"] ==
                                                contours[-2][1]), 0)
            dict_contours["id"] = contours[-2][0]

            list_annotations.append(dict_contours)
    print("Annotation List Count " + str(len(list_annotations)))
    return list_annotations



def generate_dada(ue_dict):
    json_dict = get_format_stracture()
    
    json_dict["categories"] = make_categories(ue_dict)
    json_dict["images"] = make_images(ue_dict)
    json_dict["annotations"] = make_annotations(json_dict, ue_dict)

    directory = config['output_folder']
    metadata_folder = "Metadata"
    final_directory= os.path.join(directory, metadata_folder)
    
    os.makedirs(final_directory, exist_ok= True)

    # print(json.dumps(json_dict))
    with open(os.path.join(final_directory, json_name), 'w') as outfile:
        json.dump(json_dict, outfile, indent=4)

    print(f"Exported segmentation json file in COCO format to {os.path.join(final_directory, json_name)}")

if __name__ == "__main__":
    pass    
