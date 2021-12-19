import cv2 as cv
from datetime import datetime
from PIL import Image
from object_detection.utils import visualization_utils as vis_util  # from TensorFlow object detection API
import numpy as np
import os
import time
import json


def load_yolo_model(cfg_filepath, weights_filepath: str):
    """
    Loads YOLO model with OpenCV

    :param cfg_filepath: Path to configuration file for YOLOv3
    :param weights_filepath: Path to weights file for YOLOv3
    :return: Net objects which serves as object detection model
    """

    if not os.path.exists(weights_filepath):
        raise FileNotFoundError("YOLOv3 weights file does not exist at specified path")

    # net object, used as detection model
    return cv.dnn.readNetFromDarknet(cfg_filepath, weights_filepath)


def detect_with_yolo(yolo_model, image, suppression_threshold=0.3):
    """
    Runs object detection with YOLO model on given image.

    :param yolo_model: Net object from OpenCV, used as model for object detection
    :param image: Image, saved into numpy array
    :param suppression_threshold: Threshold for non-maximum suppression application
    :return: Output dictionary in similar format as returned by TensorFlow models
    """

    # load our input image and grab its spatial dimensions
    height, width = image.shape[0], image.shape[1]

    # determine only the *output* layer names that we need from YOLO
    layer_names = yolo_model.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in yolo_model.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_model.setInput(blob)
    layer_outputs = yolo_model.forward(layer_names)

    # min x, min y, max x, max y coordinates for each detected bounding box
    # later used for visualization of bounding boxes
    min_max_boxes = []
    # min x, min y, width, height for each detected bounding box
    # used for performing non maximum suppression
    nms_boxes = []
    confidence_scores = []
    classes = []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence score of the current object detection
            results = detection[5:]
            class_id = np.argmax(results)
            confidence_score = results[class_id]

            # scale bounding box coordinates to image size
            # yolo returns center x y coordinates + width and height of bounding box
            box = detection[0:4] * np.array([width, height, width, height])
            (center_x, center_y, box_width, box_height) = box.astype("int")

            # calculate min and max coordinates of a bounding box
            min_x = int(round(center_x - (box_width / 2)))
            max_x = int(round(center_x + (box_width / 2)))
            min_y = int(round(center_y - (box_height / 2)))
            max_y = int(round(center_y + (box_height / 2)))

            # create list with min and max coordinates, divided with width and height
            # such list can be used for visualisation in the same way as with tensorflow models
            min_max_box = np.array([min_y / height, min_x / width, max_y / height, max_x / width])

            # append results to lists
            min_max_boxes.append(min_max_box)
            nms_boxes.append([min_x, min_y, int(box_width), int(box_height)])
            confidence_scores.append(float(confidence_score))
            classes.append(class_id)

    # apply non maximum suppression to suppress weak, overlapping bounding boxes
    indexes = cv.dnn.NMSBoxes(nms_boxes, confidence_scores, 0, suppression_threshold)
    indexes = np.array(indexes)
    output_dict = {"detection_boxes": [], "detection_classes": [], "detection_scores": []}
    for i in indexes.flatten():
        output_dict["detection_boxes"].append(min_max_boxes[i])
        output_dict["detection_classes"].append(classes[i])
        output_dict["detection_scores"].append(confidence_scores[i])

    # convert lists to numpy arrays so they can be used with TensorFlow visualization function
    output_dict["detection_boxes"] = np.array(output_dict["detection_boxes"])
    output_dict["detection_classes"] = np.array(output_dict["detection_classes"])
    output_dict["detection_scores"] = np.array(output_dict["detection_scores"])

    return output_dict


def get_inference_image(image_np, output_dict, categories, line_thickness=1,
                        score_thresh=0.5):
    """
    Visually applies bounding boxes and other detected data on the photo.
    Wrapper function for object_detection's visualize_boxes_and_labels_on_image_array() function

    :param image_np: Image, saved into numpy array
    :param output_dict: TensorFlow model's detection result on image
    :param categories: dictionary with all possible object classes
    :param line_thickness: Line thickness for bounding boxes
    :param score_thresh: Minimum confidence score for which bounding boxes will be drawn
    :return: Image, saved into numpy array, with visually applied bounding boxes
    """

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict["detection_boxes"],
        output_dict["detection_classes"],
        output_dict["detection_scores"],
        categories,
        instance_masks=output_dict.get("detection_masks_reframed", None),
        use_normalized_coordinates=True,
        line_thickness=line_thickness,
        min_score_thresh=score_thresh)

    return image_np


def run_detection(detection_model, categories, photos_path, results_path, score_thresh=0.5,
                  suppression_threshold=0.3):
    """
    Runs detection on photos in given directory and saves new results into given results directory.

    :param detection_model: Net object, used for detection
    :param categories: dictionary with all possible object classes
    :param photos_path: path to directory with photos, used for object detection
    :param results_path: path to where results files will be saved
    :param score_thresh: minimum confidence score; detected objects with lesser confidence will be discarded
    :param suppression_threshold: threshold for non-maximum suppersion application
    :return: Dictionary with object detection results
    """
    full_dict = {}
    for item in sorted(os.listdir(photos_path)):
        # check if file jpg/png
        if item[-4:] == ".jpg" or item[-4:] == ".png":

            # open photo as numpy array
            item_path = os.path.join(photos_path, item)
            with Image.open(item_path) as opened_image:
                photo_np = np.array(opened_image)

            # reshape grayscale image into 3 channels (tensorflow model expects RGB image)
            if len(photo_np.shape) < 3:
                photo_np = np.stack((photo_np,) * 3, axis=-1)
            # remove non-rgb channels if image has them
            elif photo_np.shape[2] > 3:
                photo_np = photo_np[..., :3]

            # dimensions
            photo_x, photo_y = photo_np.shape[1], photo_np.shape[0]

            # result photos subdir
            photos_results_path = results_path + "/result_photos"
            if not os.path.exists(photos_results_path):
                os.makedirs(photos_results_path)

            # where result photo will be saved
            result_photo_path = os.path.join(photos_results_path, item)

            # execute object detection
            print("Current photo:", item_path)
            data_dict = detect_with_yolo(detection_model, photo_np, suppression_threshold)

            # save photo with bounding boxes
            results_photo = get_inference_image(photo_np, data_dict, categories, line_thickness=2,
                                                score_thresh=score_thresh)
            img = Image.fromarray(results_photo)
            print("    Saving visualized result on location", result_photo_path)
            img.save(result_photo_path)

            photo_data = (item[:-4], results_path, photo_x, photo_y)
            full_dict[item_path] = (photo_data, data_dict)

    return full_dict


def generate_json(full_dict, categories, results_path, score_thresh=0.5):
    """
    Generates JSON file with all detection data for all photos in detection data dictionary.

    :param full_dict: Dictionary with object detection results (return of 'run_detection()' function)
    :param categories: dictionary with all possible object classes
    :param results_path: path to where JSON file will be saved
    :param score_thresh: Minimum confidence score for which bounding boxes will be drawn
    """
    # add model name to filename
    json_filename = "all_photos_data - YOLOv3.json"

    # path for JSON file
    json_path = os.path.join(results_path, json_filename)
    # list with JSON dictionaries for each photo
    photos_data = []

    # get JSON dictionaries for each photo
    for photo in full_dict:
        photo_x, photo_y = full_dict[photo][0][2:]
        data_dict = full_dict[photo][1]

        # list for objects' data
        datalist = []

        for i in range(len(data_dict["detection_classes"])):
            # read values from data_dict
            object_score = float(data_dict["detection_scores"][i])
            # skip current object if its score is under threshold
            if object_score < score_thresh:
                continue

            object_class = int(data_dict["detection_classes"][i])
            object_name = categories[object_class]["name"]

            y_min, x_min, y_max, x_max = data_dict["detection_boxes"][i]
            y_min1 = y_min * photo_y
            y_max1 = y_max * photo_y
            x_min1 = x_min * photo_x
            x_max1 = x_max * photo_x
            width = x_max1 - x_min1
            height = y_max1 - y_min1
            object_dims = {
                "width": width, "height": height, "min x": x_min1, "max x": x_max1, "min y": y_min1, "max y": y_max1
            }

            # create dictionary and append it into datalist
            object_data = {
                "Class id": object_class,
                "Class name": object_name,
                "Score": object_score,
                "Bounding box": object_dims
            }
            datalist.append(object_data)

        photo_info = {"Photo": photo, "Width": photo_x, "Height": photo_y, "Objects": datalist}
        photos_data.append(photo_info)

    # write list with all dictionaries into JSON file
    with open(json_path, "w+") as f:
        json.dump(photos_data, f, indent=4)


# ------------------------
# --- SET UP VARIABLES ---
# ------------------------
cfg_file = "data/yolov3/yolov3_testing.cfg"
weights_file = "data/yolov3/yolov3_training_1400.weights"
category_index = {0: {"id": 0, "name": "Ear"}}
model = load_yolo_model(cfg_file, weights_file)
score_threshold = 0.0

# for differing results folder according to date
today = datetime.today().strftime("%Y-%m-%d")

# full path to directory from where photos will be used
photos_dir_path = "./data/train_data/images/test"
# directory where results will be saved
results_dir = "Results/{} YOLOv3".format(today)
# ------------------------


print("---STARTING OBJECT DETECTION---")
start1 = time.time()
detection_data_dict = run_detection(model, category_index, photos_dir_path, results_dir, score_threshold)
# save detection data into json file
generate_json(detection_data_dict, category_index, results_dir, score_thresh=score_threshold)
end1 = time.time()
print("---DONE---")
time_detection = end1 - start1
time_detection_min = time_detection / 60
print("\nTime elapsed: {:.4f} s ({:.4f} m)".format(time_detection, time_detection_min))
