import numpy as np
import os
import json
import matplotlib.pyplot as plt


def read_from_txt(filename):
    bboxes = []

    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        line_list = line.strip().split(" ")
        if len(line_list) == 5:
            confidence = 1
            obj_class, x_center, y_center, width, height = line_list
        else:
            obj_class, x_center, y_center, width, height, confidence = line_list

        x_center = float(x_center) * 480
        y_center = float(y_center) * 360
        width = float(width) * 480
        height = float(height) * 360
        bbox = get_bbox(x_center, y_center, width, height)
        bboxes.append({"obj_class": int(obj_class),
                       "bbox": bbox,
                       "confidence": float(confidence)})
    return bboxes


def read_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)

    full_photo_dict = {}
    for photo in data:
        photo_name = os.path.basename(photo["Photo"])
        objects = photo["Objects"]
        bboxes = []
        for obj in objects:
            obj_class = obj["Class id"]
            confidence = obj["Score"]
            bbox_dict = obj["Bounding box"]
            bbox = [bbox_dict["width"], bbox_dict["height"], bbox_dict["min x"], bbox_dict["max x"],
                    bbox_dict["min y"], bbox_dict["max y"]]
            bboxes.append({"obj_class": obj_class, "bbox": bbox, "confidence": confidence})
        full_photo_dict[photo_name.split(".")[0]] = bboxes
    return full_photo_dict


def read_all_txt(folder_path):
    all_txt_dict = {}
    for file in os.listdir(folder_path):
        filepath = os.path.join(folder_path, file)
        bboxes = read_from_txt(filepath)
        all_txt_dict[file.split(".")[0]] = bboxes
    return all_txt_dict


def get_bbox(x_center, y_center, width, height):
    min_x = x_center - width / 2
    max_x = x_center + width / 2
    min_y = y_center - height / 2
    max_y = y_center + height / 2
    return width, height, min_x, max_x, min_y, max_y


def get_iou(b_box1, b_box2):
    width1, height1, min_x1, max_x1, min_y1, max_y1 = b_box1
    width2, height2, min_x2, max_x2, min_y2, max_y2 = b_box2

    # determine the coordinates of the intersection rectangle
    x_left = max(min_x1, min_x2)
    y_top = max(min_y1, min_y2)
    x_right = min(max_x1, max_x2)
    y_bottom = min(max_y1, max_y2)

    if x_right < x_left or y_bottom < y_top:
        # there is no intersection
        return 0.0

    # compute the area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both bounding boxes
    bb1_area = width1 * height1
    bb2_area = width2 * height2

    # compute the intersection over union
    # union = area1 + area2 - intersection
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def compare_truth(truth_data, model_data, iou_thresh=0.7):
    all_iou = []
    all_confidences = []
    tp_global, fp_global, fn_global = 0, 0, 0
    precision_global = []
    recall_global = []
    by_photo_compare = {}
    for photo in truth_data:
        truth_bboxes = []
        model_bboxes = []
        bboxes = truth_data[photo]
        for bbox_list in bboxes:
            truth_bboxes.append(bbox_list["bbox"])
        bboxes = model_data[photo]
        for bbox_list in bboxes:
            all_confidences.append(bbox_list["confidence"])
            model_bboxes.append(bbox_list["bbox"])

        tp = 0
        used_truth = []
        used_model = []

        for bbox2 in model_bboxes:
            if bbox2 in used_model:
                continue
            for bbox1 in truth_bboxes:
                if bbox1 in used_truth:
                    continue
                iou = get_iou(bbox1, bbox2)
                if iou >= iou_thresh:
                    all_iou.append(iou)
                    tp += 1
                    used_truth.append(bbox1)
                    used_model.append(bbox2)
        fp = len(model_bboxes) - len(used_model)
        fn = len(truth_bboxes) - tp

        assert tp >= 0
        assert fp >= 0
        assert fn >= 0
        tp_global += tp
        fp_global += fp
        fn_global += fn

        if (tp + fp) == 0:
            precision = 1
        else:
            precision = tp / (tp + fp)

        if (tp + fn) == 0:
            recall = 1
        else:
            recall = tp / (tp + fn)
        precision_global.append(precision)
        recall_global.append(recall)
        by_photo_compare[photo] = {"precision": precision, recall: recall,
                                   "tp": tp, "fp": fp, "fn": fn}

    return {
        "tp": tp_global,
        "fp": fp_global,
        "fn": fn_global,
        "avg_iou": np.mean(all_iou),
        "all_iou": all_iou,
        "avg_confidence": np.mean(all_confidences),
        "all_confidences": all_confidences,
        "avg_precision": np.mean(precision_global),
        "avg_recall": np.mean(recall_global),
        "precisions": precision_global,
        "recalls": recall_global,
        "by_photo_compare": by_photo_compare
    }


def run_eval(truth_data, model_data, iou):
    eval_data = compare_truth(truth_data, model_data, iou_thresh=iou)
    print("   overall TP:", eval_data["tp"])
    print("   overall FP:", eval_data["fp"])
    print("   overall FN:", eval_data["fn"])
    print("   average precision value:", eval_data["avg_precision"])
    print("   average recall value:", eval_data["avg_recall"])

    return eval_data["tp"], eval_data["fp"], eval_data["fn"],\
        eval_data["avg_precision"], eval_data["avg_recall"]


def run_eval2(truth_data, yolov3_data, yolov5_data):
    yolov3_tp, yolov5_tp = [], []
    yolov3_fp, yolov5_fp = [], []
    yolov3_fn, yolov5_fn = [], []
    yolov3_avg_precision, yolov5_avg_precision = [], []
    yolov3_avg_recall, yolov5_avg_recall = [], []
    yolov3_avg_iou, yolov5_avg_iou = [], []
    yolov3_avg_confidence, yolov5_avg_confidence = 0, 0
    yolov3_confidences, yolov5_confidences = [], []
    for i in range(1, 10):
        iou = i / 10
        yolov3_eval_data = compare_truth(truth_data, yolov3_data, iou_thresh=iou)
        yolov5_eval_data = compare_truth(truth_data, yolov5_data, iou_thresh=iou)
        print("IoU =", iou)
        print("YOLOv3:")
        print("   overall TP:", yolov3_eval_data["tp"])
        print("   overall FP:", yolov3_eval_data["fp"])
        print("   overall FN:", yolov3_eval_data["fn"])
        print("   average precision value:", yolov3_eval_data["avg_precision"])
        print("   average recall value:", yolov3_eval_data["avg_recall"])
        print("   average IoU:", yolov3_eval_data["avg_iou"])
        print("YOLOv5:")
        print("   overall TP:", yolov5_eval_data["tp"])
        print("   overall FP:", yolov5_eval_data["fp"])
        print("   overall FN:", yolov5_eval_data["fn"])
        print("   average precision value:", yolov5_eval_data["avg_precision"])
        print("   average recall value:", yolov5_eval_data["avg_recall"])
        print("   average IoU:", yolov5_eval_data["avg_iou"])

        yolov3_tp.append(yolov3_eval_data["tp"])
        yolov3_fp.append(yolov3_eval_data["fp"])
        yolov3_fn.append(yolov3_eval_data["fn"])
        yolov3_avg_precision.append(yolov3_eval_data["avg_precision"])
        yolov3_avg_recall.append(yolov3_eval_data["avg_recall"])
        yolov3_avg_iou.append(yolov3_eval_data["avg_iou"])

        yolov5_tp.append(yolov5_eval_data["tp"])
        yolov5_fp.append(yolov5_eval_data["fp"])
        yolov5_fn.append(yolov5_eval_data["fn"])
        yolov5_avg_precision.append(yolov5_eval_data["avg_precision"])
        yolov5_avg_recall.append(yolov5_eval_data["avg_recall"])
        yolov5_avg_iou.append(yolov3_eval_data["avg_iou"])

        yolov3_confidences = yolov3_eval_data["all_confidences"]
        yolov5_confidences = yolov5_eval_data["all_confidences"]
        yolov3_avg_confidence = yolov3_eval_data["avg_confidence"]
        yolov5_avg_confidence = yolov5_eval_data["avg_confidence"]

    return {"yolov3_tp": yolov3_tp, "yolov3_fp": yolov3_fp, "yolov3_fn": yolov3_fn,
            "yolov3_avg_precision": yolov3_avg_precision, "yolov3_avg_recall": yolov3_avg_recall,
            "yolov3_confidences": yolov3_confidences, "yolov3_avg_confidence": yolov3_avg_confidence,
            "yolov3_avg_iou": yolov3_avg_iou,
            "yolov5_tp": yolov5_tp, "yolov5_fp": yolov5_fp, "yolov5_fn": yolov5_fn,
            "yolov5_avg_precision": yolov5_avg_precision, "yolov5_avg_recall": yolov5_avg_recall,
            "yolov5_confidences": yolov5_confidences, "yolov5_avg_confidence": yolov5_avg_confidence,
            "yolov5_avg_iou": yolov5_avg_iou}


def line_plot(save_path, data1, data2, title, y_title, y_upper=None):
    plt.figure(figsize=(8, 6))
    x_axis = [i/10 for i in range(1, 10)]
    if y_upper is not None:
        plt.ylim(0, y_upper)
    plt.plot(x_axis, data1, label="YOLOv3")
    plt.scatter(x_axis, data1)
    plt.plot(x_axis, data2, label="YOLOv5")
    plt.scatter(x_axis, data2)
    plt.title(title)
    plt.xlabel("IoU threshold")
    plt.ylabel(y_title)
    for i, j in zip(x_axis, data1):
        if j % 1 == 0:
            annotation = str(j)
        else:
            annotation = "{:.2f}".format(j)
        plt.annotate(annotation, xy=(i, j))
    for i, j in zip(x_axis, data2):
        if j % 1 == 0:
            annotation = str(j)
        else:
            annotation = "{:.2f}".format(j)
        plt.annotate(annotation, xy=(i, j))
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    #plt.savefig(save_path)


def bar_plot(save_path, data, title, x_title):
    y_values = [0 for _ in range(10)]
    for val in data:
        y_values[int(val * 10)] += 1
    plt.ylim(0, 300)
    y_values.insert(0, data.count(0))
    x_values = ["{}-{}".format((i - 1) / 10, i / 10) for i in range(1, len(y_values))]
    x_values.insert(0, "0")
    #x_values = [str(x) for x in range(len(y_values))]
    #x_values[-1] = "10+"
    plt.bar(x_values, y_values)
    plt.xticks(x_values, rotation=45, ha="right")
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel("Number of occurrences")
    plt.subplots_adjust(bottom=0.18)
    for i, j in zip(range(len(y_values)), y_values):
        if j % 1 == 0:
            annotation = str(j)
        else:
            annotation = "{:.2f}".format(j)
        plt.annotate(annotation, xy=(i-0.2, j+3))
    plt.savefig(save_path)
    plt.show()
    #plt.imsave(save_path)


truth_data_path = "./data/yolov5/train_data/labels/test"
yolov5_data_path = "./yolov5/runs/detect/exp5/labels"
yolov3_data_path = "data/all_photos_data - YOLOv3.json"

print("reading data...")
truth_data = read_all_txt(truth_data_path)
yolov5_data = read_all_txt(yolov5_data_path)
yolov3_data = read_from_json(yolov3_data_path)
print(truth_data.keys())
print(yolov5_data.keys())
print(yolov3_data.keys())


print("EVALUATION")

eval_final = run_eval2(truth_data, yolov3_data, yolov5_data)
for t in eval_final:
    print("{}: {}".format(t, eval_final[t]))

print("PLOTTING")
line_plot("./result_plots/tp.png", eval_final["yolov3_tp"], eval_final["yolov5_tp"], "TP comparison", "Value", 300)
line_plot("./result_plots/fp.png", eval_final["yolov3_fp"], eval_final["yolov5_fp"], "FP comparison", "Value", 300)
line_plot("./result_plots/fn.png", eval_final["yolov3_fn"], eval_final["yolov5_fn"], "FN comparison", "Value", 300)
line_plot("./result_plots/precision.png", eval_final["yolov3_avg_precision"], eval_final["yolov5_avg_precision"], "Average precision values comparison", "Value")
line_plot("./result_plots/recall.png", eval_final["yolov3_avg_recall"], eval_final["yolov5_avg_recall"], "Average recall values comparison", "Value")
bar_plot("./result_plots/YOLOv3 confidences.png", eval_final["yolov3_confidences"], "YOLOv3 confidence ranges", "Confidence range")
bar_plot("./result_plots/YOLOv5 confidences.png", eval_final["yolov5_confidences"], "YOLOv5 confidence ranges", "Confidence range")
