import argparse
import os
import glob
import random
import timeit

import screen_ocr
from tesserocr import PyTessBaseAPI, RIL
import imagehash
import pytesseract
from weighted_levenshtein import lev
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageGrab, ImageOps
from skimage import filters, measure, morphology, transform
from sklearn.base import BaseEstimator
from sklearn import model_selection
from IPython.display import display

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["debug", "grid_search"])
args = parser.parse_args()

# Set seed manually for reproducibility.
random.seed(517548236)

def native_image_to_string(image, api, save_debug_images):
    api.SetImage(image)
    # return api.GetUTF8Text()
    debug_image = image.convert("RGB")
    if save_debug_images:
        draw = ImageDraw.Draw(debug_image)
    result = ""
    boxes = api.GetComponentImages(RIL.TEXTLINE, True)
    boxes = sorted(boxes, key=lambda box: (box[1]["y"], box[1]["x"]))
    text_lines = []
    for _, box, _, _ in boxes:
        api.SetRectangle(box["x"], box["y"], box["w"], box["h"])
        # if api.MeanTextConf() < 50:
        #     continue
        if save_debug_images:
            draw.line([box["x"], box["y"],
                       box["x"] + box["w"], box["y"],
                       box["x"] + box["w"], box["y"] + box["h"],
                       box["x"], box["y"] + box["h"],
                       box["x"], box["y"]],
                      fill=(255 - (api.MeanTextConf() * 255 // 100), api.MeanTextConf() * 255 // 100, 0),
                      width=4)
        text_lines.append(api.GetUTF8Text())
    if save_debug_images:
        del draw
        debug_image.save("debug_boxes_native.png")
    return "".join(text_lines)


def binary_image_to_string(image, config, save_debug_images):
    # return pytesseract.image_to_string(image, config=config)
    debug_image = image.convert("RGB")
    if save_debug_images:
        draw = ImageDraw.Draw(debug_image)
    result = ""
    boxes = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DATAFRAME)
    lines = []
    words = []
    confidences = []
    for _, box in boxes.iterrows():
        if box.level == 4:
            if words:
                line_box.text = " ".join(words)
                line_box.conf = np.mean(confidences).astype(np.int32)
                lines.append(line_box)
            words = []
            confidences = []
            line_box = box
        if box.level == 5:
            words.append(str(box.text))
            confidences.append(box.conf)
    if words:
        line_box.text = " ".join(words)
        lines.append(line_box)
    line_boxes = pd.DataFrame(lines)
    if not line_boxes.empty:
        line_boxes = line_boxes.sort_values(by=["top", "left"])
    text_lines = []
    for box in line_boxes.itertuples():
        # if not isinstance(box.conf, int) or box.conf < 0 or box.conf > 100:
        #     continue
        if save_debug_images:
            draw.line([box.left, box.top,
                       box.left + box.width, box.top,
                       box.left + box.width, box.top + box.height,
                       box.left, box.top + box.height,
                       box.left, box.top],
                      fill=(255 - (box.conf * 255 // 100), box.conf * 255 // 100, 0),
                      width=4)
        text_lines.append(str(box.text))
    if save_debug_images:
        del draw
        debug_image.save("debug_boxes_binary.png")
    return "\n".join(text_lines)


low_costs = np.ones(128, dtype=np.float64) * 0.1
zero_costs = np.zeros(128, dtype=np.float64)
def cost(result, gt, zero_delete_costs=False):
    delete_costs = zero_costs if zero_delete_costs else low_costs
    # lev() appears to require ASCII encoding.
    return lev(result.encode("ascii", errors="ignore"), gt, delete_costs=delete_costs)


class OcrEstimator(BaseEstimator):
    def __init__(self,
                 threshold_type=None,
                 block_size=None,
                 correction_block_size=None,
                 margin=None,
                 resize_factor=None,
                 convert_grayscale=None,
                 shift_channels=None,
                 label_components=None):
        self.threshold_type = threshold_type
        self.block_size = block_size
        self.correction_block_size = correction_block_size
        self.margin = margin
        self.resize_factor = resize_factor
        self.convert_grayscale = convert_grayscale
        self.shift_channels = shift_channels
        self.label_components = label_components

    def fit(self, X=None, y=None):
        if self.threshold_type == "otsu":
            self.threshold_function_ = lambda data: filters.threshold_otsu(data)
        elif self.threshold_type == "local_otsu":
            self.threshold_function_ = lambda data: filters.rank.otsu(data, morphology.square(self.block_size))
        elif self.threshold_type == "local":
            self.threshold_function_ = lambda data: filters.threshold_local(data, self.block_size)
        elif self.threshold_type == "niblack":
            self.threshold_function_ = lambda data: filters.threshold_niblack(data, self.block_size)
        elif self.threshold_type == "sauvola":
            self.threshold_function_ = lambda data: filters.threshold_sauvola(data, self.block_size)
        else:
            raise ValueError("Unknown threshold type: {}".format(self.threshold_type))

    def score(self, X, y):
        error = 0
        for image, gt_text in zip(X, y):
            image = screen_ocr.preprocess(
                image,
                threshold_function=self.threshold_function_,
                correction_block_size=self.correction_block_size,
                margin=self.margin,
                resize_factor=self.resize_factor,
                convert_grayscale=self.convert_grayscale,
                shift_channels=self.shift_channels,
                label_components=self.label_components,
                save_debug_images=False)
            # Assume "api" is set globally. This is easier than making it a
            # param because it does not support deepcopy.
            result = binary_image_to_string(image, tessdata_dir_config, save_debug_images=False)
            error += cost(result, gt_text, zero_delete_costs)
        return -error
            

os.chdir(r"C:\Users\james\Documents\OCR")
for debug_output in glob.glob("debug*"):
  os.remove(debug_output)

# Load image and crop.
# logs_example = "failure_1578861893.90"
# image_path = "train/pillow_docs.png"
# image_path = "logs/{}.png".format(logs_example)
# image = Image.open(image_path).convert("RGB") #.crop(bounding_box)
# image = ImageGrab.grab(bounding_box)

# Load ground truth.
# gt_path = "train/pillow_docs.txt"
# gt_path = "logs/{}.txt".format(logs_example)
# with open(gt_path, "r") as gt_file:
#     gt_string = gt_file.read()

text_files = set(glob.glob("logs/*.txt"))
average_hashes = set()
color_hashes = set()
success_data = []
failure_data = []
for image_file in glob.glob("logs/*.png"):
    # Skip near-duplicate images. Hash functions and parameters determined
    # experimentally.
    image = Image.open(image_file)
    average_hash = imagehash.average_hash(image, 10)
    if average_hash in average_hashes:
        continue
    average_hashes.add(average_hash)
    color_hash = imagehash.colorhash(image, 5)
    if color_hash in color_hashes:
        continue
    color_hashes.add(color_hash)

    text_file = image_file[:-3] + "txt"
    if not text_file in text_files:
        continue
    base_name = os.path.basename(text_file)
    if base_name.startswith("success"):
        success_data.append((image_file, text_file))
    elif base_name.startswith("failure"):
        failure_data.append((image_file, text_file))
    else:
        raise AssertionError("Unexpected file name: {}".format(base_name))

random.shuffle(success_data)
random.shuffle(failure_data)
# Downsample the success data so that it's proportional to the failure data.
# Only reason to do this is to save on compute resources.
labeled_data = failure_data + success_data[:len(failure_data)]

X = []
y = []
for image_file, text_file in labeled_data:
    X.append(Image.open(image_file).convert("RGB"))
    with open(text_file, "r") as f:
        y.append(f.read())

index = 0
image = X[index]
gt_string = y[index]


# Preprocess the image.
block_size = None
threshold_function = lambda data: filters.threshold_otsu(data)
# threshold_function = lambda data: filters.rank.otsu(data, morphology.square(correction_block_size))
correction_block_size = 41
margin = 60
resize_factor = 2
convert_grayscale = True
shift_channels = True
label_components = False
save_debug_images = True
preprocessing_command = """
global preprocessed_image;
preprocessed_image = screen_ocr.preprocess(image,
                                threshold_function=threshold_function,
                                correction_block_size=correction_block_size,
                                margin=margin,
                                resize_factor=resize_factor,
                                convert_grayscale=convert_grayscale,
                                shift_channels=shift_channels,
                                label_components=label_components,
                                save_debug_images=save_debug_images)"""
preprocessing_time = timeit.timeit(preprocessing_command, globals=globals(), number=1)

# Run OCR.
data_path = r"C:\Program Files\Tesseract-OCR\tessdata"
# data_path = r"C:\Users\james\tessdata_fast"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
tessdata_dir_config = r'--tessdata-dir "{}"'.format(data_path)
zero_delete_costs = True
with PyTessBaseAPI(path=data_path) as api:
    if args.mode == "debug":
        display(image)
        display(preprocessed_image)
        native_string = None
        native_time = timeit.timeit("global native_string; native_string = native_image_to_string(preprocessed_image, api, save_debug_images)", globals=globals(), number=1)
        native_cost = cost(native_string, gt_string, zero_delete_costs)
        binary_string = None
        binary_time = timeit.timeit("global binary_string; binary_string = binary_image_to_string(preprocessed_image, tessdata_dir_config, save_debug_images)", globals=globals(), number=1)
        binary_cost = cost(binary_string, gt_string, zero_delete_costs)
        with open("debug_native.txt", "w") as file:
            file.write(native_string)
        with open("debug_binary.txt", "w") as file:
            file.write(binary_string)
        print("Ground truth: {}".format(gt_string))
        print(native_string)
        print("------------------")
        print(binary_string)
        print("preprocessing time: {:f}".format(preprocessing_time))
        print("native\ttime: {:.2f}\tcost: {:.2f}".format(native_time, native_cost))
        print("binary\ttime: {:.2f}\tcost: {:.2f}".format(binary_time, binary_cost))
    elif args.mode == "grid_search":
        grid_search = model_selection.GridSearchCV(
            OcrEstimator(),
            {
                "threshold_type": ["otsu"], # , "local_otsu", "local"],  # , "niblack", "sauvola"],
                "block_size": [None], # [51, 61, 71],
                "correction_block_size": [36, 41, 46],
                "margin": [50, 60, 70],
                "resize_factor": [2], # , 3, 4],
                "convert_grayscale": [True],
                "shift_channels": [False, True],
                "label_components": [False],
            },
            # Evaluate each example separately so that standard deviation is automatically computed.
            cv=model_selection.LeaveOneOut()  # model_selection.PredefinedSplit([0] * len(y))
        )
        grid_search.fit(X, y)
        results = pd.DataFrame(grid_search.cv_results_)
        results.set_index("params", inplace=True)
        print(results["mean_test_score"].sort_values(ascending=False))
        print(grid_search.best_params_)
