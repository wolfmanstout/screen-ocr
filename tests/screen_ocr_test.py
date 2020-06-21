import argparse
import os
import glob
import random
import timeit

import screen_ocr
import test_utils
from tesserocr import PyTessBaseAPI
import imagehash
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageGrab, ImageOps
from skimage import filters, measure, morphology, transform
from sklearn import model_selection
from IPython.display import display

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["debug", "grid_search"])
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--all", action="store_true")
args = parser.parse_args()

# Set seed manually for reproducibility.
random.seed(517548236)

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


# Run OCR.
data_path = r"C:\Program Files\Tesseract-OCR\tessdata"
# data_path = r"C:\Users\james\tessdata_fast"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
tessdata_dir_config = r'--tessdata-dir "{}"'.format(data_path)
# TODO devise a better solution that doesn't allow a bunch of noise to result in zero costs
zero_delete_costs = True
with PyTessBaseAPI(path=data_path) as api:
    if args.mode == "debug":
        indices = range(len(X)) if args.all else [0]
        for index in indices:
            print("Processing: {}".format(labeled_data[index][0]))
            image = X[index]
            gt_string = y[index]
            print("Unprocessed:")
            display(image)

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
            if args.verbose:
                def debug_image_callback(name, image):
                    print("{}:".format(name))
                    display(image)
            else:
                debug_image_callback = None
            ocr_reader = screen_ocr.Reader(
                threshold_function=threshold_function,
                correction_block_size=correction_block_size,
                margin=margin,
                resize_factor=resize_factor,
                convert_grayscale=convert_grayscale,
                shift_channels=shift_channels,
                label_components=label_components,
                debug_image_callback=debug_image_callback)
            preprocessing_command = "global preprocessed_image; preprocessed_image = ocr_reader.preprocess(image)"
            preprocessing_time = timeit.timeit(preprocessing_command, globals=globals(), number=1)
            print("preprocessing time: {:f}".format(preprocessing_time))
            print("Preprocessed:")
            display(preprocessed_image)

            # Run OCR.
            print("Ground truth: {}".format(gt_string))
            print("------------------")
            native_string = None
            native_time = timeit.timeit("global native_string; native_string = test_utils.native_image_to_string(preprocessed_image, api, debug_image_callback)", globals=globals(), number=1)
            native_cost = test_utils.cost(native_string, gt_string, zero_delete_costs)
            print("native\ttime: {:.2f}\tcost: {:.2f}".format(native_time, native_cost))
            print("Native OCR: {}".format(native_string))
            print("------------------")
            binary_string = None
            binary_time = timeit.timeit("global binary_string; binary_string = test_utils.binary_image_to_string(preprocessed_image, tessdata_dir_config, debug_image_callback)", globals=globals(), number=1)
            binary_cost = test_utils.cost(binary_string, gt_string, zero_delete_costs)
            print("binary\ttime: {:.2f}\tcost: {:.2f}".format(binary_time, binary_cost))
            print("Binary OCR: {}".format(binary_string))
            print("------------------")
    elif args.mode == "grid_search":
        grid_search = model_selection.GridSearchCV(
            test_utils.OcrEstimator(),
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
