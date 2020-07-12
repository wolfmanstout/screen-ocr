import difflib

import easyocr
import numpy as np
import pandas as pd
import pytesseract
import screen_ocr
from PIL import Image, ImageDraw, ImageGrab, ImageOps
from fuzzywuzzy import fuzz
from skimage import filters, measure, morphology, transform
from sklearn.base import BaseEstimator

def binary_image_to_string(image, config, debug_image_callback):
    # return pytesseract.image_to_string(image, config=config)
    debug_image = image.convert("RGB")
    if debug_image_callback:
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
        if debug_image_callback:
            draw.line([box.left, box.top,
                       box.left + box.width, box.top,
                       box.left + box.width, box.top + box.height,
                       box.left, box.top + box.height,
                       box.left, box.top],
                      fill=(255 - (box.conf * 255 // 100), box.conf * 255 // 100, 0),
                      width=4)
        text_lines.append(str(box.text))
    if debug_image_callback:
        del draw
        debug_image_callback("debug_boxes_binary", debug_image)
    return "\n".join(text_lines)


def easyocr_image_to_string(image, reader):
    text_lines = [line for box, line, confidence in reader.readtext(np.array(image))]
    return "\n".join(text_lines)


def cost(result, gt):
    return -fuzz.partial_ratio(result.lower(), gt.lower())


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
            threshold_function = lambda data: filters.threshold_otsu(data)
        elif self.threshold_type == "local_otsu":
            threshold_function = lambda data: filters.rank.otsu(data, morphology.square(self.block_size))
        elif self.threshold_type == "local":
            threshold_function = lambda data: filters.threshold_local(data, self.block_size)
        elif self.threshold_type == "niblack":
            threshold_function = lambda data: filters.threshold_niblack(data, self.block_size)
        elif self.threshold_type == "sauvola":
            threshold_function = lambda data: filters.threshold_sauvola(data, self.block_size)
        else:
            raise ValueError("Unknown threshold type: {}".format(self.threshold_type))
        self.ocr_reader_ = screen_ocr.Reader(
            threshold_function=threshold_function,
            correction_block_size=self.correction_block_size,
            margin=self.margin,
            resize_factor=self.resize_factor,
            convert_grayscale=self.convert_grayscale,
            shift_channels=self.shift_channels,
            label_components=self.label_components,
            debug_image_callback=None)

    def score(self, X, y):
        error = 0
        for image, gt_text in zip(X, y):
            image = self.ocr_reader_._preprocess(image)
            # Assume "api" is set globally. This is easier than making it a
            # param because it does not support deepcopy.
            tessdata_dir_config = r'--tessdata-dir "{}"'.format(self.ocr_reader_.tesseract_data_path)
            pytesseract.pytesseract.tesseract_cmd = self.ocr_reader_.tesseract_command
            result = binary_image_to_string(image, tessdata_dir_config, debug_image_callback=None)
            error += cost(result, gt_text)
        return -error
            

