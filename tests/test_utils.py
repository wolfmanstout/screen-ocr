import screen_ocr
from tesserocr import RIL
import pytesseract
from weighted_levenshtein import lev
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageGrab, ImageOps
from skimage import filters, measure, morphology, transform
from sklearn.base import BaseEstimator

def native_image_to_string(image, api, debug_image_callback):
    api.SetImage(image)
    # return api.GetUTF8Text()
    debug_image = image.convert("RGB")
    if debug_image_callback:
        draw = ImageDraw.Draw(debug_image)
    result = ""
    boxes = api.GetComponentImages(RIL.TEXTLINE, True)
    boxes = sorted(boxes, key=lambda box: (box[1]["y"], box[1]["x"]))
    text_lines = []
    for _, box, _, _ in boxes:
        api.SetRectangle(box["x"], box["y"], box["w"], box["h"])
        # if api.MeanTextConf() < 50:
        #     continue
        if debug_image_callback:
            draw.line([box["x"], box["y"],
                       box["x"] + box["w"], box["y"],
                       box["x"] + box["w"], box["y"] + box["h"],
                       box["x"], box["y"] + box["h"],
                       box["x"], box["y"]],
                      fill=(255 - (api.MeanTextConf() * 255 // 100), api.MeanTextConf() * 255 // 100, 0),
                      width=4)
        text_lines.append(api.GetUTF8Text())
    if debug_image_callback:
        del draw
        debug_image_callback("debug_boxes_native", debug_image)
    return "".join(text_lines)


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


low_costs = np.ones(128, dtype=np.float64) * 0.1
zero_costs = np.zeros(128, dtype=np.float64)
def cost(result, gt, zero_delete_costs=False):
    delete_costs = zero_costs if zero_delete_costs else low_costs
    # lev() appears to require ASCII encoding.
    return lev(result.encode("ascii", errors="ignore").lower(), gt.lower(), delete_costs=delete_costs)


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
            image = self.ocr_reader_.preprocess(image)
            # Assume "api" is set globally. This is easier than making it a
            # param because it does not support deepcopy.
            result = binary_image_to_string(image, tessdata_dir_config, debug_image_callback=None)
            error += cost(result, gt_text, zero_delete_costs)
        return -error
            

