import numpy as np
import pandas as pd
import pytesseract
import six
from PIL import Image, ImageDraw, ImageGrab, ImageOps
from fuzzywuzzy import fuzz
from skimage import filters, morphology, transform

from . import _base
from . import _utils


class TesseractReader(_base.BaseReader):
    def __init__(self,
                 threshold_function,
                 correction_block_size,
                 margin,
                 resize_factor,
                 convert_grayscale,
                 shift_channels,
                 label_components,
                 radius=100,
                 confidence_threshold=0.75,
                 debug_image_callback=None,
                 backend="tesseract",
                 tesseract_data_path=r"C:\Program Files\Tesseract-OCR\tessdata",
                 tesseract_command=r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
        self.backend = backend
        self.threshold_function = threshold_function
        self.correction_block_size = correction_block_size
        self.margin = margin
        self.resize_factor = resize_factor
        self.convert_grayscale = convert_grayscale
        self.shift_channels = shift_channels
        self.label_components = label_components
        self.radius = radius
        self.confidence_threshold = confidence_threshold
        self.debug_image_callback = debug_image_callback
        self.tesseract_data_path = tesseract_data_path
        self.tesseract_command = tesseract_command

    def read_nearby(self, screen_coordinates):
        screenshot, bounding_box = _utils.screenshot_nearby(screen_coordinates, self.radius)
        ocr_df = self._find_words_in_image(screenshot)
        # Adjust bounding box offsets based on screenshot offset.
        ocr_df["left"] += bounding_box[0]
        ocr_df["top"] += bounding_box[1]
        return TesseractScreenContents(screen_coordinates=screen_coordinates,
                                       screenshot=screenshot,
                                       bounding_box=bounding_box,
                                       ocr_df=ocr_df,
                                       confidence_threshold=self.confidence_threshold,
                                       debug_image_callback=self.debug_image_callback)

    def read_image(self, image):
        ocr_df = self._find_words_in_image(image)
        return TesseractScreenContents(screen_coordinates=None,
                                       screenshot=image,
                                       bounding_box=(0, 0, image.width, image.height),
                                       ocr_df=ocr_df,
                                       confidence_threshold=self.confidence_threshold,
                                       debug_image_callback=self.debug_image_callback)

    def _find_words_in_image(self, image):
        preprocessed_image = self._preprocess(image)
        tessdata_dir_config = r'--tessdata-dir "{}"'.format(self.tesseract_data_path)
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_command
        results = pytesseract.image_to_data(preprocessed_image,
                                            config=tessdata_dir_config,
                                            output_type=pytesseract.Output.DATAFRAME)
        results[["top", "left"]] = (results[["top", "left"]] - self.margin) / self.resize_factor
        results[["width", "height"]] = results[["width", "height"]] / self.resize_factor
        return results

    def _preprocess(self, image):
        new_size = (image.size[0] * self.resize_factor, image.size[1] * self.resize_factor)
        image = image.resize(new_size, Image.NEAREST)
        if self.debug_image_callback:
            self.debug_image_callback("debug_resized", image)

        data = np.array(image)
        if self.shift_channels:
            channels = [self._shift_channel(data[:, :, i], i) for i in range(3)]
            data = np.stack(channels, axis=-1)

        if self.convert_grayscale:
            image = Image.fromarray(data)
            image = image.convert("L")
            data = np.array(image)
            data = self._binarize_channel(data, None)
            image = Image.fromarray(data)
        else:
            channels = [self._binarize_channel(data[:, :, i], i)
                        for i in range(3)]
            data = np.stack(channels, axis=-1)
            data = np.all(data, axis=-1)
            image = Image.fromarray(data)

        image = ImageOps.expand(image, self.margin, "white")
        # Ensure consistent performance measurements.
        image.load()
        if self.debug_image_callback:
            self.debug_image_callback("debug_final", image)
        return image

    def _binarize_channel(self, data, channel_index):
        if self.debug_image_callback:
            self.debug_image_callback("debug_before_{}".format(channel_index), Image.fromarray(data))
        # Necessary to avoid ValueError from Otsu threshold.
        if data.min() == data.max():
            threshold = np.uint8(0)
        else:
            threshold = self.threshold_function(data)
        if self.debug_image_callback:
            if threshold.ndim == 2:
                self.debug_image_callback("debug_threshold_{}".format(channel_index), Image.fromarray(threshold.astype(np.uint8)))
            else:
                self.debug_image_callback("debug_threshold_{}".format(channel_index), Image.fromarray(np.ones_like(data) * threshold))
        data = data > threshold
        if self.label_components:
            labels, num_labels = measure.label(data, background=-1, return_num=True)
            label_colors = np.zeros(num_labels + 1, np.bool_)
            label_colors[labels] = data
            background_labels = filters.rank.modal(labels.astype(np.uint16, copy=False),
                                                   morphology.square(self.correction_block_size))
            background_colors = label_colors[background_labels]
        else:
            white_sums = self._window_sums(data, self.correction_block_size)
            black_sums = self._window_sums(~data, self.correction_block_size)
            background_colors = white_sums > black_sums
            # background_colors = filters.rank.modal(data.astype(np.uint8, copy=False),
            #                                        morphology.square(self.correction_block_size))
        if self.debug_image_callback:
            self.debug_image_callback("debug_background_{}".format(channel_index), Image.fromarray(background_colors == True))
        # Make the background consistently white (True).
        data = data == background_colors
        if self.debug_image_callback:
            self.debug_image_callback("debug_after_{}".format(channel_index), Image.fromarray(data))
        return data

    @staticmethod
    def _window_sums(image, window_size):
        integral = transform.integral_image(image)
        radius = int((window_size - 1) / 2)
        top_left = np.zeros(image.shape, dtype=np.uint16)
        top_left[radius:, radius:] = integral[:-radius, :-radius]
        top_right = np.zeros(image.shape, dtype=np.uint16)
        top_right[radius:, :-radius] = integral[:-radius, radius:]
        top_right[radius:, -radius:] = integral[:-radius, -1:]
        bottom_left = np.zeros(image.shape, dtype=np.uint16)
        bottom_left[:-radius, radius:] = integral[radius:, :-radius]
        bottom_left[-radius:, radius:] = integral[-1:, :-radius]
        bottom_right = np.zeros(image.shape, dtype=np.uint16)
        bottom_right[:-radius, :-radius] = integral[radius:, radius:]
        bottom_right[-radius:, :-radius] = integral[-1:, radius:]
        bottom_right[:-radius, -radius:] = integral[radius:, -1:]
        bottom_right[-radius:, -radius:] = integral[-1, -1]
        return bottom_right - bottom_left - top_right + top_left

    @staticmethod
    def _shift_channel(data, channel_index):
        """Shifts each channel based on actual position in a typical LCD. This reduces
        artifacts from subpixel rendering. Note that this assumes RGB left-to-right
        ordering and a subpixel size of 1 in the resized image.
        """
        channel_shift = channel_index - 1
        if channel_shift != 0:
            data = np.roll(data, channel_shift, axis=1)
            if channel_shift == -1:
                data[:, -1] = data[:, -2]
            elif channel_shift == 1:
                data[:, 0] = data[:, 1]
        return data


class TesseractScreenContents(_base.BaseScreenContents):
    def __init__(self,
                 screen_coordinates,
                 screenshot,
                 bounding_box,
                 ocr_df,
                 confidence_threshold,
                 debug_image_callback):
        self.screen_coordinates = screen_coordinates
        self.screenshot = screenshot
        self.bounding_box = bounding_box
        self.ocr_df = ocr_df
        self.confidence_threshold = confidence_threshold
        self.debug_image_callback = debug_image_callback

    def as_string(self):
        debug_image = self.screenshot.convert("RGB")
        if self.debug_image_callback:
            draw = ImageDraw.Draw(debug_image)
        result = ""
        lines = []
        words = []
        confidences = []
        for _, box in self.ocr_df.iterrows():
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
            if self.debug_image_callback:
                draw.line([box.left, box.top,
                           box.left + box.width, box.top,
                           box.left + box.width, box.top + box.height,
                           box.left, box.top + box.height,
                           box.left, box.top],
                          fill=(255 - (box.conf * 255 // 100), box.conf * 255 // 100, 0),
                          width=4)
            text_lines.append(str(box.text))
        if self.debug_image_callback:
            del draw
            self.debug_image_callback("debug_boxes_binary", debug_image)
        return "\n".join(text_lines)

    def find_nearest_word_coordinates(self, word, cursor_position):
        if cursor_position not in ("before", "middle", "after"):
            raise ValueError("cursor_position must be either before, middle, or after")
        lowercase_word = word.lower()
        # First, find all matches with minimal edit distance from the query word.
        score_tuples = []
        for index, result in self.ocr_df.iterrows():
            text = result.text
            text = six.text_type(text, encoding="utf-8") if isinstance(text, six.binary_type) else six.text_type(text)
            # Standardize case and straighten apostrophes.
            text = text.lower().replace(u'\u2019','\'')
            # Don't match words that are significantly shorter than the query.
            if float(len(text)) / len(word) < self.confidence_threshold:
                continue
            ratio = fuzz.partial_ratio(word, text)
            # Only consider reasonably confident matches.
            if ratio / 100.0 < self.confidence_threshold:
                continue
            score_tuples.append((ratio, index))
        if not score_tuples:
            return None
        indices = [index for score, index in score_tuples if score == max(score_tuples)[0]]

        # Next, find the closest match based on screen distance.
        possible_matches = self.ocr_df.loc[indices]
        possible_matches["center_x"] = possible_matches["left"] + possible_matches["width"] / 2
        possible_matches["center_y"] = possible_matches["top"] + possible_matches["height"] / 2
        possible_matches["distance_squared"] = _utils.distance_squared(possible_matches["center_x"],
                                                                       possible_matches["center_y"],
                                                                       self.screen_coordinates[0],
                                                                       self.screen_coordinates[1])
        best_match = possible_matches.loc[possible_matches["distance_squared"].idxmin()]
        if cursor_position == "before":
            x = best_match["left"]
        elif cursor_position == "middle":
            x = best_match["center_x"]
        elif cursor_position == "after":
            x = best_match["left"] + best_match["width"]
        return (int(x), int(best_match["center_y"]))
