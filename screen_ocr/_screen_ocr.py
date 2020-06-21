#!/usr/bin/env python
# (c) Copyright 2020 by James Stout
# Licensed under the LGPL, see <http://www.gnu.org/licenses/>

"""Library for processing screen contents using OCR."""

import enum
import pytesseract
import numpy as np
from PIL import Image, ImageGrab, ImageOps
from skimage import filters, transform


class Reader(object):
    def __init__(self,
                 tesseract_data_path=r"C:\Program Files\Tesseract-OCR\tessdata",
                 tesseract_command=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                 threshold_function=lambda data: filters.threshold_otsu(data),
                 correction_block_size=41,
                 margin=60,
                 resize_factor=2,
                 convert_grayscale=True,
                 shift_channels=True,
                 label_components=False,
                 debug_image_callback=None):
        self.tesseract_data_path = tesseract_data_path
        self.tesseract_command = tesseract_command
        self.threshold_function = threshold_function
        self.correction_block_size = correction_block_size
        self.margin = margin
        self.resize_factor = resize_factor
        self.convert_grayscale = convert_grayscale
        self.shift_channels = shift_channels
        self.label_components = label_components
        self.debug_image_callback = debug_image_callback

    def read_nearby(self, screen_coordinates):
        # TODO Consider cropping within grab() for performance. Requires knowledge
        # of screen bounds.
        screenshot = ImageGrab.grab()
        bounding_box = _nearby_bounding_box(screen_coordinates, screenshot)
        screenshot = screenshot.crop(bounding_box)
        ocr_df = self._find_words_in_image(screenshot)
        # Adjust bounding box offsets based on screenshot offset.
        ocr_df["left"] += bounding_box[0]
        ocr_df["top"] += bounding_box[1]
        return ScreenContents(screen_coordinates, screenshot, bounding_box, ocr_df)

    def _find_words_in_image(self, image):
        preprocessed_image = self.preprocess(image)
        tessdata_dir_config = r'--tessdata-dir "{}"'.format(self.tesseract_data_path)
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_command
        results = pytesseract.image_to_data(preprocessed_image,
                                            config=tessdata_dir_config,
                                            output_type=pytesseract.Output.DATAFRAME)
        results[["top", "left"]] = (results[["top", "left"]] - self.margin) / self.resize_factor
        results[["width", "height"]] = results[["width", "height"]] / self.resize_factor
        return results

    def preprocess(self, image):
        new_size = (image.size[0] * self.resize_factor, image.size[1] * self.resize_factor)
        image = image.resize(new_size, Image.NEAREST)
        if self.debug_image_callback:
            self.debug_image_callback("debug_resized", image)

        data = np.array(image)
        if self.shift_channels:
            channels = [_shift_channel(data[:, :, i], i) for i in range(3)]
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
            white_sums = _window_sums(data, self.correction_block_size)
            black_sums = _window_sums(~data, self.correction_block_size)
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


class ScreenContents(object):
    def __init__(self, screen_coordinates, screenshot, bounding_box, ocr_df):
        self.screen_coordinates = screen_coordinates
        self.screenshot = screenshot
        self.bounding_box = bounding_box
        self._ocr_df = ocr_df

    def find_nearest_word_coordinates(self, word, cursor_position):
        lowercase_word = word.lower()
        # TODO Investigate why this is >10X faster than the following:
        # possible_matches = self._ocr_df[self._ocr_df.text.str.contains(word, case=False, na=False, regex=False)]
        indices = []
        for index, result in self._ocr_df.iterrows():
            text = result.text
            text = text if isinstance(text, basestring) else str(text)
            if lowercase_word in text.lower():
                indices.append(index)
        possible_matches = self._ocr_df.loc[indices]
        if possible_matches.empty:
            return None

        possible_matches["center_x"] = possible_matches["left"] + possible_matches["width"] / 2
        possible_matches["center_y"] = possible_matches["top"] + possible_matches["height"] / 2
        possible_matches["_distance_squared"] = _distance_squared(possible_matches["center_x"],
                                                                possible_matches["center_y"],
                                                                self.screen_coordinates[0],
                                                                self.screen_coordinates[1])
        best_match = possible_matches.loc[possible_matches["_distance_squared"].idxmin()]
        if cursor_position == CursorPosition.BEFORE:
            x = best_match["left"]
        elif cursor_position == CursorPosition.MIDDLE:
            x = best_match["center_x"]
        elif cursor_position == CursorPosition.AFTER:
            x = best_match["left"] + best_match["width"]
        return (x, best_match["center_y"])


class CursorPosition(enum.Enum):
    """The cursor position relative to a range of text."""

    BEFORE = 1
    """The position before the text."""

    MIDDLE = 2
    """The position in the middle of the text."""

    AFTER = 3
    """The position after the text."""


def _nearby_bounding_box(screen_coordinates, screenshot, radius=100):
    return (max(0, screen_coordinates[0] - radius),
            max(0, screen_coordinates[1] - radius),
            min(screenshot.width, screen_coordinates[0] + radius),
            min(screenshot.height, screen_coordinates[1] + radius))


def _distance_squared(x1, y1, x2, y2):
    x_dist = (x1 - x2)
    y_dist = (y1 - y2)
    return x_dist * x_dist + y_dist * y_dist


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
