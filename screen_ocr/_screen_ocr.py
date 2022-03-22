"""Library for processing screen contents using OCR."""

from collections import deque
from dataclasses import dataclass
from itertools import islice
import re
from statistics import mean
from typing import Iterable, Iterator, Optional, Sequence
import numpy as np
from PIL import Image, ImageDraw, ImageGrab, ImageOps
from rapidfuzz import fuzz
from skimage import filters, morphology, transform

from . import _base

# Optional packages.
try:
    from . import _tesseract
except (ImportError, SyntaxError):
    _tesseract = None
try:
    from . import _easyocr
except ImportError:
    _easyocr = None
try:
    from . import _winrt
except (ImportError, SyntaxError):
    _winrt = None


class Reader(object):
    """Reads on-screen text using OCR."""

    @classmethod
    def create_quality_reader(cls, **kwargs):
        """Create reader optimized for quality.

        See constructor for full argument list.
        """
        if _winrt:
            return cls.create_reader(backend="winrt", **kwargs)
        else:
            return cls.create_reader(backend="tesseract", **kwargs)

    @classmethod
    def create_fast_reader(cls, **kwargs):
        """Create reader optimized for speed.

        See constructor for full argument list.
        """
        if _winrt:
            return cls.create_reader(backend="winrt", **kwargs)
        else:
            defaults = {
               "threshold_function": lambda data: filters.threshold_otsu(data),
               "correction_block_size": 41,
               "margin": 60,
            }
            return cls.create_reader(backend="tesseract", **dict(defaults, **kwargs))

    @classmethod
    def create_reader(cls,
                      backend,
                      tesseract_data_path=None,
                      tesseract_command=None,
                      **kwargs):
        """Create reader with specified backend."""
        if backend == "tesseract":
            if not _tesseract:
                raise ValueError("Tesseract backend unavailable. To install, run pip install screen-ocr[tesseract].")
            backend = _tesseract.TesseractBackend(
                tesseract_data_path=tesseract_data_path,
                tesseract_command=tesseract_command)
            defaults = {
                "threshold_function": lambda data: filters.rank.otsu(data, morphology.square(41)),
                "correction_block_size": 31,
                "margin": 50,
                "resize_factor": 2,
                "convert_grayscale": True,
                "shift_channels": True,
                "label_components": False,
            }
            return cls(backend, **dict(defaults, **kwargs))
        elif backend == "easyocr":
            if not _easyocr:
                raise ValueError("EasyOCR backend unavailable. To install, run pip install screen-ocr[easyocr].")
            backend = _easyocr.EasyOcrBackend()
            return cls(backend, **kwargs)
        elif backend == "winrt":
            if not _winrt:
                raise ValueError("WinRT backend unavailable. To install, run pip install screen-ocr[winrt].")
            try:
                backend = _winrt.WinRtBackend()
            except ImportError:
                raise ValueError("WinRT backend unavailable. To install, run pip install screen-ocr[winrt].")
            return cls(backend,
                       **dict({"resize_factor": 2},
                              **kwargs))
        else:
            return cls(backend, **kwargs)

    def __init__(self,
                 backend,
                 threshold_function=None,
                 correction_block_size=None,
                 margin=None,
                 resize_factor=None,
                 resize_method=None,
                 convert_grayscale=False,
                 shift_channels=False,
                 label_components=False,
                 debug_image_callback=None,
                 confidence_threshold=None,
                 radius=None,
                 homophones=None):
        self._backend = backend
        self.threshold_function = threshold_function
        self.correction_block_size = correction_block_size
        self.margin = margin or 0
        self.resize_factor = resize_factor or 1
        self.resize_method = resize_method or Image.LANCZOS
        self.convert_grayscale = convert_grayscale
        self.shift_channels = shift_channels
        self.label_components = label_components
        self.debug_image_callback = debug_image_callback
        self.confidence_threshold = confidence_threshold or 0.75
        self.radius = radius or 100
        self.homophones = ScreenContents._normalize_homophones(homophones) or default_homophones()

    def read_nearby(self, screen_coordinates):
        """Return ScreenContents nearby the provided coordinates."""
        screenshot, bounding_box = self._screenshot_nearby(screen_coordinates)
        return self.read_image(screenshot,
                               offset=bounding_box[0:2],
                               screen_coordinates=screen_coordinates)

    def read_image(self, image, offset=(0, 0), screen_coordinates=(0, 0)):
        """Return ScreenContents of the provided image."""
        preprocessed_image = self._preprocess(image)
        result = self._backend.run_ocr(preprocessed_image)
        result = self._adjust_result(result, offset)
        return ScreenContents(screen_coordinates=screen_coordinates,
                              screenshot=image,
                              result=result,
                              confidence_threshold=self.confidence_threshold,
                              homophones=self.homophones)


    def _screenshot_nearby(self, screen_coordinates):
        # TODO Consider cropping within grab() for performance. Requires knowledge
        # of screen bounds.
        screenshot = ImageGrab.grab()
        bounding_box = (max(0, screen_coordinates[0] - self.radius),
                        max(0, screen_coordinates[1] - self.radius),
                        min(screenshot.width, screen_coordinates[0] + self.radius),
                        min(screenshot.height, screen_coordinates[1] + self.radius))
        screenshot = screenshot.crop(bounding_box)
        return screenshot, bounding_box

    def _adjust_result(self, result, offset):
        lines = []
        for line in result.lines:
            words = []
            for word in line.words:
                left = ((word.left - self.margin) / self.resize_factor) + offset[0]
                top = ((word.top - self.margin) / self.resize_factor) + offset[1]
                width = word.width / self.resize_factor
                height = word.height / self.resize_factor
                words.append(_base.OcrWord(word.text, left, top, width, height))
            lines.append(_base.OcrLine(words))
        return _base.OcrResult(lines)

    def _preprocess(self, image):
        if self.resize_factor != 1:
            new_size = (image.size[0] * self.resize_factor, image.size[1] * self.resize_factor)
            image = image.resize(new_size, self.resize_method)
        if self.debug_image_callback:
            self.debug_image_callback("debug_resized", image)

        data = np.array(image)
        if self.shift_channels:
            channels = [self._shift_channel(data[:, :, i], i) for i in range(3)]
            data = np.stack(channels, axis=-1)

        if self.threshold_function:
            if self.convert_grayscale:
                image = Image.fromarray(data)
                image = image.convert("L")
                data = np.array(image)
                data = self._binarize_channel(data, None)
            else:
                channels = [self._binarize_channel(data[:, :, i], i)
                            for i in range(3)]
                data = np.stack(channels, axis=-1)
                data = np.all(data, axis=-1)

        image = Image.fromarray(data)
        if self.margin:
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


def default_homophones():
    homophone_list = [
        # 0k is not actually a homophone but is frequently produced by OCR.
        ("ok", "okay", "0k"),
        ("close", "clothes"),
        ("0", "zero"),
        ("1", "one"),
        ("2", "two", "too", "to"),
        ("3", "three"),
        ("4", "four", "for"),
        ("5", "five"),
        ("6", "six"),
        ("7", "seven"),
        ("8", "eight"),
        ("9", "nine"),
        (".", "period"),
    ]
    homophone_map = {}
    for homophone_set in homophone_list:
        for homophone in homophone_set:
            homophone_map[homophone] = homophone_set
    return homophone_map


@dataclass
class WordLocation:
    """Location of a word on-screen."""

    left: int
    top: int
    width: int
    height: int
    left_char_offset: int
    right_char_offset: int
    text: str

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def middle_x(self):
        return int(self.left + self.width / 2)

    @property
    def middle_y(self):
        return int(self.top + self.height / 2)

    @property
    def start_coordinates(self):
        return (self.left, self.middle_y)

    @property
    def middle_coordinates(self):
        return (self.middle_x, self.middle_y)

    @property
    def end_coordinates(self):
        return (self.right, self.middle_y)


class ScreenContents(object):
    """OCR'd contents of a portion of the screen."""

    def __init__(self,
                 screen_coordinates,
                 screenshot,
                 result,
                 confidence_threshold,
                 homophones):
        self.screen_coordinates = screen_coordinates
        self.screenshot = screenshot
        self.result = result
        self.confidence_threshold = confidence_threshold
        self.homophones = homophones

    def as_string(self):
        """Return the contents formatted as a string."""
        lines = []
        for line in self.result.lines:
            words = []
            for word in line.words:
                words.append(word.text)
            lines.append(" ".join(words) + "\n")
        return "".join(lines)

    def find_nearest_word_coordinates(self, target_word, cursor_position):
        """Return the coordinates of the nearest instance of the provided word.

        Uses fuzzy matching.

        Arguments:
        word: The word to search for.
        cursor_position: "before", "middle", or "after" (relative to the matching word)
        """
        if cursor_position not in ("before", "middle", "after"):
            raise ValueError("cursor_position must be either before, middle, or after")
        word_location = self.find_nearest_word(target_word)
        if not word_location:
            return None
        if cursor_position == "before":
            return word_location.start_coordinates
        elif cursor_position == "middle":
            return word_location.middle_coordinates
        elif cursor_position == "after":
            return word_location.end_coordinates

    def find_nearest_word(self, target_word):
        """Return the location of the nearest instance of the provided word.

        Uses fuzzy matching.
        """
        result = self.find_nearest_words(target_word)
        return result[0] if (result and len(result) == 1) else None

    # Special-case "0k" which frequently shows up instead of the correct "OK".
    _SUBWORD_REGEX = re.compile(r"(\b0[Kk]\b|[A-Z][A-Z]+|[A-Za-z][a-z]*|.)")

    def find_nearest_words(self, target: str) -> Optional[Sequence[WordLocation]]:
        """Return the location of the nearest sequence of the provided words.

        Uses fuzzy matching.
        """
        if not target:
            raise ValueError("target is empty")
        target_words = list(map(self._normalize, 
                                (subword 
                                 for word in target.split() 
                                 for subword in re.findall(self._SUBWORD_REGEX, word))))
        # First, find all matches tied for highest score.
        scored_words = [(self._score_words(candidates, target_words), candidates)
                        for candidates in self._generate_candidates(self.result, len(target_words))]
        scored_words = [words for words in scored_words if words[0]]
        if not scored_words:
            return None
        possible_matches = [words
                            for (score, words) in scored_words
                            if score == max(score for score, _ in scored_words)]

        # Next, find the closest match based on screen distance.
        distance_to_words = [(self._distance_squared((words[0].left + words[-1].right) / 2.0,
                                                    (words[0].top + words[-1].bottom) / 2.0,
                                                    *self.screen_coordinates),
                              words)
                             for words in possible_matches]
        return min(distance_to_words, key=lambda x: x[0])[1]

    @staticmethod
    def _generate_candidates(result: _base.OcrResult, length: int) -> Iterator[Sequence[WordLocation]]:
        for line in result.lines:
            candidates = list(ScreenContents._generate_candidates_from_line(line))
            for candidate in candidates:
                # Always include the word by itself in case the target words are smashed together.
                yield [candidate]
            if length > 1:
                for window in ScreenContents._sliding_window(candidates, length):
                    yield window

    @staticmethod
    def _generate_candidates_from_line(line: _base.OcrLine) -> Iterator[WordLocation]:
        for word in line.words:
            left_offset = 0
            for match in re.finditer(ScreenContents._SUBWORD_REGEX, word.text):
                subword = match.group(0)
                right_offset = len(word.text) - (left_offset + len(subword))
                # Adjust position slightly to the right. For some reason Windows biases
                # towards the left side of whatever is clicked (not confirmed on other
                # operating systems).
                right_shift = 1
                yield WordLocation(left=int(word.left) + right_shift,
                                   top=int(word.top),
                                   width=int(word.width),
                                   height=int(word.height),
                                   left_char_offset=left_offset,
                                   right_char_offset=right_offset,
                                   text=subword)
                left_offset += len(subword)

    @staticmethod
    def _normalize(word):
        # Avoid any changes that affect word length.
        return word.lower().replace(u'\u2019', '\'')

    @staticmethod
    def _normalize_homophones(old_homophones):
        new_homophones = {}
        for k, v in old_homophones.items():
            new_homophones[ScreenContents._normalize(k)] = list(map(ScreenContents._normalize, v))
        return new_homophones

    def _score_words(self,
                     candidates: Sequence[WordLocation],
                     normalized_targets: Sequence[str]) -> float:
        if len(candidates) == 1:
            # Handle the case where the target words are smashed together.
            return self._score_word(candidates[0], "".join(normalized_targets))
        scores = list(map(self._score_word, candidates, normalized_targets))
        return mean(scores) if all(scores) else 0.0

    def _score_word(self,
                    candidate: WordLocation,
                    normalized_target: str) -> float:
        candidate_text = self._normalize(candidate.text)
        homophones = self.homophones.get(normalized_target, (normalized_target,))
        best_ratio = max(fuzz.ratio(homophone, candidate_text,
                                    score_cutoff=self.confidence_threshold*100)
                         for homophone in homophones)
        return best_ratio / 100.0

    @staticmethod
    def _distance_squared(x1, y1, x2, y2):
        x_dist = (x1 - x2)
        y_dist = (y1 - y2)
        return x_dist * x_dist + y_dist * y_dist

    @staticmethod
    # From https://docs.python.org/3/library/itertools.html
    def _sliding_window(iterable, n):
        # sliding_window('ABCDEFG', 4) -> ABCD BCDE CDEF DEFG
        it = iter(iterable)
        window = deque(islice(it, n), maxlen=n)
        if len(window) == n:
            yield tuple(window)
        for x in it:
            window.append(x)
            yield tuple(window)
