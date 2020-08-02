"""Library for processing screen contents using OCR."""

from fuzzywuzzy import fuzz
from PIL import ImageGrab
from skimage import filters, morphology

from . import _tesseract

# Optional packages.
try:
    from . import _easyocr
except ImportError:
    print("EasyOCR not supported.")
    _easyocr = None
try:
    from . import _winrt
except (ImportError, SyntaxError):
    print("WinRT not supported.")
    _winrt = None


class Reader(object):
    """Reads on-screen text using OCR."""

    @classmethod
    def create_quality_reader(cls,
                              confidence_threshold=None,
                              radius=None,
                              **kwargs):
        """Create reader optimized for quality.

        See TesseractReader constructor for full argument list.
        """
        backend = _tesseract.TesseractBackend(
            threshold_function=lambda data: filters.rank.otsu(data, morphology.square(41)),
            correction_block_size=31,
            margin=50,
            resize_factor=2,
            convert_grayscale=True,
            shift_channels=True,
            label_components=False,
            **kwargs)
        return cls(backend,
                   confidence_threshold=confidence_threshold,
                   radius=radius)

    @classmethod
    def create_fast_reader(cls,
                           confidence_threshold=None,
                           radius=None,
                           **kwargs):
        """Create reader optimized for speed.

        See TesseractReader constructor for full argument list.
        """
        backend = _tesseract.TesseractBackend(
            threshold_function=lambda data: filters.threshold_otsu(data),
            correction_block_size=41,
            margin=60,
            resize_factor=2,
            convert_grayscale=True,
            shift_channels=True,
            label_components=False,
            **kwargs)
        return cls(backend,
                   confidence_threshold=confidence_threshold,
                   radius=radius)

    @classmethod
    def create_reader(cls,
                      backend,
                      confidence_threshold=None,
                      radius=None,
                      **kwargs):
        """Create reader with specified backend."""
        if backend == "tesseract":
            return cls.create_quality_reader(
                confidence_threshold=confidence_threshold,
                radius=radius,
                **kwargs)
        elif backend == "easyocr":
            backend = _easyocr.EasyOcrBackend(**kwargs)
            return cls(backend,
                       confidence_threshold=confidence_threshold,
                       radius=radius)
        elif backend == "winrt":
            backend = _winrt.WinRtBackend(**kwargs)
            return cls(backend,
                       confidence_threshold=confidence_threshold,
                       radius=radius)
        else:
            return cls(backend,
                       confidence_threshold=confidence_threshold,
                       radius=radius)

    def __init__(self,
                 backend,
                 confidence_threshold=None,
                 radius=None):
        self._backend = backend
        self.confidence_threshold = confidence_threshold or 0.75
        self.radius = radius or 100

    def read_nearby(self, screen_coordinates):
        """Return ScreenContents nearby the provided coordinates."""
        screenshot, bounding_box = self._screenshot_nearby(screen_coordinates)
        result = self._backend.run_ocr(screenshot)
        return ScreenContents(screen_coordinates=screen_coordinates,
                              screenshot=screenshot,
                              offset=bounding_box[0:2],
                              result=result,
                              confidence_threshold=self.confidence_threshold)

    def read_image(self, image):
        """Return ScreenContents of the provided image."""
        result = self._backend.run_ocr(image)
        return ScreenContents(screen_coordinates=(0, 0),
                              screenshot=image,
                              offset=(0, 0),
                              result=result,
                              confidence_threshold=self.confidence_threshold)


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


class ScreenContents(object):
    """OCR'd contents of a portion of the screen."""

    def __init__(self,
                 screen_coordinates,
                 screenshot,
                 offset,
                 result,
                 confidence_threshold):
        self.screen_coordinates = screen_coordinates
        self.screenshot = screenshot
        self.offset = offset
        self.result = result
        self.confidence_threshold = confidence_threshold

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
        target_word = target_word.lower()
        # First, find all matches tied for highest score.
        scored_words = [(self._score_word(candidate.text, target_word), candidate)
                        for line in self.result.lines
                        for candidate in line.words]
        scored_words = [word for word in scored_words if word[0] is not None]
        if not scored_words:
            return None
        possible_matches = [word
                            for (score, word) in scored_words
                            if score == max([score for score, _ in scored_words])]

        # Next, find the closest match based on screen distance.
        distance_to_words = [(self.distance_squared(word.center[0] + self.offset[0],
                                                    word.center[1] + self.offset[1],
                                                    *self.screen_coordinates), word)
                             for word in possible_matches]
        best_match = min(distance_to_words, key=lambda x: x[0])[1]
        if cursor_position == "before":
            x = best_match.left
        elif cursor_position == "middle":
            x = best_match.center[0]
        elif cursor_position == "after":
            x = best_match.left + best_match.width
        return (int(x + self.offset[0]), int(best_match.center[1] + self.offset[1]))

    def _score_word(self, candidate, normalized_target):
        candidate = candidate.lower().replace(u'\u2019', '\'')
        if float(len(candidate)) / len(normalized_target) < self.confidence_threshold:
            return None
        ratio = fuzz.partial_ratio(normalized_target, candidate) / 100.0
        if ratio < self.confidence_threshold:
            return None
        return ratio

    @staticmethod
    def distance_squared(x1, y1, x2, y2):
        x_dist = (x1 - x2)
        y_dist = (y1 - y2)
        return x_dist * x_dist + y_dist * y_dist
