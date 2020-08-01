import easyocr
import numpy as np

from . import _base


class EasyOcrReader(_base.BaseReader):
    def __init__(self, radius=100, **kwargs):
        self.radius = radius
        self._easyocr = easyocr.Reader(["en"])

    def read_image(self, image):
        return EasyOcrScreenContents(self._easyocr.readtext(np.array(image)))


class EasyOcrScreenContents(_base.BaseScreenContents):
    def __init__(self, contents):
        self.contents = contents

    def as_string(self):
        text_lines = [line for box, line, confidence in self.contents]
        return "\n".join(text_lines)
