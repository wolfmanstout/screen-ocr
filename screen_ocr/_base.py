"""Base classes used by backend implementations."""

class OcrBackend(object):
    """Base class for backend used to perform OCR."""

    def run_ocr(self, image):
        """Return the OcrResult corresponding to the image."""
        raise NotImplementedError()


class OcrResult(object):
    def __init__(self, lines):
        self.lines = lines


class OcrLine(object):
    def __init__(self, words):
        self.words = words


class OcrWord:
    def __init__(self, text, left, top, width, height):
        self.text = text
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.center = (left + width / 2.0, top + height / 2.0)
