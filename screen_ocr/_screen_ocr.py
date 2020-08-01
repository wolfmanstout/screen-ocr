"""Library for processing screen contents using OCR."""

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
    """Contains factory methods for constructing an OCR reader."""

    @staticmethod
    def create_quality_reader(**kwargs):
        """Create reader optimized for quality.

        See TesseractReader constructor for full argument list.
        """
        return _tesseract.TesseractReader(
            threshold_function=lambda data: filters.rank.otsu(data, morphology.square(41)),
            correction_block_size=31,
            margin=50,
            resize_factor=2,
            convert_grayscale=True,
            shift_channels=True,
            label_components=False,
            **kwargs)

    @staticmethod
    def create_fast_reader(**kwargs):
        """Create reader optimized for speed.

        See TesseractReader constructor for full argument list.
        """
        return _tesseract.TesseractReader(
            threshold_function=lambda data: filters.threshold_otsu(data),
            correction_block_size=41,
            margin=60,
            resize_factor=2,
            convert_grayscale=True,
            shift_channels=True,
            label_components=False,
            **kwargs)

    @staticmethod
    def create_reader(backend, **kwargs):
        """Create reader with specified backend."""
        if backend == "tesseract":
            return Reader.create_quality_reader(**kwargs)
        elif backend == "easyocr":
            return _easyocr.EasyOcrReader(**kwargs)
        elif backend == "winrt":
            return _winrt.WinRtReader(**kwargs)
