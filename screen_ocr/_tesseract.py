import pytesseract
import six

from . import _base


class TesseractBackend(_base.OcrBackend):
    def __init__(self,
                 tesseract_data_path=None,
                 tesseract_command=None):
        self.tesseract_data_path = tesseract_data_path or r"C:\Program Files\Tesseract-OCR\tessdata"
        self.tesseract_command = tesseract_command or r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def run_ocr(self, image):
        tessdata_dir_config = r'--tessdata-dir "{}"'.format(self.tesseract_data_path)
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_command
        results = pytesseract.image_to_data(image,
                                            config=tessdata_dir_config,
                                            output_type=pytesseract.Output.DATAFRAME)
        lines = []
        words = []
        for _, box in results.iterrows():
            # Word
            if box.level == 5:
                text = box.text
                text = (six.text_type(text, encoding="utf-8")
                        if isinstance(text, six.binary_type)
                        else six.text_type(text))
                words.append(_base.OcrWord(text, box.left, box.top, box.width, box.height))
            # End of line
            if box.level == 4:
                if words:
                    lines.append(_base.OcrLine(words))
                words = []
        if words:
            lines.append(_base.OcrLine(words))
        lines.sort(key=lambda line: (line.words[0].top, line.words[0].left))
        return _base.OcrResult(lines)

