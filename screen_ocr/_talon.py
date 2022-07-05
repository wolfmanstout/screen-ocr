import functools
import operator
import re

from talon.experimental import ocr

from . import _base


class TalonBackend(_base.OcrBackend):
    def run_ocr(self, image):
        results = ocr.ocr(image)
        lines = [
            _base.OcrLine(
                [
                    _base.OcrWord(
                        match.group(),
                        *self._merge_boxes(
                            result.bounds.rects[match.start() : match.end()]
                        ),
                    )
                    for match in re.finditer(r"\S+", result.text)
                ]
            )
            for result in results
        ]
        return _base.OcrResult(lines)

    @staticmethod
    def _merge_boxes(rects):
        merged = functools.reduce(operator.add, rects)
        return (merged.x, merged.y, merged.width, merged.height)
