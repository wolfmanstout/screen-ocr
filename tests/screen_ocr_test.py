import screen_ocr
from screen_ocr import _base


def test_generate_candidates_from_line():
    line = _base.OcrLine(words=[
        _base.OcrWord(text="snake_case:=", left=0, top=0, width=10, height=10),
        _base.OcrWord(text="[TestClass]", left=12, top=0, width=10, height=10),
        _base.OcrWord(text="camelCase", left=24, top=0, width=10, height=10),
    ])
    candidates = list(
        screen_ocr.ScreenContents._generate_candidates_from_line(line))
    assert candidates == [
        screen_ocr.WordLocation(text="snake", left_char_offset=0,
                                right_char_offset=7, left=1, top=0, width=10, height=10),
        screen_ocr.WordLocation(text="_", left_char_offset=5,
                                right_char_offset=6, left=1, top=0, width=10, height=10),
        screen_ocr.WordLocation(text="case", left_char_offset=6,
                                right_char_offset=2, left=1, top=0, width=10, height=10),
        screen_ocr.WordLocation(text=":", left_char_offset=10,
                                right_char_offset=1, left=1, top=0, width=10, height=10),
        screen_ocr.WordLocation(text="=", left_char_offset=11,
                                right_char_offset=0, left=1, top=0, width=10, height=10),
        screen_ocr.WordLocation(text="[", left_char_offset=0,
                                right_char_offset=10, left=13, top=0, width=10, height=10),
        screen_ocr.WordLocation(text="Test", left_char_offset=1,
                                right_char_offset=6, left=13, top=0, width=10, height=10),
        screen_ocr.WordLocation(text="Class", left_char_offset=5,
                                right_char_offset=1, left=13, top=0, width=10, height=10),
        screen_ocr.WordLocation(text="]", left_char_offset=10,
                                right_char_offset=0, left=13, top=0, width=10, height=10),
        screen_ocr.WordLocation(text="camel", left_char_offset=0,
                                right_char_offset=4, left=25, top=0, width=10, height=10),
        screen_ocr.WordLocation(text="Case", left_char_offset=5,
                                right_char_offset=0, left=25, top=0, width=10, height=10),
    ]
