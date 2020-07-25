# Screen OCR

The `screen-ocr` package makes it easy to perform OCR on portions of the screen.

## Installation

1. Install Tesseract binaries. For Windows, see
https://github.com/UB-Mannheim/tesseract/wiki.
2. `pip install screen-ocr`

## Usage

See https://github.com/wolfmanstout/gaze-ocr for sample usage.

If not using the default Tesseract location on Windows, set
`tesseract_data_path` and `tesseract_command` paths aappropriately when
constructing a `Reader` instance.
