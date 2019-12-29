from tesserocr import PyTessBaseAPI
import pytesseract
from weighted_levenshtein import lev
import numpy as np
import timeit
from PIL import Image, ImageGrab, ImageOps
import skimage
from skimage import filters, morphology

def native_image_to_string(api, image):
    api.SetImage(image)
    return api.GetUTF8Text()


delete_costs = np.ones(128, dtype=np.float64) * 0.1
def cost(result, gt):
    # lev() appears to require ASCII encoding.
    return lev(result.encode("ascii", errors="ignore"), gt, delete_costs=delete_costs)


def binarize_channel(data):
    block_size = 51
    threshold_function = lambda data: filters.threshold_local(data, block_size)
    # threshold_function = lambda data: filters.threshold_otsu(data)
    threshold = threshold_function(data)
    data = data > threshold
    ubyte_data = skimage.img_as_ubyte(data)
    ubyte_data[ubyte_data == 255] = 1
    histograms = filters.rank.windowed_histogram(ubyte_data, morphology.square(block_size), n_bins=2)
    assert histograms.shape[2] == 2
    white_background = histograms[:, :, 1] > 0.5
    data = data == white_background
    return data


def preprocess(image):
    image = ImageOps.expand(image, 10, "white")
    resize_factor = 4
    new_size = (image.size[0] * resize_factor, image.size[1] * resize_factor)
    image = image.resize(new_size, Image.NEAREST)
    
    data = np.array(image)
    data = np.stack([binarize_channel(data[:, :, i]) for i in range(3)], axis=-1)
    data = np.all(data, axis=-1)
    image = Image.fromarray(data)

    image.load()
    return image


# Load image and crop.
bounding_box = (0, 0, 200, 200)
image_path = r"C:\Users\james\Documents\OCR\pillow_docs_cropped.png"
image = Image.open(image_path).convert("RGB")  # .crop(bounding_box)
# image = ImageGrab.grab(bounding_box)

# Preprocess the image.
preprocessing_time = timeit.timeit("global image; image = preprocess(image)", globals=globals(), number=1)
image.save(r"C:\Users\james\Documents\OCR\debug.png")

# Load ground truth.
gt_path = r"C:\Users\james\Documents\OCR\pillow_docs_cropped_gt.txt"
with open(gt_path, "r") as gt_file:
    gt_string = gt_file.read()

# Run OCR.
data_path = r"C:\Program Files\Tesseract-OCR\tessdata"
# data_path = r"C:\Users\james\tessdata_fast"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
tessdata_dir_config = r'--tessdata-dir "{}"'.format(data_path)
with PyTessBaseAPI(path=data_path) as api:
    native_string = None
    native_time = timeit.timeit("global native_string; native_string = native_image_to_string(api, image)", globals=globals(), number=1)
    native_cost = cost(native_string, gt_string)
    binary_string = None
    binary_time = timeit.timeit("global binary_string; binary_string = pytesseract.image_to_string(image, config=tessdata_dir_config)", globals=globals(), number=1)
    binary_cost = cost(binary_string, gt_string)
    print(native_string)
    print("------------------")
    print(binary_string)
    print("preprocessing time: {:f}".format(preprocessing_time))
    print("native\ttime: {:.2f}\tcost: {:.2f}".format(native_time, native_cost))
    print("binary\ttime: {:.2f}\tcost: {:.2f}".format(binary_time, binary_cost))
