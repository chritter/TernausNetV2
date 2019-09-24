
from pylab import *
from skimage.morphology import watershed
import scipy.ndimage as ndimage
from PIL import Image, ImagePalette

from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
import torch

# read image and metadata from TIFF-like files used in bioimaging.
import tifffile as tiff

# python binding for openCV
import cv2
import random
from pathlib import Path

#from torch.utils.tensorboard import SummaryWriter

# allow reproducability
random.seed(42)
NUCLEI_PALETTE = ImagePalette.random()
random.seed()

# %%

rcParams['figure.figsize'] = 15, 15

# %%

from models.ternausnet2 import TernausNetV2


# %%

def get_model(model_path):
    """
    Loads models from model_path.
    :param model_path:
    :return:
    """
    model = TernausNetV2(num_classes=2)

    # Loads an object (model) saved with torch.save from a file. Default is GPU, so I change it to cpu
    state = torch.load('weights/deepglobe_buildings.pt', map_location=torch.device('cpu'))

    # modify state_dict
    state = {key.replace('module.', '').replace('bn.', ''): value for key, value in state['model'].items()}

    # load parameters from state_dict
    model.load_state_dict(state)

    # set dropout and batch normalization layers to evaluation mode for inference
    model.eval()

    # move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()

    return model


# %%

def pad(img, pad_size=32):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (network requirement)
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """

    if pad_size == 0:
        return img

    height, width = img.shape[:2]

    if height % pad_size == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = pad_size - height % pad_size
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % pad_size == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = pad_size - width % pad_size
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    # Forms a border around an image,
    # cv2.BORDER_REFLECT_101 - Border will be mirror reflection of the border elements, like this:
    # gfedcb|abcdefgh|gfedcba
    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


# %%

def unpad(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]


# %%

def minmax(img):
    """
    Normalize image, each channel separately, to 0-1 range
    :param img: input image array
    :return:
    """
    out = np.zeros_like(img).astype(np.float32)
    if img.sum() == 0:
        return bands

    for i in range(img.shape[2]):
        c = img[:, :, i].min()
        d = img[:, :, i].max()

        t = (img[:, :, i] - c) / (d - c)
        out[:, :, i] = t
    return out.astype(np.float32)


# %%

def load_image(file_name_rgb, file_name_tif):
    """

    :param file_name_rgb: RGB image
    :param file_name_tif: MUL image with extra channels
    :return:
    """

    # Return image data from TIFF file(s) as numpy array.
    rgb = tiff.imread(str(file_name_rgb))

    # 0-1 normalization
    rgb = minmax(rgb)

    tf = tiff.imread(str(file_name_tif)).astype(np.float32) / (2 ** 11 - 1)

    return np.concatenate([rgb, tf], axis=2) * (2 ** 8 - 1)


# %%

def label_watershed(before, after, component_size=20):
    markers = ndimage.label(after)[0]

    labels = watershed(-before, markers, mask=before, connectivity=8)
    unique, counts = np.unique(labels, return_counts=True)

    for (k, v) in dict(zip(unique, counts)).items():
        if v < component_size:
            labels[labels == k] = 0
    return labels


# %%

model = get_model('weights/deepglobe_buildings.pt')

# %%

img_transform = Compose([
    # Convert a PIL Image or numpy.ndarray to tensor.
    ToTensor(),
    # Normalize a tensor image with mean and standard deviation for 11 bands as in paper
    Normalize(mean=[0.485, 0.456, 0.406, 0, 0, 0, 0, 0, 0, 0, 0],
              std=[0.229, 0.224, 0.225, 1, 1, 1, 1, 1, 1, 1, 1])
])

# %%

# example files
file_name_rgb = Path('img') / 'RGB-PanSharpen_AOI_4_Shanghai_img6917.tif' #RGB
file_name_tif = Path('img') / 'MUL-PanSharpen_AOI_4_Shanghai_img6917.tif' #other 8 bands

# %%

img = load_image(file_name_rgb, file_name_tif)

# %%

# show RGB channel image
imshow(img[:, :, :3].astype(np.uint8))

# %%

# Network contains 5 maxpool layers (see forward()) => input should be divisible by 2**5 = 32 => we pad input image and mask
img, pads = pad(img)

# %%

#input_img = torch.unsqueeze(img_transform(img / (2 ** 8 - 1)).cuda(), dim=0)
input_img = torch.unsqueeze(img_transform(img / (2 ** 8 - 1)), dim=0)

input_img.shape

# %%

input_img2 = torch.Tensor(np.ones((1, 11, 672, 672)))
input_img2.shape

prediction = torch.sigmoid(model(input_img2)).data[0].cpu().numpy()

# %%

# First predicted layer - mask
# Second predicted layer - touching areas
prediction.shape

# %%

# left mask, right touching areas
imshow(np.hstack([prediction[0], prediction[1]]))

# %%

mask = (prediction[0] > 0.5).astype(np.uint8)
contour = (prediction[1])

seed = ((mask * (1 - contour)) > 0.5).astype(np.uint8)

# %%

labels = label_watershed(mask, seed)

# %%

labels = unpad(labels, pads)

# %%

im = Image.fromarray(labels.astype(np.uint8), mode='P')
im.putpalette(NUCLEI_PALETTE)

# %%

im

# %%


