import numpy as np
from PIL import Image
from PIL import ImageOps


def importImage(path="test.png"):
    return Image.open(path)


def toArray(image):
    return np.asarray(image)



