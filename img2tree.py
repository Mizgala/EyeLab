import numpy as np
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def import_image(path="images/g1.png"):
    return Image.open(path)


def to_array(image):
    return np.asarray(image)


def bw2bin(b):
    return 1 if b else 0


def bin2bw(b):
    return True if b == 1 else False


def flip(b):
    return 1 if b == 0 else 0


def array_map(f, a):
    res = np.zeros(a.shape)
    shape = a.shape
    res_l = res.reshape(np.prod(shape))
    a_l = a.reshape(np.prod(shape))
    for i in range(0, len(a_l)):
        res_l[i] = f(a_l[i])
    return res_l.reshape(shape)


def a2coords(a):
    shape = a.shape
    res = []
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if a[i][j] == 1:
                res.append((j, i))
    return np.array(res)


def split_coords(cs):
    xs = []
    ys = []
    for c in cs:
        xs.append(c[0])
        ys.append(c[1])
    return np.array(xs), np.array(ys)


def p2ys(p, xs):
    ys = []
    for x in xs:
        ys.append(p[0] * x + p[1])
    return np.array(ys)


def image2line(image):
    a_g1 = to_array(image.convert("1"))
    res = array_map(bw2bin, a_g1)
    res = array_map(flip, res)
    res = a2coords(res)
    xs, ys = split_coords(res)
    return np.polyfit(xs, ys, 1)


def graph(image):
    a_g1 = to_array(image.convert("1"))
    res = array_map(bw2bin, a_g1)
    res = array_map(flip, res)
    res = a2coords(res)
    xs, ys = split_coords(res)
    p = np.polyfit(xs, ys, 1)
    ys_gen = p2ys(p, xs)

    shape = a_g1.shape

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('img2line')

    img = mpimg.imread("images/g1.png")
    ax1.imshow(img)
    ax1.set_xlim([0, shape[1]])
    ax1.set_ylim([0, shape[0]])
    ax1.invert_yaxis()

    ax2.plot(xs, ys_gen)
    ax2.set_xlim([0, shape[1]])
    ax2.set_ylim([0, shape[0]])
    ax2.invert_yaxis()
    ax2.set_aspect('equal')

    plt.show()
