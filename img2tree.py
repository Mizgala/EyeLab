import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def import_image(path="images/g1.png"):
    return Image.open(path)


def image2a(image):
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


def zip_coords(xs, ys):
    res = []
    for i in range(0, len(xs)):
        res.append([xs[i], ys[i]])
    return np.array(res)


def p2ys(p, xs):
    ys = []
    for x in xs:
        ys.append(p[0] * x + p[1])
    return np.array(ys)


def image2line(image):
    xs, ys = split_coords(image2coords(image))
    return np.polyfit(xs, ys, 1)


def image2coords(image):
    a = image2a(image.convert("1"))
    a = array_map(bw2bin, a)
    a = array_map(flip, a)
    return a2coords(a)


def point_width_over_x(coords):
    xs, ys = split_coords(coords)
    xs_u = list(set(xs.flatten()))
    ys_split = []
    for x in xs_u:
        tmp = []
        for c in coords:
            if c[0] == x:
                tmp.append(c[1])
        ys_split.append(tmp)
    res = []
    for i in range(0, len(xs_u)):
        res.append([xs_u[i], len(ys_split[i])])
    return res


def bound_coords(coords, axis, c_min, c_max):
    res = list(coords)
    d = -1
    if axis == 'x':
        d = 0
    elif axis == 'y':
        d = 1

    if d != -1:
        res = list(filter((lambda c: c_min <= c[d] <= c_max), res))

    return np.array(res)


def graph(image, path):
    a_g1 = image2a(image.convert("1"))
    res = array_map(bw2bin, a_g1)
    res = array_map(flip, res)
    res = a2coords(res)
    graph_coords(image, path, res, True, True)


def get_corners(path, graph_res=False):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 5, 3, 0.04)
    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    if graph_res:
        img[dst > 0.1 * dst.max()] = [0, 0, 255]
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return corners


def dist(a, b):
    y = b[1] - a[1]
    x = b[0] - a[0]
    return pow(pow(y, 2) + pow(x, 2), 0.5)


def prune_corners(coords, prune, lazy=True):
    indexes = list(range(0, len(coords)))
    i_pairs = [(a, b) for a in indexes for b in indexes]
    i_pairs = list(filter(lambda p: p[0] != p[1], i_pairs))
    dists = []
    for i in i_pairs:
        dists.append(dist(coords[i[0]], coords[i[1]]))
    cds = zip_coords(i_pairs, dists)
    is_to_remove = [0]
    for cd in cds:
        inds = cd[0]
        if inds[0] not in is_to_remove and inds[1] not in is_to_remove:
            if cd[1] < prune:
                is_to_remove.append(inds[1])

    indexes = list(filter(lambda i: i not in is_to_remove, indexes))
    return np.array(list(map(lambda i: coords[i], indexes)))


def graph_coords(image, path, coords,
                 bound_y=False, lock_aspect_ratio=False,
                 graph_type="line"):
    a_g1 = image2a(image.convert("1"))
    xs, ys = split_coords(coords)
    p = np.polyfit(xs, ys, 1)
    ys_gen = p2ys(p, xs)

    shape = a_g1.shape

    fig, (ax1, ax2) = plt.subplots(1, 2)

    img = mpimg.imread(path)
    ax1.imshow(img)
    ax1.set_xlim([0, shape[1]])
    ax1.set_ylim([0, shape[0]])
    ax1.invert_yaxis()

    if graph_type == "line":
        ax2.plot(xs, ys_gen)
    elif graph_type == "scatter":
        ax2.scatter(xs, ys)
    ax2.set_xlim([0, shape[1]])
    if bound_y:
        ax2.set_ylim([0, shape[0]])
    ax2.invert_yaxis()
    if lock_aspect_ratio:
        ax2.set_aspect('equal')

    plt.show()
