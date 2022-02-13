import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Root:
    def __init__(self, start, end, r=0.0, branches: "Tree" = None):
        self.start = start
        self.end = end
        self.r = r
        self.branches = branches
        if branches is None:
            self.has_branches = False
        else:
            self.has_branches = True

    def add_branches(self, branches: "Tree"):
        self.branches = branches
        self.has_branches |= True

    def get_length(self):
        return dist(self.start, self.end)


class Tree:
    def __init__(self, split_point, r=0.0, branches: "(Tree, Tree)" = None):
        self.split_point = split_point
        self.r = r
        self.branches = branches
        if branches is None:
            self.has_branches = False
        else:
            self.has_branches = True

    def split(self, branches):
        self.branches = branches
        self.has_branches = True

    def length_top(self):
        if self.has_branches:
            res = dist(self.split_point, self.branches[0].split_point)
        else:
            res = 0.0
        return res

    def length_bot(self):
        if self.has_branches:
            res = dist(self.split_point, self.branches[1].split_point)
        else:
            res = 0.0
        return res


def import_image(path="images/g1.png"):
    # import an image as an Image object
    return Image.open(path)


def image2a(image):
    # convert an Image object to a numpy array of color values
    return np.asarray(image)


def bw2bin(b):
    # convert boolean black and white values to their binary equivalent
    return 1 if b else 0


def bin2bw(b):
    # convert binary black and white values to their boolean equivalent
    return True if b == 1 else False


def flip(b):
    # flip a bit
    return 1 if b == 0 else 0


def array_map(f, a):
    # apply the function f to every value in array a
    res = np.zeros(a.shape)
    shape = a.shape
    res_l = res.reshape(np.prod(shape))
    a_l = a.reshape(np.prod(shape))
    for i in range(0, len(a_l)):
        res_l[i] = f(a_l[i])
    return res_l.reshape(shape)


def a2coords(a):
    # return a numpy array containing all coordinates in an array a where the value is 1
    shape = a.shape
    res = []
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if a[i][j] == 1:
                res.append((j, i))
    return np.array(res)


def split_coords(cs):
    # np.array([x, y]) -> np.array(xs), np.array(ys)
    xs = []
    ys = []
    for c in cs:
        xs.append(c[0])
        ys.append(c[1])
    return np.array(xs), np.array(ys)


def zip_coords(xs, ys):
    # np.array(xs), np.array(ys) -> np.array([x, y])
    res = []
    for i in range(0, len(xs)):
        res.append([xs[i], ys[i]])
    return np.array(res)


def p2ys(p, xs):
    # return a list of y values for p(x) for x in xs
    ys = []
    for x in xs:
        ys.append(p[0] * x + p[1])
    return np.array(ys)


def image2line(image):
    # convert an Image to a line equation
    xs, ys = split_coords(image2coords(image))
    return np.polyfit(xs, ys, 1)


def points2line(p1, p2):
    return np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1)


def image2coords(image):
    # convert an Image to black and white, binarize it, flip the values,
    # and convert to a list of coordinates where the value is 1
    a = image2a(image.convert("1"))
    a = array_map(bw2bin, a)
    a = array_map(flip, a)
    return a2coords(a)


def point_width_over_x(coords):
    # return a list of tuples (a, b) where a is the x value and b is the number of y values found at x
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
    # restrict a list of coordinates to a subset along a given axis
    res = list(coords)
    d = -1
    if axis == 'x':
        d = 0
    elif axis == 'y':
        d = 1

    if d != -1:
        res = list(filter((lambda c: c_min <= c[d] <= c_max), res))

    return np.array(res)


def double_bound_coords(coords, c1, c2):
    tmp_coords = bound_coords(coords, 'x', min(c1[0], c2[0]), max(c1[0], c2[0]))
    return bound_coords(tmp_coords, 'y', min(c1[1], c2[1]), max(c1[1], c2[1]))


def get_corners(path, graph_res=False):
    # return a list of corners found in the image at path
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
    # distance between two points
    y = b[1] - a[1]
    x = b[0] - a[0]
    return pow(pow(y, 2) + pow(x, 2), 0.5)


def get_r2(p, coords):
    # calculate the r2 value for a p
    rss = 0.0
    tss = 0.0
    xs, ys = split_coords(coords)
    y_mean = np.mean(ys)
    for i in range(0, len(xs)):
        rss += pow(ys[i] - (xs[i] * p[0] + p[1]), 2)
        tss += pow(ys[i] - y_mean, 2)
    return 1 - (rss/tss)


def prune_corners(coords, prune, lazy=True):
    # remove "duplicate" corners
    # lazy removes the weird phantom corner
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


def gen_root(corners, coords):
    scs = list(sorted(corners, key=lambda c: c[0]))
    return Root(scs[0], scs[1], branches=None)


def gen_branches(corners, coords):
    c_pairs = [(a, b) for a in corners for b in corners]
    c_pairs = list(filter(lambda c: not are_points_equal(c[0], c[1]), c_pairs))
    c_pairs = list(filter(lambda c: c[0][0] < c[1][0], c_pairs))
    prs = []
    for cp in c_pairs:
        tmp_coords = bound_coords(coords, 'x', cp[0][0], cp[1][0])
        xs, ys = split_coords(tmp_coords)
        p = np.polyfit(xs, ys, 1)
        prs.append([cp, abs(get_r2(p, tmp_coords))])
    prs = list(sorted(prs, key=lambda pr: prs[1], reverse=False))
    ps, _ = split_coords(prs)
    return ps


def are_points_equal(p1, p2):
    return p1[0] == p2[0] and p1[1] == p2[1]


def eval_branch_acc(branch, coords):
    bcs = double_bound_coords(coords,
                              [branch[0][0]-10, branch[0][1]-10],
                              [branch[1][0]+10, branch[1][1]+10])
    xs = list(range(int(branch[0][0]), int(branch[1][0])))
    p = points2line(branch[0], branch[1])
    fs = p2ys(p, xs)
    _, ys = split_coords(bcs)
    diffs = []
    for i in range(0, len(xs)):
        diffs += pow(fs[i] - ys[i], 2)
    return 1 / (sum(diffs) + 0.00001)


def are_branches_equal(b1, b2):
    return are_points_equal(b1[0], b2[0]) and are_points_equal(b1[1], b2[1])


def link_branches(b1, b2):
    # Assumes that b1 < b2 with respect to x
    return np.array([b1[0], b2[1]])


def graph(image, path):
    # graph an image along with the generated representation
    a_g1 = image2a(image.convert("1"))
    res = array_map(bw2bin, a_g1)
    res = array_map(flip, res)
    res = a2coords(res)
    graph_coords(image, path, res, True, True)


def graph_coords(image, path, coords,
                 bound_y=False, lock_aspect_ratio=False,
                 graph_type="line"):
    # graph an image and something else
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
        ax2.scatter(xs, ys, plt.rcParams['lines.markersize'] ** 0.01)
    ax2.set_xlim([0, shape[1]])
    if bound_y:
        ax2.set_ylim([0, shape[0]])
    ax2.invert_yaxis()
    if lock_aspect_ratio:
        ax2.set_aspect('equal')

    plt.show()
