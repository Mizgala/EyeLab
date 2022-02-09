# Home for the delta_p(q) implementation
import numpy as np


def calculate_L(r):
    return 7.4 * pow(r, 1.15)


class Q:
    def __init__(self, r_1, v_1):
        self.r_1 = r_1
        self.v_1 = v_1

    def calculate_v_g(self, g, r_g):
        a = pow(2, -(g-1))
        b = pow(self.r_1/r_g, 2)
        return a * b * self.v_1

    def eval(self, g, r_g):
        return np.pi * pow(r_g, 2) * self.calculate_v_g(g, r_g)


def delta_p_inner(g, r, u):
    a = 8 * u * calculate_L(r) * q.eval(g, r)
    return a / (np.pi * pow(r, 4))


def delta_p(g):
    return delta_p_inner(gs[g], diams[g - 1] * 0.5, viscs[g - 1])


# hct recommendations for now: 57.4 in arteries, 60.7 in veins, good luck in near capillaries

gs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

diams = [108, 84.7, 66.4, 52.1, 40.8, 32.0, 25.1, 19.7, 15.4, 12.1, 9.5, 7.4, 5.8, 5.1, 5.0,
         6.2, 7.9, 10.1, 12.9, 16.5, 21.0, 26.8, 34.2, 43.6, 55.6, 70.9, 90.4, 115.3, 147.0]

viscs = [3.7, 3.6, 3.4, 3.2, 2.9, 2.7, 2.4, 2.1, 1.8, 1.5, 1.2, 0.9, 2.5, 4.2, 4.6,
         2.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.7, 3.0, 3.2, 3.5, 3.7, 3.9, 4.0, 4.2]

lens = [726.9, 549.6, 415.5, 314.1, 237.5, 179.5, 135.7, 102.6, 77.6, 58.7, 44.3, 33.5, 25.3, 21.7, 500.0,
        27.3, 36.1, 47.8, 63.2, 83.6, 110.6, 146.3, 193.5, 255.9, 338.5, 447.8, 592.3, 783.4, 1036.2]

q = Q(diams[0] * 0.5, 0.000188)
