# Home for the delta_p(q) implementation
import numpy as np


class Q:
    def __init__(self, v_type="artery"):
        self.r_1 = 54.0
        self.v_1 = 2.055
        self.v_type = v_type

    def calc_v_g(self, g, r):
        a = pow(2, - (g - 1))
        b = self.r_1 / r
        c = self.v_1
        return a * pow(b, 2) * c

    def calc_q(self, g, r):
        return np.pi * pow(r, 2) * self.calc_v_g(g, r)


def calc_l(r):
    return 7.4 * pow(r, 1.15)


def calc_mu(r, v_type):
    if v_type == "artery":
        mu_inf = 1.09 * np.exp(0.024 * 57.4)
    elif v_type == "vein":
        mu_inf = 1.09 * np.exp(0.024 * 60.7)
    else:
        mu_inf = 1.09 * np.exp(0.024 * 120.0)

    return mu_inf / pow(1 + (4.29 / r), 2)


def delta_p(q, g, r):
    r = micro(r)
    a = 8
    b = calc_mu(r, q.v_type)
    c = calc_l(r)
    d = q.calc_q(g, r)
    num = a * b * c * d
    den = np.pi * pow(r, 4)
    return num / den


def centi(n):
    return n * 0.01

def mili(n):
    return n * 0.001

def micro(n):
    return n * 0.000001


def delta_p_m(mu_inf, r_1, v_1, g, r):
    delta = 4.29
    a = mu_inf * 7.4
    b = a * pow(r_1, 2)
    c = b * v_1
    num = c * pow(r, 1.15)
    a = pow(2, g-1)
    b = a * pow(r, 2)
    c = r * r + 2 * r * delta + delta * delta
    den = b * c
    return num / den
