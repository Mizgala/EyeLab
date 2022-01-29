# Home for the delta_p(q) implementation
import numpy as np


class Q:
    def __init__(self, r_1, v_1, hct, r):
        self.r_1 = r_1
        self.v_1 = v_1
        self.hct = hct
        self.r = r

    def calculate_u(self):
        u_inf = 1.09 * np.exp(0.024 * self.hct)
        div = pow((1 + 4.29) / self.r, 2)
        return u_inf / div

    def calculate_L(self):
        return 7.4 * pow(self.r, 1.15)

    def calculate_v_g(self, g, r_g):
        a = pow(2, -(g-1))
        b = pow(self.r_1/r_g, 2)
        return a * b * self.v_1

    def eval(self, g, r_g):
        return np.pi * pow(r_g, 2) * self.calculate_v_g(g, r_g)


def delta_p(q, g, r_g):
    a = 8 * q.calculate_u() * q.calculate_L() * q.eval(g, r_g)
    return a / (np.pi * pow(q.r, 4))


q = Q(0.5)
