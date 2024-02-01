import numpy as np
import torch

from color_correction.convert import XYZ_to_LAB


def CIELABDE(xyz1, xyz2, white_point, use_torch = False):
    am = torch if use_torch else np

    lab1 = XYZ_to_LAB(xyz1, white_point, use_torch)
    lab2 = XYZ_to_LAB(xyz2, white_point, use_torch)

    return am.sqrt(((lab2 - lab1) * (lab2 - lab1)).sum(axis=1))


def CIEDE2000(xyz1, xyz2, white_point, use_torch = False):
    am = torch if use_torch else np

    lab1 = XYZ_to_LAB(xyz1, white_point, use_torch)
    lab2 = XYZ_to_LAB(xyz2, white_point, use_torch)

    l1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    l2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    c_1ab = am.sqrt(a1 * a1 + b1 * b1)
    c_2ab = am.sqrt(a2 * a2 + b2 * b2)
    c_ab = (c_1ab + c_2ab) / 2
    c_ab_pow7 = c_ab ** 7
    g = 0.5 * (1 - am.sqrt(c_ab_pow7 / (c_ab_pow7 + 25 ** 7)))

    aa_1 = (1 + g) * a1
    aa_2 = (1 + g) * a2

    c_1 = am.sqrt(aa_1 * aa_1 + b1 * b1)
    c_2 = am.sqrt(aa_2 * aa_2 + b2 * b2)

    hh_1 = am.arctan2(b1, aa_1)
    hh_1 = (hh_1 + (hh_1 < 0) * 2 * am.pi) * 180 / am.pi
    hh_2 = am.arctan2(b2, aa_2)
    hh_2 = (hh_2 + (hh_2 < 0) * 2 * am.pi) * 180 / am.pi

    dll = l2 - l1
    dcc = c_2 - c_1
    hh_diff = hh_2 - hh_1
    c_12_prod = c_1 * c_2
    dhh = (c_12_prod != 0) * (hh_diff + (hh_diff < -180) * 360 - (hh_diff > 180) * 360)
    dh = 2 * am.sqrt(c_12_prod) * am.sin(dhh * am.pi / 360)

    ll = (l1 + l2) / 2
    cc = (c_1 + c_2) / 2
    hh = hh_1 + hh_2
    hh = (c_12_prod == 0) * hh + (c_12_prod != 0) \
        * (hh / 2 + (am.abs(hh_diff) > 180) * ((hh < 360) * 2 - 1) * 180)

    t = 1 - 0.17 * am.cos((hh - 30) * am.pi / 180) + 0.24 * am.cos(hh * am.pi / 90) \
        + 0.32 * am.cos((3 * hh + 6) * am.pi / 180) - 0.2 * am.cos((4 * hh - 63) * am.pi / 180)

    dteta = (hh - 275) / 25
    dteta = 30 * am.exp(-dteta * dteta)
    cc_pow7 = cc ** 7
    r_c = 2 * am.sqrt(cc_pow7 / (cc_pow7 + 25 ** 7))

    ll_moved = ll - 50
    ll_moved_pow2 = ll_moved * ll_moved
    s_l = 1 + 0.015 * (ll_moved_pow2) / am.sqrt(20 + ll_moved_pow2)
    s_c = 1 + 0.045 * cc
    s_h = 1 + 0.015 * cc * t
    r_t = -am.sin(dteta * am.pi / 90) * r_c

    de1 = dll / s_l
    de2 = dcc / s_c
    de3 = dh / s_h
    return am.sqrt(de1 * de1 + de2 * de2 + de3 * de3 + r_t * (dcc * dh) / (s_c * s_h))
