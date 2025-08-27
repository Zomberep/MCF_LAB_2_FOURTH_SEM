from math import sin, tan, atan, cos, pi, isclose, asin, e as euler
import numpy as np
import data


def calculate_e_anomaly(start: float | int, e: float, eps: float = 1745e-8) -> float:
    curr_e, prev_e = start + e * sin(start), start
    while abs(curr_e - prev_e) > eps:
        prev_e = curr_e
        curr_e = start + e * sin(prev_e)

    return curr_e # radians


def calculate_true_anomaly(anom: float | int, e: float) -> float:
    arg = tan(anom / 2) * (((1 + e) / (1 - e)) ** 0.5)
    ans = 2 * atan(arg)

    if ans < 0:
        ans += 2 * pi

    return ans ## radians


def calculate_radius_a(a: float, e: float, true_anom: float) -> float:
    return (a * (1 - e * e)) / (1 + e * cos(true_anom)) ## km


def calculate_agecs(r_a: float, true_anom: float, omega: float, w: float, i: float) -> tuple:
    u = true_anom + w
    x_a = r_a * (cos(u) * cos(omega) - sin(u) * sin(omega) * cos(i))
    y_a = r_a * (cos(u) * sin(omega) + sin(u) * cos(omega) * cos(i))
    z_a = r_a * sin(u) * sin(i)

    return x_a, y_a, z_a ## km


def calculate_speed(e: float, a: float, true_anom: float) -> tuple:
    p = a * (1 - e * e)
    mult = (398_600 / p) ** 0.5

    return mult * e * sin(true_anom), mult * (1 + e * cos(true_anom)) ## km / sec


def calculate_gecs(cords: tuple, t: float) -> tuple:
    s = 7.29e-5 * t # radians
    matrix = np.array([[cos(s), sin(s), 0],
                       [-sin(s), cos(s), 0],
                       [0, 0, 1]])

    vector = np.array(cords)

    return tuple(matrix @ vector)


def calculate_gc(cords: tuple) -> tuple:
    e23 = 0.0067385254
    a = 6378.136 # km
    x, y, z = cords # km

    d = (x * x + y * y) ** 0.5
    if isclose(d, 0.0, abs_tol=1e-12):
        b = (pi / 2) * (z / abs(z))
        l = 0
        h = z * sin(b) - a * ((1 - e23 * sin(b) * sin(b)) ** 0.5)
    else:
        l_a = asin(y / d)
        if x < 0:
            l = pi + (1 - 2 * (y > 0)) * l_a
        else:
            l = 2 * pi - l_a if y < 0 else l_a

        if isclose(z, 0.0, abs_tol=1e-12):
            b = 0
            h = d - a
        else:
            r = (x * x + y * y + z * z) ** 0.5
            c = asin(z / r)
            p = e23 * a / (2 * r)

            s1, b, s2 = 0.0, c, asin(p * sin(2 * c) / ((1 - e23 * sin(c) * sin(c)) ** 0.5))
            while abs(s2 - s1) > 1e-4:
                s1 = s2
                b += s1
                s2 = asin(p * sin(2 * b) / ((1 - e23 * sin(b) * sin(b)) ** 0.5))

            h = d * cos(b) + z * sin(b) - a * ((1 - e23 * sin(b) * sin(b)) ** 0.5)

    return b, l, h ## radians, radians, km


def calculate_density(H: float, a: tuple) -> float:
    p0 = 1.58868e-8 # kg / m ** 3

    return p0 * (euler ** (a[1] + a[2] * H + a[3] * H ** 2 + a[4] * H ** 3 + a[5] * H ** 4 + a[6] * H ** 5 + a[7] * H ** 6)) ## kg / m ** 3


def calculate_g(H: float) -> float:
    return (398_600 / ((6371 + H) ** 2)) * 1_000 # meter per sec ** 2

def calculate_f_non_central(u: float, i: float, r: float) -> tuple:
    R_e = 6378137 / 1000 # km
    alpha = 1 / 298.25
    w = 7.29e-5 # radians
    q = (w ** 2) * (R_e ** 3) / data.mu

    sigma = data.mu * (R_e ** 2) * (alpha - q / 2) # km ** 5 / sec ** 2

    f_1 = sigma * (3 * ((sin(u) * sin(i)) ** 2) - 1) / (r ** 4) # km / sec ** 2
    f_2 = -sigma * sin(i) * sin(i) * sin(2 * u) / (r ** 4) # km / sec ** 2
    f_3 = -sigma * sin(2 * i) * sin(u) / (r ** 4) # km / sec ** 2

    return f_1, f_2, f_3


def calculate_f_a(true_anom: float, r_a: float,  a: float, e: float, omega: float, w: float, i: float, t: float) -> tuple:
    agecs = calculate_agecs(r_a, true_anom, omega, w, i)
    gecs = calculate_gecs(agecs, t)

    gc = calculate_gc(gecs)
    speed = calculate_speed(e, a, true_anom)

    del agecs, gecs

    if gc[2] >= 500:
        coef_a = data.data_500_1500[250]
    else:
        coef_a = data.data_120_500[250]

    p = calculate_density(gc[2], coef_a)
    v = ((speed[0] * speed[0] + speed[1] * speed[1]) ** 0.5) * 1000 ## meter per second

    weight = data.m ## kg
    cx = data.cx_a
    square = data.s_a ## meters ** 2
    balist_coef = cx * square / (2 * weight)

    return balist_coef * p * v * v  ## meter per second ** 2