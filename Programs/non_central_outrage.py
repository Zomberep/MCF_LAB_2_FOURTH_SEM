from numpy import sin, cos, pi, sqrt, arctan2
from scipy.integrate import solve_ivp
from functions import calculate_f_non_central
import matplotlib.pyplot as plt
import data
import numpy as np


def system(u, y):
    """
    y consists of [omega, i, p, q, k, t]
                  [rad, rad, km, -, -, sec]

    u is the current position

    q = e * cos(w)
    k = e * sin(w)

    e = (q * q + k * k) ** 0.5
    w = cos⁻¹(q / e)
    r = p / (1 + e * cos(ν))
    ν = u - w
    """

    omega, i, p, q, k, t = y

    e = sqrt(q * q + k * k)
    w = arctan2(k / e, q / e) # radians
    a = p / (1 - e * e) # km

    nu = u - w # radians
    r = p / (1 + e * cos(nu)) # km

    f_1, f_2, f_3 = calculate_f_non_central(u, i, r)

    ctg = lambda x: cos(x) / sin(x)

    gamma = 1 - f_3 * (r ** 3) * sin(u) * ctg(i) / (data.mu * p)

    do_du = (r ** 3) * f_3 * sin(u) / (gamma * data.mu * p * sin(i))
    di_du = (r ** 3) * f_3 * cos(u) / (gamma * data.mu * p)
    dp_du = 2 * f_2 * (r ** 3) / (gamma * data.mu)
    dt_du = (r ** 2) / (gamma * sqrt(data.mu * p))

    dq_du = (r ** 2) / (gamma * data.mu)
    dq_du = dq_du * (f_1 * sin(u) + f_2 * ((1 + r / p) * cos(u) + r * q / p) + f_3 * r * k * ctg(i) * sin(u) / p)

    dk_du = (r ** 2) / (gamma * data.mu)
    dk_du = dk_du * ((-1) * f_1 * cos(u) + f_2 * ((1 + r / p) * sin(u) + r * k / p) - f_3 * r * q * ctg(i) * sin(u) / p)

    return [do_du, di_du, dp_du, dq_du, dk_du, dt_du]


if __name__ == "__main__":
    flag = "main"
    import_flag = 0

    u_segment = [data.u, data.u + 10 * pi] # segment for finding solution

    initial_data = [data.omega, data.i, data.p, data.e * cos(0), data.e * sin(0), 0]

    solution = solve_ivp(system, u_segment, initial_data, dense_output=True, rtol=1e-14, atol=1e-14)

    u_values = solution.t

    omega_values = solution.y[0, :]
    i_values = solution.y[1, :]
    p_values = solution.y[2, :]
    q_values = solution.y[3, :]
    k_values = solution.y[4, :]
    t_values = solution.y[5, :]

    e_values = sqrt(q_values * q_values + k_values * k_values)
    w_values = arctan2(k_values / e_values, q_values / e_values)

    # let's calculate f_1, f_2 and f_3
    nu_values = u_values - w_values
    r_values = p_values / (1 + e_values * cos(nu_values))
    l = len(u_values)

    f_a_values = np.array([calculate_f_non_central(u_values[i], i_values[i], r_values[i]) for i in range(l)])
    f_1 = f_a_values[:, 0]
    f_2 = f_a_values[:, 1]
    f_3 = f_a_values[:, 2]

    if flag == "main":
        all_values = [i_values, p_values, e_values, w_values, omega_values, f_1, f_2, f_3]
        all_titles = ["График зависимости наклонения i(u), рад", "График зависимости фокального параметра p(u), км", "График зависимости эксцентриситета e(u)",
                      "График зависимости аргумента перицента ω(u), рад", "График зависимости долготы восходящего узла Ω(u), рад",
                      "График зависимости компоненты F₁(u), км/ceк", "График зависимости компоненты F₂(u), км/сек", "График зависимости компоненты F₃(u), км/сек"
                      ]
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(25, 15))
    else:
        add_values = [t_values, sqrt(f_1 * f_1 + f_2 * f_2 + f_3 * f_3)]
        add_titles = ["График зависимости времени t(u), сек", "График зависимости полного ускорения Fₐ(u), км/c"]
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(25, 15))

    fig.set_facecolor("#edebeb")

    if flag == "main":
        for i, list_ax in enumerate(axes):
            for j, ax in enumerate(list_ax):
                ax.plot(u_values, all_values[i * 2 + j], c="#3607f2")
                ax.set_xlabel("u, рад", fontsize=13, fontweight="semibold")
                ax.locator_params(axis='y', nbins=8)
                ax.locator_params(axis='x', nbins=15)
                ax.set_title(all_titles[i * 2 + j], fontsize=14, fontweight="bold")
                ax.set_facecolor("#eeeeee")
                ax.grid()
    else:
        for i, ax in enumerate(axes):
            ax.plot(u_values, add_values[i], c="#3607f2")
            ax.set_xlabel("u, рад", fontsize=13, fontweight="semibold")
            ax.locator_params(axis='y', nbins=8)
            ax.locator_params(axis='x', nbins=15)
            ax.set_title(add_titles[i], fontsize=14, fontweight="bold")
            ax.set_facecolor("#eeeeee")
            ax.grid()

    # import values
    if import_flag and flag == "main":
        file_names = ["i(u).csv", "p(u).csv", "e(u).csv", "ω(u).csv", "Ω(u).csv", "F₁(u).csv", "F₂(u).csv", "F₃(u).csv"]

        for i, array in enumerate(all_values):
            with open(file_names[i], "w") as file:
                print(*array, sep=",", file=file)
    elif import_flag:
        file_names = ["t(u).csv", "Fₐ(u).csv", "u.csv"]

        for i, array in enumerate(add_values + [u_values]):
            with open(file_names[i], "w") as file:
                print(*array, sep=",", file=file)

    print(*u_values, sep="\n")

    plt.tight_layout(h_pad=3.5)
    plt.show()