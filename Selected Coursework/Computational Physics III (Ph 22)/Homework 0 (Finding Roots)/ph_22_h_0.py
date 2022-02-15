# Philip Carr
# Ph 22 Homework 0 (Updated)
# February 6, 2020

import random, math
from matplotlib import pyplot as plt

def sign(a):
    """Return the sign of a number."""
    if a < 0:
        return -1
    elif a == 0:
        return 0
    else:
        return 1

def bisection(f, x1, x2, tolerance, print_progress=False):
    """Bisection root-finding method."""
    assert f(x1) * f(x2) <= 0
    convergence_list = []
    x0 = x1
    convergence_list.append(x0)
    if print_progress:
        print(x0)
    while abs(f(x0)) > tolerance:
        x0 = (x1 + x2) / 2
        convergence_list.append(x0)
        if print_progress:
            print(x0)
        if sign(f(x0)) == sign(f(x1)):
            x1 = x0
        else:
            x2 = x0
    return convergence_list

def newton_raphson(f, f_prime, x1, tolerance, print_progress=False):
    """Newton-Raphson root-finding method."""
    convergence_list = []
    x2 = x1
    convergence_list.append(x2)
    if print_progress:
        print(x2)
    while abs(f(x2)) > tolerance:
        x2_old = x2
        x2 = x1 - (f(x1) / f_prime(x1))
        x1 = x2_old
        convergence_list.append(x2)
        if print_progress:
            print(x2)
    return convergence_list

def secant(f, x1, x2, tolerance, print_progress=False):
    """Secant root-finding method."""
    assert f(x1) * f(x2) <= 0
    convergence_list = []
    x3 = x1
    convergence_list.append(x3)
    if print_progress:
        print(x3)
    while (abs(f(x3)) > tolerance):
        x3 = x2 - f(x2) * (x2 - x1) / (f(x2) - f(x1))
        convergence_list.append(x3)
        if print_progress:
            print(x3)
        if sign(f(x3)) == sign(f(x1)):
            x1 = x3
        else:
            x2 = x3
    return convergence_list

def main():
    print("Assignment 1: Finding Roots (Output)")
    print("Problem 2:")
    c = random.random() * 2 - 1
    f = lambda x: math.sin(x) - c
    f_prime = lambda x: math.cos(x)
    p2_plot_title = "Convergence of Function: f(x) = sin(x) " \
                    + ("- " + str(-c)) * (c < 0) \
                    + ("+ " + str(c)) * (c >= 0)
    print("Function: f(x) = sin(x) " + ("- " + str(-c)) * (c < 0)
          + ("+ " + str(c)) * (c >= 0))
    div_term = 3
    x1 = -math.pi / div_term
    x2 = math.pi / div_term
    while f(x1) * f(x2) > 0:
        div_term -= 0.1
        x1 = -math.pi / div_term
        x2 = math.pi / div_term
    tolerance = 0.0001
    print("Bisection method")
    bisection_convergence = bisection(f, x1, x2, tolerance, print_progress=True)
    print("\nNewton-raphson method")
    newton_raphson_convergence = newton_raphson(f, f_prime, x1, tolerance,
                                                print_progress=True)
    print("\nSecant method")
    secant_convergence = secant(f, x1, x2, tolerance, print_progress=True)
    plt.figure()
    plt.title(p2_plot_title)
    plt.xlabel("Number of iterations")
    plt.ylabel("Error")
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(list(map(lambda x: abs(f(x)), bisection_convergence)), marker="o",
             label="bisection")
    plt.plot(list(map(lambda x: abs(f(x)), newton_raphson_convergence)),
             marker="x", label="newton_raphson")
    plt.plot(list(map(lambda x: abs(f(x)), secant_convergence)), marker="*",
             label="secant")
    plt.legend()
    plt.show()

    print("\nProblem 3:")
    dt = 100
    e = 0.617139
    T = 27906.98161
    a = 2.34186 * 299792.458 # in km
    t_values = list(range(0, int(T + 1), dt))
    x_coords = []
    y_coords = []
    for t in t_values:
        tolerance_p3 = 0.00001

        # Keplerian orbit function.
        xi_function = lambda xi: T / (2 * math.pi) * (xi - e * math.sin(xi)) - t
        xi1 = 0
        xi2 = 0.1
        while xi_function(xi1) * xi_function(xi2) > 0:
            xi2 += 0.1
        xi = secant(xi_function, xi1, xi2, tolerance)[-1]
        x = a * (math.cos(xi) - e)
        y = a * math.sqrt(1 - e * e) * math.sin(xi)
        x_vel = (x)
        x_coords.append(x)
        y_coords.append(y)
    plt.figure()
    plt.title("Binary pulsar 1913+16 orbit (in km)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x_coords, y_coords)
    plt.show()

    print("\nProblem 4:")
    phi = 4.7 # Closest match angle for radial velocity plot.
    phase_shift = 0
    proj_vels = []
    t_frac_values = []
    for i in range(len(t_values) - 1):
        t_frac_values.append(t_values[i] / T)
        x_vel = (x_coords[i + 1] - x_coords[i]) / dt
        y_vel = (y_coords[i + 1] - y_coords[i]) / dt
        proj_vels.append(x_vel * math.cos(phi) + y_vel * math.sin(phi))
    plt.figure()
    plt.title("Binary pulsar 1913+16 radial velocity with phi = "
              + str(round(4.7 * 180 / math.pi, 4)) + " degrees")
    plt.xlabel("Fraction of orbital period")
    plt.ylabel("Radial velocity (km s^-1)")
    plt.plot(t_frac_values, proj_vels)
    plt.show()

if __name__ == "__main__": main()
