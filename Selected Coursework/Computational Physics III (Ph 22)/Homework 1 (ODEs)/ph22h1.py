# Philip Carr
# February 2, 2020
# ph22h1.py

import math
import numpy as np
from matplotlib import pyplot as plt
from Vector import *

def explicit_euler_step(xi_old, h, f):
    """
    Return updated variables vector xi using the explicit euler method.
    """

    return xi_old + h * Vector(list(map(lambda fi: fi(xi_old), f)))

def midpoint_method_step(xi_old, h, f):
    """
    Return updated variables vector xi using the midpoint method.
    """

    xi_mid = xi_old + (h / 2) * Vector(list(map(lambda fi: fi(xi_old), f)))
    return xi_old + h * Vector(list(map(lambda fi: fi(xi_mid), f)))

def runge_kutta_4th_order_step(xi_old, h, f):
    """
    Return updated variables vector xi using the 4th order Runge-Kutta routine.
    """

    k1 = h * Vector(list(map(lambda fi: fi(xi_old), f)))
    k2 = h * Vector(list(map(lambda fi: fi(xi_old + k1/2), f)))
    k3 = h * Vector(list(map(lambda fi: fi(xi_old + k2/2), f)))
    k4 = h * Vector(list(map(lambda fi: fi(xi_old + k3), f)))

    return xi_old + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def ode_stepper(initial, f, n, ode_method, h=0.001, stop_at_one_cycle=False):
    """
    Stepper function for running ODE solvers.
    """
    xi_list = [initial]
    for i in range(n):
        if i > 0 and stop_at_one_cycle: # For problem 5 (the orbit plot)
            delta_start = xi_list[-1] - xi_list[0]
            dist = math.sqrt(delta_start.dot(delta_start))
            if dist < 0.001:
                break
        xi_list.append(ode_method(xi_list[-1], h, f))
    return xi_list

def explicit_euler_vs_midpoint_method_driver(h):
    """
    Driver function for the explicit Euler and midpoint methods comparison.
    """

    t_prime = lambda xi: 1
    x_prime = lambda xi: xi[2]
    v_prime = lambda xi: -xi[1]
    f = [t_prime, x_prime, v_prime]

    t0 = 0
    x0 = 1
    v0 = 0
    initial = Vector([t0, x0, v0])

    n = 500
    explicit_euler_results = \
        ode_stepper(initial, f, n, explicit_euler_step, h=h)
    midpoint_method_results = \
        ode_stepper(initial, f, n, midpoint_method_step, h=h)
    return explicit_euler_results, midpoint_method_results

def runge_kutta_driver():
    """
    Driver function for the Runge-Kutta 4th order-integrated orbit system.
    """

    x_prime = lambda xi: xi[2]
    y_prime = lambda xi: xi[3]
    vx_prime = lambda xi: -xi[0] \
                          / math.pow(math.sqrt(xi[0] * xi[0] + xi[1] * xi[1]),
                                     3)
    vy_prime = lambda xi: -xi[1] \
                          / math.pow(math.sqrt(xi[0] * xi[0] + xi[1] * xi[1]),
                                     3)
    f = [x_prime, y_prime, vx_prime, vy_prime]

    x0 = 1
    y0 = 0
    vx0 = 0
    vy0 = 1
    initial = Vector([x0, y0, vx0, vy0])

    n = 10000
    return ode_stepper(initial, f, n, runge_kutta_4th_order_step,
                       stop_at_one_cycle=True)

def plotPositionAndVelocity(xi_list, numericalType):
    """
    Plot position and velocity of a physical system over time.
    """
    t_values = list(map(lambda x: x[0], xi_list))
    x_values = list(map(lambda x: x[1], xi_list))
    v_values = list(map(lambda x: x[2], xi_list))
    plt.plot(t_values, x_values)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('x(t) using ' + numericalType + ' Euler method')
    plt.show()
    plt.plot(t_values, v_values)
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.title('v(t) using ' + numericalType + ' Euler method')
    plt.show()

def getHarmonicOscillatorError(xi_list):
    """
    Return the error values of the harmonic oscillator system with respect to
    given data xi_list.
    """
    t_values = list(map(lambda x: x[0], xi_list))
    x_values = list(map(lambda x: x[1], xi_list))
    v_values = list(map(lambda x: x[2], xi_list))
    xAnalyticList = np.cos(t_values[:])
    vAnalyticList = -np.sin(t_values[:])
    xErrorList = xAnalyticList - x_values
    vErrorList = vAnalyticList - v_values
    return xErrorList, vErrorList

def plotHarmonicOscillatorError(xErrorList1, vErrorList1,
                                xErrorList2, vErrorList2,
                                xi_list, numericalType1, numericalType2):
    """
    Plot harmonic oscillator error values.
    """
    t_values = list(map(lambda x: x[0], xi_list))
    plt.plot(t_values, xErrorList1, label=numericalType1)
    plt.plot(t_values, xErrorList2, label=numericalType2)
    plt.xlabel('t')
    plt.ylabel('x(t) error')
    plt.title('Error in x(t)')
    plt.legend()
    plt.show()
    plt.plot(t_values, vErrorList1, label=numericalType1)
    plt.plot(t_values, vErrorList2, label=numericalType2)
    plt.xlabel('t')
    plt.ylabel('v(t) error')
    plt.title('Error in v(t)')
    plt.legend()
    plt.show()

def plotTruncationError(h=0.1, N=5):
    """
    Plot the truncation error of the explit euler and midpoint methods as h is
    scaled down.
    """

    max_xErrorListExplicitEuler = np.zeros(N)
    max_vErrorListExplicitEuler = np.zeros(N)
    max_xErrorListMidpointMethod = np.zeros(N)
    max_vErrorListMidpointMethod = np.zeros(N)
    hList = np.zeros(N)
    for i in range(N):
        explicit_euler_results, midpoint_method_results = \
            explicit_euler_vs_midpoint_method_driver(h)
        xErrorListExplicitEuler, vErrorListExplicitEuler = \
            getHarmonicOscillatorError(explicit_euler_results)
        xErrorListMidpointMethod, vErrorListMidpointMethod = \
            getHarmonicOscillatorError(midpoint_method_results)

        hList[i] = h

        max_xErrorListExplicitEuler[i] = max(xErrorListExplicitEuler)
        max_vErrorListExplicitEuler[i] = max(vErrorListExplicitEuler)

        max_xErrorListMidpointMethod[i] = max(xErrorListMidpointMethod)
        max_vErrorListMidpointMethod[i] = max(vErrorListMidpointMethod)

        h /= 2

    plt.plot(hList, max_xErrorListExplicitEuler, label="explicit Euler method")
    plt.plot(hList, max_xErrorListMidpointMethod, label="midpoint method")
    plt.xlabel('h')
    plt.ylabel('x(t) max error')
    plt.yscale("log")
    plt.title('Max error in x(t) vs h')
    plt.legend()
    plt.show()

    plt.plot(hList, max_vErrorListExplicitEuler, label="explicit Euler method")
    plt.plot(hList, max_vErrorListMidpointMethod, label="midpoint method")
    plt.xlabel('h')
    plt.ylabel('v(t) max error')
    plt.yscale("log")
    plt.title('Max error in v(t) vs h')
    plt.legend()
    plt.show()

def plotTotalEnergy(xi_list, numericalType):
    """
    Plot the total energy of the physical system given by data xi_list.
    """

    t_values = list(map(lambda x: x[0], xi_list))
    x_values = list(map(lambda x: x[1], xi_list))
    v_values = list(map(lambda x: x[2], xi_list))
    totalEnergyList = np.power(x_values, 2) + np.power(v_values, 2)
    plt.plot(t_values, totalEnergyList)
    plt.xlabel('t')
    plt.ylabel('E(t)')
    plt.title('Total energy E(t) using ' + numericalType + ' Euler method')
    plt.show()

def main():
    print("Ph 22 Assignment 1 Program Printout")
    print("Problem 2")
    h = 0.1
    xi_list_explicit_euler, xi_list_midpoint_method = \
        explicit_euler_vs_midpoint_method_driver(h)
    xErrorListExplicitEuler, vErrorListExplicitEuler = \
        getHarmonicOscillatorError(xi_list_explicit_euler)
    xErrorListMidpointMethod, vErrorListMidpointMethod = \
        getHarmonicOscillatorError(xi_list_midpoint_method)
    plotHarmonicOscillatorError(xErrorListExplicitEuler,
                                vErrorListExplicitEuler,
                                xErrorListMidpointMethod,
                                vErrorListMidpointMethod,
                                xi_list_explicit_euler,
                                'explicit Euler method',
                                'midpoint method')

    print("Problem 3")
    plotTruncationError()

    print("Problem 5")
    xi_list = runge_kutta_driver()
    x_values = list(map(lambda x: x[0], xi_list))
    y_values = list(map(lambda x: x[1], xi_list))
    N = len(x_values)
    # Plot color changes from blue to red as orbit advances in time.
    color_list = list(map(lambda i: (i / N, 0, 1 - i / N), list(range(N))))
    plt.figure()
    plt.scatter(x_values, y_values, c=color_list, s=10)
    plt.title("Plot of orbit with initial parameters\n"
              + "[x0, y0, vx0, vy0] = [1, 0, 0, 1]")
    plt.show()

if __name__ == "__main__": main()
