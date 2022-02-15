# Philip Carr
# February 13, 2020
# ph22h2.py

from Vector import *
import numpy as np
# import matplotlib
# # matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import animation as animation

G = np.longdouble(6.6742e-11)
M_sun = np.longdouble(1.989e30)
M_jupiter = np.longdouble(1.899e27)
R = np.longdouble(778.3e9)
T_jupiter = np.longdouble(3.743e8)

r_sun = Vector([M_jupiter * R / (M_sun + M_jupiter), 0, 0])
r_jupiter = Vector([-M_sun * R / (M_sun + M_jupiter), 0, 0])

Omega = math.sqrt(G * (M_sun + M_jupiter) / math.pow(R, 3))
Omega_vector = Omega * Vector([0, 0, 1])

def runge_kutta_4th_order_step(xi_old, h, f):
    """
    Return updated variables vector xi using the 4th order Runge-Kutta routine.
    """

    k1 = h * Vector(list(map(lambda fi: fi(xi_old), f)))
    k2 = h * Vector(list(map(lambda fi: fi(xi_old + k1/2), f)))
    k3 = h * Vector(list(map(lambda fi: fi(xi_old + k2/2), f)))
    k4 = h * Vector(list(map(lambda fi: fi(xi_old + k3), f)))

    return xi_old + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def ode_stepper(initial, f, T, ode_method, h=0.001, print_progress_step=100):
    """
    Stepper function for running ODE solvers.
    """
    xi_list = [initial]
    progress = 0
    progress_step_upper_bound = print_progress_step
    print("Progress:\n" + str(progress) + "%")
    t = 0
    while t < T:
        xi_list.append(ode_method(xi_list[-1], h, f))
        progress = int(t / T * 100.0)
        if progress >= progress_step_upper_bound:
            print(str(progress) + "%")
            progress_step_upper_bound += print_progress_step
        t += h
    print("done")
    return xi_list

def restricted_three_body_problem_driver(alpha=math.pi / 3):
    """
    Driver function for the Runge-Kutta 4th order-integrated orbit system of the
    restricted three-body problem.
    """

    x_prime = lambda xi: xi[2]
    y_prime = lambda xi: xi[3]
    acceleration = lambda xi: \
        -G * M_sun / math.pow(
            (Vector([xi[0], xi[1], 0]) - r_sun).mag(), 3) \
        * (Vector([xi[0], xi[1], 0]) - r_sun) \
        - G * M_jupiter / math.pow(
            (Vector([xi[0], xi[1], 0]) - r_jupiter).mag(), 3) \
        * (Vector([xi[0], xi[1], 0]) - r_jupiter) \
        + 2 * Omega * Vector([xi[3], -xi[2], 0]) \
        + Omega * Omega * Vector([xi[0], xi[1], 0])
    vx_prime = lambda xi: acceleration(xi)[0]
    vy_prime = lambda xi: acceleration(xi)[1]

    f = [x_prime, y_prime, vx_prime, vy_prime]

    x0 = R * (M_jupiter - M_sun) / (M_sun + M_jupiter) * math.cos(alpha)
    y0 = R * math.sin(alpha)
    vx0 = 0
    vy0 = 0
    initial = Vector([x0, y0, vx0, vy0])

    res = 1000
    T = int(2 * math.pi / Omega)
    n_periods = 10
    print(T)
    return ode_stepper(initial, f, n_periods * T, runge_kutta_4th_order_step,
                       h=T/res, print_progress_step=10)

def inertial_frame_three_body_problem_driver1(alpha=math.pi / 3):
    """
    Driver function for the Runge-Kutta 4th order-integrated orbit system of the
    restricted three-body problem.
    """

    # m = [M_sun, M_jupiter, M_asteroid]
    m = [1e30, 1e30, 1e30]

    x1_prime = lambda xi: xi[6]
    y1_prime = lambda xi: xi[7]
    x2_prime = lambda xi: xi[8]
    y2_prime = lambda xi: xi[9]
    x3_prime = lambda xi: xi[10]
    y3_prime = lambda xi: xi[11]

    acceleration1 = lambda xi: \
        -G * m[1] / math.pow(
            (Vector([xi[0], xi[1], 0]) - Vector([xi[2], xi[3], 0])).mag(), 3) \
        * (Vector([xi[0], xi[1], 0]) - Vector([xi[2], xi[3], 0])) \
        - G * m[2] / math.pow(
            (Vector([xi[0], xi[1], 0]) - Vector([xi[4], xi[5], 0])).mag(), 3) \
        * (Vector([xi[0], xi[1], 0]) - Vector([xi[4], xi[5], 0]))
    vx1_prime = lambda xi: acceleration1(xi)[0]
    vy1_prime = lambda xi: acceleration1(xi)[1]

    acceleration2 = lambda xi: \
        -G * m[0] / math.pow(
            (Vector([xi[2], xi[3], 0]) - Vector([xi[0], xi[1], 0])).mag(), 3) \
        * (Vector([xi[2], xi[3], 0]) - Vector([xi[0], xi[1], 0])) \
        - G * m[2] / math.pow(
            (Vector([xi[2], xi[3], 0]) - Vector([xi[4], xi[5], 0])).mag(), 3) \
        * (Vector([xi[2], xi[3], 0]) - Vector([xi[4], xi[5], 0]))
    vx2_prime = lambda xi: acceleration2(xi)[0]
    vy2_prime = lambda xi: acceleration2(xi)[1]

    acceleration3 = lambda xi: \
        -G * m[0] / math.pow(
            (Vector([xi[4], xi[5], 0]) - Vector([xi[0], xi[1], 0])).mag(), 3) \
        * (Vector([xi[4], xi[5], 0]) - Vector([xi[0], xi[1], 0])) \
        - G * m[2] / math.pow(
            (Vector([xi[4], xi[5], 0]) - Vector([xi[2], xi[3], 0])).mag(), 3) \
        * (Vector([xi[4], xi[5], 0]) - Vector([xi[2], xi[3], 0]))
    vx3_prime = lambda xi: acceleration3(xi)[0]
    vy3_prime = lambda xi: acceleration3(xi)[1]

    f = [x1_prime, y1_prime, x2_prime, y2_prime, x3_prime, y3_prime,
         vx1_prime, vy1_prime, vx2_prime, vy2_prime, vx3_prime, vy3_prime]

    x1_0 = -R/2
    y1_0 = 0
    x2_0 = R/2
    y2_0 = 0
    x3_0 = 0
    y3_0 = R * math.sin(math.pi / 3)
    vx1_0 = math.sqrt(G * m[0] / R) * math.cos(-math.pi / 3)
    vy1_0 = math.sqrt(G * m[0] / R) * math.sin(-math.pi / 3)
    vx2_0 = math.sqrt(G * m[1] / R) * math.cos(math.pi / 3)
    vy2_0 = math.sqrt(G * m[1] / R) * math.sin(math.pi / 3)
    vx3_0 = math.sqrt(G * m[2] / R) * math.cos(math.pi)
    vy3_0 = math.sqrt(G * m[2] / R) * math.sin(math.pi)
    initial = Vector([x1_0, y1_0, x2_0, y2_0, x3_0, y3_0, vx1_0, vy1_0,
                      vx2_0, vy2_0, vx3_0, vy3_0])

    res = 1000
    # T = int(2 * math.pi / Omega)
    T = 100000000
    n_periods = 1
    print(T)
    return ode_stepper(initial, f, n_periods * T, runge_kutta_4th_order_step,
                       h=T/res, print_progress_step=10)

def inertial_frame_three_body_problem_driver2(alpha=math.pi / 3):
    """
    Driver function for the Runge-Kutta 4th order-integrated orbit system of the
    restricted three-body problem.
    """

    x1_prime = lambda xi: xi[6]
    y1_prime = lambda xi: xi[7]
    x2_prime = lambda xi: xi[8]
    y2_prime = lambda xi: xi[9]
    x3_prime = lambda xi: xi[10]
    y3_prime = lambda xi: xi[11]

    acceleration1 = lambda xi: \
        -1 / math.pow(
            (Vector([xi[0], xi[1], 0]) - Vector([xi[2], xi[3], 0])).mag(), 3) \
        * (Vector([xi[0], xi[1], 0]) - Vector([xi[2], xi[3], 0])) \
        -1 / math.pow(
            (Vector([xi[0], xi[1], 0]) - Vector([xi[4], xi[5], 0])).mag(), 3) \
        * (Vector([xi[0], xi[1], 0]) - Vector([xi[4], xi[5], 0]))
    vx1_prime = lambda xi: acceleration1(xi)[0]
    vy1_prime = lambda xi: acceleration1(xi)[1]

    acceleration2 = lambda xi: \
        -1 / math.pow(
            (Vector([xi[2], xi[3], 0]) - Vector([xi[0], xi[1], 0])).mag(), 3) \
        * (Vector([xi[2], xi[3], 0]) - Vector([xi[0], xi[1], 0])) \
        -1 / math.pow(
            (Vector([xi[2], xi[3], 0]) - Vector([xi[4], xi[5], 0])).mag(), 3) \
        * (Vector([xi[2], xi[3], 0]) - Vector([xi[4], xi[5], 0]))
    vx2_prime = lambda xi: acceleration2(xi)[0]
    vy2_prime = lambda xi: acceleration2(xi)[1]

    acceleration3 = lambda xi: \
        -1 / math.pow(
            (Vector([xi[4], xi[5], 0]) - Vector([xi[0], xi[1], 0])).mag(), 3) \
        * (Vector([xi[4], xi[5], 0]) - Vector([xi[0], xi[1], 0])) \
        -1 / math.pow(
            (Vector([xi[4], xi[5], 0]) - Vector([xi[2], xi[3], 0])).mag(), 3) \
        * (Vector([xi[4], xi[5], 0]) - Vector([xi[2], xi[3], 0]))
    vx3_prime = lambda xi: acceleration3(xi)[0]
    vy3_prime = lambda xi: acceleration3(xi)[1]

    f = [x1_prime, y1_prime, x2_prime, y2_prime, x3_prime, y3_prime,
         vx1_prime, vy1_prime, vx2_prime, vy2_prime, vx3_prime, vy3_prime]

    x1_0 = 0.97000436
    y1_0 = -0.24308753
    x2_0 = -x1_0
    y2_0 = -y1_0
    x3_0 = 0
    y3_0 = 0
    vx3_0 = -0.93240737
    vy3_0 = -0.86473146
    vx1_0 = -vx3_0 / 2
    vy1_0 = -vy3_0 / 2
    vx2_0 = -vx3_0 / 2
    vy2_0 = -vy3_0 / 2
    initial = Vector([x1_0, y1_0, x2_0, y2_0, x3_0, y3_0, vx1_0, vy1_0, vx2_0,
                      vy2_0, vx3_0, vy3_0])

    res = 10000
    # T = int(2 * math.pi / Omega)
    T = 1
    n_periods = 2
    print(T)
    return ode_stepper(initial, f, n_periods * T, runge_kutta_4th_order_step,
                       h=T/res, print_progress_step=10)

def plot_orbit_restricted(xi_list, alpha, animate=False):
    x_values = list(map(lambda x: x[0], xi_list))
    y_values = list(map(lambda x: x[1], xi_list))
    data = np.array([x_values, y_values])
    N = len(x_values)
    # Plot color changes from blue to red as orbit advances in time.
    color_list = list(map(lambda i: (i / N, 0, 1 - i / N), list(range(N))))
    fig = plt.figure()
    plt.xlim(-2 * R, 2 * R)
    plt.ylim(-2 * R, 2 * R)
    plt.title("Plot of orbit with initial parameters:\n" + str(xi_list[0])
              + "\n" + "alpha = " + str(alpha))
    plt.scatter([r_sun[0], r_jupiter[0]], [r_sun[1], r_jupiter[1]],
                c=["yellow", "orange"])
    if animate:
        asteroid = plt.scatter([], [])
        def animation_function(i, data, asteroid):
            asteroid = plt.scatter(x_values[:i], y_values[:i], c=color_list[:i],
                                   s=10)
            return asteroid,
        # Writer = animation.writers["ffmpeg"]
        # writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)
        n_frames = 100
        asteroid_animation = \
            animation.FuncAnimation(fig, animation_function,
                                    range(0, N, int(N / n_frames)),
                                    fargs=(data, asteroid), interval=1,
                                    blit=True, repeat=True, repeat_delay=3000)
        # asteroid_animation.save("ph22_hw2_p1_animation.mp4", writer=writer)
    else:
        plt.scatter(x_values, y_values, c=color_list, s=10)

    plt.show()

def plot_orbit_inertial_frame(xi_list, n_objects, animate=False, plot_number=2):
    x_values_list = []
    y_values_list = []
    for i in range(n_objects):
        x_values_list.append(list(map(lambda x: x[2*i], xi_list)))
        y_values_list.append(list(map(lambda x: x[2*i+1], xi_list)))
    data = np.array([x_values_list, y_values_list])
    N = len(x_values_list[0])
    # Plot color changes from blue to red as orbit advances in time.
    color_list = list(map(lambda i: (i / N, 0, 1 - i / N), list(range(N))))
    fig = plt.figure()
    plt.title("Plot of orbit with initial parameters:\n" + str(xi_list[0])
              + "\n")
    if animate:
        object_plots = []
        for i in range(n_objects):
            object_plots.append(plt.scatter([], []))
        def animation_function(i, data, object_plots):
            for j in range(n_objects):
                object_plots[j] = \
                    plt.scatter(x_values_list[j][:i], y_values_list[j][:i],
                                c=color_list[:i], s=10)
            return object_plots
        n_frames = 100
        orbit_animation = \
            animation.FuncAnimation(fig, animation_function,
                                          range(0, N, int(N / n_frames)),
                                          fargs=(data, object_plots),
                                          interval=50, blit=True, repeat=True,
                                          repeat_delay=3000)
        # orbit_animation.save("ph22_hw2_p" + str(plot_number) + "_animation.mp4")
    else:
        for i in range(n_objects):
            plt.scatter(x_values_list[i], y_values_list[i], c=color_list, s=10)

    plt.show()

def main():
    print("Ph 22 Homework 2 Program Printout")

    print("Problem 1")
    alpha1a = math.pi/3
    xi_list1a = restricted_three_body_problem_driver(alpha=alpha1a)
    plot_orbit_restricted(xi_list1a, alpha1a, animate=True)

    alpha1b = -math.pi/3
    xi_list1b = restricted_three_body_problem_driver(alpha=alpha1b)
    plot_orbit_restricted(xi_list1b, alpha1b, animate=True)

    alpha1c = 2
    xi_list1c = restricted_three_body_problem_driver(alpha=alpha1c)
    plot_orbit_restricted(xi_list1c, alpha1c, animate=True)

    print("Program 2")
    xi_list2 = inertial_frame_three_body_problem_driver1()
    plot_orbit_inertial_frame(xi_list2, int(len(xi_list2[0]) / 4),
                              animate=True)

    print("Program 3")
    xi_list3 = inertial_frame_three_body_problem_driver2()
    plot_orbit_inertial_frame(xi_list3, int(len(xi_list3[0]) / 4),
                              animate=True)

if __name__ == "__main__": main()
