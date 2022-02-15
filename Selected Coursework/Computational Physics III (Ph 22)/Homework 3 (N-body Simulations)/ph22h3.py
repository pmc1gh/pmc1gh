# Philip Carr
# February 21, 2020
# ph22h3.py

import random
from Vector import *
import numpy as np
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

def leapfrog_step(positions_old, velocities_old, h, accelerations):
    """
    Leapfrog symplectic integration method.
    """
    accelerations_old = accelerations(positions_old)
    positions_new = \
        positions_old + h * velocities_old + h * h * accelerations_old / 2
    velocities_new = velocities_old \
        + h * (accelerations(positions_new) + accelerations_old) / 2
    return positions_new, velocities_new

def ode_stepper(positions0, velocities0, accelerations, T, ode_method, h=0.001,
                print_progress_step=100):
    """
    Stepper function for running ODE solvers.
    """
    positions_list = [positions0]
    velocities_list = [velocities0]
    progress = 0
    progress_step_upper_bound = print_progress_step
    print("Progress:")
    print(str(progress) + "%", end="\r")
    t = 0
    while t < T:
        positions_new, velocities_new = \
            ode_method(positions_list[-1], velocities_list[-1], h,
                       accelerations)
        positions_list.append(positions_new)
        velocities_list.append(velocities_new)
        progress = int(t / T * 100.0)
        if progress >= progress_step_upper_bound:
            print(str(progress) + "%", end="\r")
            progress_step_upper_bound += print_progress_step
        t += h
    print("done")
    return positions_list, velocities_list

def n_body_simulation_driver(n_particles, v0=0.5, force_softening=0.1,
                             time_step=0.01, n_time_steps=100):
    """
    Driver function for the Runge-Kutta 4th order-integrated orbit system of the
    n-body simulation.
    """

    m = [1] * n_particles

    positions0_list = []
    velocities0_list = []

    for i in range(n_particles):
        x0 = random.uniform(-1, 1)
        y0_max = math.sqrt(1 - x0 * x0)
        y0 = random.uniform(-y0_max, y0_max)
        vx0 = random.uniform(0, v0)
        vy0_max = math.sqrt(1 - vx0 * vx0)
        if (v0 > 0):
            vy0 = random.choice([-vy0_max, vy0_max])
        else:
            vy0 = 0
        positions0_list.append(x0)
        positions0_list.append(y0)
        velocities0_list.append(vx0)
        velocities0_list.append(vy0)

    positions0 = Vector(positions0_list)
    velocities0 = Vector(velocities0_list)

    # Computes all gravitational accelerations of all n particles from all other
    # n-1 particles in the system.
    def gravitational_accelerations(positions):
        # positions = [x1, y1, x2, y2, ..., xn, yn]
        accelerations = Vector([0] * n_particles * 2)
        for i in range(n_particles):
            acceleration_i = Vector([0] * 2)
            position_i = Vector([positions[2*i], positions[2*i+1]])
            for j in range(n_particles):
                if i != j:
                    position_j = Vector([positions[2*j], positions[2*j+1]])
                    r = position_i - position_j
                    acceleration_i = acceleration_i \
                        - m[j] * r / (math.pow(r.mag(), 3)
                                      + force_softening * force_softening)
            accelerations[2*i] = acceleration_i[0]
            accelerations[2*i+1] = acceleration_i[1]
        return accelerations

    T = time_step * n_time_steps
    return ode_stepper(positions0, velocities0, gravitational_accelerations, T,
                       leapfrog_step, h=time_step, print_progress_step=1)

def plot_orbit_inertial_frame(positions_list, velocities_list, animate=False,
                              show_trail=True, n_frames=100, plot_number=2):
    """Plot the particles' positions and velocities in the N-body simulation."""
    x_values_list = []
    y_values_list = []
    vx_values_list = []
    vy_values_list = []
    n_particles = int(len(positions_list[0]) / 2)
    # print(n_particles)
    # print(positions_list)
    for i in range(n_particles):
        x_values_list.append(list(map(lambda x: x[2*i], positions_list)))
        y_values_list.append(list(map(lambda x: x[2*i+1], positions_list)))
        vx_values_list.append(list(map(lambda x: x[2*i], velocities_list)))
        vy_values_list.append(list(map(lambda x: x[2*i+1], velocities_list)))
    data = np.array([x_values_list, y_values_list])
    N = len(x_values_list[0])
    # Plot color changes from blue to red as orbit advances in time.
    color_list = (list(map(lambda j: (j / N, 0, 1 - j / N),
                           list(range(N)))))
    color_list_list = []
    for i in range(N):
        color_list_list.append(list(map(lambda j:
            (j / N, 0, 1 - j / N,
             math.pow((j / N), 3) + (N - i) / N),
            list(range(N)))))
    fig = plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.title("Plot of orbit with initial parameters:\n"
    #           + str(positions_list[0]) + "\n" + str(velocities_list[0]) + "\n")
    plt.title("Plot of N-body simulation with " + str(n_particles)
              + " particle positions")
    if animate:
        position_plots = []
        for i in range(n_particles):
            position_plots.append(plt.scatter([], []))
        def animation_function1(i, data, particle_plots):
            for j in range(n_particles):
                # plt.scatter(x_values_list[j][:i], y_values_list[j][:i],
                            # c=color_list[:i], s=10)
                if show_trail:
                    trail_lower_bound = max(0,i-int(N/20)) # i-20
                    particle_plots[j] = \
                        plt.scatter(x_values_list[j][trail_lower_bound:i],
                                    y_values_list[j][trail_lower_bound:i],
                                    c=color_list_list[i][trail_lower_bound:i],
                                    s=10)
                else:
                    particle_plots[j] = \
                        plt.scatter(x_values_list[j][i], y_values_list[j][i],
                                    c=color_list[i], s=10)
            return particle_plots
        orbit_animation1 = \
            animation.FuncAnimation(fig, animation_function1,
                                          range(0, N, int(N / n_frames)),
                                          fargs=(data, position_plots),
                                          interval=1, blit=True, repeat=True,
                                          repeat_delay=3000)
        # orbit_animation.save("ph22_hw2_p" + str(plot_number) + "_animation.mp4")
    else:
        for i in range(n_particles):
            if show_trail:
                plt.scatter(x_values_list[i], y_values_list[i],
                            c=color_list_list[-1], s=10)
            else:
                plt.scatter(x_values_list[i][-1], y_values_list[i][-1],
                            c=color_list[-1], s=10)

    plt.show()

    fig = plt.figure()
    plt.xlabel("vx")
    plt.ylabel("vy")
    # plt.title("Plot of orbit with initial parameters:\n"
    #           + str(positions_list[0]) + "\n" + str(velocities_list[0]) + "\n")
    plt.title("Plot of N-body simulation with " + str(n_particles)
              + " particle velocities")
    if animate:
        velocity_plots = []
        for i in range(n_particles):
            velocity_plots.append(plt.scatter([], []))
        def animation_function2(i, data, particle_plots):
            for j in range(n_particles):
                if show_trail:
                    trail_lower_bound = max(0,i-int(N/20)) # i-20
                    particle_plots[j] = \
                        plt.scatter(vx_values_list[j][trail_lower_bound:i],
                                    vy_values_list[j][trail_lower_bound:i],
                                    c=color_list_list[i][trail_lower_bound:i],
                                    s=10)
                else:
                    particle_plots[j] = \
                        plt.scatter(vx_values_list[j][i], vy_values_list[j][i],
                                    c=color_list[i], s=10)
            return particle_plots
        orbit_animation2 = \
            animation.FuncAnimation(fig, animation_function2,
                                          range(0, N, int(N / n_frames)),
                                          fargs=(data, velocity_plots),
                                          interval=1, blit=True, repeat=True,
                                          repeat_delay=3000)
        # orbit_animation.save("ph22_hw2_p" + str(plot_number) + "_animation.mp4")
    else:
        for i in range(n_particles):
            if show_trail:
                plt.scatter(vx_values_list[i], vy_values_list[i],
                            c=color_list_list[-1], s=10)
            else:
                plt.scatter(vx_values_list[i][-1], vy_values_list[i][-1],
                            c=color_list[-1], s=10)

    plt.show()

def main():
    print("Ph 22 Homework 3 Program Printout")

    print("Problem 2")
    print("Case of nonzero velocities")
    positions_list, velocities_list = \
        n_body_simulation_driver(20, v0=0.1, n_time_steps=500)
    plot_orbit_inertial_frame(positions_list, velocities_list, animate=True,
                              n_frames=250, show_trail=False)
    plot_orbit_inertial_frame(positions_list, velocities_list, animate=False,
                              show_trail=False)

    print("Case of zero-velocity")
    positions_list, velocities_list = \
        n_body_simulation_driver(20, v0=0, n_time_steps=2000)
    plot_orbit_inertial_frame(positions_list, velocities_list, animate=True,
                              n_frames=100, show_trail=True)
    plot_orbit_inertial_frame(positions_list, velocities_list, animate=False,
                              show_trail=True)

if __name__ == "__main__": main()
