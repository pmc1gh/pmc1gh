# Philip Carr
# March 7, 2020
# QuadTree.py

from Vector import *
from matplotlib import pyplot as plt

class Particle():
    """Class for particle"""

    def __init__(self, position, velocity, mass):
        """Constructor for quad tree node"""
        self.position = position
        self.velocity = velocity
        self.mass = mass

    def __getitem__(self, i):
        if i == 0:
            return self.position
        elif i == 1:
            return self.mass

    def __setitem__(self, i, x):
        if i == 0:
            self.position = x
        elif i == 1:
            self.mass = x

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_mass(self):
        return self.mass

    def set_position(self, position):
        self.position = position

    def set_velocity(self):
        self.velocity = velocity

    def set_mass(self, mass):
        self.mass = mass



class QuadTreeNode():
    """Class for the quad-trees of an approximated N-body simulation"""

    def __init__(self, particles, particle=None, parent=None, bounds=None,
                 layer=0):
        # particles is a 2d list of positions and
        # corresponding masses of particles
        """Constructor for quad tree node"""
        self.layer = layer
        if bounds == None:
            # compute the side length of the particle cloud using the particles
            # self.side_length = 0
            min_x = float("inf")
            min_y = float("inf")
            max_x = -float("inf")
            max_y = -float("inf")
            for p in particles:
                min_x = min(min_x, p[0][0])
                min_y = min(min_y, p[0][1])
                max_x = max(max_x, p[0][0])
                max_y = max(max_y, p[0][1])
            side_length = max(max_x - min_x, max_y - min_y)
            self.center = Vector([(min_x + max_x) / 2, (min_y + max_y) / 2])
            # bounds = [0] * 4
            # if min_x < 0:
            #     bounds[0] = min_x * 1.1
            # else:
            #     bounds[0] = min_x * 0.9
            # if min_y < 0:
            #     bounds[1] = min_y * 1.1
            # else:
            #     bounds[1] = min_y * 0.9
            # if max_x < 0:
            #     bounds[2] = max_x * 0.9
            # else:
            #     bounds[2] = max_x * 1.1
            # if max_y < 0:
            #     bounds[3] = max_y * 0.9
            # else:
            #     bounds[3] = max_y * 1.1
            self.bounds = [self.center[0] - side_length / 2,
                           self.center[1] - side_length / 2,
                           self.center[0] + side_length / 2,
                           self.center[1] + side_length / 2]
        else:
            self.bounds = bounds
            self.center = \
                Vector([(self.bounds[0] + self.bounds[2]) / 2,
                        (self.bounds[1] + self.bounds[3]) / 2])
            # print("QuadTreeNode center:", self.center)
        # print("Bounds:", self.bounds)
        self.n_particles = len(particles)
        if self.n_particles > 1:
            # print("1")
            self.particle = None
            self.mass = sum(list(map(lambda x: x[1], particles)))
            # print("2")
            # compute the center of mass here and particle cloud distance
            # divide the particles into the four regions of the quad tree and
            # recursively call the quad tree constructor again.
            self.center_of_mass = \
                sum(list(map(lambda x: x[0] * x[1] / self.mass, particles)))
            # print("3")
            particles_bl = []
            particles_ul = []
            particles_br = []
            particles_ur = []
            for i in range(self.n_particles):
                if particles[i][0][0] < self.center[0] \
                   and particles[i][0][1] < self.center[1]:
                    particles_bl.append(particles[i])
                elif particles[i][0][0] < self.center[0] \
                   and particles[i][0][1] > self.center[1]:
                    particles_ul.append(particles[i])
                elif particles[i][0][0] > self.center[0] \
                   and particles[i][0][1] < self.center[1]:
                    particles_br.append(particles[i])
                else:
                    particles_ur.append(particles[i])
            self.bl = None
            self.ul = None
            self.br = None
            self.ur = None
            if len(particles_bl) > 0:
                self.bl = QuadTreeNode(particles_bl, parent=self,
                                       bounds=[self.bounds[0], self.bounds[1],
                                               self.center[0], self.center[1]],
                                       layer=layer+1)
            if len(particles_ul) > 0:
                self.ul = QuadTreeNode(particles_ul, parent=self,
                                       bounds=[self.bounds[0], self.center[1],
                                               self.center[0], self.bounds[3]],
                                       layer=layer+1)
            if len(particles_br) > 0:
                self.br = QuadTreeNode(particles_br, parent=self,
                                       bounds=[self.center[0], self.bounds[1],
                                               self.bounds[2], self.center[1]],
                                       layer=layer+1)
            if len(particles_ur) > 0:
                self.ur = QuadTreeNode(particles_ur, parent=self,
                                       bounds=[self.center[0], self.center[1],
                                               self.bounds[2], self.bounds[3]],
                                       layer=layer+1)
        elif self.n_particles == 1:
            self.particle = particles[0]
            self.mass = particles[0][1]
            self.center_of_mass = particles[0][0]
            self.bl = None
            self.ul = None
            self.br = None
            self.ur = None
        else:
            self.particle = None
            self.mass = 0
            self.center_of_mass = None
            self.bl = None
            self.ul = None
            self.br = None
            self.ur = None

    def __repr__(self):
        r_string = ""
        if self.layer == 0:
            r_string += "Root node (layer 0)\n"
        else:
            r_string += "layer " +  str(self.layer) + "\n"
        r_string += "particle: " \
                    + {True : "None",
                       False : str(self.particle)}[self.particle == None] + "\n"
        r_string += "mass: " + str(self.mass) + "\n"
        r_string += "center of mass: " + {True : "None",
              False : str(self.center_of_mass)}[self.center_of_mass == None] \
              + "\n"
        r_string += "bottom left QuadTreeNode: " \
                    + {True : "None", False : str(self.bl)}[self.bl == None] \
                    + "\n"
        r_string += "upper left QuadTreeNode: " \
                    + {True : "None", False : str(self.ul)}[self.ul == None] \
                    + "\n"
        r_string += "bottom right QuadTreeNode: " \
                    + {True : "None", False : str(self.br)}[self.br == None] \
                    + "\n"
        r_string += "upper right QuadTreeNode:" \
                    + {True : "None", False : str(self.ur)}[self.ur == None] \
                    + "\n"
        return r_string

    def compute_force(self, particle, theta, force_softening):
        """Return the force exerted on the particle by the quad-tree."""
        if self.n_particles == 0:
            return 0
        elif self.n_particles == 1:
            displacement = particle[0] - self.particle[0]
            if displacement.mag() < 0:
                return 0
            else:
                return -self.mass * particle[1] * displacement \
                       / (math.pow(displacement.mag(), 3)
                          + force_softening * force_softening)
        else:
            displacement = particle[0] - self.center_of_mass
            side_length = self.bounds[2] - self.bounds[0]
            if side_length / displacement.mag() < theta:
                return -self.mass * particle[1] * displacement \
                       / (math.pow(displacement.mag(), 3)
                          + force_softening * force_softening)
            else:
                total_force = Vector([0, 0])
                if self.bl != None:
                    total_force = \
                        total_force + self.bl.compute_force(particle, theta,
                                                            force_softening)
                if self.ul != None:
                    total_force = \
                        total_force + self.ul.compute_force(particle, theta,
                                                            force_softening)
                if self.br != None:
                    total_force = \
                        total_force + self.br.compute_force(particle, theta,
                                                            force_softening)
                if self.ur != None:
                    total_force = \
                        total_force + self.ur.compute_force(particle, theta,
                                                            force_softening)
                return total_force

    def plot_quad_tree_node(self):
        plt.plot([self.bounds[0], self.bounds[2]],
                 [self.bounds[1], self.bounds[1]])
        plt.plot([self.bounds[0], self.bounds[2]],
                 [self.bounds[3], self.bounds[3]])
        plt.plot([self.bounds[0], self.bounds[0]],
                 [self.bounds[1], self.bounds[3]])
        plt.plot([self.bounds[2], self.bounds[2]],
                 [self.bounds[1], self.bounds[3]])
        if self.bl != None:
            self.bl.plot_quad_tree_node()
        if self.ul != None:
            self.ul.plot_quad_tree_node()
        if self.br != None:
            self.br.plot_quad_tree_node()
        if self.ur != None:
            self.ur.plot_quad_tree_node()
