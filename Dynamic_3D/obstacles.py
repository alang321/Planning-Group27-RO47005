import numpy as np
import casadi as ca

class CylinderVertical:
    def __init__(self, x, y, radius, velocity=None):
        self.x = x
        self.y = y
        self.radius = radius
        self.velocity = velocity

        self.world = None

    def init_world(self, world):
        self.world = world

    def is_colliding(self, point, margin):
        squared_dist = (point[0] - self.x) ** 2 + (point[1] - self.y) ** 2
        if squared_dist <= (self.radius + margin) ** 2:
            return True
        return False

    def update(self, dt):
        if self.velocity is not None:
            self.x += self.velocity[0] * dt
            self.y += self.velocity[1] * dt

    def plot_circle_xy(self, ax, x, y, z, radius, color):
        #this function plots only a circle in 3d
        theta = np.linspace(0, 2 * np.pi, 201)
        x_circ = x + radius * np.cos(theta)
        y_circ = y + radius * np.sin(theta)

        ax.plot(x_circ, y_circ, z, color=color)

        return ax

    def plot(self, ax, color):
        #plot a cylinder from lines and circles in 3d
        #plot the circles
        self.plot_circle_xy(ax, self.x, self.y, self.world.z_range[0], self.radius, color)
        self.plot_circle_xy(ax, self.x, self.y, self.world.z_range[1], self.radius, color)

        #plot the center line
        #ax.plot([x, x], [y, y], self.world_3d.z_range)
        ax.plot([self.x, self.x], [self.y + self.radius, self.y + self.radius], self.world.z_range, color=color)
        ax.plot([self.x, self.x], [self.y - self.radius, self.y - self.radius], self.world.z_range, color=color)
        ax.plot([self.x + self.radius, self.x + self.radius], [self.y, self.y], self.world.z_range, color=color)
        ax.plot([self.x - self.radius, self.x - self.radius], [self.y, self.y], self.world.z_range, color=color)

    def get_euclid(self, x_sym, k):
        return ca.norm_2(x_sym[0:2, k] - self.get_center_vector())

    def get_center_vector(self):
        return np.array([self.x, self.y]).reshape(2, 1)

    def get_constraint(self, x_sym, k):
        return [self.get_euclid(x_sym, k) > self.radius[-1]]

    def get_cost(self, x_sym, k, constant):
        return constant / ((self.get_euclid(x_sym, k) - self.radius) ** 2 + 0.01)

    def get_center_vector(self):
        return np.array([self.x, self.y]).reshape(2,1)


class CylinderHorizontal:
    def __init__(self, y, z, radius, velocity=None):
        self.y = y
        self.z = z
        self.radius = radius
        self.velocity = velocity

        self.world = None

    def init_world(self, world):
        self.world = world

    def is_colliding(self, point):
        squared_dist = (point[0] - self.x) ** 2 + (point[1] - self.y) ** 2
        if squared_dist <= self.radius ** 2:
            return True
        return False

    def update(self, dt):
        if self.velocity is not None:
            self.x += self.velocity[0] * dt
            self.y += self.velocity[1] * dt

    def plot_circle_yz(self, ax, x, y, z, radius, color):
        #this function plots only a circle in 3d
        theta = np.linspace(0, 2 * np.pi, 201)
        y_circ = y + radius * np.cos(theta)
        z_circ = z + radius * np.sin(theta)

        ax.plot(x, y_circ, z_circ, color=color)

        return ax

    def plot(self, ax, color):
        #plot a cylinder from lines and circles in 3d
        #plot the circles
        self.plot_circle_yz(ax, self.x, self.y, self.world.z_range[0], self.radius, color)
        self.plot_circle_yz(ax, self.x, self.y, self.world.z_range[1], self.radius, color)

        #plot the center line
        #ax.plot([x, x], [y, y], self.world_3d.z_range)
        ax.plot(self.world.x_range, [self.y, self.y], [self.z + self.radius, self.z + self.radius], color=color)
        ax.plot(self.world.x_range, [self.y, self.y], [self.z - self.radius, self.z - self.radius], self.world.z_range, color=color)
        ax.plot(self.world.x_range, [self.y + self.radius, self.y + self.radius], [self.z, self.z], self.world.z_range, color=color)
        ax.plot(self.world.x_range, [self.y - self.radius, self.y - self.radius], [self.z, self.z], self.world.z_range, color=color)
    
    def get_center_vector(self):
        return np.array([self.y, self.z]).reshape(2,1)