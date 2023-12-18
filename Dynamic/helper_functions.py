import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def path_ref(time, amplitude=1):
    target_position_x = np.sin(time * np.pi * 1.5) * amplitude
    target_position_y = np.cos(time * np.pi * 1.5) * amplitude
    return target_position_x, target_position_y

def angle_difference(a, b):
    diff = a - b
    diff = ca.fmod(diff + np.pi, 2 * np.pi) - np.pi
    return diff

def animate_trajectory(positions, dt):
    # Extract x, y, z coordinates and rotations
    x_positions, y_positions, z_positions, x_rot, y_rot, z_rot = positions

    def rotation_matrix_from_euler(x, y, z):
        Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return np.dot(Rz, np.dot(Ry, Rx))

    def set_axes_equal(ax):
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    size = 0.5
    vertices = size * np.array([[-1, -1, -0.25], [1, -1, -0.25], [1, 1, -0.25], [-1, 1, -0.25], [-1, -1, 0.25], [1, -1, 0.25], [1, 1, 0.25], [-1, 1, 0.25]])
    faces = [[0, 1, 5, 4], [7, 6, 2, 3], [0, 1, 2, 3], [7, 6, 5, 4], [0, 3, 7, 4], [1, 2, 6, 5]]
    facecolors = ['cyan', 'cyan', 'red', 'cyan', 'cyan', 'cyan']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        ax.clear()
        position = np.array([x_positions[i], y_positions[i], z_positions[i]])
        rotation = rotation_matrix_from_euler(x_rot[i], y_rot[i], z_rot[i])
        rotated_vertices = np.dot(vertices, rotation.T) + position
        rotated_faces = [[rotated_vertices[j] for j in face] for face in faces]
        ax.add_collection3d(Poly3DCollection(rotated_faces, facecolors=facecolors, linewidths=1, edgecolors='r', alpha=1))
        ax.set_xlim([min(x_positions), max(x_positions)])
        ax.set_ylim([min(y_positions), max(y_positions)])
        ax.set_zlim([min(z_positions), max(z_positions)])
        set_axes_equal(ax)

    ani = animation.FuncAnimation(fig, animate, frames=len(x_positions), interval=dt * 1000)
    plt.show()

def plot_control_inputs(inputs):
    # Control inputs
    rotor_1, rotor_2, rotor_3, rotor_4 = inputs 
    
    # Plot control inputs
    plt.figure()

    # Plot control input for each rotor
    for i, rotor in enumerate([rotor_1, rotor_2, rotor_3, rotor_4], start=1):
        plt.subplot(4, 1, i)
        plt.plot(rotor)
        plt.title(f'Control Input for Rotor {i}')

    plt.tight_layout()
    plt.show()