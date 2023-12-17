import numpy as np
import casadi as ca
from quadrotor import Quadrotor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def ref(time, amplitude=1):
    target_position_x = np.sin(time * np.pi * 1.5) * amplitude
    target_position_y = np.cos(time * np.pi * 1.5) * amplitude
    return target_position_x, target_position_y

def angle_difference(a, b):
    diff = a - b
    diff = ca.fmod(diff + np.pi, 2 * np.pi) - np.pi
    return diff

def mpc_control(quadrotor, N, x_init, x_target, num_states=4, num_inputs=2):
    # Create an optimization problem
    opti = ca.Opti()

    # State & Input matrix
    weight_pos = np.eye(3) * 120
    weight_att = np.eye(3) * 20
    weight_vel = np.eye(3) * 0
    weight_omega = np.eye(3) * 0
    R = np.eye(num_inputs) * 0.1

    # Define Variables
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N) 

    # Initialize Cost & Constraints
    cost = 0.0
    constraints = []
    time = 0

    #Loop through time steps
    for k in range(N):
        time += quadrotor.dt

        # Dynamics Constraint
        constraints += [x[:, k+1] == quadrotor.calculate_next_step(x[:, k], u[:, k])]

        # Motor Limits
        rpm_lim = 1000
        opti.subject_to(opti.bounded(0, u[0, k], rpm_lim))
        opti.subject_to(opti.bounded(0, u[1, k], rpm_lim))
        opti.subject_to(opti.bounded(0, u[2, k], rpm_lim))
        opti.subject_to(opti.bounded(0, u[3, k], rpm_lim))

        # Cost function
        x_target[0:2] = ref(time, 3)
        e_k = x_target - x[:, k]
        e_k[3:6] = angle_difference(x_target[3:6], x[3:6, k])
        Q = ca.diagcat(weight_pos, weight_att, weight_vel, weight_omega)
        cost += ca.mtimes(e_k.T, ca.mtimes(Q, e_k)) + ca.mtimes(u[:, k].T, ca.mtimes(R, u[:, k]))

    
    # Init Constraint
    constraints += [x[:, 0] == x_init] 

    # Define Problem in solver
    opti.minimize(cost)
    opti.subject_to(constraints)

    opti.solver('ipopt', {'ipopt.print_level': 5}) # Set the verbosity level (0-12, default is 5)

    # Run Solver
    try:
        sol = opti.solve()
        optimal_solution_u = sol.value(u)
        optimal_solution_x = sol.value(x)
        optimal_cost = sol.value(cost)
        print("Optimal cost:", optimal_cost)
    except RuntimeError:
        print("Solver failed to find a solution.")

    return optimal_solution_x, optimal_solution_u

x_init = np.zeros(12)
x_target = np.zeros(12)
x_target[0:3] = [0, 0, 0]
dt = 0.01

results = mpc_control(Quadrotor(dt), 300, x_init, x_target, 12, 4)
positions = results[0][0:6, :]
inputs = results[1]

# Extract x, y, z coordinates
x_positions = positions[0, :]
y_positions = positions[1, :]
z_positions = positions[2, :]
x_rot = positions[3, :]
y_rot = positions[4, :]
z_rot = positions[5, :]

# Control inputs
rotor_1 = inputs[0, :]
rotor_2 = inputs[1, :]
rotor_3 = inputs[2, :]
rotor_4 = inputs[3, :]

def animate_trajectory():
    # Function to convert Euler angles to a rotation matrix
    def rotation_matrix_from_euler(x, y, z):
        # Convert angles from degrees to radians

        # Compute rotation matrix
        Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

        R = np.dot(Rz, np.dot(Ry, Rx))

        return R

    # Function to draw the quadrotor
    def draw_quadrotor(ax, position, rotation):
        # Define the quadrotor's size
        size = 0.5
        size_x = size_y = size  # for x and z dimensions
        size_z = size / 2  # for y dimension, make it thinner

        # Define the vertices of the quadrotor
        vertices = np.array([[-size_x, -size_y, -size_z],
                            [size_x, -size_y, -size_z],
                            [size_x, size_y, -size_z],
                            [-size_x, size_y, -size_z],
                            [-size_x, -size_y, size_z],
                            [size_x, -size_y, size_z],
                            [size_x, size_y, size_z],
                            [-size_x, size_y, size_z]])

        # Rotate and translate the vertices based on the quadrotor's position and orientation
        vertices = np.dot(vertices, rotation.T) + position

        # Define the faces of the quadrotor
        faces = [[vertices[j] for j in face] for face in [[0, 1, 5, 4], [7, 6, 2, 3], [0, 1, 2, 3], [7, 6, 5, 4], [0, 3, 7, 4], [1, 2, 6, 5]]]

        # Define the colors for each face
        facecolors = ['cyan', 'cyan', 'red', 'cyan', 'cyan', 'cyan']

        # Draw the quadrotor
        ax.add_collection3d(Poly3DCollection(faces, facecolors=facecolors, linewidths=1, edgecolors='r', alpha=1))


    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        ax.clear()
        draw_quadrotor(ax, [x_positions[i], y_positions[i], z_positions[i]], rotation_matrix_from_euler(x_rot[i], y_rot[i], z_rot[i]))
        ax.set_xlim([min(x_positions), max(x_positions)])
        ax.set_ylim([min(y_positions), max(y_positions)])
        ax.set_zlim([min(z_positions), max(z_positions)])
        set_axes_equal(ax)

    print(len(x_positions))
    ani = animation.FuncAnimation(fig, animate, frames=len(x_positions), interval=dt * 1000)
    # ani.save('animation.mp4', writer='ffmpeg', fps=60)

    plt.show()

def plot_control_inputs():
    # Plot control inputs
    plt.figure()

    # Plot control input for rotor 1
    plt.subplot(4, 1, 1)
    plt.plot(rotor_1)
    plt.title('Control Input for Rotor 1')

    # Plot control input for rotor 2
    plt.subplot(4, 1, 2)
    plt.plot(rotor_2)
    plt.title('Control Input for Rotor 2')

    # Plot control input for rotor 3
    plt.subplot(4, 1, 3)
    plt.plot(rotor_3)
    plt.title('Control Input for Rotor 3')

    # Plot control input for rotor 4
    plt.subplot(4, 1, 4)
    plt.plot(rotor_4)
    plt.title('Control Input for Rotor 4')

    plt.tight_layout()
    plt.show()

plot_control_inputs()
animate_trajectory()
