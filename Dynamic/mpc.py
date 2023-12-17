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

    # Define Variables
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N) 

    # Initialize Cost & Constraints
    cost = 0.0
    constraints = []

    # Motor Limit
    rpm_lim = 1000
    
    # Weights
    weights = {
        "positions": 120,
        "attitudes": 20,
        "velocities": 0,
        "ang_velocities": 0,
        "inputs": 0.1
    }

    Q = ca.diagcat(
        np.eye(3) * weights["positions"],
        np.eye(3) * weights["attitudes"],
        np.eye(3) * weights["velocities"],
        np.eye(3) * weights["ang_velocities"]
    )

    R = np.eye(num_inputs) * weights["inputs"]

    
    time = 0
    #Loop through time steps
    for k in range(N):
        time += quadrotor.dt

        # Dynamics Constraint
        constraints += [x[:, k+1] == quadrotor.calculate_next_step(x[:, k], u[:, k])]

        # Motor Limits
        for i in range(4):
            opti.subject_to(opti.bounded(0, u[i, k], rpm_lim))

        # Cost function
        x_target[0:2] = ref(time, 3)
        e_k = x_target - x[:, k]
        e_k[3:6] = angle_difference(x_target[3:6], x[3:6, k])
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

# Run simulation
dt = 0.01
x_init = np.zeros(12)
x_target = np.zeros(12)
x_target[0:3] = [0, 0, 0]
results = mpc_control(Quadrotor(dt), 200, x_init, x_target, 12, 4)

# Extract positions and control inputs
positions = results[0][0:6, :]
inputs = results[1]

# Extract x, y, z coordinates and rotations
x_positions, y_positions, z_positions, x_rot, y_rot, z_rot = positions

# Control inputs
rotor_1, rotor_2, rotor_3, rotor_4 = inputs

def animate_trajectory():
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

def plot_control_inputs():
    # Plot control inputs
    plt.figure()

    # Plot control input for each rotor
    for i, rotor in enumerate([rotor_1, rotor_2, rotor_3, rotor_4], start=1):
        plt.subplot(4, 1, i)
        plt.plot(rotor)
        plt.title(f'Control Input for Rotor {i}')

    plt.tight_layout()
    plt.show()

plot_control_inputs()
animate_trajectory()
