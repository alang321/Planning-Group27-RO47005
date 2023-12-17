import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import casadi as ca
import operator
from ipywidgets import IntProgress
from IPython.display import display

class vehicle_SS:
    def __init__(self, dt):
        # State space matrices
        self.A_c = np.matrix([[0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

        self.B_c = np.matrix([[0,  0], 
                    [0,  0],
                    [dt, 0],
                    [0, dt]])

        self.C_c = np.matrix([[1, 0, 0, 0],
                    [0, 1, 0, 0]])

        self.D_c = np.matrix([[0, 0],
                    [0, 0]])

        self.A = np.eye(4) + self.A_c * dt
        self.B = self.B_c
        self.C = self.C_c
        self.D = self.D_c

    def CalculateNextStep(self, x, u):
        x_next = self.A.dot(x.T) + self.B.dot(u.T)
        return x_next
    

def simulate(dt, T, x_init, plan_length, control_func, num_states = 4, num_inputs = 2):
    # Initialise the output arrays
    x_real = np.zeros((num_states, T+1))
    x_all = np.zeros((num_states, plan_length+1, T+1))
    u_real = np.zeros((num_inputs, T))

    # Set time vector and initial state
    timesteps = np.linspace(0, dt, T)
    x_real[:, 0] = x_init

    for t in range(0, T):

        # Compute the control input (and apply it)
        u_out, x_out, x_all_out = control_func(x_real[:, t]) 

        # Next x is the x in the second state
        x_real[:, t+1] = x_out
        x_all[:, :, t] = x_all_out # Save the plan (for visualization)

        # Used input is the first input
        u_real[:, t] = u_out

    return x_real, u_real, x_all, timesteps

def plot_trajectories(x_real, obstacles):
    pos_x = x_real[0, :]
    pos_y = x_real[1, :]

    for idx, obstacle in enumerate(obstacles):
        circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False, label= f'Obstacle {idx}')
        plt.gca().add_patch(circle)
    
    plt.scatter(pos_x, pos_y, color = "g", label = "Vehicle Path")
    plt.title('Trajectory with Obstacle')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Set axis limits for x and y
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.axis('equal')
    plt.grid()
    plt.show()


def mpc_control(vehicle, N, x_init, x_target, pos_constraints, vel_constraints, acc_constraints, obstacles = [],  num_states=4, num_inputs=2):
    # Create an optimization problem
    opti = ca.Opti()

    # State & Input matrix
    Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    R = np.eye(2)

    # Define Variables
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N) 

    # Initialize Cost & Constraints
    cost = 0.0
    constraints = []

    #Loop through time steps
    for k in range(N):
        # State Constraints
        constraints += [x[:, k] >= [pos_constraints[0], pos_constraints[2], vel_constraints[0], vel_constraints[2]]]
        constraints += [x[:, k] <= [pos_constraints[1], pos_constraints[3], vel_constraints[1], vel_constraints[3]]]
        constraints += [u[:, k] >= [acc_constraints[0], acc_constraints[2]]]
        constraints += [u[:, k] <= [acc_constraints[1], acc_constraints[3]]]
        
        for obstacle in obstacles:
            constraints += [ca.norm_2( x[0:2,k] - np.array(obstacle[0:2]).reshape(2,1) ) >= obstacle[2]]

        # Dynamics Constraint
        constraints += [x[:, k+1] == vehicle.A @ x[:, k] + vehicle.B @ u[:, k]]

        # Cost function
        e_k = x_target - x[:, k]
        cost += ca.mtimes(e_k.T, Q @ e_k) + ca.mtimes(u[:, k].T, R @ u[:, k])
    
    # Init Constraint
    constraints += [x[:, 0] == x_init] 

    # Cost last state
    e_N = x_target - x[:, -1]
    cost += ca.mtimes(e_N.T, Q @ e_N)
    
    
    # Define Problem in solver
    opti.minimize(cost)
    opti.subject_to(constraints)

    opts = {'ipopt.print_level': 0}  # Set the verbosity level (0-12, default is 5)
    opti.solver('ipopt', {'ipopt.print_level': 0})

    # Run Solver
    try:
        sol = opti.solve()
        optimal_solution_u = sol.value(u)
        optimal_solution_x = sol.value(x)
        optimal_cost = sol.value(cost)
        print("Optimal cost:", optimal_cost)
    except RuntimeError:
        print("Solver failed to find a solution.")

    return optimal_solution_u[:, 0], optimal_solution_x[:, 1], optimal_solution_x

