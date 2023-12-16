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


def a_star(occupany_grid, start, goal, report_progress=True):
    def heuristic(node1, node2):
        return (node1.index[0] - node2.index[0]) ** 2 + (node1.index[1] - node2.index[1]) ** 2

    grid_map = occupany_grid.grid_map

    #lists for open and closed nodes
    open_nodes = []
    closed_nodes = []

    # Initialize start and goal node
    start_idx = occupany_grid.get_closest_grid_index(start)
    goal_idx = occupany_grid.get_closest_grid_index(goal)
    start_node = Node(None, start_idx)
    goal_node = Node(None, goal_idx)

    open_nodes.append(start_node)

    nodes_visited = 0
    if report_progress:
        total_nodes = occupany_grid.num_free_cells
        percent = 0
        f = IntProgress(min=0, max=total_nodes)  # instantiate the bar
        display(f)  # display the bar
    # Loop until goal is found
    while len(open_nodes) > 0:
        if(report_progress):
            f.value = nodes_visited

        # Get the cheapest open node measured by f value, f value is defined in the node class operator overloading
        min_idx, cheapest_open_node = min(enumerate(open_nodes), key=operator.itemgetter(1))

        # Move from open to closed list
        closed_nodes.append(open_nodes.pop(min_idx))

        # Check if goal is found
        if cheapest_open_node == goal_node:
            print("Goal found")
            #reconstruct the path
            path = []
            current = cheapest_open_node
            while current is not None:
                path.append(current.index)
                current = current.parent
            return path[::-1]

        # Generate children
        for neighbor_idx, cost in occupany_grid.get_adjacent_free_cells(*cheapest_open_node.index):
            # Create neighbor node
            neighbor_node = Node(cheapest_open_node, neighbor_idx)

            # Check if neighbor is in closed list
            if not neighbor_node in closed_nodes:
                # Calculate f, g, and h values
                neighbor_node.g = cheapest_open_node.g + cost
                neighbor_node.h = heuristic(neighbor_node, cheapest_open_node)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                #check if neighbor is in open list
                #if not a value error is thrown
                try:
                    index = open_nodes.index(neighbor_node)
                    if open_nodes[index].g > neighbor_node.g:
                        # if its cheaper to get to the current neighbor with the current path
                        # update the open node, otherwise discard it
                        open_nodes[index] = neighbor_node

                except ValueError:
                    # if the neighbor is not in the open list add it
                    open_nodes.append(neighbor_node)
                    nodes_visited += 1
                    pass
    return None



def plot_path(path, obstacles, occupany_grid, start, goal):
    path = np.array(path)

    #convert indeces to points
    path_points = []
    for index in path:
        path_points.append(occupany_grid.index_to_point(*index))
    path_points = np.array(path_points)
    pos_x = path_points[:, 0]
    pos_y = path_points[:, 1]

    for idx, obstacle in enumerate(obstacles):
        circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='blue', fill=False)
        plt.gca().add_patch(circle)

    plt.plot(pos_x, pos_y, color = "r", label = "Vehicle Path")
    plt.title('Path with Obstacle')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    #mark start and goal
    plt.scatter(start[0], start[1], color = "g", label = "Start")
    plt.scatter(goal[0], goal[1], color = "pink", label = "Goal")
    plt.legend()

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


class OccupancyGrid:
    def __init__(self, grid_resolution, origin, obstacles, field_size, obstacle_margin=0):
        self.grid_resolution = grid_resolution
        self.origin = origin
        self.obstacles = obstacles
        self.field_size = field_size
        self.obstacle_margin = obstacle_margin

        self.x_points = np.arange(self.origin[0], self.field_size[0], self.grid_resolution)
        self.y_points = np.arange(self.origin[1], self.field_size[1], self.grid_resolution)


        self.diag_cost = (2 ** 0.5) * self.grid_resolution
        self.straight_cost = self.grid_resolution

        self.num_free_cells = 0
        self.grid_map = self.generate_grid_map()


    def get_closest_grid_index(self, point):
        x_points = np.arange(self.origin[0], self.field_size[0], self.grid_resolution)
        y_points = np.arange(self.origin[1], self.field_size[1], self.grid_resolution)
        x_index = np.argmin(np.abs(x_points - point[0]))
        y_index = np.argmin(np.abs(y_points - point[1]))
        return x_index, y_index

    def index_to_point(self, index_x, index_y):
        x = self.x_points[index_x]
        y = self.y_points[index_y]
        return [x, y]

    def generate_grid_map(self):
        x_dim = self.x_points.size
        y_dim = self.y_points.size
        grid_map = np.zeros((x_dim, y_dim))
        for i, x in enumerate(self.x_points):
            for j, y in enumerate(self.y_points):
                if self.is_colliding([x,y], self.grid_resolution / 2):
                    grid_map[i, j] = 1
                else:
                    self.num_free_cells += 1
        return grid_map

    def is_colliding(self, point, radius):
        for obstacle in self.obstacles:
            squared_dist = (point[0] - obstacle[0]) ** 2 + (point[1] - obstacle[1]) ** 2
            if squared_dist <= (obstacle[2] + radius + self.obstacle_margin) ** 2:
                return True
        return False

    def get_adjacent_free_cells(self, index_x, index_y):
        adjacent_indices = []
        costs = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == j == 0:
                    continue
                new_index_x = index_x + i
                new_index_y = index_y + j
                if self.is_free_valid_cell(new_index_x, new_index_y):
                    #check if diagonal or straight
                    if i == 0 or j == 0:
                        cost = self.straight_cost
                    else:
                        cost = self.diag_cost
                    adjacent_indices.append(((new_index_x, new_index_y), cost))
        return adjacent_indices

    def is_free_valid_cell(self, index_x, index_y):
        return index_x >= 0 and index_x < len(self.grid_map[0]) and index_y >= 0 and index_y < len(self.grid_map[1]) and self.grid_map[index_x, index_y] != 1

    def is_occupied_index(self, index_x, index_y):
        return self.grid_map[index_x, index_y] == 1

    def plot(self):
        #flip the axes to match the grid
        plt.imshow(self.grid_map.T, cmap='Greys', origin='lower')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Occupancy Grid')
        plt.show()


class Node:
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

        self.g = 0 # Distance to start node
        self.h = 0 # Distance to goal node
        self.f = 0 # Total cost, distance to start + norm

    def __eq__(self, other):
        return self.index == other.index

    def __ne__(self, other):
        return self.index != other.index

    def __lt__(self, other):
        return self.f < other.f

    def __le__(self, other):
        return self.f <= other.f

    def __gt__(self, other):
        return self.f > other.f

    def __ge__(self, other):
        return self.f >= other.f

    def __str__(self):
        return f'Node: {self.index}, g: {self.g}'