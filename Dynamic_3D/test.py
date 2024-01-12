# Libraries
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML
from mpl_toolkits.mplot3d import Axes3D

# Files
from vehicle import *
from plotting import *
from simulate import *
from mpc import *
from global_path_planning import *
from obstacles import *


#### MPC Properties
dt = 1                  # Time step [s]
N = 20                 # Time Horizon

####  Simulation/Problem Parameters
pos_constraints = [0, 20, 0, 50, 0, 20]         # Position Constraints [m]:    [x_min, x_max, y_min, y_max, z_min, z_max	]
x_init = [10, 2, 2, 0, 0, 0]          # Initial State: [x, y, z, x_dot, y_dot, z_dot]
x_target_last = [10, 48, 2, 0, 0, 0]      # Target State: [x, y, z, x_dot, y_dot, z_dot]
T = 75                                   # Simulation time [s]

#### Drone Velocity & Acceleration Constraints
vel_constraints = [-1, 1, -1, 1, -1, 1]         # Velocity Constraints [m/s]:  [x_min, x_max, y_min, y_max, z_min, z_max]
acc_constraints = [-10, 10, -10, 10, -10, 10]         # Acceleration Constraints [m/s^2]:    [x_min, x_max, y_min, y_max, z_min, z_max]

#### Obstacles
static_cost = 300
dynamic_cost = 500

#### Waypoint Radius Threshold
waypoint_radius = 3


#### Obstacle Definition and Initialization

## Define Obstacles
V_obstacles = [[10, 20, 4], [18, 30, 3], [7, 40, 5]]
V_move_obstacles = [[20, 40, -0.7, 0, 4], [17, 0, 0, 1, 4], [5, 0, 0, 1, 2]]  # [center_x, center_y, vel_x, vel_y, radius]
H_obstacles = []         # [center_x, center_y, radius]
H_move_obstacles = []  # [center_x, center_y, vel_x, vel_y, radius]

## Initialize Obstacles
obstacles, move_obstacles = init_obstacles(V_obstacles, V_move_obstacles, H_obstacles, H_move_obstacles, static_cost, dynamic_cost)

# Create Static World
World = World_3D([pos_constraints[0], pos_constraints[1]], [pos_constraints[2], pos_constraints[3]], [pos_constraints[4], pos_constraints[5]], obstacles, [], obstacle_margin=0.5)
World.plot()

start = x_init[:3]
goal = x_target_last[:3]

# Run RRT* to find a path
path_rrt = rrt_star(World, start, goal, radius=30, max_iter=100, plot=True)
#World.plot(path_rrt)
#World.plot2d(path_rrt)

#path_rrt = path_rrt.get_subdivided_path(12)
#World.plot(path_rrt)
#World.plot2d(path_rrt)
