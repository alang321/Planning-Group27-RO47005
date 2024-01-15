# Libraries
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

# Files
from vehicle import *
from simulate import *
from mpc import *
from global_path_planning import *
from obstacles import *
from environments import *

matplotlib.rcParams['animation.embed_limit'] = 2**128



#### MPC Properties
dt = 0.1                  # Time step [s]
N = 10             # Time Horizon


# Speed limit
speed_limit = 10

# Roll and Pitch limit
angle_limit = 90

# Acceleration limit
acc_limit_x = 15
acc_limit_y = 15
acc_limit_z = 15

       # Position Constraints [m]:    [x_min, x_max, y_min, y_max, z_min, z_max	]
vel_constraints = [-speed_limit, speed_limit, -speed_limit, speed_limit, -speed_limit, speed_limit]         # Velocity Constraints [m/s]:  [x_min, x_max, y_min, y_max, z_min, z_max]
ang_constraints = [-angle_limit, angle_limit, -angle_limit, angle_limit, -360, 360]         # Angular Velocity Constraints [rad/s]:  [x_min, x_max, y_min, y_max, z_min, z_max]
acc_constraints = [-acc_limit_x, acc_limit_x, -acc_limit_y, acc_limit_y, -acc_limit_z, acc_limit_z]
max_rad_per_s = 4000

#### Obstacles
static_cost = 20
dynamic_cost = 80

#### Waypoint Radius Threshold
waypoint_radius = 3



#### Obstacle Definition and Initialization

environments = [EnvironmentMaze(static_cost, dynamic_cost),
                EnvironmentOriginal(static_cost, dynamic_cost),
                EnvironmentDynamicLasers(static_cost, dynamic_cost),
                EnvironmentChimney(static_cost, dynamic_cost)]

fig = plt.figure(figsize=(10, 3.5), constrained_layout=True)
#fig.tight_layout()
gs = gridspec.GridSpec(2, 4, figure=fig)

ax1 = fig.add_subplot(gs[:, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 2])
ax4 = fig.add_subplot(gs[:, 3])

axes = [ax1, ax2, ax3, ax4]

for idx, environment in enumerate(environments):
    pos_constraints = environment.pos_constraints       # Position Constraints [m]:    [x_min, x_max, y_min, y_max, z_min, z_max	]
    x_init = environment.start         # Initial State: [x, y, z, x_dot, y_dot, z_dot]
    x_target_last = environment.goal     # Target State: [x, y, z, x_dot, y_dot, z_dot]
    T = environment.simulation_time

    ## Initialize Obstacles
    obstacles = environment.obstacles


    # Create Static World
    World = World_3D([pos_constraints[0], pos_constraints[1]], [pos_constraints[2], pos_constraints[3]], [pos_constraints[4], pos_constraints[5]], obstacles, obstacle_margin=1)

    ax = axes[idx]

    # Run RRT* to find a path
    #path_rrt = rrt_star(World, x_init[:3], x_target_last[:3], radius=10, max_iter=1000)
    if idx == 0:
        World.plot2d_xy_ax(ax, plot_moving_obstacles=True, start=x_init[:3], goal=x_target_last[:3], title="D")
    elif idx == 1:
        World.plot2d_xy_ax(ax, plot_moving_obstacles=True, start=x_init[:3], goal=x_target_last[:3], show_legend=False, title="A")
    elif idx == 2:
        World.plot2d_xy_ax(ax, plot_moving_obstacles=True, start=x_init[:3], goal=x_target_last[:3], show_legend=False, title="B")
    elif idx == 3:
        World.plot2d_yz_ax(ax, plot_moving_obstacles=True, start=x_init[:3], goal=x_target_last[:3], show_legend=False, title="C")

plt.show()