import numpy as np
from obstacles import *

class EnvironmentMaze:
    def __init__(self, static_cost, dynamic_cost):
        self.start = [10, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Initial State: [x, y, z, x_dot, y_dot, z_dot]
        self.goal = [5, 55, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.V_obstacles = [[2, 20, 2], [6, 20, 2], [10, 20, 2], [14, 20, 2], [18, 20, 2], [22, 20, 2], [22, 16, 2],
                       [22, 12, 2], [22, 8, 2],  ## First Box
                       [58, 15, 2], [54, 15, 2], [50, 15, 2], [46, 15, 2], [42, 15, 2], [38, 15, 2],  ## Top wall
                       [10, 58, 2], [10, 54, 2], [10, 50, 2], [10, 46, 2], [10, 42, 2], [10, 38, 2], [10, 34, 2],
                       [10, 30, 2],  ## Last Wall
                       [38, 19, 2], [38, 23, 2], [38, 27, 2], [42, 27, 2], [46, 27, 2], [50, 27, 2],
                       [22, 24, 2], [22, 28, 2], [22, 32, 2], [22, 36, 2], [22, 40, 2], [22, 44, 2], [26, 40, 2],
                       [30, 40, 2], [34, 40, 2], [38, 40, 2], [42, 40, 2], [46, 40, 2], [50, 40, 2], [54, 40, 2],
                       [50, 43, 1], [54, 47, 1], [48, 54, 1], [35, 50, 3], [37, 58, 1], [28, 47, 1], [39, 49, 1],
                       [42, 58, 1], [45, 46, 2], [23, 55, 4], [30, 30, 3]]
        self.V_move_obstacles = [[30, 0, 0, 0.5, 2, 8], [60, 35, -1, 0, 2, 5], [60, 60, -0.6, -0.2, 2, 25],
                            [40, 60, 0, -0.5, 2, 20]]  # [center_x, center_y, vel_x, vel_y, radius]
        # V_move_obstacles = []
        self.H_obstacles = []  # [center_y, center_z, radius]
        self.H_move_obstacles = []  # [center_y, center_z, vel_y, vel_z, radius]

        self.static_cost = static_cost
        self.dynamic_cost = dynamic_cost

        self.obstacles = init_obstacles(self.V_obstacles, self.V_move_obstacles, self.H_obstacles, self.H_move_obstacles, self.static_cost,
                                   self.dynamic_cost)

        self.pos_constraints = [0, 60, 0, 60, 0, 20]
        self.simulation_time = 60

    def reset(self):
        self.start = init_obstacles(self.V_obstacles, self.V_move_obstacles, self.H_obstacles, self.H_move_obstacles, self.static_cost,
                                   self.dynamic_cost)
