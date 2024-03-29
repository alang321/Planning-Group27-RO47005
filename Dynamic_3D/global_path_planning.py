from ipywidgets import IntProgress
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import operator

from mpl_toolkits.mplot3d import art3d


def rrt_star(world_3d, start, goal, radius, max_iter=1000, report_progress=True):
    # Initialize start and goal node
    start_node = Node(None, start)
    goal_node = Node(None, goal)
    start_node.g = 0

    nodes = [start_node]

    # squared for faster nearest node search
    radius_sq = radius ** 2

    success = False

    if report_progress:
        f = IntProgress(min=0, max=max_iter)  # instantiate the bar
        display(f)  # display the bar

    for i in range(max_iter):
        if report_progress:
            if i % 10 == 0:
                f.value = i

        rnd_pos = world_3d.get_random_point()

        if world_3d.is_colliding(rnd_pos):
            continue

        # Find nearest valid node
        nearest_node = None
        nearest_dist_sq = None
        dist_sq = [(node.position[0] - rnd_pos[0]) ** 2 + (node.position[1] - rnd_pos[1]) ** 2 + (node.position[2] - rnd_pos[2]) ** 2 for node in nodes]

        sorted_nodes = [(x, dist) for dist, x in sorted(zip(dist_sq, nodes), key=lambda pair: pair[0])]
        for node, dist in sorted_nodes:
            #plot the line to the nearest node
            #path_points = np.array([node.position, rnd_pos])
            #is_colliding = world_3d.is_line_colliding(*node.position, *rnd_pos)
            #path = Path(path_points, node.position, rnd_pos, str(is_colliding), world_3d, valid=True)
            #world_3d.plot2d(path)
            #print(is_colliding)
            if not world_3d.is_line_colliding(*node.position, *rnd_pos):
                nearest_node = node
                nearest_dist_sq = dist
                break

        if nearest_node is None:
            continue

        new_node = Node(nearest_node, rnd_pos)
        new_node.g = nearest_dist_sq ** 0.5 + nearest_node.g

        # check if nearby nodes are close to the new node
        for idx, node in enumerate(nodes):
            dist_sq = node.distance_sq(new_node)
            if dist_sq > radius_sq:
                continue

            # check if line is colliding
            if world_3d.is_line_colliding(*node.position, *new_node.position):
                continue

            # check if new node is closer path to the node
            dist = dist_sq ** 0.5
            if new_node.g + dist < node.g:
                nodes[idx].parent = new_node
                nodes[idx].g = new_node.g + dist

            if node.g + dist < new_node.g:
                new_node.parent = node
                new_node.g = node.g + dist

        nodes.append(new_node)

        dist_sq = new_node.distance_sq(goal_node)
        if dist_sq < 4 * radius_sq:
            if not world_3d.is_line_colliding(*new_node.position, *goal_node.position):
                dist = dist_sq ** 0.5
                if goal_node.g is None or new_node.g + dist < goal_node.g:
                    goal_node.parent = new_node
                    goal_node.g = new_node.g + dist

                success = True

    if success:
        print("Goal found")
        #reconstruct the path
        path = []
        current = goal_node
        i = 0
        while current is not None:
            path.append(current.position)
            current = current.parent
            i += 1
            if i > max_iter:
                print("Infinite loop")
                return Path(None, start, goal, "RRT*", world_3d, False)
        return Path(path[::-1], start, goal, "RRT*", world_3d, success)
    else:
        print("No path found")
        return Path(None, start, goal, "RRT*", world_3d, success)




class World_3D:
    def __init__(self, x_range, y_range, z_range, obstacles, obstacle_margin=0):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.obstacles = obstacles
        self.obstacle_margin = obstacle_margin

        # initialize obstacles
        for obstacle in self.obstacles:
            obstacle.init_world(self)

    def update(self, dt):
        for obstacle in self.obstacles:
            obstacle.update(dt)

    def is_colliding(self, point):
        for obstacle in self.obstacles:
            if obstacle.is_colliding(point, self.obstacle_margin):
                return True
        return False

    def is_line_colliding(self, x0, y0, z0, x1, y1, z1, point_spacing=0.5):
        #create points along the line with given spacing
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) ** 0.5
        num_points = int(dist / point_spacing)
        point_fractions = np.linspace(0, 1, num_points, endpoint=True)

        for fraction in point_fractions:
            x = x0 + fraction * (x1 - x0)
            y = y0 + fraction * (y1 - y0)
            z = z0 + fraction * (z1 - z0)
            if self.is_colliding((x, y, z)):
                return True

    def get_random_point(self):
        x = np.random.uniform(self.x_range[0] + self.obstacle_margin, self.x_range[1] - self.obstacle_margin)
        y = np.random.uniform(self.y_range[0] + self.obstacle_margin, self.y_range[1] - self.obstacle_margin)
        z = np.random.uniform(self.z_range[0] + self.obstacle_margin, self.z_range[1] - self.obstacle_margin)
        return x, y, z

    def is_in_constraints(self, pos_x, pos_y):
        return pos_x >= self.x_range[0] and pos_x <= self.x_range[1] and pos_y >= self.y_range[0] and pos_y <= self.y_range[1]


    def plot3d(self, path=None, elev=None, azim=None, ortho=False, plot_moving_obstacles=False):
        # plot the 3d cylinder obstacles and the path from the top side and front
        # x = 0, y = 1, z = 2
        # for direction in range(3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        self.plot3d_ax(ax, elev, azim, ortho, path, plot_moving_obstacles)
        plt.show()

    def plot3d_ax(self, ax, elev=None, azim=None, ortho=False, path=None, plot_moving_obstacles=False):

        # Set the elevation and azimuth for the top-down view
        if elev is not None and azim is not None:
            ax.view_init(elev=elev, azim=azim)

        # plot obstacles
        for obstacle in self.obstacles:
            if obstacle.velocity is None:
                obstacle.plot(ax)
            elif plot_moving_obstacles:
                obstacle.plot(ax)

        if path is not None:
            if path.valid:
                path_points = np.array(path.path_points)
                ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], color='blue')

                # plot path points in blue
                ax.scatter(path_points[:, 0], path_points[:, 1], path_points[:, 2], color='blue')

                # plot start and end point
                ax.scatter(path.start[0], path.start[1], path.start[2], color='green')
                ax.scatter(path.goal[0], path.goal[1], path.goal[2], color='green')

        ax.set_aspect('equal', adjustable='box')

        if ortho:
            ax.set_proj_type('ortho')

        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.set_zlim(self.z_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def plot2d_xy(self, path=None, states=None, plot_moving_obstacles=False, show_legend=True, start=None, goal=None, title=None):
        self.plot2d_xy_ax(plt.gca(), path, states, plot_moving_obstacles, show_legend, start, goal, title)
        plt.show()

    def plot2d_xy_ax(self, ax, path=None, states=None, plot_moving_obstacles=False, show_legend=True, start=None, goal=None, legend_location='upper left', title=None):
        # plot obstacles
        # sort obstacles by y position
        sorted_obstacles = sorted(self.obstacles, key=operator.attrgetter('z'))
        for obstacle in reversed(sorted_obstacles):
            if obstacle.velocity is None:
                obstacle.plot_xy(ax)
            elif plot_moving_obstacles:
                obstacle.plot_xy(ax)

        # for move_obstacle in self.move_obstacles:
        #     move_obstacle.plot_xy(ax, 'red')

        if path is not None:
            start = path.start
            goal = path.goal

            if path.valid:
                path_points = np.array(path.path_points)
                ax.plot(path_points[:, 1], path_points[:, 0], color='blue', linestyle='--', label="RRT* Path")
                ax.scatter(path_points[:-1, 1], path_points[:-1, 0], s=20, color='blue')

        if start is not None:
            ax.scatter(start[1], start[0], color='green', label="Start")
        # plot start and end point
        if goal is not None:
            ax.scatter(goal[1], goal[0], color='red', marker='x', s=50, label="Goal")

        if states is not None:
            ax.plot(states[1], states[0], color='red', label="Trajectory")
            ax.scatter(states[1], states[0], s=7, color='red')

        ax.set_xlim(self.y_range)
        ax.set_ylim(self.x_range)
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        if show_legend:
            ax.legend(loc=legend_location)

        #equal axis scale
        ax.set_aspect('equal', adjustable='box')

        #set the title
        if title is not None:
            ax.set_title(title)

    def plot2d_xz(self, path=None, states=None, plot_moving_obstacles=False, show_legend=True, start=None, goal=None, title=None):
        self.plot2d_xz_ax(plt.gca(), path, states, plot_moving_obstacles, show_legend, start, goal, title)
        plt.show()

    def plot2d_xz_ax(self, ax, path=None, states=None, plot_moving_obstacles=False, show_legend=True, start=None, goal=None, legend_location='upper left', title=None):
        # plot obstacles

        # sort obstacles by y position
        sorted_obstacles = sorted(self.obstacles, key=operator.attrgetter('y'))
        for obstacle in reversed(sorted_obstacles):
            if obstacle.velocity is None:
                obstacle.plot_xz(ax)
            elif plot_moving_obstacles:
                obstacle.plot_xz(ax)

        # for move_obstacle in self.move_obstacles:
        #     move_obstacle.plot_xy(ax, 'red')

        if path is not None:
            start = path.start
            goal = path.goal

            if path.valid:
                path_points = np.array(path.path_points)
                ax.plot(path_points[:, 0], path_points[:, 2], color='blue', linestyle='--', label="RRT* Path")
                ax.scatter(path_points[:-1, 0], path_points[:-1, 2], s=20, color='blue')

        if start is not None:
            ax.scatter(start[0], start[2], color='green', label="Start")
        # plot start and end point
        if goal is not None:
            ax.scatter(goal[0], goal[2], color='red', marker='x', s=50, label="Goal")

        if states is not None:
            ax.plot(states[0], states[2], color='red', label="Trajectory")
            ax.scatter(states[0], states[2], s=7, color='red')

        ax.set_xlim(self.x_range)
        ax.set_ylim(self.z_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        if show_legend:
            ax.legend(loc=legend_location)

        #flip x axis
        ax.invert_xaxis()

        #equal axis scale
        ax.set_aspect('equal', adjustable='box')

        #set the title
        if title is not None:
            ax.set_title(title)

    def plot2d_yz(self, path=None, states=None, plot_moving_obstacles=False, show_legend=True, start=None, goal=None, title=None):
        self.plot2d_yz_ax(plt.gca(), path, states, plot_moving_obstacles, show_legend, start, goal, title)
        plt.show()

    def plot2d_yz_ax(self, ax, path=None, states=None, plot_moving_obstacles=False, show_legend=True, start=None, goal=None, legend_location='upper left', title=None):
        # plot obstacles
        sorted_obstacles = sorted(self.obstacles, key=operator.attrgetter('x'))
        for obstacle in reversed(sorted_obstacles):
            if obstacle.velocity is None:
                obstacle.plot_yz(ax)
            elif plot_moving_obstacles:
                obstacle.plot_yz(ax)

        # for move_obstacle in self.move_obstacles:
        #     move_obstacle.plot_xy(ax, 'red')

        if path is not None:
            start = path.start
            goal = path.goal

            if path.valid:
                path_points = np.array(path.path_points)
                ax.plot(path_points[:, 1], path_points[:, 2], color='blue', linestyle='--', label="RRT* Path")
                ax.scatter(path_points[:-1, 1], path_points[:-1, 2], s=20, color='blue')

        if start is not None:
            ax.scatter(start[1], start[2], color='green', label="Start")
        # plot start and end point
        if goal is not None:
            ax.scatter(goal[1], goal[2], color='red', marker='x', s=50, label="Goal")

        if states is not None:
            ax.plot(states[1], states[2], color='red', label="Trajectory")
            ax.scatter(states[1], states[2], s=7, color='red')

        ax.set_xlim(self.y_range)
        ax.set_ylim(self.z_range)
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        if show_legend:
            #locate the legend at the top left
            ax.legend(loc=legend_location)

        # equal axis scale
        ax.set_aspect('equal', adjustable='box')

        #set the title
        if title is not None:
            ax.set_title(title)


class Node:
    def __init__(self, parent, position):
        self.parent = parent
        self.position = position

        self.g = None # Distance to start node
        self.h = None # Heuristic
        self.f = None # Total cost, distance to start + heuristic

    def distance(self, other):
        return ((self.position[0] - other.position[0]) ** 2 + (self.position[1] - other.position[1]) ** 2 + (self.position[2] - other.position[2]) ** 2)**0.5

    def distance_sq(self, other):
        return (self.position[0] - other.position[0]) ** 2 + (self.position[1] - other.position[1]) ** 2 + (self.position[2] - other.position[2]) ** 2

    def __eq__(self, other):
        return self.position == other.position

    def __ne__(self, other):
        return self.position != other.position

    def __lt__(self, other):
        return self.f < other.f

    def __le__(self, other):
        return self.f <= other.f

    def __gt__(self, other):
        return self.f > other.f

    def __ge__(self, other):
        return self.f >= other.f

    def __str__(self):
        return f'Node: {self.position}, g: {self.g}'

class Path:
    def __init__(self, path_points, start, goal, name, world_3d, valid=True):
        self.name = name
        self.start = start
        self.goal = goal
        self.valid = valid
        self.world_3d = world_3d

        self.path_points = path_points

    def get_subdivided_path(self, max_length):
        #subdivide the path into num_points points
        if self.path_points is None:
            return None

        if len(self.path_points) < 2:
            return None

        #subdivide the path
        path_points = []
        for i in range(len(self.path_points) - 1):
            path_points.append(self.path_points[i])
            dist_sq = ((self.path_points[i][0] - self.path_points[i + 1][0]) ** 2 + (self.path_points[i][1] - self.path_points[i + 1][1]) ** 2 + (self.path_points[i][2] - self.path_points[i + 1][2]) ** 2)

            num_extra_points = int(dist_sq ** 0.5 // max_length)

            for j in range(num_extra_points):
                fraction = (j + 1) / (num_extra_points + 1)
                x = self.path_points[i][0] + (self.path_points[i + 1][0] - self.path_points[i][0]) * fraction
                y = self.path_points[i][1] + (self.path_points[i + 1][1] - self.path_points[i][1]) * fraction
                z = self.path_points[i][2] + (self.path_points[i + 1][2] - self.path_points[i][2]) * fraction
                path_points.append([x, y, z])

        path_points.append(self.path_points[-1])

        return Path(path_points, self.start, self.goal, self.name, self.world_3d, self.valid)





            
