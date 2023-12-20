import numpy as np

class Obstacle:
    def __init__(self, x, y, z, radius):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius

    def get_distance(self, x, y, z):
        return np.sqrt((self.x - x)**2 + (self.y - y)**2 + (self.z - z)**2)
    
