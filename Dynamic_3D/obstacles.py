import numpy as np

class CylinderVertical:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def is_colliding(self, point):
        squared_dist = (point[0] - self.x) ** 2 + (point[1] - self.y) ** 2
        if squared_dist <= self.radius ** 2:
            return True
        return False

    def

    def get_center_vector(self):
        return np.array([self.x, self.y]).reshape(2,1)
    


class CylinderHorizontal:
    def __init__(self, y, z, radius):
        self.x = y
        self.y = z
        self.radius = radius

    def is_colliding(self, point):
        squared_dist = (point[0] - self.x) ** 2 + (point[1] - self.y) ** 2
        if squared_dist <= self.radius ** 2:
            return True
        return False
    
    def get_center_vector(self):
        return np.array([self.y, self.z]).reshape(2,1)