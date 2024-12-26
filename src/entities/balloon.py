import pybullet as p
import numpy as np
import random

class Balloon:
    def __init__(self, position, radius=0.4, color=[0, 1, 0, 1]):
        self.radius = radius
        self.color = color
        self.id = self._create_balloon(position)
        self.target_position = self._generate_random_target()
        
    def _create_balloon(self, position):
        return p.createMultiBody(
            baseMass=0.4,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=self.radius, rgbaColor=self.color),
            basePosition=position,
        ) 