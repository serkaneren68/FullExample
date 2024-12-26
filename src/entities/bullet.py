import pybullet as p
import numpy as np

class Bullet:
    def __init__(self, position, direction, speed=1.0):
        self.speed = speed
        self.id = self._create_bullet(position)
        self.direction = direction
        
    def _create_bullet(self, position):
        return p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.05),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 1, 0, 1]),
            basePosition=position,
        ) 