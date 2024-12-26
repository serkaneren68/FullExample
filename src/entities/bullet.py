import pybullet as p
import numpy as np

class Bullet:
    def __init__(self, position, direction, speed=1.0):
        self.speed = speed
        self.direction = np.array(direction)
        self.start_pos = np.array(position)  # Başlangıç pozisyonunu kaydet
        self.bullet_id = self._create_bullet(position)
        
    def _create_bullet(self, position):
        return p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.05),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 1, 0, 1]),
            basePosition=position,
        )
    
    def move(self):
        """Move bullet forward in its direction."""
        current_pos = self.get_position()
        new_position = [
            current_pos[0] + self.speed * self.direction[0],
            current_pos[1] + self.speed * self.direction[1],
            current_pos[2] + self.speed * self.direction[2],
        ]
        p.resetBasePositionAndOrientation(self.bullet_id, new_position, [0, 0, 0, 1])
        return new_position
    
    def get_position(self):
        """Get current bullet position."""
        pos, _ = p.getBasePositionAndOrientation(self.bullet_id)
        return np.array(pos)
    
    def remove(self):
        """Remove bullet from simulation."""
        p.removeBody(self.bullet_id)
    
    def is_out_of_bounds(self, bounds=20):
        """Check if bullet is out of bounds."""
        pos = self.get_position()
        return any(abs(coord) > bounds for coord in pos) 