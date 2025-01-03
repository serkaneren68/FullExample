import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0],dtype=np.float32),  # Eklem hareketleri ve ateşleme durumu
            high=np.array([1, 1, 1],dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))
        self.balloons = []  # To keep track of balloons
        self.balloon_targets = {}  # To track balloon target positions
        self.balloon_speed = 0.01  # Speed of balloon movement
        self.bullets = []
        self.bullet_speed = 1
        
        self.stats = {
            "score": 0,
            "remaining_balloons": 0,
            "bullets_used": 0,
            "balloons_popped": 0,
        }
        
    def get_robot_info(self):
        num_joints = p.getNumJoints(self.pandaUid)
        print(num_joints)

        for joint in range(num_joints):
            print("joint info: ", p.getJointInfo(self.pandaUid, joint))
    def is_done(self):
        return len(self.balloons) == 0  # Tüm balonlar vurulmuşsa episode sona erer
    
    def compute_reward(self):
        reward = 0
        for balloon in self.balloons:
            pos, _ = p.getBasePositionAndOrientation(balloon)
            # Balona olan mesafeyi ölç
            distance = np.linalg.norm(np.array(pos) - np.array(p.getBasePositionAndOrientation(self.pandaUid)[0]))
            if distance < 0.2:  # Eğer balon çok yakında
                reward += 10  # Balon vurulmuş gibi pozitif ödül ver
        return reward
    
    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -10)

        urdfRootPath = pybullet_data.getDataPath()

        planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
        rest_poses = [0, 0]

        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[-0.5, 0, -0.65])
        tableUid2 = p.loadURDF("sahne.urdf", useFixedBase=True)
        self.pandaUid = p.loadURDF("basic_gun.urdf", useFixedBase=True)
        for i in range(2):
            p.resetJointState(self.pandaUid, i, rest_poses[i])

        observation = p.getJointState(self.pandaUid, 0)[0], p.getJointState(self.pandaUid, 1)[0]
        
        self.balloons = []
        self.balloon_targets = {}
        self.add_random_balloons(10)
    
        # Reset stats
        self.stats["score"] = 0
        self.stats["remaining_balloons"] = len(self.balloons)
        self.stats["bullets_used"] = 0
        self.stats["balloons_popped"] = 0
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return observation

    def add_random_balloons(self, num_balloons):
        """Add random balloons to the scene."""
        for _ in range(num_balloons):
            # Generate a random starting position
            x = random.uniform(9, 17)
            y = random.uniform(-10, 10)
            z = random.uniform(0.5, 2)  # Height of balloons

            balloon_id = p.createMultiBody(
                baseMass=0.4,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.4),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.4, rgbaColor=[0, 1, 0, 1]),
                basePosition=[x, y, z],
            )
            self.balloons.append(balloon_id)
            self.balloon_targets[balloon_id] = [random.uniform(9, 17), random.uniform(-10, 10), random.uniform(0.5, 2)]

    def move_balloons(self):
        """Move balloons toward random target positions."""
        for balloon_id in self.balloons:
            current_position, _ = p.getBasePositionAndOrientation(balloon_id)
            target_position = self.balloon_targets[balloon_id]

            # Move toward the target position
            new_position = [
                current_position[0] + self.balloon_speed * (target_position[0] - current_position[0]),
                current_position[1] + self.balloon_speed * (target_position[1] - current_position[1]),
                current_position[2] + self.balloon_speed * (target_position[2] - current_position[2]),
            ]

            # Update balloon position
            p.resetBasePositionAndOrientation(balloon_id, new_position, [0, 0, 0, 1])

            # If the target is reached, generate a new random target
            if all(abs(new_position[i] - target_position[i]) < 0.1 for i in range(3)):
                self.balloon_targets[balloon_id] = [random.uniform(17, 19), random.uniform(-10, 10), random.uniform(0.5, 2)]

    def check_collision(self, laser_start, laser_end):
        """Check if the laser hits any balloons."""
        # hit_info = p.rayTest(laser_start, laser_end)
        # for hit in hit_info:
        #     if hit[0] in self.balloons :  # If the laser hits a balloon
        #         print(f"Balloon {hit[0]} hit!")
        #         p.removeBody(hit[0])
        #         self.balloons.remove(hit[0])
        #         del self.balloon_targets[hit[0]]
        pass
    
    def move_bullets(self):
        """Move bullets forward and check for collisions."""
        for bullet, direction in self.bullets[:]:
            current_position, _ = p.getBasePositionAndOrientation(bullet)
            new_position = [
                current_position[0] + self.bullet_speed * direction[0],
                current_position[1] + self.bullet_speed * direction[1],
                current_position[2] + self.bullet_speed * direction[2],
            ]
            p.resetBasePositionAndOrientation(bullet, new_position, [0, 0, 0, 1])

            # Check for collisions between bullets and balloons
            for balloon_id in self.balloons[:]:
                balloon_position, _ = p.getBasePositionAndOrientation(balloon_id)
                distance = np.linalg.norm(np.array(new_position) - np.array(balloon_position))
                if distance < 0.4:  # Kurşun balona çok yaklaştığında
                    print(f"Balloon {balloon_id} hit!")
                    p.removeBody(balloon_id)
                    self.balloons.remove(balloon_id)
                    del self.balloon_targets[balloon_id]
                    p.removeBody(bullet)
                    self.bullets.remove((bullet, direction))
                    
                    self.stats["score"] += 10
                    self.stats["balloons_popped"] += 1
                    self.stats["remaining_balloons"] = len(self.balloons)
                    break  # Bir kurşun bir balonu yok ettikten sonra diğerlerini kontrol etmeye gerek yok

            # Remove bullets that go out of bounds
            # if abs(new_position[0]) > 10 or abs(new_position[1]) > 10 or abs(new_position[2]) > 10:
            #     p.removeBody(bullet)
            #     self.bullets.remove((bullet, direction))

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv

        currentOrientation = [p.getJointState(self.pandaUid, 0)[0], p.getJointState(self.pandaUid, 1)[0]]
        newAngles = [currentOrientation[0] + dx, currentOrientation[1] + dy]
        p.setJointMotorControlArray(self.pandaUid, [0, 1], p.POSITION_CONTROL, action[0:-1])
   # Ateş etme aksiyonu
        if action[2] > 0.5:  # Eğer ateşleme durumu > 0.5 ise ateş et
            link_state = p.getLinkState(self.pandaUid, 2)
            gun_position = link_state[0]
            orientation_matrix = p.getMatrixFromQuaternion(link_state[1])
            forward_direction = [orientation_matrix[0], orientation_matrix[3], orientation_matrix[6]]
            self.fire_bullet(gun_position, forward_direction)
        
        p.stepSimulation()
        self.move_balloons()
        self.move_bullets()
        reward = self.compute_reward()

        # Episode tamamlandı mı?
        done = self.is_done()
        return None, reward, done, {}
    
    def fire_bullet(self, gun_position, forward_direction):
        """Fire a bullet in the given direction."""
        bullet_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.05),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 1, 0, 1]),
            basePosition=gun_position,
        )
        self.stats["bullets_used"] += 1
        self.bullets.append((bullet_id, forward_direction))    
    
    def render(self, mode='human'):
        # Calculate laser start and end positions
        link_state = p.getLinkState(self.pandaUid, 2)  # Link 2 is the firing point
        gun_position = link_state[0]
        gun_orientation = link_state[1]  # Quaternion

        orientation_matrix = p.getMatrixFromQuaternion(gun_orientation)
        forward_direction = [orientation_matrix[0], orientation_matrix[3], orientation_matrix[6]]

        laser_length = 50.0
        laser_end_position = [
            gun_position[0] + forward_direction[0] * laser_length,
            gun_position[1] + forward_direction[1] * laser_length,
            gun_position[2] + forward_direction[2] * laser_length,
        ]

        # Draw the laser
        p.addUserDebugLine(
            lineFromXYZ=gun_position,
            lineToXYZ=laser_end_position,
            lineColorRGB=[1, 0, 0],  # Red laser
            lineWidth=2.0,
            lifeTime=1 / 240,
        )

        # Check for collisions
        #self.check_collision(gun_position, laser_end_position)
        # Display stats in the simulation
        p.addUserDebugText(
            f"Score: {self.stats['score']}", 
            textPosition=[10, -2, 2.4], 
            textColorRGB=[0, 0, 0], 
            textSize=1.2, 
            lifeTime=1/240
        )
        p.addUserDebugText(
            f"Remaining Balloons: {self.stats['remaining_balloons']}", 
            textPosition=[10, -2, 2], 
            textColorRGB=[0, 0, 0], 
            textSize=1.2, 
            lifeTime=1/240
        )
        p.addUserDebugText(
            f"Bullets Used: {self.stats['bullets_used']}", 
            textPosition=[10, -2, 1.6], 
            textColorRGB=[0, 0, 0], 
            textSize=1.2, 
            lifeTime=1/240
        )
        p.addUserDebugText(
            f"Balloons Popped: {self.stats['balloons_popped']}", 
            textPosition=[10, -2, 1.2], 
            textColorRGB=[0, 0, 0], 
            textSize=1.2, 
            lifeTime=1/240
        )
        # Stationary camera
        camera_position = [-5, 0, 5]
        camera_target = [0, 0, 0]
        up_vector = [0, 0, 1]
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=camera_target,
            cameraUpVector=up_vector,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(960) / 720, nearVal=0.1, farVal=100.0
        )
        (_, _, px, _, _) = p.getCameraImage(
            width=960, height=720, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL
        )


# Main loop
env = PandaEnv()
env.reset()
env.add_random_balloons(10)  # Add 10 balloons

keys = {}
action = [0, 0, 0]
while True:
    keys = p.getKeyboardEvents()
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        action[1] += 0.1
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        action[1] -= 0.1
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        action[0] += 0.1
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        action[0] -= 0.1

    # Fire bullet with SPACE key
    if p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
        action[2] = 1
    else:   
        action[2] = 0
    env.step(action)
    env.move_balloons()
    env.render()
