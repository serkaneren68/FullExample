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
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space = spaces.Box(np.array([-1]*2),np.array([1]*2))
        self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))
        
    def get_robot_info(self):
        num_joints = p.getNumJoints(self.pandaUid)
        print(num_joints)

        for joint in range(num_joints):
            print("joint info: ", p.getJointInfo(self.pandaUid, joint))    


    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0,0,-10)

        urdfRootPath = pybullet_data.getDataPath()

        planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition = [0,0,-0.65])
        rest_poses = [0, 0]

        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[-0.5,0,-0.65])
        tableUid2 = p.loadURDF("sahne.urdf",useFixedBase=True)
        # bikeUid = p.loadURDF("Stationary2.urdf",basePosition=[5,0,-0.65])
        self.pandaUid = p.loadURDF("basic_gun.urdf", useFixedBase=True)
        for i in range(2):
            p.resetJointState(self.pandaUid, i, rest_poses[i])
        
        
        observation = p.getJointState(self.pandaUid,0)[0], p.getJointState(self.pandaUid, 1)[0]
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) 
        
        return observation 

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        # orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])      
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv

        currentOrientation =  [p.getJointState(self.pandaUid,0)[0], p.getJointState(self.pandaUid, 1)[0]]
        newAngles = [currentOrientation[0]+dx, currentOrientation[1]+dy]
        p.setJointMotorControlArray(self.pandaUid, [0,1], p.POSITION_CONTROL,action)    
          
        p.stepSimulation()




    def render(self, mode='human'):
        # Get the position and orientation of basic_gun link 2
        link_state = p.getLinkState(self.pandaUid, 2)  # Link 2 is the firing point
        gun_position = link_state[0]
        gun_orientation = link_state[1]  # Quaternion

        # Convert orientation to a direction vector
        orientation_matrix = p.getMatrixFromQuaternion(gun_orientation)
        forward_direction = [orientation_matrix[0], orientation_matrix[3], orientation_matrix[6]]  # X-axis direction

        # Scale the direction vector to set laser length
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
            lifeTime=1/240,  # Short lifetime for dynamic update
        )

        # Set a stationary camera position at [0, 0, 10], looking at the origin
        camera_position = [-5, 0, 5]  # Stationary position
        camera_target = [10, 0, 0]  # Looking at the center of the world
        up_vector = [1, 0, 0]  # World Y-axis as the up vector for a natural view

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=camera_target,
            cameraUpVector=up_vector,
        )

        # Set projection matrix for the camera
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(960) / 720, nearVal=0.1, farVal=100.0
        )

        # Capture the camera image
        (_, _, px, _, _) = p.getCameraImage(
            width=960,
            height=720,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        # Convert the image to RGB format
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))[:, :, :3]  # Drop alpha channel

        return rgb_array




           
env = PandaEnv()
env.reset()
env.get_robot_info()

# actions = [(math.pi * i, math.pi * i) for i in np.arange(-1, 1, 0.1)]
i = (0,0)

keys = {}
action = [0,0]
while True:
    keys = p.getKeyboardEvents()

    # Başlangıç açısını sıfırla


    # Tuş girdileriyle hareket
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        action[0] += 0.1
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        action[0] -= 0.1
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        action[1] -= 0.1
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        action[1] += 0.1

    env.render()
    env.step(action)

# while True:
#     env.render()
#     env.step(action=i)
# env.close()
