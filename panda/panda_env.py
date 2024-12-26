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
        self.action_space = spaces.Box(np.array([-1]*4),np.array([1]*4))
        self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))
        
    
    
    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                    currentPosition[1] + dy,
                    currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7]

        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])        

        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        if state_object[2]>0.45:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        info= state_object
        observation = state_robot + state_fingers
        return observation, reward, done, info
    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0,0,-10)
        
        urdfRootPath = pybullet_data.getDataPath()
        
        planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition = [0,0,-0.65])
        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True)
        
        for i in range(7):
            p.resetJointState(self.pandaUid, i, rest_poses[i])
        
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
        trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])
        state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        
        observation = state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # rendering's back on again
        return observation      
        
    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                    aspect=float(960) /720,
                                                    nearVal=0.1,
                                                    farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array
    def close(self):
        p.disconnect()
    


env = PandaEnv()
observation = env.reset()
done = False
error = 0.01
fingers = 1
object_position = [0.7, 0, 0.1]

k_p = 10
k_d = 1
dt = 1./240. # the default timestep in pybullet is 240 Hz  
t = 0

for i_episode in range(20):
   observation = env.reset()
   fingers = 1
   for t in range(100):
       env.render()
       print(observation)
       dx = object_position[0]-observation[0]
       dy = object_position[1]-observation[1]
       target_z = object_position[2] 
       if abs(dx) < error and abs(dy) < error and abs(dz) < error:
           fingers = 0
       if (observation[3]+observation[4])<error+0.02 and fingers==0:
           target_z = 0.5
       dz = target_z-observation[2]
       pd_x = k_p*dx + k_d*dx/dt
       pd_y = k_p*dy + k_d*dy/dt
       pd_z = k_p*dz + k_d*dz/dt
       action = [pd_x,pd_y,pd_z,fingers]
       observation, reward, done, info = env.step(action)
       object_position = info
       if done:
           print("Episode finished after {} timesteps".format(t+1))
           break
env.close()