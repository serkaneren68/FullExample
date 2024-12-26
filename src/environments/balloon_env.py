import gym
from gym import spaces
import pybullet as p
import numpy as np
import random
import os
import pybullet_data
import time

from src.utils.constants import *
from src.entities.bullet import Bullet

class BalloonShooterEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        p.connect(p.GUI)
        # Set initial debug camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=15.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[5, 0, 0]
        )
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Frame stacking parameters
        self.frame_stack_size = 4
        self.frames = []
        
        # Image observation space (4 stacked frames, 84x84 grayscale images)
        # Using channel-first format (num_frames, height, width)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.frame_stack_size, 84, 84),  # CHW format
            dtype=np.float32
        )
        
        # Frame skipping
        self.frame_skip = 4
        
        self.balloons = []  # To keep track of balloons
        self.balloon_targets = {}  # To track balloon target positions
        self.balloon_speed = 0.01  # Speed of balloon movement
        self.bullets = []
        self.bullet_speed = 1
        
        # Episode tracking
        self.current_episode = 0
        self.episode_steps = 0
        self.max_episode_steps = 1000
        
        self.stats = {
            "score": 0,
            "remaining_balloons": 0,
            "bullets_used": 0,
            "balloons_popped": 0,
            "current_episode": 0,
            "episode_steps": 0,
        }
        
        # Camera settings
        self.camera_position = [-5, 0, 5]
        self.camera_target = [10, 0, 0]
        self.camera_up = [1, 0, 0]
        
        # Reward sabitleri
        self.HIT_REWARD = 10.0          # Temel vuruş ödülü
        self.MISS_PENALTY = -2.0        # Iskalama cezası
        self.TIME_PENALTY = -0.1        # Her adımda uygulanacak zaman cezası
        self.DISTANCE_BONUS = 5.0       # Uzak mesafe bonusu
        self.COMBO_BONUS = 2.0          # Combo vuruş bonusu
        
        # Yeni istatistikler
        self.consecutive_hits = 0        # Ardışık vuruş sayacı
        self.last_hit_time = 0          # Son vuruş zamanı
        self.bullets_missed = 0          # Iskalanan mermi sayısı
        
    def get_robot_info(self):
        num_joints = p.getNumJoints(self.pandaUid)
        print(num_joints)

        for joint in range(num_joints):
            print("joint info: ", p.getJointInfo(self.pandaUid, joint))
            
    def is_done(self):
        """Check if episode should end."""
        # End if no balloons left or max steps reached
        return len(self.balloons) == 0 or self.episode_steps >= self.max_episode_steps
    
    def compute_reward(self):
        reward = 0
        
        # Zaman cezası
        reward += self.TIME_PENALTY
        
        # Balon vuruş kontrolü ve ödülleri
        for bullet in self.bullets[:]:  # Tüm mermileri kontrol et
            bullet_pos = bullet.get_position()
            
            # Mermi sahne dışına çıktı mı kontrol et
            if bullet.is_out_of_bounds():
                self.bullets_missed += 1
                reward += self.MISS_PENALTY
                continue
            
            # Balon vuruş kontrolü
            for balloon in self.balloons[:]:
                balloon_pos = np.array(p.getBasePositionAndOrientation(balloon)[0])
                distance = np.linalg.norm(bullet_pos - balloon_pos)
                
                if distance < (BALLOON_RADIUS + BULLET_RADIUS):
                    # Temel vuruş ödülü
                    hit_reward = self.HIT_REWARD
                    reward += hit_reward
                    
                    # Uzaklık bonusu (mermi başlangıç pozisyonuna göre)
                    shot_distance = np.linalg.norm(bullet.start_pos - balloon_pos)
                    distance_multiplier = min(shot_distance / 10.0, 2.0)  # Max 2x bonus
                    distance_bonus = self.DISTANCE_BONUS * distance_multiplier
                    reward += distance_bonus
                    
                    # Combo bonus
                    current_time = self.episode_steps
                    combo_bonus = 0
                    if current_time - self.last_hit_time <= 10:  # 10 adım içinde vuruş
                        self.consecutive_hits += 1
                        combo_bonus = self.COMBO_BONUS * self.consecutive_hits
                        reward += combo_bonus
                    else:
                        self.consecutive_hits = 1
                    
                    self.last_hit_time = current_time
                    self.stats["balloons_popped"] += 1
                    
                    # Balonu ve mermiyi kaldır
                    self.balloons.remove(balloon)
                    self.bullets.remove(bullet)
                    p.removeBody(balloon)
                    bullet.remove()
                    
                    # Score'u güncelle
                    total_hit_reward = hit_reward + distance_bonus + combo_bonus
                    self.stats["score"] = reward  # Score'u toplam reward'a eşitle
                    
        return reward

    def _is_bullet_out_of_bounds(self, bullet_pos):
        # Sahne sınırlarını kontrol et
        bounds = 20  # Sahne sınırları
        return any(abs(coord) > bounds for coord in bullet_pos)

    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -10)

        # Update episode counters
        self.current_episode += 1
        self.episode_steps = 0

        urdfRootPath = pybullet_data.getDataPath()

        planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
        rest_poses = [0, 0]

        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[-0.5, 0, -0.65])
        tableUid2 = p.loadURDF("sahne.urdf", useFixedBase=True)
        self.pandaUid = p.loadURDF("basic_gun.urdf", useFixedBase=True)
        
        # Reset joint positions
        for i in range(2):
            p.resetJointState(self.pandaUid, i, rest_poses[i])

        # Clear bullets and balloons
        self.bullets = []
        self.balloons = []
        self.balloon_targets = {}
        
        # Reset stats
        self.stats["score"] = 0
        self.stats["remaining_balloons"] = 0
        self.stats["bullets_used"] = 0
        self.stats["balloons_popped"] = 0
        self.stats["current_episode"] = self.current_episode
        self.stats["episode_steps"] = self.episode_steps
        
        # Add initial balloons
        self.add_random_balloons(5)  # Start with 5 balloons
        self.stats["remaining_balloons"] = len(self.balloons)
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Get initial observation
        observation = self.get_observation()
        return observation

    def add_random_balloons(self, num_balloons):
        """Add random balloons to the scene."""
        for _ in range(num_balloons):
            x = random.uniform(9, 17)
            y = random.uniform(-10, 10)
            z = random.uniform(0.5, 2)

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

            new_position = [
                current_position[0] + self.balloon_speed * (target_position[0] - current_position[0]),
                current_position[1] + self.balloon_speed * (target_position[1] - current_position[1]),
                current_position[2] + self.balloon_speed * (target_position[2] - current_position[2]),
            ]

            p.resetBasePositionAndOrientation(balloon_id, new_position, [0, 0, 0, 1])

            if all(abs(new_position[i] - target_position[i]) < 0.1 for i in range(3)):
                self.balloon_targets[balloon_id] = [random.uniform(17, 19), random.uniform(-10, 10), random.uniform(0.5, 2)]

    def move_bullets(self):
        """Move bullets forward and check for collisions."""
        for bullet in self.bullets[:]:  # Use slice to allow modification during iteration
            try:
                # Move bullet and get new position
                new_position = bullet.move()
                
                # Check if bullet is out of bounds
                if bullet.is_out_of_bounds():
                    bullet.remove()
                    self.bullets.remove(bullet)
                    self.stats["score"] += self.MISS_PENALTY
                    continue
                
                # Check for collisions with balloons
                for balloon_id in self.balloons[:]:  # Use slice to allow modification during iteration
                    try:
                        balloon_position, _ = p.getBasePositionAndOrientation(balloon_id)
                        distance = np.linalg.norm(np.array(new_position) - np.array(balloon_position))
                        if distance < (BALLOON_RADIUS + BULLET_RADIUS):  # Hit detection radius
                            print(f"Balloon {balloon_id} hit!")
                            
                            # Calculate rewards
                            hit_reward = self.HIT_REWARD
                            
                            # Distance bonus
                            shot_distance = np.linalg.norm(bullet.start_pos - np.array(balloon_position))
                            distance_multiplier = min(shot_distance / 10.0, 2.0)  # Max 2x bonus
                            distance_bonus = self.DISTANCE_BONUS * distance_multiplier
                            
                            # Combo bonus
                            combo_bonus = 0
                            if self.episode_steps - self.last_hit_time <= 10:  # 10 adım içinde vuruş
                                self.consecutive_hits += 1
                                combo_bonus = self.COMBO_BONUS * self.consecutive_hits
                            else:
                                self.consecutive_hits = 1
                            
                            # Update timing
                            self.last_hit_time = self.episode_steps
                            
                            # Total reward for this hit
                            total_hit_reward = hit_reward + distance_bonus + combo_bonus
                            
                            # Update stats
                            self.stats["score"] += total_hit_reward
                            self.stats["balloons_popped"] += 1
                            self.stats["remaining_balloons"] = len(self.balloons) - 1  # -1 because we're about to remove it
                            
                            # Remove balloon and bullet
                            p.removeBody(balloon_id)
                            self.balloons.remove(balloon_id)
                            del self.balloon_targets[balloon_id]
                            bullet.remove()
                            self.bullets.remove(bullet)
                            
                            break
                    except Exception as e:
                        print(f"Error in balloon collision check: {e}")
                        continue
            except Exception as e:
                print(f"Error in bullet movement: {e}")
                try:
                    bullet.remove()
                    self.bullets.remove(bullet)
                except:
                    pass

    def step(self, action):
        total_reward = 0
        done = False
        
        # Update episode step counter
        self.episode_steps += 1
        self.stats["episode_steps"] = self.episode_steps
        
        # Frame skipping: Repeat action for frame_skip steps
        for _ in range(self.frame_skip):
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
            dv = 0.005
            dx = action[0] * dv
            dy = action[1] * dv

            currentOrientation = [p.getJointState(self.pandaUid, 0)[0], p.getJointState(self.pandaUid, 1)[0]]
            newAngles = [currentOrientation[0] + dx, currentOrientation[1] + dy]
            p.setJointMotorControlArray(self.pandaUid, [0, 1], p.POSITION_CONTROL, action[0:-1])

            # Fire bullet if action[2] > 0.5
            if action[2] > 0.5:
                link_state = p.getLinkState(self.pandaUid, 2)
                gun_position = link_state[0]
                orientation_matrix = p.getMatrixFromQuaternion(link_state[1])
                forward_direction = [orientation_matrix[0], orientation_matrix[3], orientation_matrix[6]]
                self.fire_bullet(gun_position, forward_direction)
            
            p.stepSimulation()
            self.move_balloons()
            self.move_bullets()
            
            # Add time penalty
            total_reward += self.TIME_PENALTY
            self.stats["score"] += self.TIME_PENALTY
            
            # Check if episode is done
            done = self.is_done()
            if done:
                break
        
        # Get observation after frame skip
        observation = self.get_observation()
        
        # Episode sonu ekstra ödül/ceza
        if done:
            if len(self.balloons) == 0:  # Tüm balonlar vuruldu
                completion_bonus = 50.0
                total_reward += completion_bonus
                self.stats["score"] += completion_bonus
            elif self.episode_steps >= self.max_episode_steps:  # Zaman aşımı
                timeout_penalty = -20.0
                total_reward += timeout_penalty
                self.stats["score"] += timeout_penalty
        
        # Create info dictionary with episode information
        info = {
            "episode_step": self.episode_steps,
            "balloons_remaining": len(self.balloons),
            "bullets_used": self.stats["bullets_used"],
            "balloons_popped": self.stats["balloons_popped"],
            "score": self.stats["score"]
        }
        
        if done:
            info["episode"] = {
                "r": total_reward,
                "l": self.episode_steps,
                "t": time.time()
            }
        
        return observation, total_reward, done, info
    
    def fire_bullet(self, gun_position, forward_direction):
        """Fire a bullet in the given direction."""
        bullet = Bullet(gun_position, forward_direction, speed=self.bullet_speed)
        self.stats["bullets_used"] += 1
        self.bullets.append(bullet)
    
    def render(self, mode='human'):
        link_state = p.getLinkState(self.pandaUid, 2)
        gun_position = link_state[0]
        gun_orientation = link_state[1]

        orientation_matrix = p.getMatrixFromQuaternion(gun_orientation)
        forward_direction = [orientation_matrix[0], orientation_matrix[3], orientation_matrix[6]]

        laser_length = 50.0
        laser_end_position = [
            gun_position[0] + forward_direction[0] * laser_length,
            gun_position[1] + forward_direction[1] * laser_length,
            gun_position[2] + forward_direction[2] * laser_length,
        ]

        p.addUserDebugLine(
            lineFromXYZ=gun_position,
            lineToXYZ=laser_end_position,
            lineColorRGB=[1, 0, 0],
            lineWidth=2.0,
            lifeTime=1/240,
        )

        # Set camera position and orientation
        camera_position = [-5, 0, 5]  # Stationary position
        camera_target = [10, 0, 0]    # Looking at the center of the world
        up_vector = [1, 0, 0]         # World Y-axis as the up vector

        # Compute view and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=camera_target,
            cameraUpVector=up_vector,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(960) / 720,
            nearVal=0.1,
            farVal=100.0
        )

        # Capture camera image
        (_, _, px, _, _) = p.getCameraImage(
            width=960,
            height=720,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Display stats with episode information
        stats_text = [
            f"Episode: {self.stats['current_episode']}",
            f"Step: {self.stats['episode_steps']}/{self.max_episode_steps}",
            f"Score: {self.stats['score']}", 
            f"Remaining Balloons: {self.stats['remaining_balloons']}", 
            f"Bullets Used: {self.stats['bullets_used']}", 
            f"Balloons Popped: {self.stats['balloons_popped']}"
        ]
        
        for i, text in enumerate(stats_text):
            p.addUserDebugText(
                text, 
                textPosition=[10, -2, 2.8 - i * 0.4], 
                textColorRGB=[0, 0, 0], 
                textSize=1.2, 
                lifeTime=1/240
            )

        if mode == 'rgb_array':
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (720, 960, 4))[:, :, :3]  # Drop alpha channel
            return rgb_array 

    def get_observation(self):
        """Get the current observation as a processed camera image."""
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=self.camera_up
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Get camera image
        (_, _, px, _, _) = p.getCameraImage(
            width=84,
            height=84,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Process image
        rgb_array = np.array(px, dtype=np.float32)
        rgb_array = np.reshape(rgb_array, (84, 84, 4))[:, :, :3]  # Drop alpha channel
        
        # Convert to grayscale
        gray_array = np.dot(rgb_array[..., :3], [0.299, 0.587, 0.114])
        gray_array = gray_array / 255.0  # Normalize to [0,1]
        
        # Update frame stack
        if len(self.frames) == 0:  # Initialize frame stack if empty
            for _ in range(self.frame_stack_size):
                self.frames.append(gray_array)
        else:
            self.frames.pop(0)  # Remove oldest frame
            self.frames.append(gray_array)  # Add new frame
        
        # Stack frames into single observation (CHW format)
        stacked_frames = np.array(self.frames)  # Shape: (4, 84, 84)
        return stacked_frames 