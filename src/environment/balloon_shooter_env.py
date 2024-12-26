import gym
from gym import spaces
import numpy as np
import pybullet as p

class BalloonShooterEnv(gym.Env):
    def __init__(self, render_width=84, render_height=84):
        super().__init__()
        
        # Image observation space (84x84 is common in RL, same as Atari)
        # Using grayscale (1 channel) to reduce complexity
        self.render_width = render_width
        self.render_height = render_height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(render_height, render_width, 1),  # (84, 84, 1) for grayscale
            dtype=np.uint8
        )
        
        # Action space remains the same
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )
        
        # Setup camera parameters
        self.camera_distance = 1.5
        self.camera_yaw = 0
        self.camera_pitch = -40
        self.camera_target_position = [0.55, -0.35, 0.2]

    def _get_observation(self):
        """Get camera-based observation."""
        # Compute view matrix
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target_position,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        # Compute projection matrix
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.render_width) / self.render_height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Get camera image
        (_, _, px, _, _) = p.getCameraImage(
            width=self.render_width,
            height=self.render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to grayscale
        rgb_array = np.array(px, dtype=np.uint8)
        gray_array = np.mean(rgb_array[:, :, :3], axis=2).astype(np.uint8)
        
        # Reshape to (H, W, 1) for the observation space
        return gray_array[:, :, None]

    def _image_preprocessing(self, image):
        """Additional preprocessing for the observation."""
        # Normalize to [0,1]
        normalized = image.astype(np.float32) / 255.0
        
        return normalized

    def step(self, action):
        # Execute action
        self._execute_action(action)
        
        # Get new observation (camera image)
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        # Additional info
        info = {
            'balloons_hit': self.balloons_hit,
            'bullets_remaining': self.bullets_remaining,
            'time_elapsed': self.time_elapsed
        }
        
        return observation, reward, done, info 