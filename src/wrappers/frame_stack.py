from collections import deque
import numpy as np
import gym

class FrameStack(gym.Wrapper):
    """Stack n_frames last frames."""
    
    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        
        # Update observation space to handle stacked frames
        old_shape = env.observation_space.shape
        new_shape = (old_shape[0], old_shape[1], n_frames)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_observation()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        # Concatenate frames along channel dimension
        return np.concatenate(list(self.frames), axis=2) 