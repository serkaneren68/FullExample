import gym
from gym import spaces
import pybullet as p
import numpy as np
from ..entities.balloon import Balloon
from ..entities.bullet import Bullet
from ..utils.constants import *

class BalloonShooterEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        # Your existing __init__ code, but more organized
        self._setup_simulation()
        self._setup_spaces()
        self._initialize_stats() 