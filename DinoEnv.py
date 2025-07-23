import gym
from gym import spaces
import numpy as np
from main_env import GameRL  # Đảm bảo bạn đặt tên đúng file class GameRL

class DinoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DinoEnv, self).__init__()
        self.game = GameRL()
        self.action_space = spaces.Discrete(3)  # 0: idle, 1: jump, 2:
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def reset(self):
        obs = self.game.reset()
        return obs

    def step(self, action):
        obs, reward, done = self.game.step(action)
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

