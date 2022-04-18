
import gym
from gym import spaces
import numpy as np

class Bitflipenv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

    def __init__(self, bit_count=20, max_steps=None, e=0.5):
        super(Bitflipenv, self).__init__()
        self.bit_count = bit_count # length of bit
        # self.e = e # error rate of goal
        
        if max_steps is None:
            self.max_steps = bit_count
        else:
            self.max_steps = max_steps      

        self.observation_space = spaces.Dict({
            'state': spaces.Box(low=0, high=1, shape=(bit_count, )),
            'goal': spaces.Box(low=0, high=1, shape=(bit_count, )),
        })
        self.action_space = spaces.Discrete(bit_count)
        self.reset()

    def step(self, action):
        # action is an integer type in range [0, self.bit_count)
        self.steps += 1
        self.state = np.copy(self.state)
        self.state[action] = self.state[action] ^ 1
        return (self.get_obs(), self.reward(), self.terminate(), None)

    def reset(self):
        # every episode has different goal
        self.steps = 0
        self.state = np.random.randint(2, size = self.bit_count)
        self.goal = np.random.randint(2, size = self.bit_count)

        while (self.state == self.goal).all(): # if goal is same as state, must change.
            self.goal = np.random.randint(2, size = self.bit_count)

        print("Goal : {}\n".format(self.goal))
        return self.get_obs()

    def get_obs(self):
        return {
            'state': self.state,
            'goal': self.goal
        }

    def terminate(self):
        return (self.state == self.goal).all() or self.steps >= self.max_steps
    
    def reward(self, state=None, action=None, goal=None):
        if goal is not None:
            state_ = np.copy(state)
            state_[action] = state_[action] ^ 1
            return 0 if np.array_equal(state_, goal) else -1

        return 0 if np.array_equal(self.state, self.goal) else -1

    def render(self, mode='human'):
        print("Current State: {}".format(self.state))

    def close(self):
        pass