import gym
import numpy as np

class PendulumTask():
    def __init__(self):
        self.env = gym.make('Pendulum-v0')
        self.action_repeat = 3
        self.state_size = self.action_repeat * 1
        self.action_size = 1
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

        self.reset()

    def step(self, action):
        total_reward = 0
        state_all = []
        for _ in range(self.action_repeat):
            _, reward, done, _ = self.env.step(action)
            total_reward += reward
            state_all.append(self.get_current_state())
        next_state = np.concatenate(state_all)
        return next_state, total_reward, done

    def reset(self):
        self.env.reset()
        state = np.concatenate([self.get_current_state()] * self.action_repeat) 
        return state

    def get_current_state(self):
        return self.env.env.state[:1]

