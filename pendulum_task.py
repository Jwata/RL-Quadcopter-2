import gym

class PendulumTask():
    def __init__(self):
        self.env = gym.make('Pendulum-v0')

        self.state_size = 2
        self.action_size = 1
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

        self.reset()

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        return self.env.env.state, reward, done

    def reset(self):
        self.env.reset()
        return self.env.env.state
