import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_z=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 

        self.state_size = len(self.get_state())
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_z = target_z

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities

        current_z = self.sim.pose[2]
        d = abs(current_z - self.target_z)
        reward = -d

        if current_z > self.target_z:
            done = True
            reward += 10.
        elif done == True:
            reward -= 10. # penalty for timeout and boundary over

        next_state = self.get_state()
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        return self.get_state()

    def get_state(self):
        return np.array(list(self.sim.pose) + list(self.sim.v) + list(self.sim.angular_v))
