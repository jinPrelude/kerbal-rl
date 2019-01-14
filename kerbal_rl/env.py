import krpc
import time
import numpy as np
"""
Kerbal Space Program reinforcement learning environment
Author : Uijin Jung
github : https://github.com/jinPrelude/kerbal-rl
"""

# hover_v0 returns continuous reward
class hover_v0:
    def __init__(self, sas=True, max_altitude = 500, max_step=100, interval = 0.085):
        self.conn = krpc.connect(name='hover')
        self.vessel = self.conn.space_center.active_vessel
        self.step_count = 0
        self.done = False
        self.reward = 0
        self.interval = interval
        self.max_altitude = max_altitude
        self.observation_space = 1

        # Action space : 0.0 ~ 1.0 (Thrust ratio)
        self.action_space = 1
        self.action_max = 1.
        self.action_min = -1.

        self.initial_throttle = self.action_min

        # Initializing
        self.sas = sas
        self.target_altitude = 100
        self.relative_goal = self.target_altitude
        self.max_step = max_step

    # reset() returns target_altitude, while step() don't.
    def reset(self):

        # Quicksave initial state
        self.conn.space_center.quicksave()

        # Initialize sas
        self.vessel.control.sas = self.sas

        # Initialize throttle
        self.vessel.control.throttle = self.initial_throttle

        # Set target altitude
        self.target_altitude = np.random.randint(100, self.max_altitude)

        print('Target altitude : ', self.target_altitude)
        self.relative_goal -= self.vessel.flight().mean_altitude

        self.step_count = 0
        self.done = False
        self.reward = 0

        # Launch
        self.vessel.control.activate_next_stage()

        return [self.vessel, self.vessel.flight(), self.relative_goal]

    def step(self, action):
        self.decision(action)

        if self.step_count >= self.max_step :
            # Revert to launch pad when the step is reached to the max_step.
            self.done = True
            self.conn.space_center.quickload()
        else :
            self.step_count += 1

            # Return the reward is given proportion to the distance between the target altitude & current altitude
            # and inverse proportion to the speed of the vehicle.
            self.reward = -0.6 * abs(self.vessel.flight().mean_altitude - self.target_altitude) + \
                          -0.4 * abs(self.vessel.flight().speed)

        self.relative_goal = self.target_altitude - self.vessel.flight().mean_altitude

        time.sleep(self.interval)

        # obs, reward, done
        return [self.vessel, self.vessel.flight(), self.relative_goal], self.reward, self.done, []

    # Return action
    def decision(self, action):
        action = action[0] * 0.5 + 0.5
        self.vessel.control.throttle = float(action)

    def sample_action_space(self):
        return np.random.uniform(-1, 1, 1)

# hover_v1 returns sparse reward
class hover_v1:
    def __init__(self, sas=True, max_altitude = 1000, max_step=100, epsilon=1, interval = 0.085):
        self.conn = krpc.connect(name='hover')
        self.vessel = self.conn.space_center.active_vessel
        self.step_count = 0
        self.done = False
        self.reward = 0
        self.interval = interval
        self.max_altitude = max_altitude
        self.observation_space = 1


        # error tolerance(meter).
        # If epsilon is 1 and target_altitude is 100m, the error tolerance is between 99m and 101m
        self.epsilon = epsilon

        # Action space : 0.0 ~ 1.0 (Thrust ratio)
        self.action_space = 1
        self.action_max = 1.
        self.action_min = 0.0

        self.initial_throttle = self.action_min

        # Initializing
        self.sas = sas
        self.target_altitude = 100
        self.max_step = max_step

    def reset(self):

        # Quicksave initial state
        self.conn.space_center.quicksave()

        # Initialize sas
        self.vessel.control.sas = self.sas

        # Initialize throttle
        self.vessel.control.throttle = self.initial_throttle

        # Set target altitude
        self.target_altitude = np.random.randint(100, self.max_altitude)

        print('Target altitude : ', self.target_altitude)
        self.step_count = 0
        self.done = False
        self.reward = 0

        # Launch
        self.vessel.control.activate_next_stage()

        return self.vessel.thrust, self.vessel.mass, self.target_altitude

    def step(self, action):
        self.decision(action)

        if self.step_count >= self.max_step :
            # Revert to launch pad
            self.done = True
            self.conn.space_center.quickload()
        else :
            self.step_count += 1

            # Return the reward if the current altitude is between the error tolerance and the speed is 0.
            if (self.vessel.flight().mean_altitude <= self.target_altitude + self.epsilon) and \
                    (self.vessel.flight().mean_altitude >= self.target_altitude - self.epsilon) and \
                    abs(self.vessel.flight().speed) < 0.01 :
                self.reward = 1
            else : self.reward = 0

        time.sleep(self.interval)

        # obs, reward, done
        return (self.vessel.thrust, self.vessel.mass, self.target_altitude), self.reward, self.done

    # Return action
    def decision(self, action):
        self.vessel.control.throttle = float(action[0])

envs = {
    'hover_v0' : hover_v0,
    'hover_v1' : hover_v1
}

def make(id) :
    return envs[id]