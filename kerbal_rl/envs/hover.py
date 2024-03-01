import gymnasium as gym
from gymnasium import spaces
import numpy as np
import krpc
import time

class HoverV0(gym.Env):
    """
    Kerbal Space Program reinforcement learning environment for hovering a vehicle.
    This environment is designed to be compatible with Gymnasium 0.28.0 or later.
    """
    
    metadata = {'render.modes': ['console']}
    
    def __init__(self, max_altitude=500, max_step=100, interval=0.085):
        super(HoverV0, self).__init__()

        self.initial_throttle = 0.0  # Initial throttle setting
        self.init_connection()
        self.conn = krpc.connect(name='hover')
        self.vessel = self.conn.space_center.active_vessel
        self.ref_frame = self.conn.space_center.ReferenceFrame.create_hybrid(
            position=self.vessel.orbit.body.reference_frame,
            rotation=self.vessel.surface_reference_frame
        )

        self.max_altitude = max_altitude
        self.max_step = max_step
        self.interval = interval
        self.step_count = 0
        self.done = False
        self.reward = 0
        self.target_altitude = 300
        

        # Define the action and observation space
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        # Assuming the observation is the [altitude_difference, speed], but could be expanded to include more information
        self.observation_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

    def init_connection(self):
        self.conn = krpc.connect(name='hover')
        self.conn.space_center.quicksave()
        self.vessel = self.conn.space_center.active_vessel
        self.vessel.control.sas = True
        self.vessel.control.throttle = 0.0
        

    def get_obs(self):
        current_altitude = self.vessel.flight().mean_altitude
        altitude_difference = self.target_altitude - current_altitude
        vertical_vel = self.vessel.flight(self.ref_frame).velocity[0]

        normalized_current_altitude = current_altitude / self.max_altitude
        normalized_target_altitude = self.target_altitude / self.max_altitude
        normalized_altitude_difference = altitude_difference / self.max_altitude
        normalized_speed = vertical_vel / 200

        obs = np.array([normalized_current_altitude, normalized_target_altitude, normalized_altitude_difference, normalized_speed], dtype=np.float32)
        reward = -0.8 * abs(normalized_altitude_difference) + -0.2 * abs(normalized_speed)

        return obs, reward, current_altitude

    def activate_engine(self):
        self.vessel.control.activate_next_stage()

    def reset(self):
        self.conn.space_center.quickload()
        self.vessel.control.sas = True
        self.vessel.control.rcs = True
        self.vessel.control.throttle = self.initial_throttle
        # self.target_altitude = np.random.randint(100, self.max_altitude)
        self.step_count = 0
        self.done = False
        self.reward = 0
        self.activate_engine()
        # Return the initial observation
        obs, self.reward, _ = self.get_obs()
        return obs

    def get_reward(self, alitude_difference: float, speed: float) -> float:
        normalized_altitude_difference = alitude_difference / self.max_altitude
        normalized_speed = speed / 500
        reward = 0.8 * normalized_altitude_difference + 0.2 * normalized_speed
        return reward

    def step(self, action):
        time.sleep(self.interval)
        self._apply_action(action)
        self.step_count += 1

        obs, self.reward, current_altitude = self.get_obs()
        if (self.step_count >= self.max_step):
            self.done = True
        if current_altitude > self.max_altitude:
            self.reward = -10
            self.done = True
        
        return obs, self.reward, self.done, {}

    def _apply_action(self, action):
        # Normalize action to [0, 1]
        throttle_setting = (action[0] * 0.5 + 0.5)*0.5
        self.vessel.control.throttle = float(throttle_setting)
        
    def render(self, mode='console'):
        if mode == 'console':
            print(f"Step: {self.step_count}, Altitude: {self.vessel.flight().mean_altitude}, Target: {self.target_altitude}, Reward: {self.reward}")

    def close(self):
        # Clean up resources, if necessary
        pass
