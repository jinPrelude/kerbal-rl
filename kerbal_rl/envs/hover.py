import time
from typing import Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import krpc

class HoverV0(gym.Env):
    """
    Kerbal Space Program reinforcement learning environment for hovering a vehicle.
    This environment is designed to be compatible with Gymnasium 0.28.0 or later.
    """
    metadata = {'render_modes': ['console']}
    
    def __init__(self,
            render_mode: str = None,
            target_altitude: int = 300,
            max_altitude: int = 500,
            max_speed: int = 200,
            max_step: int = 500,
            max_thrust_ratio: float = 1.0,
            frame_interval: float = 0.02
        ):
        """
        Args:
            render_mode (str): the mode to render the environment. Currently only 'console' is supported.
            target_altitude (int): the target altitude to hover at.
            max_altitude (int): the maximum altitude before the episode is considered done.
            max_speed (int): the maximum speed before the episode is considered done.
            max_step (int): the maximum number of steps before the episode is considered done.
            max_thrust_ratio (float): the maximum thrust ratio to apply to the engine.
        """
        super(HoverV0, self).__init__()

        self.conn = krpc.connect(name='hover')
        self.conn.space_center.quicksave()
        self.vessel = self.conn.space_center.active_vessel
        self.ref_frame = self.conn.space_center.ReferenceFrame.create_hybrid(
            position=self.vessel.orbit.body.reference_frame,
            rotation=self.vessel.surface_reference_frame
        )

        self.target_altitude = target_altitude
        self.max_altitude = max_altitude
        self.max_speed = max_speed
        self.max_step = max_step
        self.max_thrust_ratio = max_thrust_ratio
        self.interval = frame_interval

        self.step_count = None
        self.done = None
        self.reward = None
        
        # Define the action and observation space
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        # Assuming the observation is the [altitude_difference, speed], but could be expanded to include more information
        self.observation_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self) -> Tuple[np.ndarray, dict]:
        self._init_connection()

        self.step_count = 0
        self.done = False
        self.reward = 0

        self._activate_engine()
        # Return the initial observation
        state_dict = self._get_frame_infos()
        obs = self._get_obs(
            current_altitude=state_dict["altitude"],
            vertical_vel=state_dict["vertical_vel"]
        )
        return obs, {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        time.sleep(self.interval)
        self._apply_action(action)
        self.step_count += 1
        state_dict = self._get_frame_infos()
        obs = self._get_obs(
            current_altitude=state_dict["altitude"],
            vertical_vel=state_dict["vertical_vel"]
        )
        self.reward = self._get_reward(
            current_altitude=state_dict["altitude"],
            vertical_vel=state_dict["vertical_vel"]
        )
        self.done = self._check_done(state_dict["altitude"], state_dict["vertical_vel"])
        
        return obs, self.reward, self.done, {}
        
    def render(self, mode='console'):
        if mode == 'console':
            print(f"Step: {self.step_count}, Altitude: {self.vessel.flight().mean_altitude}, Target: {self.target_altitude}, Reward: {self.reward}")

    def close(self):
        # Clean up resources, if necessary
        pass

    def _get_frame_infos(self) -> Dict[str, float]:
        return {
            "altitude": self.vessel.flight().mean_altitude,
            "vertical_vel": self.vessel.flight(self.ref_frame).velocity[0]
        }

    def _get_obs(
            self,
            current_altitude: float,
            vertical_vel: float,
        ) -> np.ndarray:
        altitude_difference = self.target_altitude - current_altitude

        normalized_current_altitude = current_altitude / self.max_altitude
        normalized_target_altitude = self.target_altitude / self.max_altitude
        normalized_altitude_difference = altitude_difference / self.max_altitude
        normalized_speed = vertical_vel / self.max_speed

        obs = np.array(
            [
            normalized_current_altitude,
            normalized_target_altitude,
            normalized_altitude_difference,
            normalized_speed
            ], dtype=np.float32
        )
        return obs

    def _get_reward(
            self,
            current_altitude: float,
            vertical_vel: float
        ) -> float:
        reward = 0

        altitude_difference = self.target_altitude - current_altitude
        normalized_altitude_difference = altitude_difference / self.max_altitude
        normalized_speed = vertical_vel / self.max_speed
        if (current_altitude > self.max_altitude) or (abs(vertical_vel) > self.max_speed):
            reward = -10
        else:
            reward = -0.8 * abs(normalized_altitude_difference) + -0.2 * abs(normalized_speed)
        return reward

    def _check_done(self, current_altitude: float, vertical_vel: float) -> bool:
        if (self.step_count >= self.max_step) or (current_altitude > self.max_altitude) or (abs(vertical_vel) > self.max_speed):
            return True

    def _activate_engine(self):
        self.vessel.control.activate_next_stage()

    def _apply_action(self, action):
        # Normalize action to [0, 1]
        throttle_setting = (action[0] * 0.5 + 0.5) * self.max_thrust_ratio
        self.vessel.control.throttle = float(throttle_setting)
        
    def _init_connection(self):
        self.conn.space_center.quickload()
        self.vessel = self.conn.space_center.active_vessel
        self.vessel.control.rcs = True
        self.vessel.control.sas = True
        self.vessel.control.throttle = 0.0
