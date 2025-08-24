from pdb import set_trace as T

import gymnasium as gym 
import functools
from collections import deque
from gymnasium.spaces import Box
from gymnasium.wrappers.frame_stack import LazyFrames

from super_mario_land_rl.mario_env import SuperMarioLandEnv
import numpy as np
import pufferlib.emulation
import pufferlib.postprocess
import time

def env_creator(name="super_mario_land"):
    return functools.partial(make, name)


def make(
    name,
    headless: bool = True,
    state_path=None,
    buf=None,
    render_mode="rgb_array",
    framestack=8,
):
    """Super Mario Land"""
    env = SuperMarioLandEnv(render_mode=render_mode)
    env = RenderWrapper(env)
    env = pufferlib.postprocess.EpisodeStats(env)

    if framestack > 1:
        env = gym.wrappers.FrameStack(env, framestack)

    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env

    @property
    def render_mode(self):
        return "human"

    def render(self):
        return self.env.screen.ndarray

class FrameStackWrapper(gym.wrappers.FrameStack):
    """Unfinished Hack to make FrameStackWrapper work with dict observation spaces"""
    
    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
        lz4_compress: bool = False,
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, num_stack=num_stack, lz4_compress=lz4_compress
        )
        gym.ObservationWrapper.__init__(self, env)

        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        low = self.env.observation_space['screen'].low
        high = self.env.observation_space['screen'].high
        dtype = self.env.observation_space['screen'].dtype
        self.observation_space = Box(
            low=low, high=high, dtype=dtype
        )
    
    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, info = self.env.reset(**kwargs)

        [self.frames.append(obs) for _ in range(self.num_stack)]

        return self.observation(None), info
