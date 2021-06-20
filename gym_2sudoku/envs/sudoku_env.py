import sys
from contextlib import closing

import numpy as np
from io import StringIO
import random
import gym
from typing import Any, Dict, List, Tuple
from gym import spaces
from gym.utils import seeding
from gym import utils

class Sudoku2Env(gym.Env):
  """
  Defines a 4x4 Sudoku Game 
  """
  metadata = {'render.modes': ['human']}
  gold_set = set([1,2,3,4])

  def __init__(self) -> None:
    self.__version__ = "0.1.0"
    self.is_sudoku_finished = False
    
    # Define what the agent can do
    self.action_space = spaces.MultiDiscrete([4,4,4])
    self.observation_space = spaces.MultiDiscrete([5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5])
    self.square = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    self.count = 0
    self.is_sudoku_finished = False
    self.valid_action_reward = 0
    self.seed()

  def step(self, action: int) -> Tuple[List[int], float, bool, Dict[Any, Any]]:
    """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : List[int]
                an environment-specific object representing your observation of
                the environment.
            reward : float
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : Dict
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
    """
    if self.is_sudoku_finished:
      raise RuntimeError("Episode is done")
    #import pdb;pdb.set_trace()
    self._take_action(action)
    reward = self._get_reward() + self.valid_action_reward
    return np.array(self.square).reshape(16), reward, self.is_sudoku_finished, {}

  def _take_action(self, action: int) -> None:
#    import pdb;pdb.set_trace()
    #if random.random() < 0.1:
    #   action = random.randint(0, 63)
    i = action[0]
    j = action[1]
    v = action[2] + 1
    if self.square[i][j] != 0:
      self.is_sudoku_finished = True
      self.valid_action_reward = 0
    else:
      self.square[i][j] = v
      self.valid_action_reward = 1
    self.count += 1
    if self.count == 16:
      self.is_sudoku_finished = True
    
  def _get_reward(self) -> float:
    for i in range(4):
      if len(Sudoku2Env.gold_set.intersection(set(self.square[i]))) != 4:
        return 0
      if len(Sudoku2Env.gold_set.intersection(set(self.square[:][i]))) != 4:
        return 0
      a = i // 2
      b = i % 2
      if len(Sudoku2Env.gold_set.intersection(set([self.square[2*a][2*b],self.square[2*a+1][2*b],self.square[2*a][2*b+1],self.square[2*a+1][2*b+1]]))) != 4:
        return 0
    return 100

  def reset(self):
    self.is_sudoku_finished = False
    self.square = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    self.count = 0
    self.valid_action_reward = 0
    self.seed()
    return np.array(self.square).reshape(16)

  def render(self, mode='human'):
    for i in range(4):
      sss = ""
      for j in range(4):
        sss += str(self.square[i][j]) + " " 
      print(sss)

  def _get_state(self) -> str:
    sss = "8"
    for i in range(4):
      for j in range(4):
        sss += str(self.square[i][j]) 
    return str(sss)

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


