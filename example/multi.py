import time

import gym
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from typing import Callable

from multiprocessing import Process, freeze_support 

from typing import Callable

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():

	env_id = "CartPole-v1"

	num_cpu = 8  # Number of processes to use
	# Create the vectorized environment
	vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
	#import pdb;pdb.set_trace()

	model = A2C('MlpPolicy', vec_env, verbose=1)
	#model = PPO("MlpPolicy", vec_env, verbose=1)

	# We create a separate environment for evaluation
	eval_env = gym.make(env_id)

	# Random Agent, before training
	mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
	print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

	n_timesteps = 25000

	# Multiprocessed RL Training
	start_time = time.time()
	model.learn(n_timesteps)
	total_time_multi = time.time() - start_time

	mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
	print(f'Trained Mean reward: {mean_reward} +/- {std_reward:.2f}')

	print(f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS")

	# Single Process RL Training
	single_process_model = A2C('MlpPolicy', env_id, verbose=0)

	start_time = time.time()
	single_process_model.learn(n_timesteps)
	total_time_single = time.time() - start_time
	mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
	print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

	print(f"Took {total_time_single:.2f}s for single process version - {n_timesteps / total_time_single:.2f} FPS")

	print("Multiprocessed training is {:.2f}x faster!".format(total_time_single / total_time_multi))



if __name__ == '__main__':
    freeze_support()
    Process(target=main).start()
