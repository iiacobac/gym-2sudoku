import time

import gym
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import gym_2sudoku



from typing import Callable

from multiprocessing import Process, freeze_support 

from typing import Callable


def main():

    env_id = "sudoku2-v0"

    num_cpu = 64  # Number of processes to use
    # Create the vectorized environment
    print("aca")
    #vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    vec_env = make_vec_env(env_id, n_envs=num_cpu)
    print("aca1")
    model = A2C('MlpPolicy', vec_env, verbose=1)
    #model = PPO("MlpPolicy", vec_env, verbose=1)
    #model = A2C.load("a2c_sudoku")
    env = gym.make("sudoku2-v0")
    #model.set_env(vec_env)
    print("aca2")
    # We create a separate environment for evaluation
    eval_env = gym.make(env_id)
    print("aca3")
    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

    n_timesteps = 1000000

    # Multiprocessed RL Training
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_multi = time.time() - start_time

    model.save("a2c_sudoku")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f'Trained Mean reward: {mean_reward} +/- {std_reward:.2f}')

    print(f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS")
    #import pdb;pdb.set_trace()
    obs = eval_env.reset()
    done = False
    for i in range(160):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        if done:
            obs = eval_env.reset()
            print(obs, reward, done, info)



if __name__ == '__main__':
    freeze_support()
    Process(target=main).start()
