import gym

from stable_baselines3 import PPO
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import gym_2sudoku

env = gym.make('sudoku2-v0')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)

obs = env.reset()
for i in range(160):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    if done:
      print(obs, reward, done, info)
      obs = env.reset()

env.close()

