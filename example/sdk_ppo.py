import gym

from stable_baselines3 import PPO
from stable_baselines3 import A2C
import gym_2sudoku
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('sudoku2-v0')

#model.load("a2c_sudoku")

#import pdb;pdb.set_trace()

model = PPO("MlpPolicy", env, verbose=1)
#model = A2C('MlpPolicy', env, verbose=1)


mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

model.learn(total_timesteps=10000)
#                            1000000
model.save("a2c_sudoku_ppo")
#import pdb;pdb.set_trace()
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')


obs = env.reset()
for i in range(160):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    if done:
      print(obs, reward, done, info)
      obs = env.reset()

env.close()

