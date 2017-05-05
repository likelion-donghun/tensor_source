import sys
import tensorflow
import gym
import msvcrt
from gym.envs.registration import register

env = gym.make('FrozenLake-v8')
observation = env.reset()

inkey = mscvrt()

for episode in range(1000):
    env.render()
    action = env.action_sample.sample()
    observation, reward, done, info = env.step(action)
    total_reward = 0

    for t in range(300):

        total_reward += reward

        if done:
            print
            "Reward : " + str(total_reward)
            total_reward = 0
            break