#!/usr/bin/python3

#https://stackoverflow.com/questions/45068568/is-it-possible-to-create-a-new-gym-environment-in-openai

import gym
import gym_banana
import baselines.run as r

time_cycles = 2464

env = gym.make('Banana-v0')
env.num_envs = 1
learn = r.get_learn_function("ppo2")
model = learn(
    network='mlp',
    env=env,
    total_timesteps=time_cycles
)

