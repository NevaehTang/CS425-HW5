import gym
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from matplotlib import animation
from ipywidgets import widgets
from IPython.display import display
from collections import deque
import cv2
import sys
import random
from DeepRLAgent import DeepRLAgent
env = gym.make('SpaceInvaders-v0')
stack = deque([np.zeros((84,84),dtype=np.int) for i in range(4)], length = 4)
agent = DeepRLAgent(observation_space_dim=4, action_space=env.action_space, exploration_rate=0.5, learning_rate=0.1, discount=0.2,
                               exploration_decay_rate=0.95)
episodes = 100
num_rounds = 1000

for i_episode in range(5):
    observation = env.reset()
    while True:
        t=0
        env.render()  #comment out for faster training!
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    env = gym.make('SpaceInvaders-v0')
    for i_episode in range(episodes):
        eachEpisodeReward = []
        observation = env.reset()
        state,stack = stack(stack, observation, True)
            while True:
                action = agent.act(observation)
                next_observation, reward, done, info = env.step(action)
                eachEpisodeReward.append(reward)
                if done:
                    next_observation = np.zeros((84,84), dtype = np.int)
                    nextstate,  stack = stack(stack, next_observation, False)
                    totalreward = np.sum(eachEpisodeReward)
                    print('Episode: {}'.format(i_episode),
                          'Total reward: {}'.format(totalreward),
                          'Training loss: {:.4f}'.format(agent.self.loss),
                          # 'Explore P: {:.4f}'.format(explore_probability)
                          )
                else:
                    nextstate,stack = stack(stack, next_observation, False)
                    state = nextstate


def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return observation

def stack(stack, observation, newStart):
    frame = preprocess(observation)
    if newStart:
        stack = deque([np.zeros((84, 84), dtype=np.int) for i in range(4)], length=4)
        for i in range (4):
            stack.append(frame)
            state = np.stack(stack, axis=2)
    else:
        stack.append(frame)
        np.stack(stack, axis=2)

    return state, stack