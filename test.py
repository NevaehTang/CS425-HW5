import matplotlib.pyplot as plt

from ipywidgets import widgets
from IPython.display import display

from matplotlib import animation
# from JSAnimation.IPython_display import display_animation
from time import gmtime, strftime
import random
import cv2
import sys
# from BrainDQN_Nature import *
import numpy as np

import gym


env = gym.make('SpaceInvaders-v0')
env.reset()
actions = env.action_space.n


def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))


action0 = 0  # do nothing
observation0, reward0, terminal, info = env.step(action0)
print("Before processing: " + str(np.array(observation0).shape))
plt.imshow(np.array(observation0))
plt.show()
observation0 = preprocess(observation0)
print("After processing: " + str(np.array(observation0).shape))
plt.imshow(np.array(np.squeeze(observation0)))
plt.show()

# brain.setInitState(observation0)
# brain.currentState = np.squeeze(brain.currentState)