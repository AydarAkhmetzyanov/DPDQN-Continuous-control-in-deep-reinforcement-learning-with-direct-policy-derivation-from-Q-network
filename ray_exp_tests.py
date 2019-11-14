import torch.optim as optim
from ray import tune
from ray.tune import track
from ray.tune.schedulers import ASHAScheduler
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test

from dpdqn_v1 import DPDQN1
import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from shutil import copyfile
from utils import *
import pickle
import pandas as pd

time_steps = 1e4 #testrun
envname = "BipedalWalker-v2"
env = gym.make(envname)
model = DPDQN1(env, verbose=1) #params here
model.learn(total_timesteps=int(time_steps))

