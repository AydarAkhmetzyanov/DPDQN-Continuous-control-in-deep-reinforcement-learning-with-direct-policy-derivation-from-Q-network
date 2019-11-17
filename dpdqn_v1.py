import gym
import sys
import os
import numpy as np

from IPython.display import clear_output
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from replay_buffer import ReplayBuffer
sys.setrecursionlimit(10000)
import utils
import copy
from ray import tune
from ray.tune import track
import math
from tqdm import trange

def remap( x, oMin, oMax, nMin, nMax ):
    reverseInput = False
    oldMin = min( oMin, oMax )
    oldMax = max( oMin, oMax )
    if not oldMin == oMin:
        reverseInput = True

    reverseOutput = False
    newMin = min( nMin, nMax )
    newMax = max( nMin, nMax )
    if not newMin == nMin :
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return math.floor(result)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ActionLossFunc(torch.nn.Module):

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-y)
        return loss

class BQNAgent(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_size1, hidden_size2, greedy_lr, greedy_opt_steps):
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space

        self.flatten = Flatten()

        self.action = torch.ones((1, len(self.action_space.low)), requires_grad=True)
        self.action.requires_grad = True
        self.greedy_lr=greedy_lr
        self.greedy_opt_steps=greedy_opt_steps

        # self.statedim = 1
        # for x in self.observation_space.shape:
        #     self.statedim*=x
        #self.hid = nn.Linear(len(self.action_space.low) + self.statedim, hidden_size1)
        self.hid = nn.Linear(len(self.action_space.low) + len(self.observation_space.low), hidden_size1)
        self.hid2 = nn.Linear(hidden_size1, hidden_size2)
        self.hid3 = nn.Linear(hidden_size2, 1)

    def forward(self, states_t, actions_t):
        #states_t = self.flatten(states_t) # if we have more complex state (image for example)
        x = torch.cat((states_t, actions_t), dim=1)

        x = F.relu(self.hid(x))
        x = F.relu(self.hid2(x))
        x = self.hid3(x)
        qvalue = x

        return qvalue

    def get_action(self, state, greedy=False, num_action_samples=10):

        action_array = []
        states = []

        for i in range(num_action_samples):
            action_array.append(self.action_space.sample())
            states.append(state)

        model_device = next(self.parameters()).device
        actions = torch.tensor(action_array, device=model_device, dtype=torch.float)
        states = torch.tensor(states, device=model_device, dtype=torch.float)

        q_values = self.forward(states, actions)

        best_action = actions[q_values.argmax()]

        if greedy==True:
            loss_fn = ActionLossFunc()
            for param in self.parameters():
                param.requires_grad = False
            state_tensor = torch.tensor([state], device=model_device, dtype=torch.float)

            self.action = torch.stack([best_action])
            self.action.requires_grad = True
            self.action.data.clamp_(min=self.action_space.low[0], max=self.action_space.high[0])

            for int_opt in range(self.greedy_opt_steps):
                q_value = self.forward(state_tensor, self.action)
                loss = loss_fn(q_value)
                loss.backward()

                with torch.no_grad():
                    self.action -= self.greedy_lr * self.action.grad


                self.action.data.clamp_(min=self.action_space.low[0], max=self.action_space.high[0])
                self.action.grad.zero_()

            self.action.requires_grad = False
            for param in self.parameters():
                param.requires_grad = True

            best_action=self.action[0]


        action = best_action.data.cpu().numpy()
        return action

class DPDQN1:
    def __init__(self, env, verbose=0, ray_tune=False,
                 replay_size=5000, lr=1e-4, batch_size=16,
                 refresh_target_network_freq = 3000, max_grad_norm=50,
                 gamma=0.99, hidden_size1 = 64, hidden_size2=16, opt_steps_per_step=2,
                 num_action_samples=9, num_action_samples_final=11,
                 greedy_lr=0.05, greedy_opt_steps=0, warm_up=2000):
        self.env = env
        self.verbose = verbose
        self.ray_tune = ray_tune

        self.opt_steps_per_step=opt_steps_per_step
        self.gamma = gamma
        self.batch_size = batch_size
        self.refresh_target_network_freq = refresh_target_network_freq
        self.max_grad_norm = max_grad_norm
        self.warm_up = warm_up
        self.replay_size = replay_size
        self.lr = lr
        self.num_action_samples = num_action_samples
        self.num_action_samples_final = num_action_samples_final

        self.hidden_size1=hidden_size1
        self.hidden_size2 = hidden_size2

        self.greedy_lr=greedy_lr
        self.greedy_opt_steps=greedy_opt_steps
        self.prepare_network()



    def prepare_network(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.agent = BQNAgent(self.env.observation_space, self.env.action_space, self.hidden_size1, self.hidden_size2, self.greedy_lr, self.greedy_opt_steps).to(self.device)
        self.target_network = BQNAgent(self.env.observation_space, self.env.action_space, self.hidden_size1, self.hidden_size2, self.greedy_lr, self.greedy_opt_steps).to(self.device)

        self.exp_replay = ReplayBuffer(self.replay_size)

        self.opt = torch.optim.Adam(self.agent.parameters(), lr=self.lr)


    def evaluate(self, n_games=1, greedy=True, t_max=10000):
        rewards = []
        for _ in range(n_games):
            s = self.env.reset()
            reward = 0
            for _ in range(t_max):
                action = self.predict(s, greedy=greedy, num_action_samples=30)[0]
                s, r, done, _ = self.env.step(action)
                reward += r
                if done:
                    break

            rewards.append(reward)
        return np.mean(rewards)

    def learn(self, total_timesteps=1e5):
        loss_freq = 50
        eval_freq = total_timesteps/100

        mean_rw_history = []
        td_loss_history = []
        grad_norm_history = []
        initial_state_v_history = []

        self.target_network.load_state_dict(self.agent.state_dict())
        state = self.env.reset()

        _, state = self.play_and_record(state, n_steps=self.warm_up, current_exploration_actions=3)

        state = self.env.reset()

        if self.verbose==0:
            trange = range
        else:
            from tqdm import trange

        for step in trange(total_timesteps-self.warm_up):
            current_exploration_actions = remap(step, 0, total_timesteps-self.warm_up+2, self.num_action_samples, self.num_action_samples_final)

            _, state = self.play_and_record(state, current_exploration_actions=current_exploration_actions)

            s_, a_, r_, next_s_, done_ = self.exp_replay.sample(self.batch_size)

            for opt_step in range(self.opt_steps_per_step):
                loss = self.compute_td_loss(s_, a_, r_, next_s_, done_)
                self.opt.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.opt.step()


            if step % loss_freq == 0:
                td_loss_history.append(loss.data.cpu().item())
                grad_norm_history.append(grad_norm)

            if step % self.refresh_target_network_freq == 0:
                self.target_network.load_state_dict(self.agent.state_dict())

            #skip for verbose
            if self.verbose and step % eval_freq == 0:
                mean_rw_history.append(self.evaluate(n_games=3, greedy=True, t_max=10000))
                if self.ray_tune:
                    tune.track.log(mean_reward=mean_rw_history[-1])

            if self.verbose and step % eval_freq == 0:
                clear_output(True)

                plt.figure(figsize=[16, 9])
                plt.subplot(2, 2, 1)
                plt.title("Mean reward per episode")
                plt.plot(mean_rw_history)
                plt.grid()

                plt.subplot(2, 2, 2)
                plt.title("TD loss history (smoothened)")
                plt.plot(utils.smoothen(td_loss_history))
                plt.grid()

                plt.subplot(2, 2, 3)
                plt.title("Initial state V")
                plt.plot(initial_state_v_history)
                plt.grid()

                plt.subplot(2, 2, 4)
                plt.title("Grad norm history (smoothened)")
                plt.plot(utils.smoothen(grad_norm_history))
                plt.grid()

                plt.show()


    def predict(self, obs, greedy=False, num_action_samples=10):
        action = self.agent.get_action(obs, greedy=greedy, num_action_samples=num_action_samples)
        return action, None

    def save(self, filename):
        d = copy.copy(self.__dict__)
        del d["agent"]
        del d["target_network"]
        del d["device"]
        del d["exp_replay"]
        del d["opt"]
        del d["env"]
        pickle.dump(d, open(filename+".p", "wb"), pickle.HIGHEST_PROTOCOL)
        torch.save(self.agent, filename+".agent")


    def load(filename, env):
        model = DPDQN1(env)
        model.__dict__.update(pickle.load(open(filename+".p", "rb")))
        model.prepare_network()
        model.agent = torch.load(filename+".agent")
        return model


    def compute_td_loss(self, states, actions, rewards, next_states_np, is_done):
        # for param in self.target_network.parameters():
        #     param.requires_grad = False
        # for param in self.agent.parameters():
        #     param.requires_grad = True

        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.float)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_states = torch.tensor(next_states_np, device=self.device, dtype=torch.float)

        is_done = torch.tensor(
            is_done.astype('float32'),
            device=self.device,
            dtype=torch.float
        )
        is_not_done = 1 - is_done

        predicted_qvalues = self.agent(states, actions).squeeze()

        next_actions = [self.target_network.get_action(next_state) for next_state in next_states_np]
        next_actions = torch.tensor(next_actions, device=self.device, dtype=torch.float)
        predicted_next_qvalues = self.target_network(next_states, next_actions).squeeze()

        target_qvalues_for_actions = rewards + is_not_done * self.gamma * predicted_next_qvalues

        loss = torch.mean((predicted_qvalues - target_qvalues_for_actions.detach()) ** 2)

        return loss

    def play_and_record(self, initial_state, n_steps=1, current_exploration_actions=10):
        s = initial_state
        if hasattr(self.env, "needs_reset"):
            if self.env.needs_reset:
                s = self.env.reset()
        sum_rewards = 0

        for t in range(n_steps):
            a = self.predict(s, num_action_samples=current_exploration_actions, greedy=False)[0]
            next_s, r, done, _ = self.env.step(a)

            self.exp_replay.add(s, a, r, next_s, done)

            sum_rewards += r
            s = next_s

            if done:
                s = self.env.reset()

        return sum_rewards, s

