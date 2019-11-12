import gym
import sys
import os
import numpy as np
from tqdm import trange
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

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class BQNAgent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space

        self.flatten = Flatten()

        self.hid = nn.Linear(len(self.action_space.low) + len(self.observation_space.low), 64)
        self.hid2 = nn.Linear(64, 16)
        self.hid3 = nn.Linear(16, 1)

    def forward(self, states_t, actions_t):
        """
        TODO update
        takes agent's observation (tensor) and actions (tensor), returns qvalues (tensor)
        :param state_t: shape = [batch_size, 8]
        :param actions_t: shape = [batch_size, 2]
        """
        # Use your network to compute qvalues for given state

        # state_t = self.flatten(state_t) # if we have more complex state (image for example)
        x = torch.cat((states_t, actions_t), dim=1)

        x = F.relu(self.hid(x))
        x = F.relu(self.hid2(x))
        x = self.hid3(x)
        qvalue = x

        assert qvalue.requires_grad, "qvalue must be a torch tensor with grad"

        return qvalue

    def get_action(self, state, greedy=False):
        # stochastic exploration with randomly generated actions, more samples->more determenistic prediction
        # todo: for greedy use backprop to optimize input for the best sample
        RAND_SAMPLES = 10
        #actions = torch.rand((RAND_SAMPLES, len(self.action_space.low)))
        #actions = torch.zeros((RAND_SAMPLES, len(self.action_space.low)))
        # todo: normal fit to environment constraints for action space
        action_array = []
        for i in range(RAND_SAMPLES):
            action_array.append(self.action_space.sample())

        model_device = next(self.parameters()).device
        actions = torch.tensor(action_array, device=model_device, dtype=torch.float)

        # todo shitty code RAND_SAMPLES
        states = [state, state, state, state, state, state, state, state, state, state]
        states = torch.tensor(states, device=model_device, dtype=torch.float)

        qvalues = self.forward(states, actions)

        action = actions[qvalues.argmax()].data.cpu().numpy()

        return action

class DPDQN1:

    def __init__(self, env, verbose=0,
                 replay_size=5000, lr=1e-4, batch_size=16, total_steps=3 * 10 ** 6,
                 decay_steps = 10 ** 6, refresh_target_network_freq = 3000, max_grad_norm=50,
                 gamma=0.99, warm_up=2000,
                 actions_per_sampling = 10, hidden_size1 = 64, hidden_size2=16,
                 opt_steps_per_step=2):
        self.env = env
        self.verbose = verbose
        self.opt_steps_per_step=opt_steps_per_step
        self.gamma = gamma
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.decay_steps = decay_steps
        self.refresh_target_network_freq = refresh_target_network_freq
        self.max_grad_norm = max_grad_norm
        self.warm_up = warm_up
        self.replay_size = replay_size
        self.lr = lr

        self.prepare_network()



    def prepare_network(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.agent = BQNAgent(self.env.observation_space, self.env.action_space).to(self.device)
        self.target_network = BQNAgent(self.env.observation_space, self.env.action_space).to(self.device)

        self.exp_replay = ReplayBuffer(self.replay_size)

        self.opt = torch.optim.Adam(self.agent.parameters(), lr=self.lr)


    def evaluate(self, n_games=1, greedy=False, t_max=10000):
        rewards = []
        for _ in range(n_games):
            s = self.env.reset()
            reward = 0
            for _ in range(t_max):
                action = self.predict(s, greedy=True)[0]
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

        total_timesteps=total_timesteps-self.warm_up
        _, state = self.play_and_record(state, n_steps=self.warm_up)

        state = self.env.reset()
        for step in trange(total_timesteps + 1):
            _, state = self.play_and_record(state)

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

            if step % eval_freq == 0:
                mean_rw_history.append(self.evaluate(
                    n_games=3, greedy=True, t_max=1000)
                )

                clear_output(True)

                plt.figure(figsize=[16, 9])
                plt.subplot(2, 2, 1)
                plt.title("Mean reward per episode")
                plt.plot(mean_rw_history)
                plt.grid()

                assert not np.isnan(td_loss_history[-1])
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


    def predict(self, obs, greedy=False):
        action = self.agent.get_action(obs, greedy=greedy)
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


    def compute_td_loss(self, states, actions, rewards, next_states_np, is_done, ):
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

    def play_and_record(self, initial_state, n_steps=1):
        s = initial_state
        if self.env.needs_reset:
            s = self.env.reset()
        sum_rewards = 0

        for t in range(n_steps):
            a = self.predict(s)[0]
            next_s, r, done, _ = self.env.step(a)

            self.exp_replay.add(s, a, r, next_s, done)

            sum_rewards += r
            s = next_s

            if done:
                s = self.env.reset()

        return sum_rewards, s

