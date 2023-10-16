import numpy as np
import os
from common.rollout import RolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import pandas as pd
import time

class Runner:
    def __init__(self, env, args):
        self.env = env
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.episode_rewards = []
        self.save_path = self.args.result_dir + '/' + args.alg
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self):
        self.loss, self.episode_rewards, self.episode_rewards_cost, self.episode_rewards_punish = [], [], [], []
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while time_steps <= self.args.n_steps:
            episodes = []
            for episode_idx in range(self.args.n_episodes):
                episode, episode_reward, steps, _, _, _, reward_cost, reward_punish, _ = self.rolloutWorker.generate_episode(time_steps)
                self.episode_rewards.append(episode_reward*5)
                self.episode_rewards_cost.append(reward_cost)
                self.episode_rewards_punish.append(reward_punish)
                episodes.append(episode)
                time_steps += steps

            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            self.buffer.store_episode(episode_batch)
            if time_steps > 1000 * self.args.episode_limit:
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.loss.append(self.agents.train(mini_batch, train_steps, time_steps))#.cpu().detach().numpy())
                    train_steps += 1

    def evaluate(self):
        EVINFO_ta_N = []
        EVINFO_tl_N = []
        EVINFO_soc_N = []
        POWER_N = []
        SOC_N = []
        COST_N = []
        MAX_P_N = np.zeros(self.args.evaluate_epoch)
        for epoch in range(self.args.evaluate_epoch):
            _, _, _, EVINFO, POWER, SOC, COST, PUNISH, P_tot = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            EVINFO_ta_N.append(EVINFO[0])
            EVINFO_tl_N.append(EVINFO[1])
            EVINFO_soc_N.append(EVINFO[2])
            POWER_N.append(POWER)
            SOC_N.append(SOC)
            COST_N.append(COST)
            MAX_P_N[epoch] = max(P_tot)

        # -------------------------write output---------------------------#
        writer = pd.ExcelWriter(self.args.model_dir + '/'
                                + self.args.alg + '/' + 'EVresult.xlsx')
        df1 = pd.DataFrame(EVINFO_ta_N)
        df2 = pd.DataFrame(EVINFO_tl_N)
        df3 = pd.DataFrame(EVINFO_soc_N)
        df4 = pd.DataFrame(POWER_N)
        df5 = pd.DataFrame(SOC_N)
        df6 = pd.DataFrame(COST_N)
        df7 = pd.DataFrame(MAX_P_N)
        df1.to_excel(writer, sheet_name='info_ta')
        df2.to_excel(writer, sheet_name='info_tl')
        df3.to_excel(writer, sheet_name='info_soc')
        df4.to_excel(writer, sheet_name='power')
        df5.to_excel(writer, sheet_name='SOC')
        df6.to_excel(writer, sheet_name='COST')
        df7.to_excel(writer, sheet_name='MAX_P')
        writer.save()








