import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    @torch.no_grad()
    def generate_episode(self, time_steps, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded, power_N, soc_N, P_tot = [], [], [], [], [], [], [], [], [], [], []
        EVINFO = self.env.reset()
        COST = 0
        PUNISH = 0
        terminated = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        # if self.args.epsilon_anneal_scale == 'episode':
        #     epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot, power_n, soc_n = [], [], [], [], []
            # print(step, obs[0][-2])
            for agent_id in range(self.n_agents):
                soc = obs[agent_id][0]
                avail_action = self.env.get_avail_agent_actions(step, agent_id, obs[agent_id])
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                power = action - 6
                actions.append(np.int(action))
                power_n.append(power)
                # power_n.append(int(power.detach().numpy()))
                soc_n.append(soc)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            power_N.append(power_n)
            soc_N.append(soc_n)
            reward, cost, punish, p_tot, terminated = self.env.step(actions)
            COST += cost
            PUNISH += punish
            P_tot.append(p_tot)
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            # if time_steps > self.args.buffer_size*self.args.episode_limit and epsilon > self.min_epsilon and step % 2 == 0:
            #     epsilon = epsilon * 0.9998
            #     print(epsilon)
            if time_steps > 1000 and self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(step, agent_id, obs[agent_id])
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        episode = dict(o=o.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       s=s.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       # padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        # if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
        #     self.env.save_replay()
        #     self.env.close()

        return episode, episode_reward, step, EVINFO, power_N, soc_N, COST, PUNISH, P_tot