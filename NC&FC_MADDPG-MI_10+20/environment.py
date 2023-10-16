import numpy as np
from gym import spaces
import random
#from pypowertest import power
class MultiAgentEnv():
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 ag_observation_callback=None, st_observation_callback=None):

        self.world = world
        self.agents = self.world.agents
        self.n = len(world.agents)
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.ag_observation_callback = ag_observation_callback
        self.st_observation_callback = st_observation_callback
        self.action_space = []
        self.observation_space = []
        # self.b = 0.2
        for agent in self.agents:
            self.action_space.append(spaces.Box(low=-agent.p_range, high=+agent.p_range, shape=(1,), dtype=np.float32))
            # obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(8,), dtype=np.float32))

    def reset(self):
        obs_N = []
        ag_obs_N = []
        soc_, e_up, e_down, t_stay_ = self.reset_callback(self.world)  #soc_   dim= (num_agent, agent.num_piles, t)

        for i, agent in enumerate(self.agents):
            ag_obs_n = []
            for j in range(agent.n_piles):
                ag_obs_n.append(self._get_ag_obs(agent, soc_[i][j], e_up[i][j], e_down[i][j], t_stay_[i][j]))
            ag_obs_N.append(ag_obs_n)

        st_obs, num_ev = self._get_st_obs()

        for i, agent in enumerate(self.agents):
            obs_n = []
            for j in range(agent.n_piles):
                obs_n.append(np.concatenate((ag_obs_N[i][j], st_obs), axis=-1))
            obs_N.append(obs_n)

        return obs_N, soc_, e_up, e_down, t_stay_, num_ev

    def step(self, action_n, obs_n, soc, e_up, e_down, t_stay):
        ag_obs_N = []
        obs_N = []
        reward_N = []
        rp1_N = []

        self.agents = self.world.agents
        self.stations = self.world.stations

        p_n, soc_, t_stay_ = self.world.step(action_n, obs_n, soc, t_stay)   # agent.state/station.state更新

        reward_dim2 = self._get_reward(p_n)

        for i, agent in enumerate(self.agents):
            rp1_n = []
            for j in range(agent.n_piles):
                rp1_n.append(reward_dim2[1])
            rp1_N.append(rp1_n)

        rc_a_n = (reward_dim2[0])

        for i, agent in enumerate(self.agents):
            reward_n = []
            for j in range(agent.n_piles):
                # print("3", i, j, rc_a_n[i][j])
                reward_n.append((rc_a_n[i][j] / 6 + 2*rp1_N[i][j]*(1-obs_n[i][j][-i-1])))
                # reward_n.append(rc_a_n[i][j] / 6 + rp1_N[i][j]*1.5 + rp2)
            reward_N.append(reward_n)

        for i, agent in enumerate(self.agents):
            agent.state.eme = 0
            agent.state.n_ev = 0

        for station in self.stations:
            station.state.t_cur += 1
            station.state.n_ev = 0
            station.state.eme_tot = 0

        for i, agent in enumerate(self.agents):
            ag_obs_n = []
            for j in range(agent.n_piles):
                ag_obs_n.append(self._get_ag_obs(agent, soc_[i][j], e_up[i][j], e_down[i][j], t_stay_[i][j]))
            ag_obs_N.append(ag_obs_n)
        st_obs, num_ev = self._get_st_obs()

        for i, agent in enumerate(self.agents):
            obs_n = []
            for j in range(agent.n_piles):
                obs_n.append(np.concatenate((ag_obs_N[i][j], st_obs), axis=-1))
            obs_N.append(obs_n)

        # if rp1_N[0][0] != 0:
        #    print(rp1_N, reward_N)
        return obs_N, p_n, reward_N, rc_a_n, rp1_N, num_ev#, rp2_n

    # get observation for a particular agent
    def _get_ag_obs(self, agent, soc_, e_up, e_down, t_stay):
        return self.ag_observation_callback(agent, self.world, soc_, e_up, e_down, t_stay)

    # get observation for a particular agent
    def _get_st_obs(self):
        return self.st_observation_callback(self.world)

    # get reward for a particular agent
    def _get_reward(self, p_n):
        return self.reward_callback(self.world, p_n)

    # set env action for a particular agent
    def _set_action(self, action, agent):
        # print(action[0])
        agent.action.p = action[0]
