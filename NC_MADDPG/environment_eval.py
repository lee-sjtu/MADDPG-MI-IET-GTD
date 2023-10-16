import numpy as np
from gym import spaces
import random

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
        for agent in self.agents:
            self.action_space.append(spaces.Box(low=-agent.p_range, high=+agent.p_range, shape=(1,), dtype=np.float32))   # p_range -1/1
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(7,), dtype=np.float32))

    def reset(self, num_EVs_per_agent):
        obs_N = []
        ag_obs_n = []
        st_obs = []
        soc_, e_up, e_down, t_stay_, t_arrive_n, t_leave_n, e_arrive_n = self.reset_callback(self.world, num_EVs_per_agent)

        for i in range(num_EVs_per_agent):
            for j, agent in enumerate(self.agents):
                ag_obs_n.append(self._get_ag_obs(agent, soc_[i][j], e_up[i][j], e_down[i][j], t_stay_[i][j]))
        for i, agent in enumerate(self.agents):
            st_obs.append(self._get_st_obs(i, num_EVs_per_agent))
        for i in range(num_EVs_per_agent):
            obs_n = []
            for j, agent in enumerate(self.agents):
                obs_n.append(np.concatenate((ag_obs_n[i*len(self.agents)+j], st_obs[j]), axis=-1))
            obs_N.append(obs_n)
        return obs_N, soc_, e_up, e_down, t_stay_, t_arrive_n, t_leave_n, e_arrive_n

    def step(self, action_n, obs_n, num_EVs_per_agent, soc, e_up, e_down, t_stay):
        ag_obs_n = []
        obs_N = []
        eme = 0
        st_obs = []

        self.agents = self.world.agents
        self.stations = self.world.stations

        p_n, soc_, t_stay_ = self.world.step(action_n, obs_n, soc, t_stay)   # agent.state/station.state更新
        cost = self._get_reward(p_n)
        for i, agent in enumerate(self.agents):
            eme += agent.state.eme

        for i, agent in enumerate(self.agents):
            agent.state.eme = 0
            agent.state.n_ev = 0

        for station in self.stations:
            station.state.t_cur += 1
            station.state.n_ev = 0
            station.state.eme_tot = 0
            # station.state.p_pv_ = station.state.p_pv
            # station.state.p_pv = random.uniform(
            #     0.9 * station.state.pre_p_pv[station.state.t_cur],
            #     1.1 * station.state.pre_p_pv[station.state.t_cur])
            # station.state.p_wt_ = station.state.p_wt
            # station.state.p_wt = random.uniform(
            #     0.9 * station.state.pre_p_wt[station.state.t_cur],
            #     1.1 * station.state.pre_p_wt[station.state.t_cur])
            # station.state.p_load_ = station.state.p_load
            # station.state.p_load = random.uniform(
            #     0.9 * station.state.pre_p_load[station.state.t_cur],
            #     1.1 * station.state.pre_p_load[station.state.t_cur])

        for i in range(num_EVs_per_agent):
            for j, agent in enumerate(self.agents):
                ag_obs_n.append(self._get_ag_obs(agent, soc_[i][j], e_up[i][j], e_down[i][j], t_stay_[i][j]))
        for i, agent in enumerate(self.agents):
            st_obs.append(self._get_st_obs(i, num_EVs_per_agent))
        for i in range(num_EVs_per_agent):
            obs_n = []
            for j, agent in enumerate(self.agents):
                obs_n.append(np.concatenate((ag_obs_n[i*len(self.agents)+j], st_obs[j]), axis=-1))
            obs_N.append(obs_n)
        return obs_N, p_n, cost

    # get observation for a particular agent
    def _get_ag_obs(self, agent, soc_, e_up, e_down, t_stay):
        return self.ag_observation_callback(agent, self.world, soc_, e_up, e_down, t_stay)

    # get observation for a particular agent
    def _get_st_obs(self, agent_index, num_EVs_per_agent):
        return self.st_observation_callback(self.world, agent_index, num_EVs_per_agent)

    def _get_reward(self, p_n):
        return self.reward_callback(self.world, p_n)
    # set env action for a particular agent
    def _set_action(self, action, agent):
        agent.action.p = action[0]
