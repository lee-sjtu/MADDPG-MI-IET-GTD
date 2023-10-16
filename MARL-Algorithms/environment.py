import numpy as np
from gym import spaces
from scenarios.one_EV_station import Scenario
import math
import random
class MultiAgentEnv():
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None):
        self.ev_scenario = Scenario()
        self.world = world
        self.n_actions = 13
        self.stations = self.world.stations
        self.world_agents = self.world.agents
        self.n_agents = len(world.agents)
        self.reset_callback = reset_callback
        self.action_space = []
        self.observation_space = []
        # self.b = 0.2
        # self.alpha = 0.3
        for i in range(len(self.world_agents)):
            self.action_space.append(spaces.Discrete(13))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(7,), dtype=np.float32))

    def reset(self):
        t_arrive_n = []
        t_leave_n = []
        e_arrive_n = []
        for i, agent in enumerate(self.world_agents):
            t_arrive, t_leave, e_arrive = self.ev_scenario.ev_info(agent)     # 随机产生n组接入EV数据
            agent.state.soc, agent.state.e_up, agent.state.e_down, agent.state.t_stay = \
                self.ev_scenario.agents_state(t_arrive, t_leave, e_arrive, agent, self.world)   # 传回了agent.state.soc/e_up/e_down/t_stay
            t_arrive_n.append(t_arrive)
            t_leave_n.append(t_leave)
            e_arrive_n.append(e_arrive)
        self.stations = self.ev_scenario.station_state(self.world)
        EVinfo = [t_arrive_n, t_leave_n, e_arrive_n]
        # print(EVinfo)
        return EVinfo

    def step(self, action_n):
        terminated = False
        p_tot = 0
        self.step_1(action_n)   # agent.state/station.state更新
        reward, cost, punish = self._get_reward()
        for station in self.stations:
            station.state.t_cur += 1
            p_tot = station.state.p_tot
            if station.state.t_cur == 22:
                terminated = True
            # print(station.state.t_cur, terminated)
        return reward, cost, punish, p_tot, terminated

    def _get_st_obs(self):
        for station in self.stations:
            t = station.state.t_cur
            # print(t)
            sta_state = []
            for i, agent in enumerate(self.world_agents):
                if station.state.n_ev == 0:
                    eme_agent = 0
                # elif station.state.eme_tot == 0:
                #     eme_agent = 0
                else:
                #     eme_agent = agent.state.eme / station.state.eme_tot
                    eme_agent = station.state.eme_tot / station.state.n_ev
                sta_state.append([[t / 24] + [station.state.price_b[t]] + [eme_agent]][0])
            return sta_state

    def _get_reward(self):
        for station in self.world.stations:
            r_c = -1 * station.state.p_tot * station.state.price_b[station.state.t_cur]
            if station.p_ex_max < abs(station.state.p_tot):
                rew_punish1 = (station.p_ex_max - abs(station.state.p_tot))
            else:
                rew_punish1 = 0
            reward = (r_c/6/5 + rew_punish1/6)/2
            # reward = r_c / 6 / 5
            # if rew_punish1 != 0:
                # print(station.state.p_tot, r_c, rew_punish1)
            return reward, r_c, rew_punish1

    # set env action for a particular agent
    def _set_action(self, action, agent):
        agent.action.p = action[0]

    def get_obs_agent(self, agent):
        for station in self.stations:
            t = station.state.t_cur
            if agent.state.soc[t] != 0:
                agent.state.n_ev += 1
                station.state.n_ev += 1
                p_max = min(agent.max_power, (agent.state.e_up[t + 1] - agent.state.soc[t]) / self.world.ita * agent.cap)
                # p_min = max(-agent.max_power, (e_down[t+1] - soc_[t]) * world.ita * agent.cap)
                if agent.state.e_down[t + 1] > agent.state.soc[t]:
                    p_min = max(-agent.max_power,
                                (agent.state.e_down[t + 1] - agent.state.soc[t]) / self.world.ita * agent.cap)
                else:
                    p_min = max(-agent.max_power,
                                (agent.state.e_down[t + 1] - agent.state.soc[t]) * self.world.ita * agent.cap)
                if agent.state.soc[t] >= agent.soc_need:
                    agent.state.eme += 0
                else:
                    agent.state.eme += (agent.soc_need - agent.state.soc[t]) * agent.cap / agent.max_power / \
                                       agent.state.t_stay[t]
            else:
                p_max = 0
                p_min = 0
                agent.state.eme += 0
            station.state.eme_tot += agent.state.eme
            # print(agent.state.eme)
            ag_state = [agent.state.soc[t], p_max / agent.max_power, p_min / agent.max_power,
                        agent.state.t_stay[t] / 24]
            return np.array(ag_state)

    def get_obs(self):
        agents_obs = []
        for i, agent in enumerate(self.world_agents):
            agents_obs.append(self.get_obs_agent(agent))
        st_obs = self._get_st_obs()
        # print(st_obs)
        obs_N = [np.concatenate((agents_obs[i], st_obs[i]), axis=-1) for i in range(self.n_agents)]
        # print(obs_N)
        for i, agent in enumerate(self.world_agents):
            agent.state.eme = 0
            agent.state.n_ev = 0
        for station in self.stations:
            # print('111', station.state.n_ev)
            station.state.n_ev = 0
            station.state.eme_tot = 0
        return obs_N

    def get_state(self):
        obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
        return obs_concat

    def get_avail_agent_actions(self, step, agent_id, obs_agent):
        """Returns the available actions for agent_id."""
        avail_actions = [0] * self.n_actions
        avail_actions[int(self.n_actions/2)] = 1
        for i, agent in enumerate(self.world_agents):
            if i == agent_id:
                avail_act_up = int(obs_agent[1]*agent.max_power)
                avail_act_down = math.ceil(obs_agent[2]*agent.max_power)
                if avail_act_up+6 < int(self.n_actions/2) or avail_act_down+6 > int(self.n_actions/2):
                    avail_actions[int(self.n_actions / 2)] = 0
                for j in range(avail_act_down+6, avail_act_up+7):
                    avail_actions[j] = 1
        return avail_actions

    # update state of the world
    def step_1(self, action_n):
        p_tot = 0
        p_real = np.zeros(len(self.world_agents))
        for station in self.stations:
            t = station.state.t_cur
            station.state.p_tot = 0
            for i, agent in enumerate(self.world_agents):
                if agent.state.soc[t] != 0:
                    p_real[i] = action_n[i] - 6
                    # p_real[i] = (action_n[i] - (-1)) / 2 * agent.max_power * (agent.state.e_up[t] - agent.state.e_down[t]) + agent.state.e_down[t] * agent.max_power
                    if p_real[i] > 0:
                        agent.state.soc[t + 1] = agent.state.soc[t] + p_real[i] * self.world.ita * self.world.dt / agent.cap
                    else:
                        agent.state.soc[t + 1] = agent.state.soc[t] + p_real[i] / self.world.ita * self.world.dt / agent.cap
                if agent.state.t_stay[t] != 0:
                    agent.state.t_stay[t + 1] = agent.state.t_stay[t] - 1
                if agent.state.t_stay[t + 1] == 0:
                    agent.state.soc[t + 1] = 0
            for i in range(len(self.world_agents)):
                station.state.p_tot += p_real[i]







