import numpy as np
from core import World, Agent, Station
from scenario import BaseScenario
import math
import random
random.seed(0)
import pandas as pd
# data_orig = pd.read_excel('/Users/lihang/Desktop/EVdata.xlsx')
data_orig = pd.read_excel('.\EVdata1.xlsx')
price_station = data_orig.iloc[0:24, 4]
price_fast = price_station + 0.8
price_norm = price_station

class Scenario(BaseScenario):
    def make_world(self, num_fast_piles, num_norm_piles, num_evs):
        world = World()
        # set any world properties first
        # num_agents = 3
        self.num_norm_chargers = num_norm_piles
        self.num_fast_chargers = num_fast_piles
        num_agents = self.num_norm_chargers + self.num_fast_chargers
        num_stations = 1
        # add agents/stations
        world.agents = [Agent() for i in range(num_agents)]
        world.stations = [Station() for i in range(num_stations)]
        for station in world.stations:
            station.p_ex_max = 24
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.soc_need = 0.8
            agent.fast = True if i < self.num_fast_chargers else False
            agent.max_power = 30.0 if agent.fast else 6.0  # fast/norm charge power
            agent.max_soc = 0.9
            agent.min_soc = 0.2
            agent.cap = 180.0 if agent.fast else 24.0  # fast/norm charge power
            agent.state.price_s = price_fast if agent.fast else price_norm
        # make initial conditions
        # self.reset_world(world)
        return world

    def reset_world(self, world, num_EVs_per_agent):
        soc_ = []
        e_up = []
        e_down = []
        t_stay = []
        for i in range(num_EVs_per_agent):
            t_arrive, t_leave, e_arrive = self.ev_info(world)     # 随机产生n组接入EV数据
            ag_soc_, ag_e_up, ag_e_down, ag_t_stay = self.agents_state(t_arrive, t_leave, e_arrive, world)   # 传回了agent.state.soc/e_up/e_down/t_stay
            soc_.append(ag_soc_)
            e_up.append(ag_e_up)
            e_down.append(ag_e_down)
            t_stay.append(ag_t_stay)
        self.station_state(world)
        # print(soc_)
        return soc_, e_up, e_down, t_stay

    def station_state(self, world):
        # day-ahead prediction data
        for station in world.stations:
            # station.state.pre_p_pv = p_pv
            # station.state.pre_p_load = p_load
            # station.state.pre_p_wt = p_wt
            # p_mg = p_load - p_pv - p_wt
            station.state.price_b = price_station
            station.state.t_cur = 7
            station.state.eme_tot = 0
            station.state.n_ev = 0
            station.state.p_tot = 0


    def ev_info(self, world):
        t_arrive = []
        t_leave = []
        soc_orig = []
        for i in range(len(world.agents)):
            if i < self.num_fast_chargers:
                t_arrive.append(int(9))
                t_leave.append(int(19))
                soc_orig.append(random.normalvariate(0.4, 0.1))

                # soc ~ (0.2,0.8)
                if soc_orig[i] < 0.2:
                    soc_orig[i] = 0.2
                elif soc_orig[i] > 0.6:
                    soc_orig[i] = 0.6
            else:
                t_arrive.append(int(round(random.normalvariate(9, 1))))
                t_leave.append(int(round(random.normalvariate(19, 1))))
                soc_orig.append(random.normalvariate(0.4, 0.1))
                # t_arrive ~ (6,11)
                if t_arrive[i] < 7:
                    t_arrive[i] = 7
                elif t_arrive[i] > 11:
                    t_arrive[i] = 11
                # t_leave ~ (15,21)
                if t_leave[i] < 17:
                    t_leave[i] = 17
                elif t_leave[i] > 21:
                    t_leave[i] = 21
                # soc ~ (0.2,0.8)
                if soc_orig[i] < 0.2:
                    soc_orig[i] = 0.2
                elif soc_orig[i] > 0.6:
                    soc_orig[i] = 0.6
            # print(t_arrive, soc_orig)
        # print(t_arrive, t_leave, soc_orig)
        return t_arrive, t_leave, soc_orig

    def agents_state(self, t_arrive, t_leave, e_arrive, world):
        soc_ = []
        e_up = []
        e_down = []
        t_stay = []
        for i, agent in enumerate(world.agents):
            agent.state.n_ev = 0
            agent.state.eme = 0
            agent.state.t_stay = np.zeros(24)
            agent.state.soc = np.zeros(24)
            agent.state.e_up = np.zeros(24)
            agent.state.e_down = np.zeros(24)
            agent.state.t_stay[t_arrive[i]] = t_leave[i] - t_arrive[i]
            # print("t_stay", agent.state.t_stay[t_arrive[i]])
            agent.state.soc[t_arrive[i]] = e_arrive[i]
            agent.state.e_up[t_arrive[i]] = e_arrive[i]
            agent.state.e_down[t_arrive[i]] = e_arrive[i]
            t_need = math.ceil((agent.soc_need-agent.min_soc) * agent.cap / agent.max_power / world.ita)
            # print("t_need", t_need)
            for t in range(t_arrive[i]+1, t_leave[i]+1):
                # ----------time before leave-----------
                agent.state.t_stay[t] = agent.state.t_stay[t-1] - 1
                # ----------boundary of e_up------------
                agent.state.e_up[t] = agent.state.e_up[t-1] + world.ita * agent.max_power / agent.cap
                if agent.state.e_up[t] > agent.max_soc:
                    agent.state.e_up[t] = agent.max_soc
                # ----------boundary of e_down------------
                # ---------------decrease---------------
                if t <= (t_leave[i] - t_need):
                    agent.state.e_down[t] = agent.state.e_down[t-1] - agent.max_power / world.ita / agent.cap
                    if agent.state.e_down[t] < agent.min_soc:
                        agent.state.e_down[t] = agent.min_soc
                # ---------------increase---------------
                # else:
                #     agent.state.e_down[t] = agent.state.e_down[t-1] + world.ita * agent.max_power / agent.cap
                #     if agent.state.e_down[t] > agent.soc_need:
                #         agent.state.e_down[t] = agent.soc_need
            # print("e_up,e_down", agent.state.e_up, agent.state.e_down)
            agent.state.e_down[t_leave[i]] = agent.soc_need
            agent.state.e_down[t_leave[i] - 1] = agent.soc_need - agent.max_power * world.ita / agent.cap
            agent.state.e_down[t_leave[i] - 2] = agent.state.e_down[
                                                     t_leave[i] - 1] - agent.max_power * world.ita / agent.cap
            agent.state.e_down[t_leave[i] - 3] = agent.state.e_down[
                                                     t_leave[i] - 2] - agent.max_power * world.ita / agent.cap
            if agent.state.e_down[t_leave[i] - 3] < agent.min_soc:
                agent.state.e_down[t_leave[i] - 3] = agent.min_soc
            agent.state.e_down[t_leave[i] - 4] = agent.state.e_down[
                                                     t_leave[i] - 3] - agent.max_power * world.ita / agent.cap
            if agent.state.e_down[t_leave[i] - 4] < agent.min_soc:
                agent.state.e_down[t_leave[i] - 4] = agent.min_soc
            soc_.append(agent.state.soc)
            e_up.append(agent.state.e_up)
            e_down.append(agent.state.e_down)
            t_stay.append(agent.state.t_stay)
        return soc_, e_up, e_down, t_stay

    def reward(self, world, p_n):  # reward for a particular agent
        for station in world.stations:
            rew_cost = []
            # print("1", p_n, len(p_n))
            for i in range(len(p_n)):
                # print("0",p_n[i])
                rew_cost.append((-1 * p_n[i] * station.state.price_b[station.state.t_cur])) #/ station.p_ex_max
            # print("2", rew_cost)
            # print(sum(station.state.p_tot))
            if (station.p_ex_max) < abs(sum(station.state.p_tot)):
                # rew_punish1 = - ((np.e**(abs(sum(station.state.p_tot)) - 0.8*station.p_ex_max)) - 1) / (np.e**(36 - 0.8 * station.p_ex_max) - 1)
                # rew_punish1 = (- (abs(sum(station.state.p_tot)) - 0.9 * station.p_ex_max) - abs(sum(station.state.p_tot)) * 0.05) / 10
                # rew_punish1 = - (np.log(abs(sum(station.state.p_tot)) - station.p_ex_max + 1)) / (np.log(60 - station.p_ex_max + 1))
                rew_punish1 = (station.p_ex_max - abs(sum(station.state.p_tot))) / (30 - station.p_ex_max)
            else:
                rew_punish1 = 0

            return rew_cost, rew_punish1

    def agent_observation(self, agent, world, soc_, e_up, e_down, t_stay):
        for station in world.stations:
            t = station.state.t_cur
            if soc_[t] != 0:
                agent.state.n_ev += 1
                station.state.n_ev += 1
                p_max = min(agent.max_power, (e_up[t+1] - soc_[t]) / world.ita * agent.cap)
                if e_down[t + 1] > soc_[t]:
                    p_min = max(-agent.max_power,
                                            (e_down[t + 1] - soc_[t]) / world.ita * agent.cap)
                else:
                    p_min = max(-agent.max_power,
                                            (e_down[t + 1] - soc_[t]) * world.ita * agent.cap)
                if soc_[t] >= agent.soc_need:
                    agent.state.eme += 0
                else:
                    agent.state.eme += (agent.soc_need - soc_[t]) * agent.cap / agent.max_power / t_stay[t]
            else:
                p_max = 0
                p_min = 0
                agent.state.eme += 0

            station.state.eme_tot += agent.state.eme

            ag_state = [soc_[t], p_max/agent.max_power, p_min/agent.max_power,
                        t_stay[t] / 24]

            return np.array(ag_state)

    def station_observation(self, world):
        for station in world.stations:

            eme_agent = 0
            t = station.state.t_cur

            if station.state.n_ev == 0:
                eme_agent = 0
            else:
                eme_agent = station.state.eme_tot/5

            sta_state = [[t / 24] + [station.state.price_b[t]] + [eme_agent]][0]# + num_evs_agent + eme_agent][0] [station.p_ex_max/60] +
            num_ev = station.state.n_ev
            return np.array(sta_state), num_ev
