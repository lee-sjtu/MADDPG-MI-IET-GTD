import numpy as np
from core import World, Agent, Station
from scenario import BaseScenario
import math
import random
random.seed(0)
import pandas as pd
# random.seed(1)
# data_orig = pd.read_excel('/Users/lihang/Desktop/EVdata.xlsx')
# data_orig = pd.read_excel('/Users/lihang/OneDrive - sjtu.edu.cn/科研/小论文/基于多智能体强化学习的电动汽车充电站分散协同优化调度/程序/EVdata.xlsx')
# dataEVinfo = pd.read_excel('/Users/lihang/OneDrive - sjtu.edu.cn/科研/小论文/基于多智能体强化学习的电动汽车充电站分散协同优化调度/程序/MADDPG_一个智能体多个车0503/output6ev23.xlsx')
data_orig = pd.read_excel('L:\OneDrive - sjtu.edu.cn\【科研】\小论文\【3】IET-GTD\程序\EVdata1.xlsx')
# dataEVinfo = pd.read_excel('L:\OneDrive - sjtu.edu.cn\科研\小论文\基于多智能体强化学习的电动汽车充电站分散协同优化调度\程序\MADDPG_一个智能体多个车0503\output6ev23.xlsx')
# arrive = dataEVinfo.iloc[0, 0:6]
# leave = dataEVinfo.iloc[1, 0:6]
# earrive = dataEVinfo.iloc[2, 0:6]
price_station = data_orig.iloc[0:24, 4]
price_fast = price_station + 0.8
price_norm = price_station
# p_load = data_orig.iloc[0:24, 16]
# p_pv = data_orig.iloc[0:24, 18]
# p_wt = data_orig.iloc[0:24, 17]
# print(arrive)
class Scenario(BaseScenario):
    def make_world(self, num_fags, num_nags, num_fcp, num_ncp):
        world = World()
        self.num_nag= num_nags
        self.num_fag = num_fags
        self.num_agents = self.num_nag + self.num_fag
        self.num_fcp = num_fcp
        self.num_ncp = num_ncp
        num_stations = 1
        # add agents/stations
        world.agents = [Agent() for i in range(self.num_agents)]
        world.stations = [Station() for i in range(num_stations)]
        for station in world.stations:
            station.p_ex_max = 24
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.soc_need = 0.8
            agent.fast = True if i < self.num_fag else False
            agent.n_piles = self.num_fcp if agent.fast else self.num_ncp
            agent.max_power = 30.0 if agent.fast else 6.0  # fast/norm charge power
            agent.max_soc = 0.9
            agent.min_soc = 0.2
            agent.cap = 180.0 if agent.fast else 24.0  # fast/norm charge power
            agent.state.price_s = price_fast if agent.fast else price_norm
        # make initial conditions
        # self.reset_world(world)
        return world

    def reset_world(self, world):
        t_leave_n = []
        e_arrive_n = []
        t_stay_n = []
        soc_ = []
        e_up = []
        e_down = []
        t_stay = []
        for i, agent in enumerate(world.agents):
            t_arrive, t_leave, e_arrive = self.ev_info(agent)     # 随机产生n组接入EV数据
            ag_soc_, ag_e_up, ag_e_down, ag_t_stay = self.agents_state(t_arrive, t_leave, e_arrive, agent, world)    # 传回了agent.state.soc/e_up/e_down/t_stay
            soc_.append(ag_soc_)
            e_up.append(ag_e_up)
            e_down.append(ag_e_down)
            t_stay.append(ag_t_stay)
            t_leave_n.extend(np.array(t_arrive))
            e_arrive_n.extend(np.array(t_leave))
            t_stay_n.extend(np.array(e_arrive))
        self.station_state(world)
        # print(soc_)
        return soc_, e_up, e_down, t_stay, t_leave_n, e_arrive_n, t_stay_n

    def station_state(self, world):
        # day-ahead prediction data
        for station in world.stations:
            station.state.price_b = price_station
            station.state.t_cur = 7
            station.state.eme_tot = 0
            station.state.n_ev = 0
            station.state.p_tot = 0

    def ev_info(self, agent):
        t_arrive = []
        t_leave = []
        soc_orig = []
        if agent.fast: #如果是快充站
            for i in range(agent.n_piles):
                t_arrive.append(int(round(random.normalvariate(9, 1))))
                t_leave.append(int(round(random.normalvariate(19, 1))))
                soc_orig.append(random.normalvariate(0.4, 0.1))
                # t_arrive ~ (6,11)
                if t_arrive[i] < 8:
                    t_arrive[i] = 8
                elif t_arrive[i] > 10:
                    t_arrive[i] = 10
                # t_leave ~ (15,21)
                if t_leave[i] < 18:
                    t_leave[i] = 18
                elif t_leave[i] > 20:
                    t_leave[i] = 20
                # soc ~ (0.2,0.8)
                if soc_orig[i] < 0.2:
                    soc_orig[i] = 0.2
                elif soc_orig[i] > 0.6:
                    soc_orig[i] = 0.6
        else:#如果是慢充站
            for i in range(agent.n_piles):
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

        return t_arrive, t_leave, soc_orig  #dim=num of pile

    def reward(self, world, p_n):  # reward for a particular agent
        for station in world.stations:
            cost = 0
            for i, agent in enumerate(world.agents):
                for j in range(agent.n_piles):
                    cost += (-1 * p_n[i][j] * station.state.price_b[station.state.t_cur])
            return cost

    def agents_state(self, t_arrive, t_leave, e_arrive, agent, world):
        agent.state.soc = []
        agent.state.e_up = []
        agent.state.e_down = []
        agent.state.t_stay = []
        agent.state.n_ev = 0
        agent.state.eme = 0
        for i in range(agent.n_piles):
            t_stay_p = np.zeros(24)
            soc_p = np.zeros(24)
            e_up_p = np.zeros(24)
            e_down_p = np.zeros(24)
            t_stay_p[t_arrive[i]] = t_leave[i] - t_arrive[i]
            # print("t_stay", agent.state.t_stay[t_arrive[i]])
            soc_p[t_arrive[i]] = e_arrive[i]
            e_up_p[t_arrive[i]] = e_arrive[i]
            e_down_p[t_arrive[i]] = e_arrive[i]
            t_need = math.ceil((agent.soc_need-agent.min_soc) * agent.cap / agent.max_power / world.ita)
            # print("t_need", t_need)
            for t in range(t_arrive[i]+1, t_leave[i]+1):
                # ----------time before leave-----------
                t_stay_p[t] = t_stay_p[t-1] - 1
                # ----------boundary of e_up------------
                e_up_p[t] = e_up_p[t-1] + world.ita * agent.max_power / agent.cap
                if e_up_p[t] > agent.max_soc:
                    e_up_p[t] = agent.max_soc
                # ----------boundary of e_down------------
                # ---------------decrease---------------
                if t <= (t_leave[i] - t_need):
                    e_down_p[t] = e_down_p[t-1] - agent.max_power / world.ita / agent.cap
                    if e_down_p[t] < agent.min_soc:
                        e_down_p[t] = agent.min_soc
                # ---------------increase---------------
                # else:
                #     agent.state.e_down[t] = agent.state.e_down[t-1] + world.ita * agent.max_power / agent.cap
                #     if agent.state.e_down[t] > agent.soc_need:
                #         agent.state.e_down[t] = agent.soc_need
            # print("e_up,e_down", agent.state.e_up, agent.state.e_down)
            e_down_p[t_leave[i]] = agent.soc_need
            e_down_p[t_leave[i] - 1] = agent.soc_need - agent.max_power * world.ita / agent.cap
            e_down_p[t_leave[i] - 2] = e_down_p[
                                                     t_leave[i] - 1] - agent.max_power * world.ita / agent.cap
            e_down_p[t_leave[i] - 3] = e_down_p[
                                                     t_leave[i] - 2] - agent.max_power * world.ita / agent.cap
            if e_down_p[t_leave[i] - 3] < agent.min_soc:
                e_down_p[t_leave[i] - 3] = agent.min_soc
            e_down_p[t_leave[i] - 4] = e_down_p[
                                                     t_leave[i] - 3] - agent.max_power * world.ita / agent.cap
            if e_down_p[t_leave[i] - 4] < agent.min_soc:
                e_down_p[t_leave[i] - 4] = agent.min_soc
            agent.state.soc.append(soc_p)
            agent.state.e_up.append(e_up_p)
            agent.state.e_down.append(e_down_p)
            agent.state.t_stay.append(t_stay_p)
        return agent.state.soc, agent.state.e_up, agent.state.e_down, agent.state.t_stay

    def agent_observation(self, agent, world, soc_, e_up, e_down, t_stay):
        for station in world.stations:
            t = station.state.t_cur
            if soc_[t] != 0:
            # if t_stay[t] != 0:
                agent.state.n_ev += 1
                station.state.n_ev += 1
                p_max = min(agent.max_power, (e_up[t+1] - soc_[t]) / world.ita * agent.cap)
                # p_min = max(-agent.max_power, (e_down[t+1] - soc_[t]) * world.ita * agent.cap)
                if e_down[t + 1] > soc_[t]:
                    p_min = max(-agent.max_power,
                                            (e_down[t + 1] - soc_[t]) / world.ita * agent.cap)
                else:
                    p_min = max(-agent.max_power,
                                            (e_down[t + 1] - soc_[t]) * world.ita * agent.cap)
                # if ((agent.soc_need - soc_[t]) * agent.cap / agent.max_power / t_stay[t]) > 0.5:
                if soc_[t] >= agent.soc_need:
                    agent.state.eme += 0
                    station.state.eme_tot += 0
                    # agent.state.eme +=
                else:
                    eme_11 = (agent.soc_need - soc_[t]) * agent.cap / agent.max_power / world.ita / t_stay[t]
                    agent.state.eme += eme_11
                    station.state.eme_tot += eme_11
            else:
                p_max = 0
                p_min = 0
                agent.state.eme += 0
                station.state.eme_tot += 0
            # if soc_[t] >= agent.soc_need:
            #     agent.state.eme = 0
            # print("st", station.state.eme_tot)
            ag_state = [soc_[t], p_max/agent.max_power, p_min/agent.max_power,
                        t_stay[t] / 24]
                        # agent.state.eme / agent.max_power]
                        # agent.state.price_s[t]]
            return np.array(ag_state)

    def station_observation(self, world):
        for station in world.stations:
            eme_agent = []
            # eme_agent = 0
            t = station.state.t_cur
            # print(station.state.eme_tot, station.state.n_ev)
            if station.state.eme_tot == 0:
                for i, agent in enumerate(world.agents):
                    eme_agent.append(0)
            else:
                for i, agent in enumerate(world.agents):
                    eme_agent.append(agent.state.eme/station.state.eme_tot)
            #     print(t, agent.state.eme, agent.n_piles)
            # print(eme_agent)
            # if station.state.n_ev == 0:
            #     eme_agent = 0
            # else:
            #     eme_agent = station.state.eme_tot/station.state.n_ev

            sta_state = [[t / 24] + [station.state.price_b[t]] + eme_agent][0]  # + num_evs_agent + eme_agent][0] [station.p_ex_max/60] +
            # print(sta_state)
            num_ev = station.state.n_ev
            return np.array(sta_state), num_ev
