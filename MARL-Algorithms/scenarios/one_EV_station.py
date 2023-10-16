import numpy as np
from core import World, Agent, Station
from scenario import BaseScenario
import math
import random
random.seed(0)
import pandas as pd
data_orig = pd.read_excel('./EVdata1.xlsx')
price_station = data_orig.iloc[0:24, 4]

class Scenario(BaseScenario):
    def make_world(self, num_fags, num_nags):
        world = World()
        # set any world properties first
        # num_agents = 3
        self.num_nag= num_nags
        self.num_fag = num_fags
        self.num_agents = self.num_nag + self.num_fag
        # self.num_fcp = num_fcp
        # self.num_ncp = num_ncp
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
            agent.max_power = 30.0 if agent.fast else 6.0  # fast/norm charge power
            agent.max_soc = 0.9
            agent.min_soc = 0.2
            agent.cap = 180.0 if agent.fast else 24.0  # fast/norm charge power
        return world

    def station_state(self, world):
        # day-ahead prediction data
        for station in world.stations:
            station.state.price_b = price_station
            station.state.t_cur = 7
            station.state.eme_tot = 0
            station.state.n_ev = 0
            station.state.p_tot = 0
        return world.stations

    def ev_info(self, agent):
        if agent.fast:
            t_arrive = int(9)
            t_leave = int(19)
            soc_orig = random.normalvariate(0.4, 0.1)
            if soc_orig < 0.2:
                soc_orig = 0.2
            elif soc_orig > 0.6:
                soc_orig = 0.6
        else:
            t_arrive = int(round(random.normalvariate(9, 1)))
            t_leave = int(round(random.normalvariate(19, 1)))
            soc_orig = random.normalvariate(0.4, 0.1)
            # t_arrive ~ (6,11)
            if t_arrive < 7:
                t_arrive = 7
            elif t_arrive > 11:
                t_arrive = 11
            # t_leave ~ (15,21)
            if t_leave < 17:
                t_leave = 17
            elif t_leave > 21:
                t_leave = 21
            # soc ~ (0.2,0.6)
            if soc_orig < 0.2:
                soc_orig = 0.2
            elif soc_orig > 0.6:
                soc_orig = 0.6
        return t_arrive, t_leave, soc_orig

    def agents_state(self, t_arrive, t_leave, e_arrive, agent, world):
        agent.state.n_ev = 0
        agent.state.eme = 0
        t_stay_p = np.zeros(24)
        soc_p = np.zeros(24)
        e_up_p = np.zeros(24)
        e_down_p = np.zeros(24)
        t_stay_p[t_arrive] = t_leave - t_arrive
        soc_p[t_arrive] = e_arrive
        e_up_p[t_arrive] = e_arrive
        e_down_p[t_arrive] = e_arrive
        t_need = math.ceil((agent.soc_need-agent.min_soc) * agent.cap / agent.max_power / world.ita)
        for t in range(t_arrive+1, t_leave+1):
            # ----------time before leave-----------
            t_stay_p[t] = t_stay_p[t-1] - 1
            # ----------boundary of e_up------------
            e_up_p[t] = e_up_p[t-1] + world.ita * agent.max_power / agent.cap
            if e_up_p[t] > agent.max_soc:
                e_up_p[t] = agent.max_soc
            # ----------boundary of e_down----------
            # ---------------decrease---------------
            if t <= (t_leave - t_need):
                e_down_p[t] = e_down_p[t-1] - agent.max_power / world.ita / agent.cap
                if e_down_p[t] < agent.min_soc:
                    e_down_p[t] = agent.min_soc
            # ---------------increase---------------
            e_down_p[t_leave] = agent.soc_need
            e_down_p[t_leave - 1] = agent.soc_need - agent.max_power * world.ita / agent.cap
            e_down_p[t_leave - 2] = e_down_p[t_leave - 1] - agent.max_power * world.ita / agent.cap
            e_down_p[t_leave - 3] = e_down_p[t_leave - 2] - agent.max_power * world.ita / agent.cap
            if e_down_p[t_leave - 3] < agent.min_soc:
                e_down_p[t_leave - 3] = agent.min_soc
            e_down_p[t_leave - 4] = e_down_p[t_leave - 3] - agent.max_power * world.ita / agent.cap
            if e_down_p[t_leave - 4] < agent.min_soc:
                e_down_p[t_leave - 4] = agent.min_soc
        agent.state.soc = soc_p
        agent.state.e_up = e_up_p
        agent.state.e_down = e_down_p
        agent.state.t_stay = t_stay_p
        return agent.state.soc, agent.state.e_up, agent.state.e_down, agent.state.t_stay


