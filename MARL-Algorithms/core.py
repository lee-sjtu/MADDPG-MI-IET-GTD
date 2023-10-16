import numpy as np

# state of agents (internal/mental state)
class AgentState(object):
    def __init__(self):
        super(AgentState, self).__init__()
        # soc
        self.soc = None
        # energy up boundary
        self.e_up = None
        # energy down boundary
        self.e_down = None
        # energy up boundary
        self.p_max = None
        # energy down boundary
        self.p_min = None
        # time for stay
        self.t_stay = None
        # price for service
        self.price_s = None  #serve
        # emergency
        self.eme = None
        # t ev number
        self.n_ev = None


class StationState(object):
    def __init__(self):
        super(StationState, self).__init__()
        # day-ahead prediction pv power
        self.pre_p_pv = None
        # t-1 photovoltaics power
        self.p_pv_ = None
        # t photovoltaics power
        self.p_pv = None
        # day-ahead prediction pv power
        self.pre_p_load = None
        # t-1 photovoltaics power
        self.p_load_ = None
        # t photovoltaics power
        self.p_load = None
        # day-ahead prediction pv power
        self.pre_p_wt = None
        # t-1 photovoltaics power
        self.p_wt_ = None
        # t photovoltaics power
        self.p_wt = None
        # t-1 photovoltaics power
        self.p_avg = None
        # current time
        self.t_cur = None
        # t price of buying from gird
        self.price_b = None  #buy
        # emergency
        self.eme_tot = None


# action of the agent
class Action(object):
    def __init__(self):
        # charge power action
        self.p = None
        # charge power action
        self.p_real = None


# properties of agents
class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()
        # name
        self.name = ''
        # max charge/discharge power
        self.max_power = None
        # limited of the lowest soc when leave
        self.soc_need = None
        # minimum soc
        self.min_soc = None
        # maximum soc
        self.max_soc = None
        # charge capacity
        self.cap = None
        # agents can serve by default
        self.price = None
        # control range
        self.p_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # piles number
        self.n_piles = None

# properties of station
class Station(object):
    def __init__(self):
        super(Station, self).__init__()
        # name
        self.name = ''
        # max exchange power with grid
        self.p_ex_max = None
        # state
        self.state = StationState()


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        # station
        self.stations = []
        # state of charge dimensionality
        self.dim_e = 1
        # simulation time step
        self.dt = 1    # 15 min
        # charge/discharge ita
        self.ita = 0.95

    # update state of the world
    def step(self, action_n):
        # integrate physical state
        p_real = self.integrate_agent_state(action_n)  # update 1 station state 1 agent state
        # integrate station state
        # self.integrate_station_state()  # update 7 station states
        return p_real

    # integrate physical state
    def integrate_agent_state(self, action_n):
        p_real = np.zeros(len(self.agents))

        for station in self.stations:
            t = station.state.t_cur
            station.state.p_tot = 0
            for i, agent in enumerate(self.agents):
                if agent.state.soc[t] != 0:
                    p_real[i] = (action_n[i] - (-1)) / 2 * agent.max_power * (agent.state.e_up[t] - agent.state.e_down[t]) + agent.state.e_down[t] * agent.max_power
                    if p_real[i] > 0:
                        agent.state.soc[t + 1] = agent.state.soc[t] + p_real[i] * self.ita * self.dt / agent.cap
                    else:
                        agent.state.soc[t + 1] = agent.state.soc[t] + p_real[i] / self.ita * self.dt / agent.cap
                if agent.state.t_stay_[t] != 0:
                    agent.state.t_stay_[t + 1] = agent.state.t_stay_[i][t] - 1
                if agent.state.t_stay_[t + 1] == 0:
                    agent.state.soc[t + 1] = 0
                    # print(p_real)
                    # agent.state.p_tot += sum(p_real[i][j])
            # print("p_real", p_real)
            for i in range(len(self.agents)):
                station.state.p_tot += np.sum(p_real[i])
            # print("p_total", station.state.p_tot)
        return p_real
