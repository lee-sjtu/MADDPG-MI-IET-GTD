import os.path as osp
import imp

def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)

class AgentTrainer(object):
    def __init__(self, name, model, obs_shape, act_space, args):
        raise NotImplemented()

    def action(self, ag_obs, i):
        raise NotImplemented()

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def update(self, agents, i):
        raise NotImplemented()

class AgentTrainer_test(object):
    def __init__(self, name, model, obs_shape, act_space, args):
        raise NotImplemented()

    def action(self, ag_obs):
        raise NotImplemented()
