
# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self, num_fast_agents, num_norm_agents):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()
