
# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self, fast_n, norm_n, num_evs):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world, num_evs):
        raise NotImplementedError()
