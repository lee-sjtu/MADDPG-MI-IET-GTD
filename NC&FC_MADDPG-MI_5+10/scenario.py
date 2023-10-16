
# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self, fag_n, nag_n, fcp_n, ncp_n):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()
