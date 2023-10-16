from runner import Runner
import time
# from scenarios.one_EV_station import Scenario
from common.arguments import get_common_args, get_mixer_args
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def make_env(scenario_name, num_fast_agents, num_norm_agents):
    from environment import MultiAgentEnv
    import scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(num_fast_agents, num_norm_agents)
    env = MultiAgentEnv(world, scenario.reset_world)#, scenario.reward)#, scenario.agent_observation,
                       # scenario.station_observation)
    return env

if __name__ == '__main__':

    args = get_common_args()
    args = get_mixer_args(args)
    env = make_env(args.scenario, args.num_fca, args.num_nca)
    args.n_actions = 13
    args.n_agents = env.n_agents
    args.obs_shape = 7
    args.state_shape = args.obs_shape * args.n_agents
    args.episode_limit = 15
    args.algrithm = ['qmix']  # vdn
    # -------------------------------------
    if not args.evaluate:
        for i in range(len(args.algrithm)):
            args.alg = args.algrithm[i]
            runner = Runner(env, args)
            # start = time.perf_counter()
            runner.run()
            # end = time.perf_counter()

    else:
        for i in range(len(args.algrithm)):
            args.alg = args.algrithm[i]
            runner = Runner(env, args)
            reward = runner.evaluate()


