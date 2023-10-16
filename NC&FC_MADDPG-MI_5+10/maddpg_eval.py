import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import time
import pandas as pd
from maddpg_test import MADDPGAgentTrainer

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="one_EV_station_test", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=13, help="maximum episode length")
    parser.add_argument("--num-nca", type=int, default=1, help="number of agents for norm charge stations")
    parser.add_argument("--num-fca", type=int, default=1, help="number of agents for fast charge stations")
    parser.add_argument("--num-ncp", type=int, default=80, help="number of norm charge piles")
    parser.add_argument("--num-fcp", type=int, default=20, help="number of fast charge piles")
    # Core training parameters
    parser.add_argument("--num-units1", type=int, default=64, help="number of units1 in the mlp")
    parser.add_argument("--num-units2", type=int, default=64, help="number of units2 in the mlp")
    parser.add_argument("--num-units3", type=int, default=64, help="number of units3 in the mlp")
    # Checkpointing
    return parser.parse_args()


def mlp_model_actor(input, num_outputs, num_units1, num_units2, num_units3, scope, reuse=False):   # p function
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = tf.layers.dense(out, num_units1, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_units1, activation=tf.nn.relu)
        # out = tf.layers.dense(out, num_units3, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_outputs, activation=tf.nn.tanh)
        return tf.multiply(out, 1.0)


def make_env(scenario_name, num_fast_agents, num_norm_agents, num_EVs_fcp, num_EVs_ncp):
    from environment_eval import MultiAgentEnv
    import scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(num_fast_agents, num_norm_agents, num_EVs_fcp, num_EVs_ncp)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.agent_observation, scenario.station_observation)
    return env


def get_trainers(env, obs_shape_n, arglist):
    trainers = []
    model_a = mlp_model_actor
    trainer = MADDPGAgentTrainer
    for i, agent in enumerate(env.agents):
        trainers.append(trainer(
            "agent_%d" % i, env, agent, model_a, obs_shape_n, i, arglist))
    return trainers


def maddpg_eval(arglist, env):
    with tf.Session() as sess:

        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_piles = env.n
        trainers = get_trainers(env, obs_shape_n, arglist)
        episode_step = 0
        SOC = []
        POWER = []
        # action_N = []
        obs_n, soc_, e_up, e_down, t_stay_, t_arrive_n, t_leave_n, e_arrive_n = env.reset()
        # print(e_up, e_down)
        # print('Starting iterations...')
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, './save_pl_mi_10+5-0815/params')
        COST = 0
        MAX_P_T = []
        MAX_P_t = []
        while True:
            action_N = []
            POWER_t = []
            # MAX_P_t = []
            # print("1", obs_n)
            # obs_n1 = np.transpose(obs_n, (1, 0, 2))
            # print("2", obs_n1)
            soc = []
            start = time.perf_counter()
            for i, agent in enumerate(env.agents):
                action_n = []
                for j in range(agent.n_piles):
                    action_n.append(trainers[i].action(obs_n[i][j], j))
                action_N.append(action_n)
            for i, agent in enumerate(env.agents):
                for j in range(agent.n_piles):
                    soc.append(obs_n[i][j][0])
                    if obs_n[i][j][0] == 0:
                        action_N[i][j] = np.array([0])

            new_obs_n, p_n, cost = env.step(action_N, obs_n, soc_, e_up, e_down, t_stay_)
            COST += cost
            # print("1", p_n, sum(sum(p_n)))
            # print(p_n, "total", sum(p_n))
            for i, agent in enumerate(env.agents):
                for j in range(agent.n_piles):
                    POWER_t.append(p_n[i][j])
                    # SOC_t.append(soc[i][j])
            POWER.append(POWER_t)
            SOC.append(soc)
            MAX_P_t.append(sum(POWER_t))
            # for i in range(len(p_n)):
            #     POWER.append(p_n[i])
            #     SOC.append(soc[i])
            episode_step += 1
            terminal = (episode_step > arglist.max_episode_len)
            obs_n = new_obs_n
            end = time.perf_counter()
            if terminal:
                # MAX_P_T.append(max(MAX_P_t))
                break
        # print(end-start)
        # EVINFO = info
        # print(MAX_P_t)
    return t_arrive_n, t_leave_n, e_arrive_n, POWER, SOC, COST, MAX_P_t

if __name__ == '__main__':
    arglist = parse_args()
    sample_number = 10
    EVINFO_ta_N = []
    EVINFO_tl_N = []
    EVINFO_soc_N = []
    POWER_N = []
    SOC_N = []
    COST_N = []
    MAX_P_N = np.zeros(sample_number)
    env = make_env(arglist.scenario, arglist.num_fca, arglist.num_nca, arglist.num_fcp, arglist.num_ncp)
    for i in range(sample_number):
        EVINFO_ta, EVINFO_tl, EVINFO_soc, POWER, SOC, COST, MAX_P_T = maddpg_eval(arglist, env)
        EVINFO_ta_N.append(EVINFO_ta)
        EVINFO_tl_N.append(EVINFO_tl)
        EVINFO_soc_N.append(EVINFO_soc)
        POWER_N.append(POWER)
        SOC_N.append(SOC)
        COST_N.append(COST)
        MAX_P_N[i] = max(MAX_P_T)
    # print(EVINFO, POWER, SOC)
    #-------------------------write output---------------------------#
    writer = pd.ExcelWriter('E:\OneDrive - sjtu.edu.cn\【科研】\小论文\【5】AE\MADDPG_异构充电站-5+10\save_pl_mi_20+80-0816.xlsx')
    df1 = pd.DataFrame(EVINFO_ta_N)
    df2 = pd.DataFrame(EVINFO_tl_N)
    df3 = pd.DataFrame(EVINFO_soc_N)
    df4 = pd.DataFrame(POWER_N)
    df5 = pd.DataFrame(SOC_N)
    df6 = pd.DataFrame(COST_N)
    df7 = pd.DataFrame(MAX_P_N)
    df1.to_excel(writer,sheet_name='info_ta')
    df2.to_excel(writer, sheet_name='info_tl')
    df3.to_excel(writer, sheet_name='info_soc')
    df4.to_excel(writer,sheet_name='power')
    df5.to_excel(writer,sheet_name='SOC')
    df6.to_excel(writer, sheet_name='COST')
    df7.to_excel(writer, sheet_name='MAX_P')
    writer.save()

