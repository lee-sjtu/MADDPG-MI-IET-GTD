import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import time
import tf_util as U
from maddpg_test import MADDPGAgentTrainer
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="one_EV_station_test", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=15, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=4000, help="number of episodes")
    parser.add_argument("--memory-size", type=int, default=2000, help="size of memory")
    parser.add_argument("--num-norm-piles", type=int, default=3, help="number of norm charge piles")
    parser.add_argument("--num-fast-piles", type=int, default=0, help="number of fast charge piles")
    parser.add_argument("--num-EVs-per-agent", type=int, default=1, help="number of EVs")
    # Core training parameters
    parser.add_argument("--lr-a", type=float, default=1e-3, help="learning rate for actor network")
    parser.add_argument("--lr-c", type=float, default=1e-4, help="learning rate for critic network")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units1", type=int, default=128, help="number of units1 in the mlp")
    parser.add_argument("--num-units2", type=int, default=128, help="number of units2 in the mlp")
    parser.add_argument("--num-units3", type=int, default=128, help="number of units3 in the mlp")
    parser.add_argument("--num-units4", type=int, default=128, help="number of units4 in the mlp")
    parser.add_argument("--num-units5", type=int, default=128, help="number of units5 in the mlp")
    return parser.parse_args()


def restore(sess):
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, './save2_2/params')


def mlp_model_actor(input, num_outputs, num_units1, num_units2, num_units3, num_units4, num_units5, scope, reuse=False):   # p function
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = tf.layers.dense(out, num_units1, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_units2, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_units3, activation=tf.nn.relu)
        # out = tf.layers.dense(out, num_units4, activation=tf.nn.relu)
        # out = tf.layers.dense(out, num_units5, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_outputs, activation=tf.nn.tanh)
        return tf.multiply(out, 1.0)


def mlp_model_critic(input, num_outputs, num_units1, num_units2, num_units3, num_units4, num_units5, scope, reuse=False):    # q_function
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = tf.layers.dense(out, num_units1, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_units2, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_units3, activation=tf.nn.relu)
        # out = tf.layers.dense(out, num_units4, activation=tf.nn.relu)
        # out = tf.layers.dense(out, num_units5, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_outputs)
        return out


def make_env(scenario_name, num_fast_agents, num_norm_piles, num_EVs_per_agent):
    from environment_eval import MultiAgentEnv
    import scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(num_fast_agents, num_norm_piles, num_EVs_per_agent)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.agent_observation, scenario.station_observation)
    return env


def get_trainers(num_adversaries, obs_shape_n, arglist):
    trainers = []
    model_a = mlp_model_actor
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model_a, obs_shape_n, i, arglist))
    return trainers


def train(arglist):
    with tf.Session() as sess:
        env = make_env(arglist.scenario, arglist.num_fast_piles, arglist.num_norm_piles, arglist.num_EVs_per_agent)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_piles = env.n
        trainers = get_trainers(num_piles, obs_shape_n, arglist)

        restore(sess)
        power_n = []

        obs_n, soc_, e_up, e_down, t_stay_, t_arrive_n, t_leave_n, e_arrive_n = env.reset(arglist.num_EVs_per_agent)

        episode_step = 0

        print('Starting iterations...')
        start = time.perf_counter()

        while True:
            action_N = []
            obs_n1 = np.transpose(obs_n, (1, 0, 2))
            for i in range(arglist.num_EVs_per_agent):
                action_n = [agent.action(obs[i], i) for agent, obs in zip(trainers, obs_n1)]
                action_N.append(action_n)

            new_obs_n, p_n = env.step(action_N, obs_n, arglist.num_EVs_per_agent, soc_, e_up, e_down, t_stay_)

            power_n.append(p_n)
            episode_step += 1
            terminal = (episode_step >= arglist.max_episode_len)

            obs_n = new_obs_n

            if terminal:
                break

        end = time.perf_counter()
        print(start, end)
    return t_arrive_n, t_leave_n, e_arrive_n, power_n, soc_
if __name__ == '__main__':
    arglist = parse_args()
    t_arrive_n, t_leave_n, soc_arr, power_n, soc_ = train(arglist)
    EVinfo = np.concatenate((t_arrive_n, t_leave_n, soc_arr), axis = 1)
    #-------------------------write output---------------------------#
    writer = pd.ExcelWriter('E:/OneDrive - sjtu.edu.cn/科研/小论文/基于多智能体强化学习的电动汽车充电站分散协同优化调度/程序/results.xlsx')
    df1 = pd.DataFrame(EVinfo)
    df2 = pd.DataFrame(power_n)
    df3 = pd.DataFrame(soc_)
    df1.to_excel(writer, sheet_name='info')
    df2.to_excel(writer, sheet_name='power')
    df3.to_excel(writer, sheet_name='SOC')
    writer.save()
