import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import time
import tf_util as U
from maddpg import MADDPGAgentTrainer
from plotmoving import plotma
# from pypowertest import power
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="one_EV_station", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=13, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=3000, help="number of episodes")
    parser.add_argument("--memory-size", type=int, default=1000, help="size of memory")
    parser.add_argument("--num-nca", type=int, default=1, help="number of agents for norm charge stations")
    parser.add_argument("--num-fca", type=int, default=1, help="number of agents for fast charge stations")
    parser.add_argument("--num-ncp", type=int, default=10, help="number of norm charge piles")
    parser.add_argument("--num-fcp", type=int, default=5, help="number of fast charge piles")
    # Core training parameters
    parser.add_argument("--lr-a", type=float, default=1e-3, help="learning rate for actor network")
    parser.add_argument("--lr-c", type=float, default=1e-3, help="learning rate for critic network")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units1", type=int, default=128, help="number of units1 in the mlp")
    parser.add_argument("--num-units2", type=int, default=64, help="number of units2 in the mlp")
    parser.add_argument("--num-units3", type=int, default=32, help="number of units3 in the mlp")
    # parser.add_argument("--num-units4", type=int, default=128, help="number of units4 in the mlp")
    # parser.add_argument("--num-units5", type=int, default=128, help="number of units5 in the mlp")
    return parser.parse_args()


def mlp_model_actor1(input, num_outputs, num_units1, num_units2, num_units3, scope, reuse=False):   # p function
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = tf.layers.dense(out, num_units3, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_units3, activation=tf.nn.relu)
        # out = tf.layers.dense(out, num_units2, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_outputs, activation=tf.nn.tanh)
        return tf.multiply(out, 1.0)

def mlp_model_actor2(input, num_outputs, num_units1, num_units2, num_units3, scope, reuse=False):   # p function
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = tf.layers.dense(out, num_units3, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_units3, activation=tf.nn.relu)
        # out = tf.layers.dense(out, num_units2, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_outputs, activation=tf.nn.tanh)
        return tf.multiply(out, 1.0)

def mlp_model_critic(input, num_outputs, num_units1, num_units2, num_units3, scope, reuse=False):    # q_function
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = tf.layers.dense(out, num_units1, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_units1, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_units1, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_outputs)
        return out


def make_env(scenario_name, num_fast_agents, num_norm_agents, num_EVs_fcp, num_EVs_ncp):
    from environment import MultiAgentEnv
    import scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(num_fast_agents, num_norm_agents, num_EVs_fcp, num_EVs_ncp)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.agent_observation, scenario.station_observation)
    return env


def get_trainers(env, obs_shape_n, arglist):
    trainers = []
    model_c = mlp_model_critic
    trainer = MADDPGAgentTrainer
    for i, agent in enumerate(env.agents):
        if agent.fast == True:
            trainers.append(trainer(
                "agent_%d" % i, env, agent, mlp_model_actor1, model_c, obs_shape_n, i, arglist))
        else:
            trainers.append(trainer(
                "agent_%d" % i, env, agent, mlp_model_actor2, model_c, obs_shape_n, i, arglist))
    return trainers


def train(arglist):
    with tf.Session() as sess:
        env = make_env(arglist.scenario, arglist.num_fca, arglist.num_nca, arglist.num_fcp, arglist.num_ncp)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, obs_shape_n, arglist)

        # Initialize
        U.initialize()

        agent_tot_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        agent_c_rewards = [[0.0] for _ in range(env.n)]
        agent_p1_rewards = [[0.0] for _ in range(env.n)]
        agent_tot_loss = [[0.0] for _ in range(env.n)]
        loss_n = [0.0 for _ in range(env.n)]

        obs_n, soc_, e_up, e_down, t_stay_, num_ev = env.reset()

        episode_step = 0
        tot_step = 0
        train_step = 0
        # print(obs_n)
        noise = 2
        print('Starting iterations...')
        start = time.perf_counter()

        while True:
            action_N = []
            # print("1", obs_n)
            # obs_n1 = np.transpose(obs_n, (1, 0, 2))
            # print("2", obs_n1)
            for i, agent in enumerate(env.agents):
                action_n = []
                for j in range(agent.n_piles):
                    # print(i, j)
                    action_n.append(trainers[i].action(obs_n[i][j], j))
                    # action_n = [agent.action(obs[i], i) for agent, obs in zip(trainers, obs_n)]
                action_N.append(action_n)
            for i, agent in enumerate(env.agents):
                for j in range(agent.n_piles):
                    if obs_n[i][j][0] == 0:
                        action_N[i][j] = np.array([0])
                    else:
                        action_N[i][j] = np.clip(np.random.normal(action_N[i][j], noise), -1, 1)
                    # elif not explore:
                    #     action_N[i][j] = np.clip(action_n[i]+np.random.normal(0, noise) * decay, -1, 1)

            new_obs_n, p_n, rew_n, rc_a_n, rp1_n, num_ev1 = env.step(action_N, obs_n, soc_, e_up, e_down, t_stay_)

            if tot_step > (arglist.num_episodes - 50):
                print(obs_n)
                print("price", obs_n[0][0][5])
                print("power0", p_n, "r_n", rew_n, "rc_n", rc_a_n, "rp1_n", rp1_n)

            episode_step += 1
            train_step += 1
            terminal = (episode_step >= arglist.max_episode_len)   # max_episode_len = 15
            if num_ev != 0:
            # if num_ev >= (arglist.num_fca*arglist.num_fcp+arglist.num_nca*arglist.num_ncp)*0.6:
                for i, agent_train in enumerate(trainers):
                    agent_train.experience1(obs_n, action_N, rew_n, new_obs_n, terminal)
            # else:
            #     for i, agent_train in enumerate(trainers):
            #         agent_train.experience2(obs_n, action_N, rew_n, new_obs_n, terminal)
            num_ev = num_ev1
            if tot_step > 1000 and train_step % 2 == 0:
                # for k in range(2):
                loss_n = [0.0 for _ in range(env.n)]
                for i, agent_train in enumerate(trainers):
                    agent_train.preupdate()
                for i, agent_train in enumerate(trainers):
                    loss_n[i] = agent_train.update(trainers, i)[0]

            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                agent_tot_rewards[i][-1] += sum(rew)
            for i, rew in enumerate(rc_a_n):
                agent_c_rewards[i][-1] += sum(rew)
            for i, rew in enumerate(rp1_n):
                agent_p1_rewards[i][-1] += sum(rew)

            for a in agent_tot_loss:
                a.append(0)
            for i, loss in enumerate(loss_n):
                agent_tot_loss[i][-1] += loss_n[i]

            if terminal:
                if tot_step % 100 == 0:
                    print(tot_step)
                obs_n, soc_, e_up, e_down, t_stay_, num_ev = env.reset()
                episode_step = 0
                if tot_step < arglist.num_episodes:
                    for a in agent_tot_rewards:
                        a.append(0)
                    for b in agent_c_rewards:
                        b.append(0)
                    for a in agent_p1_rewards:
                        a.append(0)
                    # for a in agent_p2_rewards:
                    #     a.append(0)
                # agent_p_rewards.append(0)
                # max_tot_power.append(0)
                # min_tot_power.append(0)
                tot_step += 1
            if tot_step > arglist.num_episodes:
                end = time.perf_counter()
                break
            # if tot_step > arglist.memory_size and train_step % 5 == 0:
            if train_step % 2 == 0 and noise > 0.1 and tot_step > 1000:  #tot_step > arglist.memory_size
                noise *= 0.9998  # 不协同的reward取0.99  1-（tot_step / num-episodes）

        print(start, end)

        saver = tf.compat.v1.train.Saver()
        saver.save(sess, './save_pl_mi_10+5-0815/params', write_meta_graph=False)

        plotma(
               agent_tot_rewards[0], agent_tot_rewards[1], #agent_tot_rewards[2],
               #max_tot_power, min_tot_power,
               agent_c_rewards[0], agent_c_rewards[1], #, agent_c_rewards[2],#, agent_c_rewards[3],# agent_c_rewards[4],
               agent_p1_rewards[0], agent_p1_rewards[0], #agent_p_rewards[1], agent_p_rewards[2], # agent_p_rewards[3], agent_p_rewards[4],
               #agent_p2_rewards[0],# agent_p2_rewards[1],# agent_p2_rewards[2],# agent_p2_rewards[3],
               agent_tot_loss[0], agent_tot_loss[1])#, agent_tot_loss[2])#, agent_tot_loss[3]) #, agent_tot_loss[3], agent_tot_loss[4])


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)

