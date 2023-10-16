import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import time
import tf_util as U
from maddpg import MADDPGAgentTrainer
from plotmoving import plotma

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="one_EV_station", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=14, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=5000, help="number of episodes")
    parser.add_argument("--memory-size", type=int, default=2000, help="size of memory")
    parser.add_argument("--num-norm-piles", type=int, default=1, help="number of norm charge piles")
    parser.add_argument("--num-fast-piles", type=int, default=0, help="number of fast charge piles")
    parser.add_argument("--num-EVs-per-agent", type=int, default=5, help="number of EVs")
    # Core training parameters
    parser.add_argument("--lr-a", type=float, default=1e-3, help="learning rate for actor network")
    parser.add_argument("--lr-c", type=float, default=1e-3, help="learning rate for critic network")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units1", type=int, default=64, help="number of units1 in the mlp")
    parser.add_argument("--num-units2", type=int, default=64, help="number of units2 in the mlp")
    parser.add_argument("--num-units3", type=int, default=64, help="number of units3 in the mlp")
    # parser.add_argument("--num-units4", type=int, default=128, help="number of units4 in the mlp")
    # parser.add_argument("--num-units5", type=int, default=128, help="number of units5 in the mlp")
    return parser.parse_args()


def mlp_model_actor(input, num_outputs, num_units1, num_units2, num_units3, scope, reuse=False):   # p function
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


def mlp_model_critic(input, num_outputs, num_units1, num_units2, num_units3, scope, reuse=False):    # q_function
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
    from environment import MultiAgentEnv
    import scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(num_fast_agents, num_norm_piles, num_EVs_per_agent)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.agent_observation, scenario.station_observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model_a = mlp_model_actor
    model_c = mlp_model_critic
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):  # num adversary agent
        trainers.append(trainer(
            "agent_%d" % i, model_a, model_c, obs_shape_n, env.action_space, i, arglist))
    return trainers


def train(arglist):
    with tf.Session() as sess:
        env = make_env(arglist.scenario, arglist.num_fast_piles, arglist.num_norm_piles, arglist.num_EVs_per_agent)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_piles = env.n   # , arglist.num_piles)
        trainers = get_trainers(env, num_piles, obs_shape_n, arglist)

        # Initialize
        U.initialize()

        agent_tot_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        agent_c_rewards = [[0.0] for _ in range(env.n)]
        agent_p_rewards = [[0.0] for _ in range(env.n)]
        # max_tot_power = [0.0]
        # min_tot_power = [0.0]
        # agent_p2_rewards = [[0.0] for _ in range(env.n)]
        tot_rewards = [0.0]
        agent_tot_loss = [[0.0] for _ in range(env.n)]
        loss_n = [0.0 for _ in range(env.n)]

        obs_n, soc_, e_up, e_down, t_stay_, num_ev = env.reset(arglist.num_EVs_per_agent)

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
            obs_n1 = np.transpose(obs_n, (1, 0, 2))
            # print("2", obs_n1)
            for i in range(arglist.num_EVs_per_agent):
                action_n = [agent.action(obs[i], i) for agent, obs in zip(trainers, obs_n1)]
                action_N.append(action_n)
            for i in range(arglist.num_EVs_per_agent):
                for j in range(len(action_n)):
                    if obs_n[i][j][0] == 0:
                        action_N[i][j] = np.array([0])
                    else:
                        action_N[i][j] = np.clip(np.random.normal(action_N[i][j], noise), -1, 1)
                    # elif not explore:
                    #     action_N[i][j] = np.clip(action_n[i]+np.random.normal(0, noise) * decay, -1, 1)

            new_obs_n, p_n, rew_n, rc_a_n, rp1_n, num_ev1 = env.step(action_N, obs_n, arglist.num_EVs_per_agent, soc_, e_up, e_down, t_stay_, tot_step)
            # print(rp1_n)
            if tot_step > (arglist.num_episodes - 50):
                # soc = []
                for i in range(arglist.num_EVs_per_agent):
                    for j in range(len(action_n)):
                #         soc.append(obs_n[i][j][0])
                        print(rew_n[i][j], rc_a_n[i][j], rp1_n[i][j])
                # print("soc0", soc)#, "soc2", obs_n[0][2][0])
                # print("price", obs_n[0][0][5])
                # print("price", obs_n[0][1][7])
                # print("power0", p_n, "total", sum(sum(p_n)), "r_n", rew_n, "rc_n", rc_a_n, "rp1_n", rp1_n)
            # if sum(sum(p_n)) > 30:
            #     print("power0", p_n, "total", sum(sum(p_n)), "r_n", rew_n, "rc_n", rc_a_n, "rp1_n", rp1_n)
            episode_step += 1
            train_step += 1
            terminal = (episode_step >= arglist.max_episode_len)   # max_episode_len = 15
            # print(np.shape(obs_n))
            if num_ev != 0:
                for i, agent in enumerate(trainers):
                    agent.experience1(obs_n, action_N, rew_n, new_obs_n, terminal)
            # if num_ev != 0 and rp1_n[0][0] != 0:
            #     for i, agent in enumerate(trainers):
            #         agent.experience2(obs_n, action_N, rew_n, new_obs_n, terminal)
            num_ev = num_ev1
            if tot_step > 1000 and train_step % 2 == 0: #tot_step > arglist.memory_size
            # if train_step % 10 == 0:
                loss_n = [0.0 for _ in range(env.n)]
                for i, agent in enumerate(trainers):
                    agent.preupdate()
                for i, agent in enumerate(trainers):
                    loss_n[i] = agent.update(trainers, i)[0]

            obs_n = new_obs_n
            # print("4", rew_n)
            rew_n = np.transpose(rew_n, (1, 0))
            rc_a_n = np.transpose(rc_a_n, (1, 0))
            rp1_n = np.transpose(rp1_n, (1, 0))
            # print("5", rew_n)
            for i, rew in enumerate(rew_n):
                # print(i, len(rew_n), agent_tot_rewards)
                agent_tot_rewards[i][-1] += sum(rew)
            for i, rew in enumerate(rc_a_n):
                # print("4", i, rc_a_n)
                agent_c_rewards[i][-1] += sum(rew)
            for i, rew in enumerate(rp1_n):
                # print("5", i, rp1_n)
                agent_p_rewards[i][-1] += sum(rew)/5
            # tot_rewards [-1] =
            # print(agent_p_rewards[-1], rp1_n)
            # agent_p_rewards[-1] += rp1_n

            # if max_tot_power[-1] < sum(sum(p_n)):
            #     max_tot_power[-1] = sum(sum(p_n))
            # if min_tot_power[-1] > sum(sum(p_n)):
            #     min_tot_power[-1] = sum(sum(p_n))

            for a in agent_tot_loss:
                a.append(0)
            for i, loss in enumerate(loss_n):
                agent_tot_loss[i][-1] += loss_n[i]

            if terminal:
                if tot_step % 100 == 0:
                    print(tot_step)
                obs_n, soc_, e_up, e_down, t_stay_, num_ev = env.reset(arglist.num_EVs_per_agent)
                episode_step = 0
                # if tot_step < arglist.num_episodes:
                for a in agent_tot_rewards:
                    a.append(0)
                for b in agent_c_rewards:
                    b.append(0)
                for a in agent_p_rewards:
                    a.append(0)
                # tot_rewards.append(0)
                # max_tot_power.append(0)
                # min_tot_power.append(0)
                tot_step += 1

            # if tot_step > arglist.memory_size and train_step % 5 == 0:
            if train_step % 2 == 0 and noise > 0.1 and tot_step > 1000:  #tot_step > arglist.memory_size
                noise *= 0.9995  # 不协同的reward取0.99  1-（tot_step / num-episodes）

            # if tot_step > arglist.memory_size:
            #     if decay > 0.1:
            #         decay = 1-(train_step / 6000 / arglist.max_episode_len)
            #     else:
            #         decay = 0

            if tot_step > arglist.num_episodes:
                end = time.perf_counter()
                break

        print(start, end)

        # saver = tf.compat.v1.train.Saver()
        # saver.save(sess, './save17-MI-5EV-poli-0915/params', write_meta_graph=False)

        plotma(
               agent_tot_rewards[0], #agent_tot_rewards[1], agent_tot_rewards[2],
               #max_tot_power, min_tot_power,
               agent_c_rewards[0],# agent_c_rewards[1],#, agent_c_rewards[2],#, agent_c_rewards[3],# agent_c_rewards[4],
               agent_p_rewards[0],# agent_p_rewards[0], agent_p_rewards[1], agent_p_rewards[2], # agent_p_rewards[3], agent_p_rewards[4],
               #agent_p2_rewards[0],# agent_p2_rewards[1],# agent_p2_rewards[2],# agent_p2_rewards[3],
               agent_tot_loss[0])#,# agent_tot_loss[1])#, agent_tot_loss[2])#, agent_tot_loss[3]) #, agent_tot_loss[3], agent_tot_loss[4])


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)

