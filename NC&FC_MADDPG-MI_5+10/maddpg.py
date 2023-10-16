import numpy as np
import tensorflow.compat.v1 as tf
import tf_util as U
from scenarios import AgentTrainer
from replay_buffer import ReplayBuffer


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    # for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
    for var, var_target in zip(vals, target_vals):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def p_train(make_obs_ph_n, make_act_ph_n, num_piles, p_func, q_func, num_units1, num_units2, num_units3, s_idx, optimizer, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        obs_ph_n = make_obs_ph_n
        act_ph_n = make_act_ph_n
        act = []
        target_act = []
        p_values = []
        act_input_n = act_ph_n + []
        # print("1", act_ph_n)
        for i in range(num_piles):
            p_input = tf.concat([obs_ph_n[s_idx+i]], axis=1)
            p = p_func(p_input, 1, num_units1, num_units2, num_units3, scope="p_func", reuse=tf.AUTO_REUSE)
            act.append(U.function(inputs=[obs_ph_n[s_idx + i]], outputs=p))
            act_input_n[s_idx + i] = p
            target_p = p_func(p_input, 1, num_units1, num_units2, num_units3, scope="target_p_func", reuse=tf.AUTO_REUSE)
            target_act.append(U.function(inputs=[obs_ph_n[s_idx + i]], outputs=target_p))
            p_values.append(U.function([obs_ph_n[s_idx + i]], p))
        # print("2", act_ph_n)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        q_input = tf.concat(obs_ph_n + act_input_n, axis=-1)
        q = q_func(q_input, 1, num_units1, num_units2, num_units3, scope="q_func", reuse=True)[:, 0]
        pg_loss = -tf.reduce_mean(q)
        loss = pg_loss #+ p_reg * 1e-4
        optimize_expr = optimizer.minimize(loss, var_list=p_func_vars)
        # optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars)
        # print(obs_ph_n + act_ph_n)
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        # p_values = U.function([obs_ph_n[p_index]], p)

        # target_p = p_func(p_input, 1, num_units1, num_units2, num_units3, scope="target_p_func")
        # target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
        # target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_p)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}  #  'p_values': p_values,


def q_train(make_obs_ph_n, make_act_ph_n, q_func, num_units1, num_units2, num_units3, optimizer, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        obs_ph_n = make_obs_ph_n
        act_ph_n = make_act_ph_n
        # print(np.shape(obs_ph_n), np.shape(act_ph_n))
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        q_input = tf.concat(obs_ph_n + act_ph_n, axis=-1)
        q = q_func(q_input, 1, num_units1, num_units2, num_units3, scope="q_func")[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        q_loss = tf.reduce_mean(tf.square(q - target_ph))
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss
        optimize_expr = optimizer.minimize(loss, var_list=q_func_vars)
        # optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss,
                           updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        target_q = q_func(q_input, 1, num_units1, num_units2, num_units3, scope="target_q_func")[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)  # soft update
        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, env, agent, model_a, model_c, obs_shape_n, agent_index, args):
        self.name = name
        self.n = len(obs_shape_n)
        self.env = env
        self.args = args
        self.act = []
        obs_ph_N = []
        act_ph_N = []
        state_ind = 0
        for i, agent1 in enumerate(env.agents):
            if i < agent_index:
                state_ind += agent1.n_piles
            for j in range(agent1.n_piles):
                obs_ph_N.append(U.BatchInput(obs_shape_n[i], name="cs"+str(i)+"pile"+str(j)).get())
                act_ph_N.append(tf.placeholder(shape=[None, 1], name="act"+str(i)+"pile"+str(j), dtype=tf.float32))
        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_N,
            make_act_ph_n=act_ph_N,
            q_func=model_c,  # net
            num_units1=args.num_units1,
            num_units2=args.num_units2,
            num_units3=args.num_units3,
            # num_units4=args.num_units4,
            # num_units5=args.num_units5,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr_c),
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_N,
            make_act_ph_n=act_ph_N,
            num_piles=agent.n_piles,
            p_func=model_a,
            q_func=model_c,
            num_units1=args.num_units1,
            num_units2=args.num_units2,
            num_units3=args.num_units3,
            s_idx=state_ind,
            # num_units4=args.num_units4,
            # num_units5=args.num_units5,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr_a),
        )

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(args.memory_size * args.max_episode_len)  # * args.num_EVs_per_agent)
        self.replay_sample_index = None

    def action(self, ag_obs, pile_index):
        # print(self.act)
        return self.act[pile_index](ag_obs[None])[0]

    def experience1(self, obs, act, rew, new_obs, done):
        self.replay_buffer.add1(obs, act, rew, new_obs, float(done))

    def experience2(self, obs, act, rew, new_obs, done):
        self.replay_buffer.add2(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, agent_index):
        state_ind = 0
        for i, agent in enumerate(self.env.agents):
            if i < agent_index:
                state_ind += agent.n_piles

        # self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # index = self.replay_sample_index
        # obs_n, act_n, rew_n, obs_next_n, done_n = self.replay_buffer.sample_index(index)
        self.replay_sample_index1 = self.replay_buffer.make_index1(int(self.args.batch_size*1))
        # self.replay_sample_index2 = self.replay_buffer.make_index2(int(self.args.batch_size*1/4))
        index1 = self.replay_sample_index1
        # index2 = self.replay_sample_index2
        obs_n, act_n, rew_n, obs_next_n, done_n = self.replay_buffer.sample_index(index1)
        #-----------------------------------------------------
        # obs_n = np.transpose(obs_n, (1, 2, 0, 3))
        # act_n = np.transpose(act_n, (1, 2, 0, 3))
        # rew_n = np.transpose(rew_n, (1, 2, 0))
        # obs_next_n = np.transpose(obs_next_n, (1, 2, 0, 3))
        # obs_N = []
        # act_N = []
        # obs_N_next = []
        # for i, agent in enumerate(self.env.agents):
        #     for j in range(agent.n_piles):
        #         obs_N.append(obs_n[i][j])
        #         act_N.append(act_n[i][j])
        #         obs_N_next.append(obs_next_n[i][j])
        # print(np.shape(obs_N))
        #----------------------------------------------------
        obs_N = []
        act_N = []
        rew_N = []
        obs_N_next = []
        for i, agent in enumerate(self.env.agents):
            rew_N2 = []
            for j in range(agent.n_piles):
                obs_N1 = []
                act_N1 = []
                rew_N1 = []
                obs_N_next1 = []
                for k in range(self.args.batch_size):
                    obs_N1.append(obs_n[k][i][j])
                    act_N1.append(act_n[k][i][j])
                    rew_N1.append(rew_n[k][i][j])
                    obs_N_next1.append(obs_next_n[k][i][j])
                obs_N.append(obs_N1)
                act_N.append(act_N1)
                obs_N_next.append(obs_N_next1)
                rew_N2.append(rew_N1)
            rew_N.append(rew_N2)
        # print(np.shape(obs_N_next))
        #-----------------------------------------------------

        target_q = 0.0

        target_act_next_n = []
        for i, agent in enumerate(self.env.agents):
            for j in range(agent.n_piles):
                target_act_next_n.append(agents[i].p_debug['target_act'][j](
                    *([obs_N_next[state_ind + j]])))# + np.random.normal(0, 0.1))

        target_q_next = self.q_debug['target_q_values'](*(obs_N_next + target_act_next_n))

        n_piles = self.env.agents[agent_index].n_piles
        for k in range(n_piles):
            target_q += rew_N[agent_index][k] + self.args.gamma * (1.0 - done_n) * target_q_next
        target_q /= n_piles
        q_loss = self.q_train(*(obs_N + act_N + [target_q]))
        p_loss = self.p_train(*(obs_N + act_N))
        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew_N[agent_index]), np.mean(target_q_next), np.std(target_q)]
