import tensorflow.compat.v1 as tf
import tf_util as U
from scenarios import AgentTrainer

def p_train(make_obs_ph_n, p_index, num_piles, p_func, num_units1, num_units2, num_units3, s_idx, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        obs_ph_n = make_obs_ph_n
        act = []
        for i in range(num_piles):
            # p_input = tf.concat([obs_ph_n[p_index * num_piles + i]], axis=1)
            p_input = tf.concat([obs_ph_n[s_idx + i]], axis=1)
            p = p_func(p_input, 1, num_units1, num_units2, num_units3, scope="p_func", reuse=tf.AUTO_REUSE)
            act.append(U.function(inputs=[obs_ph_n[s_idx + i]], outputs=p))
        return act

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, env, agent, model_a, obs_shape_n, agent_index, args):
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
        self.act = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_N,
            p_index=agent_index,
            num_piles=agent.n_piles,
            p_func=model_a,
            num_units1=args.num_units1,
            num_units2=args.num_units2,
            num_units3=args.num_units3,
            s_idx=state_ind,
        )

    def action(self, ag_obs, pile_index):
        return self.act[pile_index](ag_obs[None])[0]
