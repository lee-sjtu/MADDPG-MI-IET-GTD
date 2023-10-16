import tensorflow.compat.v1 as tf
import tf_util as U
from scenarios import AgentTrainer

def p_train(make_obs_ph_n, p_index, num_piles, p_func, num_units1, num_units2, num_units3, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        obs_ph_n = make_obs_ph_n
        act = []

        for i in range(num_piles):
            p_input = tf.concat([obs_ph_n[p_index * num_piles + i]], axis=1)
            p = p_func(p_input, 1, num_units1, num_units2, num_units3, scope="p_func", reuse=tf.AUTO_REUSE)
            act.append(U.function(inputs=[obs_ph_n[p_index * num_piles + i]], outputs=p))

        return act

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model_a, obs_shape_n, agent_index, args):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        # self.act = []
        obs_ph_N = []
        for i in range(self.n):
            for j in range(args.num_EVs_per_agent):
                obs_ph_N.append(U.BatchInput(obs_shape_n[i], name="obs"+str(i)+"pile"+str(j)).get())

        self.act = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_N,
            p_index=agent_index,
            num_piles=args.num_EVs_per_agent,
            p_func=model_a,
            num_units1=args.num_units1,
            num_units2=args.num_units2,
            num_units3=args.num_units3
        )

    def action(self, ag_obs, pile_index):
        return self.act[pile_index](ag_obs[None])[0]
