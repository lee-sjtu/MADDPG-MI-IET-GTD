import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--scenario", type=str, default="one_EV_station", help="name of the scenario script")
    parser.add_argument("--num-nca", type=int, default=5, help="number of agents for norm charge stations")
    parser.add_argument("--num-fca", type=int, default=0, help="number of agents for fast charge stations")
    # parser.add_argument("--num-ncp", type=int, default=3, help="number of norm charge piles")
    # parser.add_argument("--num-fcp", type=int, default=0, help="number of fast charge piles")
    # Core training parameters
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    # parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    parser.add_argument('--n_steps', type=int, default=15*5000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="Adam", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=15*1, help='how often to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=100, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
    args = parser.parse_args()
    return args


# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 64
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 1e-3

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.1
    anneal_steps = 15000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 4

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(2000)

    # how often to save the model
    args.save_cycle = 50000

    # how often to update the target_net
    args.target_update_cycle = 50

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args

