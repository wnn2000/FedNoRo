import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')

    # basic setting
    parser.add_argument('--exp', type=str,
                        default='Fed', help='experiment name')
    parser.add_argument('--dataset', type=str,
                        default='ICH', help='dataset name')
    parser.add_argument('--model', type=str,
                        default='Resnet18', help='model name')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float,  default=3e-4,
                        help='base learning rate')

    # for FL
    parser.add_argument('--n_clients', type=int,  default=20,
                        help='number of users') 
    parser.add_argument('--iid', type=int, default=0, help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float,
                        default=0.9, help='parameter for non-iid')
    parser.add_argument('--alpha_dirichlet', type=float,
                        default=2.0, help='parameter for non-iid')
    parser.add_argument('--local_ep', type=int, default=5, help='local epoch')
    parser.add_argument('--rounds', type=int,  default=100, help='rounds')

    parser.add_argument('--s1', type=int,  default=10, help='stage 1 rounds')
    parser.add_argument('--begin', type=int,  default=10, help='ramp up begin')
    parser.add_argument('--end', type=int,  default=49, help='ramp up end')
    parser.add_argument('--a', type=float,  default=0.8, help='a')

    # noise
    parser.add_argument('--level_n_system', type=float, default=0.4, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.3, help="lower bound of noise level")
    parser.add_argument('--level_n_upperb', type=float, default=0.5, help="upper bound of noise level")
    parser.add_argument('--n_type', type=str, default="instance", help="type of noise")

    
    args = parser.parse_args()
    return args