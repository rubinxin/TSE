import os
import argparse
import pickle
import numpy as np
import random
from nas201func import NAS201

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', default="nasbench201_cifar100", type=str, nargs='?',
                    help='specifies the benchmark: nasbench101_cifar10')
parser.add_argument('--n_iters', default=2000, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results/", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../data/", type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--n_repeat', type=int, default=20, help='number of repeats of experiments')
parser.add_argument('--fixed_query_seed', type=int, default=3,
                    help='Whether to use deterministic objective function as NAS-Bench-101 has 3 different seeds for '
                         'validation and test accuracies. Options in [None, 0, 1, 2]. If None the query will be '
                         'random.')
parser.add_argument('--n_init', type=int, default=10, help='number of initial data')
parser.add_argument('--eval_metric', default="tseema", type=str, help='evaluation metric: final_val, early_stop, sum_trainloss')
parser.add_argument('--es_budget', type=int, default=100, help='number of early-stop epochs')

args = parser.parse_args()

if args.eval_metric != 'final_val':
    args.n_iters = args.n_iters*int(200/args.es_budget)

# load init data
if "nasbench201" in args.benchmark:
    init_data = pickle.load(open('../data/init_data_nasbench201', 'rb'))

if "nasbench201" in args.benchmark:
    task_name = args.benchmark.split('_')[-1]
    b = NAS201(data_dir=args.data_dir, task=task_name, seed=args.fixed_query_seed, metric=args.eval_metric, es_budget=args.es_budget)

output_dir = os.path.join(args.output_path, args.benchmark)
os.makedirs(output_dir, exist_ok=True)
result_path = os.path.join(output_dir, f"random_search_test_{args.eval_metric}{args.es_budget}")

cs = b.get_configuration_space()

results_all_seed = []
for s in range(args.n_repeat):

    np.random.seed(s)
    random.seed(s)

    b.reset_tracker()

    if init_data is not None:
        for i in range(args.n_init):
            x = init_data[s]['x'][i]
            config = x
            b.objective_function(config)

    for i in range(args.n_iters):
        config = cs.sample_configuration()
        b.objective_function(config)

    res = b.get_results(ignore_invalid_configs=True)
    results_all_seed.append(res)
pickle.dump(results_all_seed, open(result_path, 'wb'))

