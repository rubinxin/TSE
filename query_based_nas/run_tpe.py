import os
from copy import deepcopy
import ConfigSpace
import argparse
import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import random
from nas201func import NAS201

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', default="nasbench201_cifar100", type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--n_iters', default=20, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results/", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../data/", type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--n_repeat', type=int, default=1, help='number of repeats of experiments')
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

if "nasbench201" in args.benchmark:
    task_name = args.benchmark.split('_')[-1]
    b = NAS201(data_dir=args.data_dir, task=task_name, seed=args.fixed_query_seed, metric=args.eval_metric, es_budget=args.es_budget)

output_dir = os.path.join(args.output_path, args.benchmark)
os.makedirs(output_dir, exist_ok=True)
result_path = os.path.join(output_dir, f"tpe_{args.eval_metric}{args.es_budget}")

cs = b.get_configuration_space()

space = {}
for h in cs.get_hyperparameters():
    if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
        space[h.name] = hp.quniform(h.name, 0, len(h.sequence)-1, q=1)
    elif type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
        space[h.name] = hp.choice(h.name, h.choices)
    elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
        space[h.name] = hp.quniform(h.name, h.lower, h.upper, q=1)
    elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
        space[h.name] = hp.uniform(h.name, h.lower, h.upper)


def objective(x):
    config = deepcopy(x)
    for h in cs.get_hyperparameters():
        if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
            
            config[h.name] = h.sequence[int(x[h.name])]

        elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:

            config[h.name] = int(x[h.name])
    y, c = b.objective_function(config)

    return {
        'config': config,
        'loss': y,
        'cost': c,
        'status': STATUS_OK}

# load init data
if "nasbench201" in args.benchmark:
    init_data = pickle.load(open('../data/init_data_nasbench201', 'rb'))

results_all_seed = []
for s in range(args.n_repeat):
    np.random.seed(s)
    random.seed(s)

    b.reset_tracker()

    # prepare initial data in tpe format
    init_data_for_TPE = []
    for i in range(args.n_init):
        x   = init_data[s]['x'][i]
        out = objective(x)
        init_data_for_TPE.append(out)

    # run tpe
    trials = Trials()
    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=args.n_iters,
                points_to_evaluate=init_data_for_TPE,
                trials=trials)

    res = b.get_results(ignore_invalid_configs=True)

    results_all_seed.append(res)
pickle.dump(results_all_seed, open(result_path, 'wb'))

