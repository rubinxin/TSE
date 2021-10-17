"""
Regularized evolution as described in:
Real, E., Aggarwal, A., Huang, Y., and Le, Q. V.
Regularized Evolution for Image Classifier Architecture Search.
In Proceedings of the Conference on Artificial Intelligence (AAAI’19)

The code is based one the original regularized evolution open-source implementation:
https://colab.research.google.com/github/google-research/google-research/blob/master/evolution/regularized_evolution_algorithm/regularized_evolution.ipynb

NOTE: This script has certain deviations from the original code owing to the search space of the benchmarks used:
1) The fitness function is not accuracy but error and hence the negative error is being maximized.
2) The architecture is a ConfigSpace object that defines the model architecture parameters.

"""

import argparse
import collections
import os
import random
from copy import deepcopy

import ConfigSpace
import numpy as np
import pickle
from nas201func import NAS201


class Model(object):
    """A class representing a model.

    It holds two attributes: `arch` (the simulated architecture) and `accuracy`
    (the simulated accuracy / fitness). See Appendix C for an introduction to
    this toy problem.

    In the real case of neural networks, `arch` would instead hold the
    architecture of the normal and reduction cells of a neural network and
    accuracy would be instead the result of training the neural net and
    evaluating it on the validation set.

    We do not include test accuracies here as they are not used by the algorithm
    in any way. In the case of real neural networks, the test accuracy is only
    used for the purpose of reporting / plotting final results.

    In the context of evolutionary algorithms, a model is often referred to as
    an "individual".

    Attributes:  (as in the original code)
      arch: the architecture as an int representing a bit-string of length `DIM`.
          As a result, the integers are required to be less than `2**DIM`. They
          can be visualized as strings of 0s and 1s by calling `print(model)`,
          where `model` is an instance of this class.
      accuracy:  the simulated validation accuracy. This is the sum of the
          bits in the bit-string, divided by DIM to produce a value in the
          interval [0.0, 1.0]. After that, a small amount of Gaussian noise is
          added with mean 0.0 and standard deviation `NOISE_STDEV`. The resulting
          number is clipped to within [0.0, 1.0] to produce the final validation
          accuracy of the model. A given model will have a fixed validation
          accuracy but two models that have the same architecture will generally
          have different validation accuracies due to this noise. In the context
          of evolutionary algorithms, this is often known as the "fitness".
    """

    def __init__(self):
        self.arch = None
        self.accuracy = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return '{0:b}'.format(self.arch)


def train_and_eval(config):
    y, cost = b.objective_function(config)
    # returns negative error (similar to maximizing accuracy)
    return -y


def random_architecture():
    config = cs.sample_configuration()
    return config


def mutate_arch(parent_arch):
    # pick random parameter
    dim = np.random.randint(len(cs.get_hyperparameters()))
    hyper = cs.get_hyperparameters()[dim]

    if type(hyper) == ConfigSpace.OrdinalHyperparameter:
        choices = list(hyper.sequence)
    else:
        choices = list(hyper.choices)
    # drop current values from potential choices
    choices.remove(parent_arch[hyper.name])

    # flip parameter
    idx = np.random.randint(len(choices))

    child_arch = deepcopy(parent_arch)
    child_arch[hyper.name] = choices[idx]
    return child_arch


def regularized_evolution(cycles, population_size, sample_size, init_data=None):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    Args:
      cycles: the number of cycles the algorithm should run for.
      population_size: the number of individuals to keep in the population.
      sample_size: the number of individuals that should participate in each
          tournament.

    Returns:
      history: a list of `Model` instances, representing all the models computed
          during the evolution experiment.
    """
    population = collections.deque()
    history = []  # Not used by the algorithm, only used to report results.

    # Initialize the population with initial data if present.
    if init_data is not None:
        for x, y in zip(init_data['x'], init_data['nlogy']):
            model = Model()
            model.arch = x
            # model.accuracy = np.exp(-y)
            model.accuracy = train_and_eval(model.arch)
            population.append(model)

    # Add up to the population size by random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture()
        model.accuracy = train_and_eval(model.arch)
        population.append(model)
        history.append(model)

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.accuracy)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch)
        child.accuracy = train_and_eval(child.arch)
        population.append(child)
        history.append(child)

        # Remove the oldest model.
        population.popleft()

    return history


parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', default="nasbench201_cifar100", type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--n_iters', default=200, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results/", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../data/", type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--pop_size', default=100, type=int, nargs='?', help='population size')
parser.add_argument('--sample_size', default=10, type=int, nargs='?', help='sample_size')
parser.add_argument('--n_repeat', type=int, default=20, help='number of repeats of experiments')
parser.add_argument('--fixed_query_seed', type=int, default=2,
                    help='Whether to use deterministic objective function as NAS-Bench-101 has 3 different seeds for '
                         'validation and test accuracies. Options in [None, 0, 1, 2]. If None the query will be '
                         'random.')
parser.add_argument('--n_init', type=int, default=10, help='number of initial data')
parser.add_argument('--eval_metric', default="final_val", type=str, help='evaluation metric: final_val, early_stop, tseema')
parser.add_argument('--es_budget', type=int, default=200, help='number of early-stop epochs: =200 for final_val and =10 for early stop and tseema')

args = parser.parse_args()

if args.eval_metric != 'final_val':
    args.n_iters = args.n_iters*int(200/args.es_budget)

# load init data
if "nasbench201" in args.benchmark:
    init_data = pickle.load(open(f'{args.data_dir}init_data_nasbench201', 'rb'))

for data in init_data:
    data['x'] = data['x'][:args.n_init]
    data['nlogy'] = data['nlogy'][:args.n_init]

if "nasbench201" in args.benchmark:
    task_name = args.benchmark.split('_')[-1]
    b = NAS201(data_dir=args.data_dir, task=task_name, seed=args.fixed_query_seed, metric=args.eval_metric, es_budget=args.es_budget)

output_dir = os.path.join(args.output_path, args.benchmark)
os.makedirs(output_dir, exist_ok=True)
result_path = os.path.join(output_dir, f"regularized_evolution_{args.eval_metric}{args.es_budget}")

cs = b.get_configuration_space()

results_all_seed = []
for s in range(args.n_repeat):
    np.random.seed(s)
    random.seed(s)

    b.reset_tracker()

    history = regularized_evolution(cycles=args.n_iters, population_size=args.pop_size, sample_size=args.sample_size,
                                    init_data=init_data[s])

    res = b.get_results(ignore_invalid_configs=True)

    results_all_seed.append(res)
pickle.dump(results_all_seed, open(result_path, 'wb'))

