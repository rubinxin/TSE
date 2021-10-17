import os
from nas_201_api import NASBench201API as API

import ConfigSpace
import numpy as np
import random


class NAS201(object):

    def __init__(self, data_dir, task='cifar10-valid', seed=None, metric='final_val', es_budget=None):

        self.api = API(os.path.join(data_dir, 'NAS-Bench-201-v1_1-096897.pth'))
        self.task = task
        self.X = []
        self.y_valid = []
        self.y_final_valid = []
        self.y_test = []
        self.costs = []
        self.seed = seed
        self.metric = metric
        self.es_budget = es_budget
        if metric == 'final_val':
            self.es_budget = None


        if task == 'cifar10-valid':
            best_val_arch_index = 6111
            self.y_star_valid = 1 - 91.60666665039064 / 100
            best_test_arch_index = 1459
            self.y_star_test = 1 - 91.52333333333333 / 100
        elif task == 'cifar100':
            best_val_arch_index = 9930
            self.y_star_valid = 1 - 73.49333323567708 / 100
            best_test_arch_index = 9930
            self.y_star_test = 1 - 73.51333326009114 / 100
        elif task == 'ImageNet16-120':
            best_val_arch_index = 10676
            self.y_star_valid = 1 - 46.766666727701825 / 100
            best_test_arch_index = 857
            self.y_star_test = 1 - 47.311111097547744 / 100
        else:
            raise NotImplementedError("task" + str(task) + " is not implemented in the dataset.")


    def reset_tracker(self):
        self.X = []
        self.y_valid = []
        self.y_final_valid = []
        self.y_test = []
        self.costs = []

    def objective_function(self, config, budget=199):
        window_size = 1
        mu = 0.9

        if self.es_budget is not None:
            budget = self.es_budget
        #  set random seed for evaluation
        seed_list = [777, 888, 999]
        if self.seed is None:
            seed = random.choice(seed_list)
        elif self.seed >= 3:
            seed = self.seed
        else:
            seed = seed_list[self.seed]

        # find architecture index
        op_node_labelling = [config["edge_%d" % i] for i in range(len(config.keys()))]
        arch_str = f'|{op_node_labelling[0]}~0|+' \
                   f'|{op_node_labelling[1]}~0|{op_node_labelling[2]}~1|+' \
                   f'|{op_node_labelling[3]}~0|{op_node_labelling[4]}~1|{op_node_labelling[5]}~2|'

        try:
            arch_index = self.api.query_index_by_arch(arch_str)
            if self.seed >= 3:
                # some architectures only contain 1 seed result
                # get training losses
                train_loss_i = []

                for e in range(budget):

                    train_loss_i_e = self.api.get_more_info(arch_index, self.task, e, False, False)['train-loss']
                    if e <= 0:
                        ema = train_loss_i_e
                    else:
                        ema = ema * (1 - mu) + mu * train_loss_i_e
                    train_loss_i.append(train_loss_i_e)
                sum_train_loss = np.sum(train_loss_i[-window_size:])
                sum_ema_train_loss = ema

                if self.metric == 'final_val':
                    acc_results = self.api.get_more_info(arch_index, self.task, None, False, False)
                    val_acc = acc_results['valid-accuracy'] / 100

                else:
                    acc_results = self.api.get_more_info(arch_index, self.task, budget, False, False)
                    if self.task == 'cifar10-valid':
                        val_acc = acc_results['valid-accuracy'] / 100
                    else:
                        val_acc = acc_results['valtest-accuracy'] / 100
                train_time = acc_results['train-all-time']

                final_acc_results = self.api.get_more_info(arch_index, self.task, None, False, False)
                test_acc = final_acc_results['test-accuracy'] / 100
                final_val_acc = final_acc_results['valid-accuracy'] / 100
            else:
                try:
                    # some architectures only contain 1 seed result
                    # get training losses
                    train_loss_i = []
                    for e in range(budget):
                        train_loss_i_e = self.api.get_more_info(arch_index, self.task, e, False, is_random=seed)['train-loss']
                        if e <= 0:
                            ema = train_loss_i_e
                        else:
                            ema = ema * (1 - mu) + mu * train_loss_i_e
                        train_loss_i.append(train_loss_i_e)
                    sum_train_loss = np.sum(train_loss_i[-window_size:])
                    sum_ema_train_loss = ema

                    if self.metric == 'final_val':
                        acc_results = self.api.get_more_info(arch_index, self.task, None, False, is_random=seed)
                        val_acc = acc_results['valid-accuracy'] / 100

                    else:
                        acc_results = self.api.get_more_info(arch_index, self.task, budget, False, is_random=seed)
                        if self.task == 'cifar10-valid':
                            val_acc = acc_results['valid-accuracy'] / 100
                        else:
                            val_acc = acc_results['valtest-accuracy'] / 100
                    train_time = acc_results['train-all-time']

                    final_acc_results = self.api.get_more_info(arch_index, self.task, None, False, is_random=seed)
                    test_acc = final_acc_results['test-accuracy'] / 100
                    final_val_acc = final_acc_results['valid-accuracy'] / 100

                except:
                    # some architectures only contain 1 seed result
                    train_loss_i = []

                    for e in range(budget):
                        train_loss_i_e = self.api.get_more_info(arch_index, self.task, e, False, False)['train-loss']
                        if e <= 0:
                            ema = train_loss_i_e
                        else:
                            ema = ema * (1 - mu) + mu * train_loss_i_e

                        train_loss_i.append(train_loss_i_e)
                    sum_train_loss = np.sum(train_loss_i[-window_size:])
                    sum_ema_train_loss = ema

                    if self.metric == 'final_val':
                        acc_results = self.api.get_more_info(arch_index, self.task, None, False, False)
                        val_acc = acc_results['valid-accuracy'] / 100

                    else:
                        acc_results = self.api.get_more_info(arch_index, self.task, budget, False, False)
                        if self.task == 'cifar10-valid':
                            val_acc = acc_results['valid-accuracy'] / 100
                        else:
                            val_acc = acc_results['valtest-accuracy'] / 100
                    train_time = acc_results['train-all-time']

                    final_acc_results = self.api.get_more_info(arch_index, self.task, None, False, False)
                    test_acc = final_acc_results['test-accuracy'] / 100
                    final_val_acc = final_acc_results['valid-accuracy'] / 100

            cost_info = self.api.get_cost_info(arch_index, self.task)
            # auxiliary cost results such as number of flops and number of parameters
            cost_results = {'flops': cost_info['flops'], 'params': cost_info['params'],
                            'latency': cost_info['latency'], 'train_time': train_time}

        except FileNotFoundError:

            cost_results = {'flops': None, 'params': None,
                            'latency': None, 'train_time': -1}
            self.record_invalid(config, 1, 1, 1, cost_results)
            return 1

        val_err = 1 - val_acc
        test_err = 1 - test_acc
        final_val_err = 1 - final_val_acc
        self.record_valid(config, val_err, test_err, final_val_err, cost_results)
        if self.metric == 'tse':
            y_score = sum_train_loss
        elif self.metric == 'tseema':

            y_score = sum_ema_train_loss

        elif self.metric == 'early_stop':
            y_score = val_err
        elif self.metric == 'final_val':
            y_score = val_err
        else:
            print('not implemented')
            assert False
        # minimise y_score
        return y_score, -1


    def record_invalid(self, config, valid, test, final_val, costs):
        self.X.append(config)
        self.y_valid.append(valid)
        self.y_final_valid.append(final_val)
        self.y_test.append(test)
        self.costs.append(costs)

    def record_valid(self, config, valid, test, final_val, cost):

        self.X.append(config)
        # compute mean test error for the final budget
        self.y_test.append(test)
        self.y_final_valid.append(final_val)
        # compute validation error for the chosen budget
        self.y_valid.append(valid)
        self.costs.append(cost)

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = ['nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none']
        for i in range(6):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, ops_choices))
        return cs

    def get_results(self, ignore_invalid_configs=False):

        regret_validation = []
        regret_test = []

        inc_valid = np.inf
        inc_test = np.inf
        inc_final_valid = np.inf

        for i in range(len(self.X)):

            if ignore_invalid_configs and self.costs[i] == 0:
                continue

            if inc_valid > self.y_valid[i]:
                inc_valid = self.y_valid[i]
                inc_test = self.y_test[i]
                inc_final_valid = self.y_final_valid[i]

            regret_validation.append(float(inc_final_valid - self.y_star_valid))
            regret_test.append(float(inc_test - self.y_star_test))

        res = dict()
        res['y_test'] = self.y_test
        res['y_valid'] = self.y_valid
        res['y_final_valid'] = self.y_final_valid
        res['regret_validation'] = regret_validation
        res['regret_test'] = regret_test
        res['costs'] = self.costs

        return res
