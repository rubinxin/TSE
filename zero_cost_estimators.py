import random

import numpy as np
import torch
import torch.nn as nn

from zero_cost_utils.datasets import get_datasets
from zero_cost_utils.models import get_cell_based_tiny_net

def get_batch_jacobian(net, x, target):
    net.zero_grad()

    x.requires_grad_(True)

    _, y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()

def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1. / (v + k))


class zero_cost_estimator(object):

    def __init__(self, method_name='JacCov', search_space='nas201',
                 dataset='cifar10', batch_size = 256, seed=1):

        # Reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.batch_size = batch_size
        self.method_name = method_name
        self.search_space = search_space
        if 'nas201' in search_space or 'darts' in search_space or 'nas301' in search_space or 'nas101' in search_space:
            trainval = False
            self.dataset = dataset.split('-')[0]
            # specify the directory for image data
            data_loc = f'./zero_cost_utils/datasets/{self.dataset}/'
            self.seed = seed
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            train_data, valid_data, xshape, class_num = get_datasets(self.dataset, data_loc, cutout=0)
            self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                       num_workers=0, pin_memory=True)

    def predict(self, arch):
        data_iterator = iter(self.train_loader)
        x, target = next(data_iterator)
        x, target = x.to(self.device), target.to(self.device)

        if self.method_name == 'JacCov':

            if 'nas201' in self.search_space:
                config = {'name': 'infer.tiny', 'C': 16, 'N': 5, 'arch_str': arch, 'num_classes': 1}
                network = get_cell_based_tiny_net(config)  # create the network from configuration

            network = network.to(self.device)

            try:
                jacobs, labels = get_batch_jacobian(network, x, target)
                jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
                s = eval_score(jacobs, labels)
            except Exception as e:
                print(e)
                s = -10e8

            return s

        elif self.method_name == 'SNIP':

            num_classes_dic = {'cifar10': 10, 'cifar100': 100, 'ImageNet16': 120}
            if 'nas201' in self.search_space:
                config = {'name': 'infer.tiny', 'C': 16, 'N': 5, 'arch_str': arch}
                config['num_classes'] = num_classes_dic[self.dataset]
                network = get_cell_based_tiny_net(config)  # create the network from configuration


            # run forward and backward passes
            try:
                network = network.to(self.device)
                criterion = nn.CrossEntropyLoss()
                network.zero_grad()
                _, y = network(x)
                loss = criterion(y, target)
                loss.backward()
                grads = [p.grad.detach().clone().abs() for p in network.parameters() if p.grad is not None]

                with torch.no_grad():
                    saliences = [(grad * weight).view(-1).abs() for weight, grad in zip(network.parameters(), grads)]
                    saliences_score = torch.sum(torch.cat(saliences)).cpu().numpy()
            except Exception as e:
                print(e)
                saliences_score = -10e8
            return saliences_score


