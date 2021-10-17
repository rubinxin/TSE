import torch
import sys
import os
from pathlib import Path
from copy import deepcopy
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

def fo_grad_if_possible(args, fnetwork, criterion, all_arch_inputs, all_arch_targets, arch_inputs, arch_targets, cur_grads, inner_step, inner_steps, step, outer_iter, first_order_grad, first_order_grad_for_free_cond, logger=None):
    if first_order_grad_for_free_cond: 

        with torch.no_grad():
          if first_order_grad is None:
            first_order_grad = cur_grads
          else:
            first_order_grad = [g1 + g2 for g1, g2 in zip(first_order_grad, cur_grads)]
    return first_order_grad

def hyper_higher_step(network, inner_rollouts, higher_grads, args, data_step, logger = None, model_init=None, outer_iters=1, epoch=0):

    if args.higher_algo in ["darts_higher"]: assert args.higher_params == "arch"

    with torch.no_grad():
        if args.higher_reduction_outer == "sum":
            avg_higher_grad = [sum([g if g is not None else 0 for g in grads]) for grads in zip(*higher_grads)]
        elif args.higher_reduction_outer == "mean":
            avg_higher_grad = [sum([g if g is not None else 0 for g in grads]) / outer_iters for grads in
                                zip(*higher_grads)]

    return avg_higher_grad

