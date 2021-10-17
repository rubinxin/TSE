import torch
import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
import higher
def format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, search_loader_iter, inner_steps, args, loader_type="train-val"):

    
    
    
    base_inputs, base_targets = base_inputs.cuda(non_blocking=True), base_targets.cuda(non_blocking=True)
    arch_inputs, arch_targets = arch_inputs.cuda(non_blocking=True), arch_targets.cuda(non_blocking=True)
    if args.higher_method == "tse":
        arch_inputs, arch_targets = None, None
    all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = [base_inputs], [base_targets], [arch_inputs], [arch_targets]
    for extra_step in range(inner_steps-1):
        try:
            if loader_type == "train-val" or loader_type == "train-train":
              (extra_base_inputs, extra_base_targets), (extra_arch_inputs, extra_arch_targets)= next(search_loader_iter)
            else:
              extra_base_inputs, extra_base_targets = next(search_loader_iter)
              extra_arch_inputs, extra_arch_targets = None, None
        except Exception as e:
            continue
        
        
        extra_base_inputs, extra_base_targets = extra_base_inputs.cuda(non_blocking=True), extra_base_targets.cuda(non_blocking=True)
        if extra_arch_inputs is not None and extra_arch_targets is not None:
          extra_arch_inputs, extra_arch_targets = extra_arch_inputs.cuda(non_blocking=True), extra_arch_targets.cuda(non_blocking=True)
        
        all_base_inputs.append(extra_base_inputs)
        all_base_targets.append(extra_base_targets)
        all_arch_inputs.append(extra_arch_inputs)
        all_arch_targets.append(extra_arch_targets)

    return all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets


import torch
import sys
import os
from copy import deepcopy
from typing import *

def avg_state_dicts(state_dicts: List):
  if len(state_dicts) == 1:
    return state_dicts[0]
  else:
    mean_state_dict = {}
    for k in state_dicts[0].keys():
      mean_state_dict[k] = sum([network[k] for network in state_dicts])/len(state_dicts)
    return mean_state_dict

def fo_grad_if_possible(args, fnetwork, criterion, all_arch_inputs, all_arch_targets, arch_inputs, arch_targets, 
                        cur_grads, inner_step, step, outer_iter, first_order_grad, first_order_grad_for_free_cond, logger=None):
    if first_order_grad_for_free_cond: 
        if inner_step < 3 and step == 0:
            msg = f"Adding cur_grads to first_order grads at inner_step={inner_step}, step={step}, outer_iter={outer_iter}. First_order_grad is head={str(first_order_grad)[0:100]}, cur_grads is {str(cur_grads)[0:100]}"
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)
        with torch.no_grad():
          if first_order_grad is None:
            first_order_grad = cur_grads
          else:
            first_order_grad = [g1 + g2 for g1, g2 in zip(first_order_grad, cur_grads)]

    return first_order_grad

def hyper_higher_step(network, inner_rollouts, higher_grads, args, data_step, logger = None, model_init=None, outer_iters=1, epoch=0):

    if args.higher_algo in ["darts_higher"]: assert args.higher_params == "arch"
    
    if epoch < 2 and logger is not None:
        msg = f"Reductioning in the outer loop (len(higher_grads)={len(higher_grads)}, head={str(higher_grads)[0:150]}) with outer reduction={args.higher_reduction_outer}, outer_iters={outer_iters}"
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
    with torch.no_grad():
        if args.higher_reduction_outer == "sum":
            avg_higher_grad = [sum([g if g is not None else 0 for g in grads]) for grads in zip(*higher_grads)]
        elif args.higher_reduction_outer == "mean":
            avg_higher_grad = [sum([g if g is not None else 0 for g in grads]) / outer_iters for grads in
                                zip(*higher_grads)]

    
    with torch.no_grad():  
        for (n, p), g in zip(network.named_parameters(), avg_higher_grad):
            cond = 'arch' not in n if args.higher_params == "weights" else 'arch' in n  
            if cond:
                if g is not None and p.requires_grad:
                    p.grad = g
    return avg_higher_grad

def hypergrad_outer(
    args,
    fnetwork,
    criterion,
    arch_targets,
    arch_inputs,
    all_arch_inputs,
    all_arch_targets,
    all_base_inputs,
    all_base_targets,
    tse,
    inner_step,
    inner_steps,
    inner_rollouts,
    first_order_grad_for_free_cond,
    higher_grads,
    step,
    epoch,
    logger=None,
):
    if args.higher_algo:
        if args.higher_method == "tse":
            if args.higher_order == "first":
                if not (
                    first_order_grad_for_free_cond
                ):  
                    all_logits = [
                        fnetwork(
                            all_base_inputs[i], params=fnetwork.parameters(time=i)
                        )[1]
                        for i in range(0, inner_steps)
                    ]
                    arch_loss = [
                        criterion(all_logits[i], all_base_targets[i])
                        for i in range(len(all_logits))
                    ]
                    all_grads = [
                        torch.autograd.grad(arch_loss[i], fnetwork.parameters(time=i))
                        for i in range(0, inner_steps)
                    ]
                    if step == 0 and epoch < 2:
                        if logger is not None:
                            logger.info(
                                f"Reductioning all_grads (len={len(all_grads)} with reduction={args.higher_reduction}, inner_steps={inner_steps}"
                            )
                            logger.info(f"Grads sample before: {all_grads[0]}")
                        else:
                            print(
                                f"Reductioning all_grads (len={len(all_grads)} with reduction={args.higher_reduction}, inner_steps={inner_steps}"
                            )
                            print(f"Grads sample before: {all_grads[0]}")
                    with torch.no_grad():
                        if args.higher_reduction == "sum":
                            fo_grad = [sum(grads) for grads in zip(*all_grads)]
                        elif args.higher_reduction == "mean":
                            fo_grad = [
                                sum(grads) / inner_steps for grads in zip(*all_grads)
                            ]
                    if step == 0:
                        print(f"Grads sample after: {fo_grad[0]}")
                    higher_grads.append(fo_grad)

                else:
                    pass
    return higher_grads, inner_rollouts
