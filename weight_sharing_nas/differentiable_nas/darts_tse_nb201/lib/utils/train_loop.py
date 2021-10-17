
import os, sys, time, random, argparse, math
import numpy as np
from copy import deepcopy
from collections import defaultdict
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config
from procedures   import get_optim_scheduler
from log_utils    import AverageMeter
from utils        import obtain_accuracy
from typing import *


def sample_arch_and_set_mode_search(args, outer_iter, sampled_archs, api, network, algo, arch_sampler, step, logger, epoch, all_archs):

    sampled_arch = None
    if algo.startswith('darts'):
        network.set_cal_mode('joint', None)
        sampled_arch = network.genotype

    elif "random" in algo and args.multipath is not None and args.multipath > 1 and args.multipath_mode == "quartiles":
        assert args.multipath == 4 

        sampled_arch = sampled_archs[outer_iter] 
        network.set_cal_mode('dynamic', sampled_arch)

    elif "random" in algo and args.multipath is not None and args.multipath > 1 and args.multipath_mode == "fairnas":
        assert args.multipath == len(network._op_names)
        sampled_arch = sampled_archs[outer_iter] 

        network.set_cal_mode('dynamic', sampled_arch)
    elif algo == 'random': 
        if all_archs is not None:
            if all_archs is not None:
                sampled_arch = random.sample(all_archs, 1)[0]
                network.set_cal_mode('dynamic', sampled_arch)
            else:
                sampled_arch = arch_sampler.sample(mode="random")[0]
                network.set_cal_mode('dynamic', sampled_arch)
        else:
            network.set_cal_mode('urs', None)
    else:
        raise ValueError('Invalid algo name : {:}'.format(algo))
    return sampled_arch

def format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, search_loader_iter, inner_steps, args, loader_type="train-val"):

    base_inputs, base_targets = base_inputs.cuda(non_blocking=True), base_targets.cuda(non_blocking=True)
    arch_inputs, arch_targets = arch_inputs.cuda(non_blocking=True), arch_targets.cuda(non_blocking=True)
    all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = [base_inputs], [base_targets], [arch_inputs], [arch_targets]
    for extra_step in range(inner_steps-1):
        if args.inner_steps_same_batch:
            all_base_inputs.append(base_inputs)
            all_base_targets.append(base_targets)
            all_arch_inputs.append(arch_inputs)
            all_arch_targets.append(arch_targets)
            continue 
        try:
            if loader_type == "train-val" or loader_type == "train-train":
              extra_base_inputs, extra_base_targets, extra_arch_inputs, extra_arch_targets = next(search_loader_iter)
            else:
              extra_base_inputs, extra_base_targets = next(search_loader_iter)
              extra_arch_inputs, extra_arch_targets = None, None
        except:
            continue

        extra_base_inputs, extra_base_targets = extra_base_inputs.cuda(non_blocking=True), extra_base_targets.cuda(non_blocking=True)
        if extra_arch_inputs is not None and extra_arch_targets is not None:
          extra_arch_inputs, extra_arch_targets = extra_arch_inputs.cuda(non_blocking=True), extra_arch_targets.cuda(non_blocking=True)
        
        all_base_inputs.append(extra_base_inputs)
        all_base_targets.append(extra_base_targets)
        all_arch_inputs.append(extra_arch_inputs)
        all_arch_targets.append(extra_arch_targets)

    return all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets


def get_finetune_scheduler(scheduler_type, config, xargs, network2, epochs=None, logger=None):

    if scheduler_type in ['linear_warmup', 'linear']:
        config = config._replace(scheduler=scheduler_type, warmup=1, eta_min=0, decay = 0.0005)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
    elif scheduler_type == "cos_reinit":
        
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
    elif scheduler_type == "cos_restarts":
        config = config._replace(scheduler='cos_restarts', warmup=0, epochs=epochs, decay = 0.0005)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config, xargs)
    elif scheduler_type in ['cos_adjusted']:
        config = config._replace(scheduler='cos', warmup=0, epochs=epochs, decay = 0.0005)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
    elif scheduler_type in ['cos_fast']:
        config = config._replace(scheduler='cos', warmup=0, LR=0.001 if xargs.lr is None else xargs.lr, epochs=epochs, eta_min=0, decay = 0.0005)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
    elif scheduler_type in ['cos_warmup']:
        config = config._replace(scheduler='cos', warmup=1, LR=0.001 if xargs.lr is None else xargs.lr, epochs=epochs, eta_min=0, decay = 0.0005)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
    elif scheduler_type in ["scratch"]:
        config_opt = load_config('./configs/nas-benchmark/hyper-opts/200E.config', None, logger)
        config_opt = config_opt._replace(LR=0.1 if xargs.lr is None else xargs.lr, decay = 0.0005)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config_opt)
    elif scheduler_type in ["scratch12E"]:
        config_opt = load_config('./configs/nas-benchmark/hyper-opts/12E.config', None, logger)
        config_opt = config_opt._replace(LR=0.1 if xargs.lr is None else xargs.lr, decay = 0.0005)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config_opt)
    elif scheduler_type in ["scratch1E"]:
        config_opt = load_config('./configs/nas-benchmark/hyper-opts/01E.config', None, logger)
        config_opt = config_opt._replace(LR=0.1 if xargs.lr is None else xargs.lr, decay = 0.0005)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config_opt)
    elif (xargs.lr is not None or (xargs.lr is None)) and scheduler_type == 'constant':
        config = config._replace(scheduler='constant', constant_lr=xargs.lr, decay = 0.0005)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
    elif scheduler_type == "constant":
        config = config._replace(scheduler='constant', constant_lr=xargs.lr, decay = 0.0005)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
    else:
        print(f"Unrecognized scheduler at {scheduler_type}")
        raise NotImplementedError
    return w_optimizer2, w_scheduler2, criterion
    
def sample_arch_and_set_mode(network, algo, arch_sampler, all_archs, parsed_algo, args, step, logger, sampled_archs, outer_iter):
    sampled_arch = None
    if algo.startswith('darts'):
        network.set_cal_mode('joint', None)
        sampled_arch = network.genotype

    elif "random" in algo and args.multipath is not None and args.multipath > 1 and args.multipath_mode == "quartiles":
        if args.search_space_paper == "nats-bench":
            assert args.multipath == 4 
            sampled_arch = sampled_archs[outer_iter] 
            network.set_cal_mode('dynamic', sampled_arch)
        else:
            network.set_cal_mode('urs')
    elif "random" in algo and args.multipath is not None and args.multipath > 1 and args.multipath_mode == "fairnas":
        assert args.multipath == len(network._op_names)
        sampled_arch = sampled_archs[outer_iter] 

        network.set_cal_mode('dynamic', sampled_arch)
    elif algo == 'random': 
        if all_archs is not None:
            sampled_arch = random.sample(all_archs, 1)[0]
            network.set_cal_mode('dynamic', sampled_arch)
        else:
            if args.search_space_paper == "nats-bench":
                sampled_arch = arch_sampler.sample(mode="random")[0]
                network.set_cal_mode('dynamic', sampled_arch)
            else:
                network.set_cal_mode('urs', None)
    else:
        raise ValueError('Invalid algo name : {:}'.format(algo))
    return sampled_arch

def valid_func(xloader, network, criterion, algo, logger, steps=None, grads=False):
  data_time, batch_time = AverageMeter(), AverageMeter()
  loss, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  end = time.time()
  with torch.set_grad_enabled(grads):
    network.eval()
    for step, (arch_inputs, arch_targets) in enumerate(xloader):
      if steps is not None and step >= steps:
        break
      arch_targets = arch_targets.cuda(non_blocking=True)
      
      data_time.update(time.time() - end)
      
      _, logits = network(arch_inputs.cuda(non_blocking=True))
      arch_loss = criterion(logits, arch_targets)
      if grads:
        arch_loss.backward()
      
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      loss.update(arch_loss.item(),  arch_inputs.size(0))
      top1.update  (arch_prec1.item(), arch_inputs.size(0))
      top5.update  (arch_prec5.item(), arch_inputs.size(0))
      
      batch_time.update(time.time() - end)
      end = time.time()
  network.train()
  return loss.avg, top1.avg, top5.avg

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


def _hessian_vector_product(vector, network, criterion, base_inputs, base_targets, r=1e-2):
  R = r / _concat(vector).norm()
  for p, v in zip(network.weights, vector):
    p.data.add_(R, v)
  _, logits = network(base_inputs)
  loss = criterion(logits, base_targets)
  grads_p = torch.autograd.grad(loss, network.alphas)

  for p, v in zip(network.weights, vector):
    p.data.sub_(2*R, v)
  _, logits = network(base_inputs)
  loss = criterion(logits, base_targets)
  grads_n = torch.autograd.grad(loss, network.alphas)

  for p, v in zip(network.weights, vector):
    p.data.add_(R, v)
  return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


def backward_step_unrolled(network, criterion, base_inputs, base_targets, w_optimizer, arch_inputs, arch_targets):
  
  _, logits = network(base_inputs)
  loss = criterion(logits, base_targets)
  LR, WD, momentum = w_optimizer.param_groups[0]['lr'], w_optimizer.param_groups[0]['weight_decay'], w_optimizer.param_groups[0]['momentum']
  with torch.no_grad():
    theta = _concat(network.weights)
    try:
      moment = _concat(w_optimizer.state[v]['momentum_buffer'] for v in network.weights)
      moment = moment.mul_(momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, network.weights)) + WD*theta
    params = theta.sub(LR, moment+dtheta)
  unrolled_model = deepcopy(network)
  model_dict  = unrolled_model.state_dict()
  new_params, offset = {}, 0
  for k, v in network.named_parameters():
    if 'arch_parameters' in k: continue
    v_length = np.prod(v.size())
    new_params[k] = params[offset: offset+v_length].view(v.size())
    offset += v_length
  model_dict.update(new_params)
  unrolled_model.load_state_dict(model_dict)

  unrolled_model.zero_grad()
  _, unrolled_logits = unrolled_model(arch_inputs)
  unrolled_loss = criterion(unrolled_logits, arch_targets)
  unrolled_loss.backward()

  dalpha = unrolled_model.arch_parameters.grad
  vector = [v.grad.data for v in unrolled_model.weights]
  [implicit_grads] = _hessian_vector_product(vector, network, criterion, base_inputs, base_targets)
  
  dalpha.data.sub_(LR, implicit_grads.data)

  if network.arch_parameters.grad is None:
    network.arch_parameters.grad = deepcopy( dalpha )
  else:
    network.arch_parameters.grad.data.copy_( dalpha.data )
  return unrolled_loss.detach(), unrolled_logits.detach()
                

def update_running(running, valid_loss=None, valid_acc = None, valid_acc_top5=None, loss=None, train_acc_top1=None, 
                   train_acc_top5=None, sogn=None, sogn_norm=None, total_train_loss_for_tse_aug=None):
  if loss is not None:
    running["tse"] -= loss 
  return running

def update_base_metrics(metrics, running, metrics_keys=None,
                        valid_acc=None, train_acc=None, loss=None, valid_loss=None, arch_str=None, epoch_idx = None):
  if metrics_keys is None:
    metrics_keys = metrics.keys()
  for k in running.keys():
    metrics[k][arch_str][epoch_idx].append(running[k])
  metrics["val_acc"][arch_str][epoch_idx].append(valid_acc)
  metrics["train_acc"][arch_str][epoch_idx].append(train_acc)
  metrics["train_loss"][arch_str][epoch_idx].append(-loss)
  metrics["val_loss"][arch_str][epoch_idx].append(-valid_loss)
  return metrics