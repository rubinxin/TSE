# Example run of FairNAS
# python ./search-cell.py --algo=random --data_path=$TORCH_HOME/cifar.python --lr=0.001 --rand_seed=1 --multipath=5 --multipath_mode=fairnas --search_epochs=100

# Example run of MultiPath
# python ./search-cell.py --algo=random --data_path=$TORCH_HOME/cifar.python --lr=0.001 --rand_seed=1 --multipath=4 --multipath_mode=quartiles --search_epochs=100 

# Example run of RandomNAS
# python ./search-cell.py --algo=random --data_path=$TORCH_HOME/cifar.python --lr=0.001 --rand_seed=1 --scheduler=constant --search_epochs=100 

import os, sys, time, random, argparse, math
import numpy as np
from copy import deepcopy
from collections import defaultdict
import torch
import torch.nn as nn
import pickle
from pathlib import Path
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces
from nats_bench   import create
from utils.tse_utils import (query_all_results_by_arch, summarize_results_by_dataset,
  eval_archs_on_batch, 
  calc_corrs_after_dfs, get_true_rankings, SumOfWhatever, 
  ValidAccEvaluator, DefaultDict_custom, 
  mutate_topology_func)
from utils.train_loop import (format_input_data, get_finetune_scheduler, 
                              valid_func, 
                              backward_step_unrolled, sample_arch_and_set_mode_search, 
                              update_running, update_base_metrics)
from utils.higher_loop import fo_grad_if_possible, hyper_higher_step
from models.cell_searchs.generic_model import ArchSampler
from log_utils import Logger
import time

from argparse import Namespace
from typing import *
from tqdm import tqdm
import higher
import higher.patch
import higher.optim

def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, algo, logger, xargs=None, epoch=None,
  api=None,   all_archs=None, 
  checkpoint_freq=3, val_loader=None, train_loader=None, higher_optimizer=None):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(track_std=True), AverageMeter(track_std=True), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(track_std=True), AverageMeter(track_std=True), AverageMeter()
  end = time.time()
  network.train()
  parsed_algo = algo.split("_")
  arch_sampler = ArchSampler(api=api, model=network, mode="perf", prefer="random", op_names=network._op_names, max_nodes = xargs.max_nodes, search_space = xargs.search_space_paper) 
    
  if xargs.higher_algo is not None:
    model_init = deepcopy(network)
    w_optim_init = deepcopy(w_optimizer) 

  else:
    model_init, w_optim_init = None, None
  before_rollout_state = {} 
  before_rollout_state["model_init"] = model_init
  before_rollout_state["w_optim_init"] = w_optim_init  
  arch_overview = {"cur_arch": None, "all_cur_archs": [], "all_archs": [], "top_archs_last_epoch": [], "train_loss": [], "train_acc": [], "val_acc": [], "val_loss": []}
  search_loader_iter = iter(xloader)
  if xargs.inner_steps is not None:
    inner_steps = xargs.inner_steps
  else:
    inner_steps = 1 
  logger.log(f"Starting search with batch_size={len(next(iter(xloader))[0])}, len={len(xloader)}")
  use_higher_cond = xargs.higher_algo
  diffopt_higher_grads_cond = False
  monkeypatch_higher_grads_cond = False
  first_order_grad_for_free_cond = xargs.higher_order == "first" and xargs.higher_method == "tse"
  
  for data_step, (base_inputs, base_targets, arch_inputs, arch_targets) in tqdm(enumerate(search_loader_iter), desc = "Iterating over SearchDataset", total = round(len(xloader)/(inner_steps if not xargs.inner_steps_same_batch else 1))): 
    all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, search_loader_iter, inner_steps, xargs)
    network.zero_grad()

    scheduler.update(None, 1.0 * data_step / len(xloader))
    
    data_time.update(time.time() - end)

    if (xargs.multipath is None or xargs.multipath == 1):
      outer_iters = 1
    else:
      outer_iters = xargs.multipath
 
    inner_rollouts, higher_grads = [], [] 
    if xargs.multipath_mode in ["quartiles", "fairnas"]:
      sampled_archs = arch_sampler.sample(mode = xargs.multipath_mode, subset = all_archs, candidate_num=xargs.multipath) 
    else:
      sampled_archs = None
      
    for outer_iter in range(outer_iters):
      sampled_arch = sample_arch_and_set_mode_search(xargs, outer_iter, sampled_archs, api, network, algo, arch_sampler, 
                                                     data_step, logger, epoch, all_archs)
      

      weights_mask = [1 if 'arch' not in n else 0 for (n, p) in network.named_parameters()] 
      zero_arch_grads = lambda grads: [g*x if g is not None else None for g,x in zip(grads, weights_mask)]
      if use_higher_cond: 
        
        fnetwork = higher.patch.monkeypatch(network, device='cuda', copy_initial_weights=True if xargs.higher_loop == "bilevel" else False, track_higher_grads = monkeypatch_higher_grads_cond)
        diffopt = higher.optim.get_diff_optim(w_optimizer, network.parameters(), fmodel=fnetwork, grad_callback=zero_arch_grads, device='cuda', override=None, track_higher_grads = diffopt_higher_grads_cond) 
        fnetwork.zero_grad() 
      else: 
        fnetwork = network
        diffopt = w_optimizer

      tse, first_order_grad = [], None
      assert inner_steps == 1 or xargs.higher_algo is not None or xargs.implicit_algo is not None
      assert xargs.higher_algo is None or xargs.higher_loop is not None
      if xargs.higher_algo is not None and "higher" in xargs.higher_algo:
        assert xargs.higher_order is not None
        
      for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in tqdm(enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)), desc="Iterating over inner batches", 
                                                                                     disable=True if round(len(xloader)/(inner_steps if not xargs.inner_steps_same_batch else 1)) > 1 else False, total= len(all_base_inputs)):

        _, logits = fnetwork(base_inputs)
        base_loss = criterion(logits, base_targets) * (1 if xargs.multipath is None else 1/xargs.multipath)
        tse.append(base_loss)
            
        if (not xargs.higher_algo):
          base_loss.backward() 

        elif xargs.higher_algo: 
          new_params, cur_grads = diffopt.step(base_loss)
          cur_grads = list(cur_grads)
          for idx, (g, p) in enumerate(zip(cur_grads, fnetwork.parameters())):
            if g is None:
              cur_grads[idx] = torch.zeros_like(p)
          
          first_order_grad = fo_grad_if_possible(args=xargs, fnetwork=fnetwork, criterion=criterion, 
                                                 all_arch_inputs=all_arch_inputs, all_arch_targets=all_arch_targets, arch_inputs=arch_inputs, arch_targets=arch_targets, cur_grads=cur_grads,
                                                 inner_step=inner_step, inner_steps=inner_steps,
                                                 step=data_step, outer_iter=outer_iter,
                                                 first_order_grad=first_order_grad, first_order_grad_for_free_cond=first_order_grad_for_free_cond,
                                                 logger=logger)
        else:
          pass 

        base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
        if inner_step == 0:
          base_losses.update(base_loss.item() / (1 if xargs.multipath is None else 1/xargs.multipath),  base_inputs.size(0))
          base_top1.update  (base_prec1.item(), base_inputs.size(0))
          base_top5.update  (base_prec5.item(), base_inputs.size(0))
          arch_overview["train_acc"].append(base_prec1)
          arch_overview["train_loss"].append(base_loss.item())
      

      if first_order_grad is not None:
        assert first_order_grad_for_free_cond

        if xargs.higher_reduction == "sum": 
          higher_grads.append(first_order_grad)
        else:
          higher_grads.append([g/inner_steps if g is not None else g for g in first_order_grad])
        
      if all_archs is not None: 
        assert sampled_arch in all_archs 

    if xargs.higher_algo is None:
      w_optimizer.step()
      network.zero_grad()

    arch_loss = torch.tensor(10) 
    
    if algo.startswith('darts'):
      network.set_cal_mode('joint', None)
    elif 'random' in algo:
      network.set_cal_mode('urs', None)
    else:
      raise ValueError('Invalid algo name : {:}'.format(algo))
    network.zero_grad()
    if algo == 'darts-v2' and not xargs.higher_algo:
      arch_loss, logits = backward_step_unrolled(network, criterion, base_inputs, base_targets, w_optimizer, arch_inputs, arch_targets)
      a_optimizer.step()
    elif (algo == 'random' or 'random' in algo ) and not xargs.higher_algo:
      if algo == "random" and xargs.merge_train_val_supernet:
        arch_loss = torch.tensor(10) 
      else:
        with torch.no_grad():
          _, logits = network(arch_inputs)
          arch_loss = criterion(logits, arch_targets)
    elif xargs.higher_algo:
      avg_higher_grad = hyper_higher_step(network, inner_rollouts, higher_grads, xargs, data_step, logger, before_rollout_state["model_init"], outer_iters, epoch)
      
      network.load_state_dict(
        before_rollout_state["model_init"].state_dict())  
      
      with torch.no_grad():  
          for (n, p), g in zip(network.named_parameters(), avg_higher_grad):
              cond = 'arch' not in n if xargs.higher_params == "weights" else 'arch' in n  
              if cond:
                  if g is not None and p.requires_grad:
                      p.grad = g
              else:
                p.grad = None
      higher_optimizer.step()
      higher_optimizer.zero_grad()
      pass
    else:
      network.zero_grad()
      _, logits = network(arch_inputs)
      arch_loss = criterion(logits, arch_targets)
      arch_loss.backward()
      a_optimizer.step()

    if use_higher_cond and xargs.higher_loop == "bilevel" and xargs.higher_params == "arch":
      for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)):
        if inner_step == 1 and xargs.inner_steps_same_batch: 
          break

        _, logits = network(base_inputs)
        base_loss = criterion(logits, base_targets) * (1 if xargs.multipath is None else 1/xargs.multipath)
        network.zero_grad()
        base_loss.backward()
        w_optimizer.step()
    elif use_higher_cond and xargs.higher_loop == "joint" and xargs.higher_params == "arch" and outer_iters == 1:
      if epoch == 0 and data_step < 3:
        logger.log(f"Updating meta-weights by copying from the rollout model")
      with torch.no_grad():
        for (n1, p1), p2 in zip(network.named_parameters(), fnetwork.parameters()):
          if 'arch' not in n1: 
            p1.data = p2.data

    if xargs.higher_algo and use_higher_cond:
      del fnetwork
      del diffopt

    
    if arch_targets is not None:
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      val_batch_size = arch_inputs.size(0)
    else:
      arch_prec1, arch_prec5, val_batch_size = torch.tensor(0), torch.tensor(0), torch.tensor(1)
    arch_losses.update(arch_loss.item(),  val_batch_size)
    arch_top1.update  (arch_prec1.item(), val_batch_size)
    arch_top5.update  (arch_prec5.item(), val_batch_size)

    if xargs.higher_algo is not None: 

      before_rollout_state["model_init"].load_state_dict(network.state_dict())
      before_rollout_state["w_optim_init"] = w_optim_init

    arch_overview["all_cur_archs"] = [] 
    network.zero_grad()
    
    batch_time.update(time.time() - end)
    end = time.time()

    if data_step % print_freq == 0 or data_step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, data_step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=arch_losses, top1=arch_top1, top5=arch_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
  
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg



def get_best_arch(train_loader, valid_loader, network, n_samples, algo, logger, criterion,
  additional_training=True, api=None, style:str='val_acc', w_optimizer=None, 
  config: Dict=None, epochs:int=1, steps_per_epoch:int=100, 
  val_loss_freq:int=2, overwrite_additional_training:bool=False, 
  scheduler_type:str=None, xargs:Namespace=None, train_loader_stats=None, val_loader_stats=None, 
  all_archs=None, checkpoint_freq=1, search_epoch=None):
  true_archs = None
  with torch.no_grad():
    network.eval()
    if 'random' in algo:
      if api is not None and xargs is not None:
        archs, decision_metrics = network.return_topK(n_samples, True, api=api, dataset=xargs.dataset), []
      else:
        archs, decision_metrics = network.return_topK(n_samples, True), []
      if xargs.archs_split is not None:
        logger.log(f"Loading archs from {xargs.archs_split} to use as sampled architectures in finetuning with algo={algo}")
        with open(f'./configs/nas-benchmark/arch_splits/{xargs.archs_split}', 'rb') as f:
          archs = pickle.load(f)
      elif xargs.save_archs_split is not None:
        logger.log(f"Savings archs split to {xargs.archs_split} to use as sampled architectures in finetuning with algo={algo}")
        with open(f'./configs/nas-benchmark/arch_splits/{xargs.save_archs_split}', 'wb') as f:
          pickle.dump(archs, f)

    elif algo.startswith('darts'):
      arch = network.get_genotype(original_darts_format=True)
      true_archs, true_decision_metrics = [arch], [] 
      archs, decision_metrics = network.return_topK(n_samples, False, api=api, dataset=xargs.dataset), []
    else:
      raise ValueError('Invalid algorithm name : {:}'.format(algo))
  

    if steps_per_epoch is not None and steps_per_epoch != "None":
      steps_per_epoch = int(steps_per_epoch)
    elif steps_per_epoch in [None, "None"]:
      steps_per_epoch = len(train_loader)
    else:
      raise NotImplementedError
  if style in ['val_acc', 'val']:
    
    if len(archs) >= 1:
      corrs = {"archs": [arch.tostr() for arch in archs]}
      decision_metrics_eval = {"archs": [arch.tostr() for arch in archs]}
      search_summary_stats = {"search":defaultdict(lambda: defaultdict(dict)), "epoch": search_epoch}
      for data_type in ["val", "train"]:
        for metric in ["acc", "loss"]:
          cur_loader = valid_loader if data_type == "val" else train_loader

          decision_metrics_computed, decision_sum_metrics_computed = eval_archs_on_batch(xloader=cur_loader, archs=archs, network=network, criterion=criterion, metric=metric, 
            train_loader=train_loader, w_optimizer=w_optimizer, train_steps=xargs.eval_arch_train_steps, same_batch = True) 

          best_idx_search = np.argmax(decision_metrics_computed)
          best_arch_search, best_valid_acc_search = archs[best_idx_search], decision_metrics_computed[best_idx_search]
          search_results_top1 = summarize_results_by_dataset(best_arch_search, api=api, iepoch=199, hp='200')

          decision_metrics_eval["supernet_" + data_type + "_" + metric] = decision_metrics_computed

          search_summary_stats["search"][data_type][metric]["mean"] = np.mean(decision_metrics_computed)
          search_summary_stats["search"][data_type][metric]["std"] = np.std(decision_metrics_computed)
          search_summary_stats["search"][data_type][metric]["top1"] = search_results_top1

      try:
        decision_metrics = decision_metrics_eval["supernet_val_acc"]
      except Exception as e:
        logger.log(f"Failed to get decision metrics - decision_metrics_eval={decision_metrics_eval}")
    else:
      decision_metrics, decision_sum_metrics = eval_archs_on_batch(xloader=valid_loader, archs=archs, network=network, 
                                                                  train_loader=train_loader, w_optimizer=w_optimizer, train_steps = xargs.eval_arch_train_steps)

  if style == 'tse':
    true_rankings, final_accs = get_true_rankings(archs, api)
    upper_bound = {}
    for n in [1,5,10]:
      upper_bound[f"top{n}"] = {"cifar10":0, "cifar10-valid":0, "cifar100":0, "ImageNet16-120":0}
      for dataset in true_rankings.keys():
        upper_bound[f"top{n}"][dataset] += sum([x["metric"] for x in true_rankings[dataset][0:n]])/min(n, len(true_rankings[dataset][0:n]))
    upper_bound = {"upper":upper_bound}
    logger.log(f"Upper bound: {upper_bound}")
    
    cond = logger.path('corr_metrics').exists() and not overwrite_additional_training
    total_metrics_keys = ["total_val", "total_train", "total_val_loss", "total_train_loss", "total_arch_count"]
    sum_metrics_keys = ["tse"]
    
    metrics_keys = ["val_acc", "train_acc", "train_loss", "val_loss", "gap_loss", *sum_metrics_keys, *total_metrics_keys]
    must_restart = False
    start_arch_idx = 0

    if cond: 
      logger.log("=> loading checkpoint of the last-checkpoint '{:}' start".format(logger.path('corr_metrics')))
      try:
        checkpoint = torch.load(logger.path('corr_metrics'))
        logger.log(f"Loaded corr metrics checkpoint at {logger.path('corr_metrics')}")
      except Exception as e:
        logger.log("Failed to load corr_metrics checkpoint, trying backup now")
        checkpoint = torch.load(os.fspath(logger.path('corr_metrics'))+"_backup")

      checkpoint_config = checkpoint["config"] if "config" in checkpoint.keys() else {}
      try:
        metrics = {k:checkpoint["metrics"][k] for k in checkpoint["metrics"].keys()}
      except Exception as e:
        print("Errored due to exception below")
        print(e)
        print("Unknown reason but must restart!")
        must_restart = True

      decision_metrics = checkpoint["decision_metrics"] if "decision_metrics" in checkpoint.keys() else []
      start_arch_idx = checkpoint["start_arch_idx"]
      cond1={k:v for k,v in checkpoint_config.items() if ('path' not in k and 'dir' not in k)}
      cond2={k:v for k,v in vars(xargs).items() if ('path' not in k and 'dir' not in k)}
      logger.log(f"Checkpoint config: {cond1}")
      logger.log(f"Newly input config: {cond2}")
      different_items = {k: cond1[k] for k in cond1 if k in cond2 and cond1[k] != cond2[k]}
      if (cond1 == cond2 or len(different_items) == 0):
        logger.log("Both configs are equal.")
      else:
        logger.log("Checkpoint and current config are not the same! need to restart")
        logger.log(f"Different items are : {different_items}")
      
      if set([x.tostr() if type(x) is not str else x for x in checkpoint["archs"]]) != set([x.tostr() if type(x) is not str else x for x in archs]):
        print("Checkpoint has sampled different archs than the current seed! Need to restart")
        print(f"Checkpoint: {checkpoint['archs'][0]}")
        print(f"Current archs: {archs[0]}")
        if all_archs is not None or xargs.archs_split is not None:
          logger.log(f"Architectures do not match up to the checkpoint but since all_archs (or archs_split={xargs.archs_split}) was supplied, it might be intended")
        else:
            if not ('eval_candidate_num' in different_items) and not 'darts' in algo:
              logger.log("Using the checkpoint archs as ground-truth for current run. But might be better to investigate what went wrong")
              archs = checkpoint["archs"]
              true_rankings, final_accs = get_true_rankings(archs, api)
              upper_bound = {}
              for n in [1,5,10]:
                upper_bound[f"top{n}"] = {"cifar10":0, "cifar10-valid":0, "cifar100":0, "ImageNet16-120":0}
                for dataset in true_rankings.keys():
                  upper_bound[f"top{n}"][dataset] += sum([x["metric"] for x in true_rankings[dataset][0:n]])/min(n, len(true_rankings[dataset][0:n]))
              upper_bound = {"upper":upper_bound}
            else:
              logger.log("Cannot reuse archs from checkpoint because they use different arch-picking parameters")
    if xargs.restart:
      must_restart=True
    if (not cond) or must_restart or (xargs is None) or (cond1 != cond2 and len(different_items) > 0): 
      if not cond:
        logger.log(f"Did not find a checkpoint for supernet post-training at {logger.path('corr_metrics')}")
      else:
        logger.log(f"Starting postnet training with fresh metrics")

      metrics_factory = {arch.tostr():[[] for _ in range(epochs)] for arch in archs}
      metrics = DefaultDict_custom()
      metrics.set_default_item(metrics_factory)
      decision_metrics = []    
      start_arch_idx = 0

    train_start_time = time.time()
    arch_rankings = sorted([(arch.tostr(), summarize_results_by_dataset(genotype=arch, api=api, avg_all=True)["avg"]) for arch in archs], reverse=True, key=lambda x: x[1])
    
    arch_rankings_dict = {k: {"rank":rank, "metric":v} for rank, (k,v) in enumerate(arch_rankings)}


    network_init = deepcopy(network.state_dict())
    logger.log(f"Starting finetuning at {start_arch_idx} with total len(archs)={len(archs)}")
    for arch_idx, sampled_arch in tqdm(enumerate(archs[start_arch_idx:], start_arch_idx), desc="Iterating over sampled architectures", total = len(archs)-start_arch_idx):
      assert (all_archs is None) or (sampled_arch in all_archs), "There must be a bug since we are training an architecture that is not in the supplied subset"

      true_step = 0 
      arch_str = sampled_arch.tostr() 

      logger.log("Started deepcopying")
      network2 = network
      network2.set_cal_mode('dynamic', sampled_arch)
      network2.load_state_dict(network_init)
      logger.log("Finished deepcopying")

      arch_param_count = api.get_cost_info(api.query_index_by_arch(sampled_arch), xargs.dataset if xargs.dataset != "cifar5m" else "cifar10")['params'] 
      print(f"Arch param count: {arch_param_count}MB")

      if hasattr(train_loader.sampler, "reset_counter"):
        train_loader.sampler.reset_counter()

      if xargs.lr is not None and scheduler_type is None:
        scheduler_type = "constant"

      w_optimizer2, w_scheduler2, criterion = get_finetune_scheduler(scheduler_type, config, xargs, network2, logger=logger)
      
      if arch_idx == start_arch_idx: 
        logger.log(f"Optimizers for the supernet post-training: {w_optimizer2}, {w_scheduler2}")

      running = defaultdict(int)

      start = time.time()
      train_loss_total, train_acc_total, _ = valid_func(xloader=train_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps if not xargs.drop_fancy else 4)

      if not xargs.merge_train_val_postnet or (xargs.val_dset_ratio is not None and xargs.val_dset_ratio < 1):
        val_loss_total, val_acc_total, _ = valid_func(xloader=val_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps)

      else:
        val_loss_total, val_acc_total = train_loss_total, train_acc_total
      val_loss_total, train_loss_total = -val_loss_total, -train_loss_total
      logger.log(f"Computed total_metrics in {time.time()-start} time")

      if arch_idx == 0: 
        logger.log(f"Time taken to compute total_train/total_val statistics once with {xargs.total_estimator_steps} estimator steps is {time.time()-start}")

      for epoch_idx in range(epochs):
        if epoch_idx < 5:
          logger.log(f"New epoch (len={len(train_loader)}) of arch; for debugging, those are the indexes of the first minibatch in epoch with idx up to 5: {epoch_idx}: {next(iter(train_loader))[1][0:15]}")
          logger.log(f"Weights LR before scheduler update: {w_scheduler2.get_lr()[0]}")

        if epoch_idx == 0: 
          total_mult_coef = min(len(train_loader)-1, steps_per_epoch)
        else:
          total_mult_coef = min(len(train_loader)-1, steps_per_epoch)
        val_acc_evaluator = ValidAccEvaluator(valid_loader, None)
        total_metrics_dict = {"total_val":val_acc_total, "total_train":train_acc_total, "total_val_loss":val_loss_total, "total_train_loss": train_loss_total, "total_arch_count":arch_param_count}
        for batch_idx, data in tqdm(enumerate(train_loader), desc = "Iterating over batches", total=len(train_loader), disable=True):
          stop_early_cond = ((steps_per_epoch is not None and steps_per_epoch != "None") and batch_idx > steps_per_epoch) or ((args.steps_per_epoch_postnet is not None and args.steps_per_epoch_postnet != "None") and batch_idx > args.steps_per_epoch_postnet)
          if stop_early_cond:
            break
          for metric, metric_val in total_metrics_dict.items():
            metrics[metric][arch_str][epoch_idx].append(metric_val)
        
          with torch.set_grad_enabled(mode=additional_training): 
            if scheduler_type in ["linear", "linear_warmup"]:
              w_scheduler2.update(epoch_idx, 1.0 * batch_idx / min(len(train_loader), steps_per_epoch))
            elif scheduler_type == "cos_adjusted":
              w_scheduler2.update(epoch_idx , batch_idx/min(len(train_loader), steps_per_epoch))
            elif scheduler_type == "cos_reinit":
              w_scheduler2.update(epoch_idx, 0.0)
            elif scheduler_type in ['cos_fast', 'cos_warmup']:
              w_scheduler2.update(epoch_idx , batch_idx/min(len(train_loader), steps_per_epoch))
            else:
              w_scheduler2.update(epoch_idx, 1.0 * batch_idx / len(train_loader))

            network2.zero_grad()
            inputs, targets = data
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            _, logits = network2(inputs)
            train_acc_top1, train_acc_top5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
            loss = criterion(logits, targets)
            if additional_training:
              loss.backward()
              w_optimizer2.step()
            loss, train_acc_top1, train_acc_top5 = loss.item(), train_acc_top1.item(), train_acc_top5.item()
            
          true_step += 1

          if (batch_idx % val_loss_freq == 0) and (batch_idx % 100 == 0 or not xargs.drop_fancy):
            if batch_idx == 0 or not xargs.merge_train_val_postnet or xargs.postnet_switch_train_val or (xargs.val_dset_ratio is not None and xargs.val_dset_ratio < 1):
              w_optimizer2.zero_grad() 
              valid_acc, valid_acc_top5, valid_loss = val_acc_evaluator.evaluate(arch=sampled_arch, network=network2, criterion=criterion)
              w_optimizer2.zero_grad() 
            else:
              valid_acc, valid_acc_top5, valid_loss = 0, 0, 0

          running = update_running(running=running, valid_loss=valid_loss, valid_acc=valid_acc, valid_acc_top5=valid_acc_top5, loss=loss, 
                         train_acc_top1=train_acc_top1, train_acc_top5=train_acc_top5
                          )
          
          metrics = update_base_metrics(metrics=metrics, running=running, 
                              valid_acc=valid_acc, train_acc=train_acc_top1, loss=loss, 
                              valid_loss=valid_loss, arch_str=arch_str, epoch_idx=epoch_idx)
  

          special_metrics = {k:metrics[k][arch_str][epoch_idx][-1] for k in metrics.keys() if len(metrics[k][arch_str][epoch_idx])>0}
          
          if additional_training and (batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1) and batch_idx < 400 and not (batch_idx == 0 and epoch_idx == 0): 
            start = time.time()
            if not xargs.drop_fancy or xargs.merge_train_val_postnet:
              train_loss_total, train_acc_total, _ = valid_func(xloader=train_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps if not xargs.drop_fancy else 4)
            network2.zero_grad() 
            if not xargs.merge_train_val_postnet or (xargs.val_dset_ratio is not None and xargs.val_dset_ratio < 1):
              val_loss_total, val_acc_total, _ = valid_func(xloader=val_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps)
            else:
              val_loss_total, val_acc_total = train_loss_total, train_acc_total
            val_loss_total, train_loss_total = -val_loss_total, -train_loss_total
            
            total_metrics_dict["total_val"], total_metrics_dict["total_train"] = val_acc_total, train_acc_total
            total_metrics_dict["total_val_loss"], total_metrics_dict["total_train_loss"] = val_loss_total, train_loss_total


        if hasattr(train_loader.sampler, "reset_counter"): 
          train_loader.sampler.counter += 1

      final_metric = None 
      if style == "tse":
        final_metric = running["tse"]

      decision_metrics.append(final_metric)
      
      if arch_idx % checkpoint_freq == 0 or arch_idx == len(archs)-1:
        corr_metrics_path = save_checkpoint({"corrs":{}, "metrics":metrics, "archs":archs, "start_arch_idx": arch_idx+1, "config":vars(xargs), "decision_metrics":decision_metrics},   
          logger.path('corr_metrics'), logger, quiet=True, backup=False)

    train_total_time = time.time()-train_start_time
    print(f"Train total time: {train_total_time}")

    logger.log("Deepcopying metrics")
    original_metrics = deepcopy(metrics)
    logger.log("Calculating transforms of original metrics:")
    metrics_factory = {arch.tostr():[[] for _ in range(epochs)] for arch in archs}
                
    if epochs >= 1:
      metrics_E1 = {metric+"E1": {arch.tostr():SumOfWhatever(measurements=metrics[metric][arch.tostr()], e=1).get_time_series(chunked=True, name=metric) for arch in archs} for metric,v in tqdm(metrics.items(), desc = "Calculating E1 metrics") if not metric.startswith("so") and not 'accum' in metric and not 'total' in metric and not 'standalone' in metric}
      metrics.update(metrics_E1)
    
    start=time.time()
    corrs = {}
    to_logs = []
    
    core_metrics = ["tse", "train_loss", "val_loss", "val_acc", "train_acc", "total_train", "total_val", "total_train_loss", "total_val_loss"]
    for idx, (k,v) in tqdm(enumerate(metrics.items()), desc="Calculating correlations", total = len(metrics)):
      if xargs.drop_fancy and k not in core_metrics:
        continue
      tqdm.write(f"Started computing correlations for {k}")
      if torch.is_tensor(v[next(iter(v.keys()))]):
        v = {inner_k: [[batch_elem.item() for batch_elem in epoch_list] for epoch_list in inner_v] for inner_k, inner_v in v.items()}
      
      constant_metric = True if ("upper" in k) else False      
      if len(archs) > 1:
        try:
          corr, to_log = calc_corrs_after_dfs(epochs=epochs, xloader=train_loader, steps_per_epoch=steps_per_epoch, metrics_depth_dim=v, 
        final_accs = final_accs, archs=archs, true_rankings = true_rankings, prefix=k, api=api, corrs_freq = xargs.corrs_freq, 
        constant=constant_metric, xargs=xargs, nth_tops = [1, 5, 10] if k in core_metrics else [1], 
        top_n_freq=1 if xargs.search_space_paper != "darts" else 100)
          corrs["corrs_"+k] = corr
          to_logs.append(to_log)
        except Exception as e:
          logger.log(f"Failed to compute corrs for {k} due to {e}")
          raise e

    arch_ranking_inner = [{"arch":arch, "metric":metrics["total_arch_count"][arch][0][0]} for arch in metrics["total_arch_count"].keys()]
    arch_ranking_inner = sorted(arch_ranking_inner, key=lambda x: x["metric"], reverse=True)
    arch_true_rankings = {"cifar10":arch_ranking_inner, "cifar100":arch_ranking_inner,"cifar10-valid":arch_ranking_inner, "ImageNet16-120":arch_ranking_inner}
    for k in ["train_lossE1", "tse"]:
      
      if k not in metrics.keys():
        print(f"WARNING! Didnt find {k} in metrics keys: {list(metrics.keys())}")
        continue
      v = metrics[k]
      if torch.is_tensor(v[next(iter(v.keys()))]):
        v = {inner_k: [[batch_elem.item() for batch_elem in epoch_list] for epoch_list in inner_v] for inner_k, inner_v in v.items()}
      corr, to_log = calc_corrs_after_dfs(epochs=epochs, xloader=train_loader, steps_per_epoch=steps_per_epoch, metrics_depth_dim=v, 
    final_accs = final_accs, archs=archs, true_rankings = arch_true_rankings, corr_funs=None, prefix=k+"P", api=api, corrs_freq = xargs.corrs_freq, constant=None, xargs=xargs)
      corrs["param_corrs_"+k] = corr
      to_logs.append(to_log) 



  if style in ["tse"] and n_samples-start_arch_idx > 0: 
    corr_metrics_path = save_checkpoint({"metrics":original_metrics, "corrs": corrs, 
      "archs":archs, "start_arch_idx":arch_idx+1, "config":vars(xargs), "decision_metrics":decision_metrics},
      logger.path('corr_metrics'), logger, backup=False)

  best_idx = np.argmax(decision_metrics)
  try:
    best_arch, best_valid_acc = archs[best_idx], decision_metrics[best_idx]
  except Exception as e:
    logger.log(f"Failed to get best arch via decision_metrics due to {e}")
    logger.log(f"Decision metrics: {decision_metrics}")
    logger.log(f"Best idx: {best_idx}, length of archs: {len(archs)}")
    best_arch,best_valid_acc = archs[0], decision_metrics[0]

  if true_archs is not None: 
    return true_archs[0], best_valid_acc
  else: 
    return best_arch, best_valid_acc

def main(xargs):
  import warnings 
  warnings.filterwarnings("ignore", category=UserWarning)
  warnings.filterwarnings("ignore")

  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( max(int(xargs.workers), 1))
  if xargs.search_space_paper == "darts": 
    assert xargs.num_cells == 2
    assert xargs.max_nodes == 7
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(xargs)
  gpu_mem = torch.cuda.get_device_properties(0).total_memory

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  if xargs.overwite_epochs is None:
    extra_info = {'class_num': class_num, 'xshape': xshape}
  else:
    extra_info = {'class_num': class_num, 'xshape': xshape, 'epochs': xargs.overwite_epochs}
  config = load_config(xargs.config_path, extra_info, logger)
  if xargs.search_epochs is not None:
    config = config._replace(epochs=xargs.search_epochs)

  if os.environ.get("TORCH_WORKERS", None) is not None:
    dataloader_workers = int(os.environ["TORCH_WORKERS"])
  else:
    dataloader_workers = xargs.workers
  resolved_train_batch_size, resolved_val_batch_size = xargs.train_batch_size if xargs.train_batch_size is not None else config.batch_size, xargs.val_batch_size if xargs.val_batch_size is not None else config.test_batch_size
  
  logger.log("Instantiating the Search loaders")
  search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', 
    (config.batch_size if xargs.search_batch_size is None else xargs.search_batch_size, config.test_batch_size), workers=dataloader_workers, epochs=config.epochs + config.warmup, determinism=xargs.deterministic_loader, 
    merge_train_val = xargs.merge_train_val_supernet, merge_train_val_and_use_test = xargs.merge_train_val_and_use_test, 
    valid_ratio=xargs.val_dset_ratio, use_only_train=xargs.use_only_train_supernet, xargs=xargs)
  logger.log("Instantiating the postnet loaders")
  train_data_postnet, valid_data_postnet, xshape_postnet, class_num_postnet = get_datasets(xargs.dataset, xargs.data_path, -1)
  search_loader_postnet, train_loader_postnet, valid_loader_postnet = get_nas_search_loaders(train_data_postnet, valid_data_postnet, xargs.dataset, 'configs/nas-benchmark/', 
    (resolved_train_batch_size, resolved_val_batch_size), workers=dataloader_workers, valid_ratio=xargs.val_dset_ratio, determinism=xargs.deterministic_loader, 
    epochs=xargs.eval_epochs, merge_train_val=xargs.merge_train_val_postnet, 
    merge_train_val_and_use_test = xargs.merge_train_val_and_use_test, use_only_train=xargs.use_only_train_supernet, xargs=xargs)
  logger.log("Instantiating the stats loaders")
  _, train_loader_stats, val_loader_stats = get_nas_search_loaders(train_data_postnet, valid_data_postnet, xargs.dataset, 'configs/nas-benchmark/', 
    (512 if gpu_mem < 8147483648 else 1024, 512 if gpu_mem < 8147483648 else 1024), workers=dataloader_workers, valid_ratio=xargs.val_dset_ratio, determinism="all", 
     epochs=xargs.eval_epochs, merge_train_val=xargs.merge_train_val_postnet, 
    merge_train_val_and_use_test = xargs.merge_train_val_and_use_test,  use_only_train=xargs.use_only_train_supernet, xargs=xargs)
  logger.log(f"Using train batch size: {resolved_train_batch_size}, val batch size: {resolved_val_batch_size}")
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = get_search_spaces(xargs.search_space, xargs.search_space_paper)

  if xargs.model_name is None:
    model_config = dict2config(
    dict(name='generic', super_type = "basic", C=xargs.channel, N=xargs.num_cells, max_nodes=xargs.max_nodes, num_classes=class_num,
          space=search_space, affine=bool(xargs.affine), track_running_stats=bool(xargs.track_running_stats)), None)
  else:
    super_type = super_type = "basic" if xargs.search_space in ["nats-bench", None] else "nasnet-super"
    model_config = dict2config(
        dict(name=xargs.model_name, super_type = super_type, C=xargs.channel, N=xargs.num_cells, max_nodes=xargs.max_nodes, 
        num_classes=class_num, stem_multiplier=3, multiplier=4, steps=4,
            space=search_space, affine=bool(xargs.affine), track_running_stats=bool(xargs.track_running_stats)), None)
  logger.log('search space : {:}'.format(search_space))
  logger.log('model config : {:}'.format(model_config))
  search_model = get_cell_based_tiny_net(model_config)
  search_model.set_algo(xargs.algo)
  search_model = search_model.cuda()
  
  
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.weights, config)
  a_optimizer = torch.optim.Adam(search_model.alphas, lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay, eps=xargs.arch_eps)
  higher_optimizer = a_optimizer

  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  logger.log('search-space : {:}'.format(search_space))
  if bool(xargs.use_api):
    api = create(None, 'topology', fast_mode=True, verbose=False)
  else:
    api = None
  logger.log('{:} create API = {:} done'.format(time_string(), api))

  network, criterion = search_model, criterion.cuda()  
  last_info_orig, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  arch_sampler = ArchSampler(api=api, model=network, op_names=network._op_names, 
                             max_nodes = xargs.max_nodes, search_space = xargs.search_space_paper)
  network.arch_sampler = arch_sampler 
  network.xargs = xargs
  messed_up_checkpoint = False

  if last_info_orig.exists() and not xargs.reinitialize and not xargs.force_overwrite: 
    try:
      
      logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info_orig))
      if os.name == 'nt': 
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
      try:
        last_info   = torch.load(last_info_orig.resolve())
        checkpoint  = torch.load(last_info['last_checkpoint'])
      except Exception as e:
        logger.log(f"Failed to load checkpoints due to {e} but will try to load backups now")
        try:
          last_info   = torch.load(os.fspath(last_info_orig)+"_backup")
          checkpoint  = torch.load(os.fspath(last_info['last_checkpoint'])+"_backup") 
        except Exception as e:
          logger.log(f"Failed to load checkpoint backups at last_info: {os.fspath(last_info_orig)+'_backup'}, checkpoint: {os.fspath(last_info['last_checkpoint'])+'_backup'}")
      start_epoch, epoch = last_info['epoch'], last_info['epoch']
      genotypes   = checkpoint['genotypes']
      baseline  = checkpoint['baseline']
      try:
        search_tse_stats = checkpoint["search_tse_stats"]
      except Exception as e:
        search_tse_stats = {arch: {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []} for arch in arch_sampler.archs}
      valid_accuracies = checkpoint['valid_accuracies']
      search_model.load_state_dict( checkpoint['search_model'] )
      w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
      w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
      a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
      
      logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
    except Exception as e:
      logger.log(f"Checkpoint got messed up and cannot be loaded due to {e}! Will have to restart")
      messed_up_checkpoint = True

  if not (last_info_orig.exists() and not xargs.reinitialize and not xargs.force_overwrite) or messed_up_checkpoint or (xargs.supernet_init_path is not None and not last_info_orig.exists()):
    logger.log(f"""=> do not find the last-info file (or was given a checkpoint as initialization): {last_info_orig}, whose existence status is {last_info_orig.exists()}. Also, reinitialize={xargs.reinitialize}, 
      force_overwrite={xargs.force_overwrite}, messed_up_checkpoint={messed_up_checkpoint}, supernet_init_path={xargs.supernet_init_path}""")
    start_epoch, valid_accuracies, genotypes, search_tse_stats = 0, {'best': -1}, {-1: network.return_topK(1, True)[0]}, {arch: {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []} for arch in arch_sampler.archs}
    baseline = None

  start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup if xargs.search_epochs is None else xargs.search_epochs
  
  if start_epoch > total_epoch: 
    start_epoch = total_epoch


  valid_a_loss , valid_a_top1 , valid_a_top5 = 0, 0, 0 
  for epoch in range(start_epoch if not xargs.reinitialize else 0, total_epoch if not xargs.reinitialize else 0):
    
    w_scheduler.update(epoch if epoch < total_epoch else epoch-total_epoch, 0.0)
    need_time = 'Time Left: {:}'.format(convert_secs2time(epoch_time.val * (total_epoch-epoch), True))
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr())))

    network.set_drop_path(float(epoch+1) / total_epoch, xargs.drop_path_rate)
    if epoch < total_epoch: 
      archs_to_sample_from = None
    
    search_w_loss, search_w_top1, search_w_top5, search_a_loss, search_a_top1, search_a_top5 \
                = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str, xargs.print_freq, xargs.algo, logger, 
                  api=api, epoch=epoch,
                  all_archs=archs_to_sample_from, xargs=xargs, val_loader=valid_loader_postnet, train_loader=train_loader_postnet,
                  higher_optimizer=higher_optimizer)


    search_time.update(time.time() - start_time)
    logger.log('[{:}] search [base] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))
    logger.log('[{:}] search [arch] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, search_a_loss, search_a_top1, search_a_top5))

    if epoch % xargs.search_eval_freq == 0 or epoch == total_epoch - 1 or epoch == total_epoch or len(genotypes) == 0 or 'random' not in xargs.algo:
      genotype, temp_accuracy = get_best_arch(train_loader, valid_loader, network, xargs.eval_candidate_num, xargs.algo, 
                                              xargs=xargs, criterion=criterion, logger=logger, api=api, search_epoch=epoch)
      logger.log('[{:}] - [get_best_arch] : {:} -> {:}'.format(epoch_str, genotype, temp_accuracy))
      valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion, xargs.algo, logger, steps=5)
      logger.log('[{:}] evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}% | {:}'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5, genotype))
      genotypes[epoch] = genotype
    elif len(genotypes) > 0:
      genotype = genotypes[-1]
      temp_accuracy = 0
    if xargs.algo.startswith('darts'):
      network.set_cal_mode('joint', None)
    elif 'random' in xargs.algo:
      network.set_cal_mode('urs', None)
    else:
      raise ValueError('Invalid algorithm name : {:}'.format(xargs.algo))

    valid_accuracies[epoch] = valid_a_top1

    if hasattr(search_loader.sampler, "reset_counter"):
      search_loader.sampler.counter += 1

    genotypes[epoch] = genotype

    with torch.no_grad():
      logger.log('{:}'.format(search_model.show_alphas()))
    logger.log(f"Querying genotype {genotypes[epoch]} at epoch={epoch}")
    if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[epoch], '200')))

    logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
    
    if epoch % xargs.checkpoint_freq == 0 or epoch == total_epoch-1 or epoch in [49, 99]:
      save_path = save_checkpoint({'epoch' : epoch + 1,
                  'args'  : deepcopy(xargs),
                  'baseline'    : baseline,
                  'search_model': search_model.state_dict(),
                  'w_optimizer' : w_optimizer.state_dict(),
                  'a_optimizer' : a_optimizer.state_dict(),
                  'w_scheduler' : w_scheduler.state_dict(),
                  'genotypes'   : genotypes,
                  'valid_accuracies' : valid_accuracies,
                  "search_tse_stats": search_tse_stats
                  },
                  model_base_path, logger, backup=False)
      last_info = save_checkpoint({
            'epoch': epoch + 1,
            'args' : deepcopy(args),
            'last_checkpoint': save_path,
          }, logger.path('info'), logger, backup=False)
      if epoch == total_epoch - 1:
        save_path = save_checkpoint({'epoch' : epoch + 1,
                  'args'  : deepcopy(xargs),
                  'baseline'    : baseline,
                  'search_model': search_model.state_dict(),
                  'w_optimizer' : w_optimizer.state_dict(),
                  'a_optimizer' : a_optimizer.state_dict(),
                  'w_scheduler' : w_scheduler.state_dict(),
                  'genotypes'   : genotypes,
                  'valid_accuracies' : valid_accuracies,
                  "search_tse_stats": search_tse_stats
                  },
                  model_base_path, logger, backup=False)
        last_info = save_checkpoint({
              'epoch': epoch + 1,
              'args' : deepcopy(args),
              'last_checkpoint': save_path,
            }, logger.path('info'), logger, backup=False)
      
    
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  
  start_time = time.time()

  if xargs.cand_eval_method in ['val_acc', 'val'] or "random" not in xargs.algo:
    genotype, temp_accuracy = get_best_arch(train_loader_postnet, valid_loader_postnet, network, xargs.eval_candidate_num, xargs.algo, xargs=xargs, 
                                            criterion=criterion, logger=logger, style="val", api=api, search_epoch=epoch, config=config)

  else:
    if xargs.finetune_search == "uniform":
      genotype, temp_accuracy = get_best_arch(train_loader_postnet, valid_loader_postnet, network, xargs.eval_candidate_num, xargs.algo, criterion=criterion, logger=logger, style=xargs.cand_eval_method, 
        w_optimizer=w_optimizer,config=config, epochs=xargs.eval_epochs, steps_per_epoch=xargs.steps_per_epoch, 
        api=api, additional_training = xargs.additional_training, val_loss_freq=xargs.val_loss_freq, 
        overwrite_additional_training=xargs.overwrite_additional_training, scheduler_type=xargs.scheduler, xargs=xargs, train_loader_stats=train_loader_stats, val_loader_stats=val_loader_stats
        )

  if xargs.algo.startswith('darts'):
    network.set_cal_mode('joint', None)
  elif 'random' in xargs.algo:
    network.set_cal_mode('urs', None)
  else:
    raise ValueError('Invalid algorithm name : {:}'.format(xargs.algo))
  search_time.update(time.time() - start_time)

  valid_a_loss , valid_a_top1 , valid_a_top5 = valid_func(valid_loader_postnet, network, criterion, xargs.algo, logger)
  logger.log('Last : the gentotype is : {:}, with the validation accuracy of {:.3f}%.'.format(genotype, valid_a_top1))

  logger.log('\n' + '-'*100)
  
  logger.log('[{:}] run {:} epochs, cost {:.1f} s, last-geno is {:}.'.format(xargs.algo, total_epoch, search_time.sum, genotype))
  if api is not None: logger.log('{:}'.format(api.query_by_arch(genotype, '200') ))
  results_by_dataset = summarize_results_by_dataset(genotype, api, separate_mean_std=False)
  logger.close()
  



if __name__ == '__main__':
  parser = argparse.ArgumentParser("Weight sharing NAS methods to search for cells.")
  parser.add_argument('--data_path'   ,       type=str,   help='Path to dataset')
  parser.add_argument('--dataset'     ,       type=str,   choices=['cifar10'], default="cifar10", help='Choose between Cifar10/100 and ImageNet-16.')
  parser.add_argument('--search_space',       type=str,   default='tss', choices=['tss'], help='The search space name.')
  parser.add_argument('--algo'        ,       type=str,   help='The search space name.')
  parser.add_argument('--use_api'     ,       type=int,   default=1, choices=[0,1], help='Whether use API or not (which will cost much memory).')
  
  parser.add_argument('--tau_min',            type=float, default=0.1,  help='The minimum tau for Gumbel Softmax.')
  parser.add_argument('--tau_max',            type=float, default=10,   help='The maximum tau for Gumbel Softmax.')
  
  parser.add_argument('--max_nodes'   ,       type=int,   default=4,  help='The maximum number of nodes.')
  parser.add_argument('--channel'     ,       type=int,   default=16, help='The number of channels.')
  parser.add_argument('--num_cells'   ,       type=int,   default=5,  help='The number of cells in one stage.')
  
  parser.add_argument('--eval_candidate_num', type=int,   default=200, help='The number of selected architectures to evaluate.')
  
  parser.add_argument('--track_running_stats',type=int,   default=0, choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--affine'      ,       type=int,   default=0, choices=[0,1],help='Whether use affine=True or False in the BN layer.')
  parser.add_argument('--config_path' ,       type=str,   default='./configs/nas-benchmark/algos/weight-sharing.config', help='The path of configuration.')
  parser.add_argument('--overwite_epochs',    type=int,   help='The number of epochs to overwrite that value in config files.')
  
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay' , type=float, default=1e-3, help='weight decay for arch encoding')
  parser.add_argument('--arch_eps'          , type=float, default=1e-8, help='weight decay for arch encoding')
  parser.add_argument('--drop_path_rate'  ,  type=float, help='The drop path rate.')
  
  parser.add_argument('--workers',            type=int,   default=0,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   default='./output/search', help='Folder to save checkpoints and log.')
  parser.add_argument('--print_freq',         type=int,   default=200,  help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  parser.add_argument('--cand_eval_method',          type=str,   help='tse or normal val acc', default='tse', choices = ['tse', 'val_acc', 'val'])

  parser.add_argument('--steps_per_epoch', type=int,           default=None,  help='Number of minibatches to train for when evaluating candidate architectures with tse')
  parser.add_argument('--eval_epochs',          type=int, default=1,   help='Number of epochs to train for when evaluating candidate architectures with tse')
  parser.add_argument('--additional_training',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=True,   help='Whether to train the supernet samples or just go through the training loop with no grads')
  parser.add_argument('--val_batch_size',          type=int, default=64,   help='Batch size for the val loader')
  parser.add_argument('--val_dset_ratio',          type=float, default=1)
  parser.add_argument('--val_loss_freq',          type=int, default=1,   help='How often to calculate val loss during training. Probably better to only this for smoke tests as it is generally better to record all and then post-process if different results are desired')
  parser.add_argument('--overwrite_additional_training',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False,   help='Whether to load checkpoints of additional training')
  parser.add_argument('--scheduler',          type=str, default="constant",   help='Whether to use different training protocol for the postnet training')
  parser.add_argument('--train_batch_size',          type=int, default=64,   help='Training batch size for the POST-SUPERNET TRAINING!')
  parser.add_argument('--lr',          type=float, default=0.001,   help='Constant LR for the POST-SUPERNET TRAINING!')

  parser.add_argument('--deterministic_loader',          type=str, default='all', choices=['None', 'train', 'val', 'all'],   help='Whether to choose SequentialSampler or RandomSampler for data loaders')
  parser.add_argument('--reinitialize',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Whether to use trained supernetwork weights for initialization')
  parser.add_argument('--total_estimator_steps',          type=int, default=10000, help='Number of batches for evaluating the total_val/total_train etc. metrics')
  parser.add_argument('--corrs_freq',          type=int, default=4, help='Calculate corrs based on every i-th minibatch')
  parser.add_argument('--search_epochs',          type=int, default=None, help='Can be used to explicitly set the number of search epochs')
  parser.add_argument('--restart',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=None, help='WHether to force or disable restart of training via must_restart')

  
  parser.add_argument('--multipath',          type=int, default=None, help='Number of architectures in each multipath sample')
  parser.add_argument('--multipath_mode',          type=str, default=None, choices=["fairnas", "quartiles", None], help='Special sampling like quartiles/FairNAS etc.')
  parser.add_argument('--multipath_computation',          type=str, default="serial", choices=["serial"])

  parser.add_argument('--force_overwrite',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Load saved seed or not')

  parser.add_argument('--merge_train_val_postnet',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Whether to merge train/val sets in finetuning')
  parser.add_argument('--merge_train_val_supernet',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Whether to merge train/val sets')
  parser.add_argument('--postnet_switch_train_val',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Whether to switch train and val sets in finetuning')
  parser.add_argument('--use_only_train_supernet',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Whether to use validation set')
  
  parser.add_argument('--merge_train_val_and_use_test',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Merges CIFAR10 train/val into one (ie. not split in half) AND then also treats test set as validation')
  parser.add_argument('--search_batch_size',          type=int, default=None, help='Controls batch size for the supernet training')
  parser.add_argument('--search_eval_freq',          type=int, default=5, help='How often to run get_best_arch during supernet training')
  parser.add_argument('--overwrite_supernet_finetuning',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=True, help='Whether to load additional checkpoints on top of the normal training -')
  parser.add_argument('--eval_arch_train_steps',          type=int, default=None, help='Whether to load additional checkpoints on top of the normal training -')
  parser.add_argument('--supernet_init_path' ,       type=str,   default=None, help='The path of pretrained checkpoint')
  parser.add_argument('--search_space_paper' ,       type=str,   default="nats-bench", choices=["nats-bench"], help='Number of adaptation steps in MetaProx')
  parser.add_argument('--checkpoint_freq' ,       type=int,   default=3, help='How often to pickle checkpoints')
  
  parser.add_argument('--higher_method' ,       type=str, choices=['tse'],   default='tse', help='Dataset to take gradien with respect to')
  parser.add_argument('--higher_params' ,       type=str, choices=['arch'],   default='arch', help='Whether to do meta-gradients with respect to the meta-weights or architecture')
  parser.add_argument('--higher_order' ,       type=str, choices=['first'],   default='first', help='First order approximation for gradients')
  parser.add_argument('--higher_loop' ,       type=str, choices=['bilevel'],   default='bilevel', help='Alternating/bilevel optim')
  parser.add_argument('--higher_reduction' ,       type=str, choices=['sum'],   default='sum', help='Reduction across inner steps - relevant for first-order approximation')
  parser.add_argument('--higher_reduction_outer' ,       type=str, choices=['sum'],   default='sum', help='Reduction across the outer steps')

  parser.add_argument('--first_order_strategy' ,       type=str, choices=['every'],   default='every', help='Whether to make a copy of network for the Higher rollout or not. If we do not copy, it will be as in joint training')

  parser.add_argument('--higher_algo' ,       type=str, choices=['darts_higher'],   default=None, help='Whether to do meta-gradients with respect to the meta-weights or architecture')
  
  parser.add_argument('--inner_steps' ,       type=int,   default=None, help='Number of steps to do in the inner loop of bilevel meta-learning')
  parser.add_argument('--inner_steps_same_batch' ,       type=lambda x: False if x in ["False", "false", "", "None"] else True,   default=False, help='Number of steps to do in the inner loop of bilevel meta-learning')

  parser.add_argument('--finetune_search' ,       type=str,   default="uniform", choices=["uniform"], help='Search method for finetnuing')

  parser.add_argument('--model_name' ,       type=str,   default=None, choices=["generic"], help='Picking the right model to instantiate')
  parser.add_argument('--drop_fancy' ,       type=lambda x: False if x in ["False", "false", "", "None"] else True,   default=True, help='Drop special metrics in get_best_arch to make the finetuning proceed faster')
  parser.add_argument('--archs_split' ,       type=str,   default=None, help='Use a pre-determined architectures sample')
  parser.add_argument('--save_archs_split' ,       type=str,   default=None, help='Drop special metrics in get_best_arch to make the finetuning proceed faster')
  parser.add_argument('--save_train_split' ,       type=str,   default=None, help='Save train split somewhere')
  parser.add_argument('--train_split' ,       type=str,   default=None, help='Load train split somewhere')


  parser.add_argument('--steps_per_epoch_postnet' ,       type=int,   default=None, help='Drop special metrics in get_best_arch to make the finetuning proceed faster')


  args = parser.parse_args()


  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  if args.overwite_epochs is None:
    args.save_dir = os.path.join('{:}-{:}'.format(args.save_dir, args.search_space),
        args.dataset,
        '{:}-affine{:}_BN{:}-{:}'.format(args.algo, args.affine, args.track_running_stats, args.drop_path_rate))
  else:
    args.save_dir = os.path.join('{:}-{:}'.format(args.save_dir, args.search_space),
        args.dataset,
        '{:}-affine{:}_BN{:}-E{:}-{:}'.format(args.algo, args.affine, args.track_running_stats, args.overwite_epochs, args.drop_path_rate))

  main(args)
