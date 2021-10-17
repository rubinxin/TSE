import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
from collections import defaultdict
import torch
import torch.nn as nn
from pathlib import Path
import functools
from pprint import pprint
lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from utils import obtain_accuracy
from nats_bench import create
from typing import *
import itertools
import scipy.stats
import pickle
from tqdm import tqdm
import torch
from torch.autograd import Variable

def load_arch_overview(mode="perf"):
  """Load the precomputed performances of all architectures because querying NASBench is slow"""
  from pathlib import Path
  try:
    with open(f'./configs/nas-benchmark/percentiles/{mode}_all_dict.pkl', 'rb') as f:
      archs_dict = pickle.load(f)
    print(f"Suceeded in loading architectures from ./configs/nas-benchmark/percentiles/configs/nas-benchmark/percentiles/{mode}_all_dict.pkl! We have archs with len={len(archs_dict)}.")
    return archs_dict

  except Exception as e:
    print(f"Failed to load {mode} all dict! Need to run training with perf_percentile=0.9 to generate it. The error was {e}")
    raise NotImplementedError
  
def get_true_rankings(archs, api, hp='200', avg_all=False, decimals=None, is_random=False):
  """Extract true rankings of architectures on NASBench """
  final_accs = {genotype.tostr(): summarize_results_by_dataset(genotype, api, separate_mean_std=False, avg_all=avg_all, hp=hp, is_random=is_random) for genotype in tqdm(archs, desc=f"Getting true rankings from API with is_random={is_random}")}
  true_rankings = {}
  
  for dataset in final_accs[archs[0].tostr()].keys():
    if decimals is None:
      acc_on_dataset = [{"arch":arch.tostr(), "metric": final_accs[arch.tostr()][dataset]} for i, arch in enumerate(archs)]
    elif decimals is not None:
      acc_on_dataset = [{"arch":arch.tostr(), "metric": np.round(final_accs[arch.tostr()][dataset], decimals = decimals)} for i, arch in enumerate(archs)]

    acc_on_dataset = sorted(acc_on_dataset, key=lambda x: x["metric"], reverse=True)

    true_rankings[dataset] = acc_on_dataset
  
  return true_rankings, final_accs


def calc_corrs_val(archs, valid_accs, final_accs, true_rankings, corr_funs=None):
  if corr_funs is None:
    corr_funs = {"kendall": lambda x,y: scipy.stats.kendalltau(x,y).correlation, 
      "spearman":lambda x,y: scipy.stats.spearmanr(x,y).correlation, 
      "pearson":lambda x, y: scipy.stats.pearsonr(x,y)[0]}
  
  corr_per_dataset = {}
  for dataset in tqdm(final_accs[archs[0].tostr()].keys(), desc = "Calculating corrs per dataset"):
    ranking_pairs = []
    for val_acc_ranking_idx, archs_idx in enumerate(np.argsort(-1*np.array(valid_accs))):
      arch = archs[archs_idx].tostr()
      for true_ranking_dict in [tuple2 for tuple2 in true_rankings[dataset]]:
        if arch == true_ranking_dict["arch"]:
          ranking_pairs.append((valid_accs[val_acc_ranking_idx], true_ranking_dict["metric"]))
          break

    ranking_pairs = np.array(ranking_pairs)
    corr_per_dataset[dataset] = {method:fun(ranking_pairs[:, 0], ranking_pairs[:, 1]) for method, fun in corr_funs.items()}
    
  return corr_per_dataset

def mutate_topology_func(op_names):
  """Computes the architecture for a child of the given parent architecture.
  The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
  """
  def mutate_topology_func(parent_arch):
    child_arch = deepcopy( parent_arch )
    node_id = random.randint(0, len(child_arch.nodes)-1)
    node_info = list( child_arch.nodes[node_id] )
    snode_id = random.randint(0, len(node_info)-1)
    xop = random.choice( op_names )
    while xop == node_info[snode_id][0]:
      xop = random.choice( op_names )
    node_info[snode_id] = (xop, node_info[snode_id][1])
    child_arch.nodes[node_id] = tuple( node_info )
    return child_arch
  return mutate_topology_func

def avg_nested_dict(d):
  
  try:
    d = list(d.values()) 
  except: 
    pass 
  _data = sorted([i for b in d for i in b.items()], key=lambda x:x[0])
  _d = [(a, [j for _, j in b]) for a, b in itertools.groupby(_data, key=lambda x:x[0])]
  return {a:avg_nested_dict(b) if isinstance(b[0], dict) else round(sum(b)/float(len(b)), 1) for a, b in _d}

def calc_corrs_after_dfs(epochs:int, xloader, steps_per_epoch:int, metrics_depth_dim, final_accs, archs, true_rankings, 
  prefix, api, corr_funs=None, corrs_freq=4, nth_tops=[1,5,10], top_n_freq=1, constant=False, xargs=None):
  """Main function for producing correlation curves """
  if corrs_freq is None:
    corrs_freq = 1
  if corr_funs is None:

    corr_funs = {"kendall": lambda x,y: scipy.stats.kendalltau(x,y).correlation, 
      "spearman":lambda x,y: scipy.stats.spearmanr(x,y).correlation, 
      "pearson":lambda x, y: scipy.stats.pearsonr(x,y)[0],
      }

  tse_rankings = [] 
  for epoch_idx in range(epochs):
    rankings_per_epoch = []
    for batch_idx, data in enumerate(xloader):
      if ((steps_per_epoch is not None and steps_per_epoch != "None") and batch_idx >= steps_per_epoch-1) or ((xargs.steps_per_epoch_postnet is not None and xargs.steps_per_epoch_postnet != "None") and batch_idx >= xargs.steps_per_epoch_postnet-1):
        break
      if constant == True and batch_idx > 0:
        rankings_per_epoch.append(rankings_per_epoch[-1])
        continue
      relevant_tses = []
      metrics_depth_dim_keys = list(metrics_depth_dim.keys())

      for i, arch in enumerate(metrics_depth_dim_keys):
        try:
          metric = metrics_depth_dim[arch][epoch_idx][batch_idx]
          relevant_tses.append({"arch":arch, "metric": metric})
        except Exception as e:
          print(f"{e} for key={prefix}")
      
      relevant_tses = sorted(relevant_tses, key=lambda x: x["metric"] if x["metric"] is not None else 0, reverse=True) 
      rankings_per_epoch.append(relevant_tses)
    tse_rankings.append(rankings_per_epoch)
   
  corrs = []
  to_log = [[] for _ in range(epochs)]
  true_step = 0
  for epoch_idx in range(epochs):
    corrs_per_epoch = []
    for batch_idx, data in enumerate(xloader):
      if ((steps_per_epoch is not None and steps_per_epoch != "None") and batch_idx >= steps_per_epoch-1) or ((xargs.steps_per_epoch_postnet is not None and xargs.steps_per_epoch_postnet != "None") and batch_idx >= xargs.steps_per_epoch_postnet-1):
        break
      if batch_idx % corrs_freq != 0:
        continue
      
      if constant == True and batch_idx > 0:
        
        to_log[epoch_idx].append(to_log[epoch_idx][-1])
        corrs_per_epoch.append(corr_per_dataset)
        continue

      corr_per_dataset = {}
      for dataset in final_accs[archs[0].tostr()].keys(): 
        ranking_pairs = [] 
        hash_index = {(true_ranking_dict["arch"] if type(true_ranking_dict["arch"]) is str else true_ranking_dict["arch"].tostr()):true_ranking_dict['metric'] for pos, true_ranking_dict in enumerate(true_rankings[dataset])}
        for tse_dict in [tuple2 for tuple2 in tse_rankings[epoch_idx][batch_idx]]: 
          arch, tse_metric = tse_dict["arch"], tse_dict["metric"]
          true_ranking_idx = hash_index[arch if type(arch) is str else arch.tostr()]
          ranking_pairs.append((tse_metric, true_ranking_idx))
        if len([tuple2 for tuple2 in tse_rankings[epoch_idx][batch_idx]]) == 0:
          continue
        ranking_pairs = np.array(ranking_pairs)
        approx_ranking = scipy.stats.rankdata(ranking_pairs[:, 0])


        try:
          corr_per_dataset[dataset] = {**{method:fun(ranking_pairs[:, 0], ranking_pairs[:, 1]) for method, fun in corr_funs.items()}}
        except Exception as e:
          pprint(f"Failed calc corrs due to {e}! Dataset: {dataset}, prefix: {prefix}, X: {ranking_pairs[:, 0]} \n, Y: {ranking_pairs[:, 1]} \n")
          
      top1_perf = summarize_results_by_dataset(tse_rankings[epoch_idx][batch_idx][0]["arch"], api, separate_mean_std=False)
      top_perfs = {}
      if batch_idx % top_n_freq == 0:
        for top in nth_tops:
          top_perf = {nth_top: final_accs[tse_rankings[epoch_idx][batch_idx][nth_top]["arch"]]
            for nth_top in range(min(top, len(tse_rankings[epoch_idx][batch_idx])))}
          top_perf = avg_nested_dict(top_perf)
          top_perfs["top"+str(top)] = top_perf

      stats_to_log = {prefix:{**corr_per_dataset, "top1_backup":top1_perf, **top_perfs, "batch": batch_idx, "epoch":epoch_idx}, "true_step_corr":true_step}

      to_log[epoch_idx].append(stats_to_log)
      corrs_per_epoch.append(corr_per_dataset)
      
      true_step += corrs_freq
      
      if batch_idx % 100 == 0 and prefix in ["tse", "val_acc", "total_val_loss", "train_loss"]:
        print(f"Stats for metric {prefix} at batch={batch_idx}:")
        print(f"Corrs per dataset: {corr_per_dataset}")
        print(f"Top performances: {top_perfs}")

    corrs.append(corrs_per_epoch)
  
  return corrs, to_log

class ValidAccEvaluator:
  def __init__(self, valid_loader, valid_loader_iter=None):
    self.valid_loader = valid_loader
    self.valid_loader_iter=valid_loader_iter
    super().__init__()

  def evaluate(self, arch, network, criterion, grads=False):
    network.eval()
    sampled_arch = arch
    with torch.set_grad_enabled(grads):
      network.set_cal_mode('dynamic', sampled_arch)
      try:
        inputs, targets = next(self.valid_loader_iter)
      except:
        self.valid_loader_iter = iter(self.valid_loader)
        inputs, targets = next(self.valid_loader_iter)
      _, logits = network(inputs.cuda(non_blocking=True))
      loss = criterion(logits, targets.cuda(non_blocking=True))
      val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))
      val_acc_top1 = val_top1.item()
      val_acc_top5 = val_top5.item()

      if grads:
        loss.backward()


    network.train()
    return val_acc_top1, val_acc_top5, loss.item()

class DefaultDict_custom(dict):
  """
  default dict created by Teast Ares.
  """

  def set_default_item(self, default_item):
      self.default_item = default_item
      
  def __missing__(self, key):
      x = deepcopy(self.default_item)
      self[key] = x
      return x


def eval_archs_on_batch(xloader, archs, network, criterion, same_batch=False, metric="acc", train_steps = None, epochs=1, train_loader = None, w_optimizer=None, progress_bar=True):
  arch_metrics = []
  sum_metrics = {"loss":[], "acc": []}
  loader_iter = iter(xloader)
  inputs, targets = next(loader_iter)
  
  if w_optimizer is not None or train_steps is not None:
    init_state_dict = deepcopy(network.state_dict()) 
    init_w_optim_state_dict = deepcopy(w_optimizer.state_dict())

  for i, sampled_arch in tqdm(enumerate(archs), desc = f"Evaling archs on a batch of data with metric={metric}", disable = not progress_bar):

    network.set_cal_mode('dynamic', sampled_arch)
    if train_steps is not None:
      network.train()
      network.requires_grad_(True)
      tse = 0
      soacc = 0
      assert train_loader is not None and w_optimizer is not None, "Need to supply train loader in order to do quick training for quick arch eval"
      for epoch in range(epochs):
        for step, (inputs, targets) in enumerate(train_loader):
          if step >= train_steps:
            break
          w_optimizer.zero_grad()
          inputs = inputs.cuda(non_blocking=True)
          targets = targets.cuda(non_blocking=True)
          _, logits = network(inputs)
          loss = criterion(logits, targets)
          loss.backward()
          acc_top1, acc_top5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
          tse -= loss.item()
          soacc += acc_top1.item()
          w_optimizer.step()

      sum_metrics["loss"].append(tse)
      sum_metrics["acc"].append(soacc)
      network.eval()

    with torch.no_grad():
      network.eval()
      if not same_batch:
        try:
          inputs, targets = next(loader_iter)
        except Exception as e:
          loader_iter = iter(xloader)
          inputs, targets = next(loader_iter)
      _, logits = network(inputs.cuda(non_blocking=True))
      loss = criterion(logits, targets.cuda(non_blocking=True))

      acc_top1, acc_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))
      if metric == "acc":
        arch_metrics.append(acc_top1.item())
      elif metric == "loss":
        arch_metrics.append(-loss.item()) 

      if w_optimizer is not None:
        network.load_state_dict(init_state_dict)
        w_optimizer.load_state_dict(init_w_optim_state_dict)
  network.train()
  return arch_metrics, sum_metrics


def query_all_results_by_arch(
    arch: str,
    api,
    iepoch: bool = 11,
    hp: str = "12",
    is_random: bool = False,
    accs_only: bool = True,
):
    index = api.query_index_by_arch(arch)
    datasets = ["cifar10", "cifar10-valid", "cifar100", "ImageNet16-120"]
    results = {dataset: {} for dataset in datasets}
    for dataset in datasets:
        results[dataset] = api.get_more_info(
            index, dataset, iepoch=iepoch, hp=hp, is_random=is_random
        )
    if accs_only is True:
        for dataset in datasets:
            if (
                "test-accuracy" in results[dataset].keys()
            ):  
                results[dataset] = results[dataset]["test-accuracy"]
            else:
                results[dataset] = results[dataset]["valtest-accuracy"]
    return results

def summarize_results_by_dataset(genotype: str = None, api=None, results_summary=None, separate_mean_std=False, avg_all=False, iepoch=None, is_random=False, hp = '200') -> dict:
  if hp == '200' and iepoch is None:
    iepoch = 199
  elif hp == '12' and iepoch is None:
    iepoch = 11

  if results_summary is None:
    abridged_results = query_all_results_by_arch(genotype, api, iepoch=iepoch, hp=hp, is_random=is_random)
    results_summary = [abridged_results] 
  else:
    assert genotype is None
  interim = {}
  if not avg_all:
    for dataset in results_summary[0].keys():

      if separate_mean_std:
          interim[dataset]= {"mean":round(sum([result[dataset] for result in results_summary])/len(results_summary), 2),
          "std": round(np.std(np.array([result[dataset] for result in results_summary])), 2)}
      else:
          interim[dataset] = round(sum([result[dataset] for result in results_summary])/len(results_summary), 2)
  else:
    interim["avg"] = round(sum([result[dataset] for result in results_summary for dataset in results_summary[0].keys()])/len(results_summary[0].keys()), 2)
        
  return interim


def rolling_window(a, window):
    if type(a) is list:
      a = np.array(a)
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
  
class SumOfWhatever:
  def __init__(self, measurements=None, e = 1, epoch_steps=None, mode="sum"):
    if measurements is None:
      self.measurements = []
      self.measurements_flat = []
    else:
      self.measurements = measurements
      self.measurements_flat = list(itertools.chain.from_iterable(measurements))
    self.epoch_steps = epoch_steps
    self.e =e
    self.mode = mode

  def update(self, epoch, val):

    while epoch >= len(self.measurements):
      self.measurements.append([])
    self.measurements[epoch].append(val)
    self.measurements_flat.append(val)

  def get_time_series(self, e=None, mode=None, window_size = None, chunked=False, name=None):

    if mode is None:
      mode = self.mode

    params = self.guess(e=e, mode=mode, epoch_steps=None)
    return_fun, e, epoch_steps = params["return_fun"], params["e"], params["epoch_steps"]
    window_size = e*epoch_steps if window_size is None else window_size
    ts = []
    for step_idx in range(len(self.measurements_flat)):
      at_the_time = self.measurements_flat[max(step_idx-window_size+1,0):step_idx+1]
      try:
        ts.append(return_fun(at_the_time))
      except Exception as e:
        ts.append(-1)
    if chunked is False:
      return ts
    else:
      return list(chunks(ts, epoch_steps))

    
  def guess(self, epoch_steps, e, mode):
    if mode == "sum":
      return_fun = sum
    elif mode == "last":
      return_fun = lambda x: x[-1]
    elif mode == "first":
      return_fun = lambda x: x[0]
    elif mode == "fd":
      return_fun = lambda x: x[-1] - x[-2] if len(x) >= 2 else 0
    elif mode == "R":
      return_fun = lambda x: -(x[-1] - x[-2]) + x[0] if len(x) >= 2 else x[0]


    if self.epoch_steps is None:
      epoch_steps = len(self.measurements[0])
    else:
      epoch_steps = self.epoch_steps

    if e is None:
      e = self.e

    return {"e":e, "epoch_steps":epoch_steps, "return_fun":return_fun}

    
  def get(self, measurements_flat=None, e=None, mode=None):
    if mode is None:
      mode = self.mode

    params = self.guess(e=e, mode=mode, epoch_steps=None)
    return_fun, e, epoch_steps = params["return_fun"], params["e"], params["epoch_steps"]
    
    if measurements_flat is None:
      measurements_flat = self.measurements_flat

    desired_start = e*epoch_steps
    
    return return_fun(measurements_flat[-desired_start:])
  
  def __repr__(self):
    return str(self.measurements)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def test_SumOfWhatever():
  x=SumOfWhatever()
  epochs = 3
  steps_per_epoch = 5
  returned_vals = []
  for i in range(epochs):
    for j in range(steps_per_epoch):
      x.update(i, j)
      returned_vals.append(x.get())
  assert returned_vals == [0, 1, 3, 6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

  x=SumOfWhatever()
  epochs = 3
  steps_per_epoch = 5
  returned_vals = []
  for i in range(epochs):
    for j in range(steps_per_epoch):
      x.update(i, j+i)
      returned_vals.append(x.get())
  assert returned_vals == [0, 1, 3, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    