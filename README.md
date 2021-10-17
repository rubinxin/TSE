# Speedy Performance Estimation for Neural Architecture Search

This is the code repository for reproducing NAS-Bench-201 results in our paper.

## Requirements

To install the following dependencies:
 - Python >= 3.6.0
 - scikit-learn 
 - nas-bench-201==1.3
 - ConfigSpace==0.4.11
 - hyperopt==0.2.2

Download the NAS-Bench-201 dataset from [here](https://github.com/D-X-Y/NAS-Bench-201])
and put it in the `./data` folder.

## Compute Rank Correlation
### Prestore data for unique valid architectures
```
python prestore_nas201_arch_data.py
```

### Compare rank correlation performance of various performance estimators 
```
python rank_correlation_comparison.py
```

## Run NAS with TSE
### Run Query-based NAS
Go to `cd ./query_based_nas/` first

E.g. run regularised evolution with TSE-EMA with 10 epochs of training budget 
```
python run_regularized_evolution.py --eval_metric=tseema  --es_budget=10
```
E.g. run regularised evolution with final validation accuracy with 200 epochs of training budget
```
python run_regularized_evolution.py --eval_metric=final_val --es_budget=200 
```

### Run Differentiable and One-shot NAS
Go to `cd ./weight_sharing_nas/` first

E.g. run DrNAS-TSE on NASBench 201
```
python differentiable_nas/drnas_nb201/201-space/train_search_tse.py --seed=1
```
E.g. run DARTS-TSE on NASBench 201
```
python differentiable_nas/darts_tse_nb201/search-cell.py --algo=darts_higher --data_path=$TORCH_HOME/cifar.python --higher_algo=darts_higher --rand_seed=1 --search_epochs=50 --inner_steps=100
```
E.g. run FairNAS on NASBench 201
```
python oneshot_nas/oneshot_nb201/search-cell.py --algo=random --data_path=$TORCH_HOME/cifar.python --lr=0.001 --rand_seed=1 --multipath=5 --multipath_mode=fairnas --search_epochs=100
```
