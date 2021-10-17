
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pickle
from zero_cost_estimators import zero_cost_estimator

# specify some setup hyperparameters for comparison
sum_window_E = 1
dataset_list = ['cifar10-valid','cifar100', 'ImageNet16-120']
dataset = 'cifar10-valid'
search_space = 'nas201'
max_epochs = 200
n_arch = 6704
arch_dataset = f'./{dataset}_n{n_arch}_arch_info'

# define all estimators to be compared
method_list = [
    {'name': 'TSE-EMA', 'E': None, 'metric': 'train_loss', 'color': 'red', 'style': '-'},
    {'name': 'TSE', 'E': None, 'metric': 'train_loss', 'color': 'orange', 'style': '-'},
    {'name': 'TSE-E', 'E': sum_window_E, 'metric': 'train_loss', 'color': 'blue', 'style': '-'},
    {'name': 'VAccES', 'E': None, 'metric': 'val_acc', 'color': 'green', 'style': '-'},
    {'name': 'LcSVR', 'E': None, 'metric': ['val_acc', 'HP', 'AP'], 'color': 'cyan',
     'style': '-', 'ntrain': 200, 'interval': 25},
    {'name': 'SoVL', 'E': None, 'metric': 'val_loss', 'color': 'deeppink', 'style': '-'},
    {'name': 'SynFlow', 'label': 'SynFlow', 'E': None, 'metric': 'arch', 'color': 'lime', 'style': '-',
     'batch_size': 64},
    {'name': 'SNIP', 'E': None, 'metric': 'arch', 'color': 'blueviolet', 'style': '-', 'batch_size': 64},
    {'name': 'JacCov', 'E': None, 'metric': 'arch', 'color': 'brown', 'style': '-', 'batch_size': 64},
    # {'name': 'TestL', 'label': 'TestL(T=200)', 'E': None, 'metric': 'test_loss', 'color': 'black', 'style': '-'},
    ]
zero_cost_estimator_list = ['SynFlow', 'JacCov', 'SNIP']

# load prestored arch data
with open(arch_dataset, 'rb') as outfile:
    res = pickle.load(outfile)
n_arch = len(res['test_acc'])
print(f'total_n_arch ={n_arch}')

# compute rank correlation for each estimator method
dic_for_plot = {}
for method in method_list:

    test_acc_all_arch = [np.max(test_acc) for test_acc in res['test_acc']]
    window_size = method['E']
    method_name = method['name']
    metric_name = method['metric']
    style = method['style']
    indices = range(n_arch)
    print(f'compute rank correlation for {method_name}')

    test_acc_all_arch_array = np.vstack(test_acc_all_arch)

    if 'So' in method_name or 'TSE' in method_name:
        metric_all_arch = res[metric_name]
        sum_metric_all_arch = []
        for i in range(n_arch):
            metric_one_arch = metric_all_arch[i]
            if window_size is not None:
                so_metric = [np.sum(metric_one_arch[se - window_size:se]) for se in range(window_size, max_epochs)]
            else:
                if 'EMA' in method_name:
                    so_metric = []
                    mu = 0.9
                    for se in range(max_epochs):
                        if se <= 0:
                            ema = metric_one_arch[se]
                        else:
                            ema = ema * (1 - mu) + mu * metric_one_arch[se]

                        so_metric.append(ema)
                else:
                    so_metric = [np.sum(metric_one_arch[:se]) for se in range(max_epochs)]
            sum_metric_all_arch.append(so_metric)

        metric_all_arch_array = np.vstack(sum_metric_all_arch)
    elif 'LcSVR' in method_name:
        from svr_estimator import SVR_Estimator
        n_train = method['ntrain']
        svr_interval = method['interval']
        metric_all_arch_list = [res[metric] for metric in metric_name]
        svr_regressor = SVR_Estimator(metric_all_arch_list, test_acc_all_arch,
                                      all_curve=True, n_train=n_train)

    elif method_name in zero_cost_estimator_list:
        batch_size = method['batch_size']
        metric_all_arch = res[metric_name]
        estimator = zero_cost_estimator(method_name=method_name, search_space=search_space,
                                        dataset=dataset, batch_size=batch_size)
        score_all_arch = []
        for i in range(n_arch):
            metric_one_arch = metric_all_arch[i]
            score_one_arch = estimator.predict(metric_one_arch)
            score_all_arch.append(score_one_arch)
            print(f'arch={i}: score={score_one_arch}')
        metric_all_arch_array = np.vstack(score_all_arch)
    else:
        metric_all_arch = res[metric_name]
        metric_all_arch_array = np.vstack(metric_all_arch)

    # save method scores
    if 'LcSVR' not in method_name:
        method['score_all_arch'] = metric_all_arch_array

    # compute rank correlation
    if 'LcSVR' in method_name:
        rank_correlation_metric = []
        score_all_epochs = []
        epoch_list = range(svr_interval, max_epochs + 1, svr_interval)
        for epoch in epoch_list:
            best_hyper, time_taken = svr_regressor.learn_hyper(epoch)
            rank_coeff = svr_regressor.extrapolate()
            rank_correlation_metric.append(rank_coeff)
            score_all_epochs.append(svr_regressor.y_pred)
        method['score_all_arch'] = np.hstack(score_all_epochs)

    elif method_name in zero_cost_estimator_list or method_name == 'TestL':
        rank_correlation_metric = []
        if 'loss' in metric_name:
            metric_all_arch_array = - metric_all_arch_array
        rank_coeff, _ = stats.spearmanr(test_acc_all_arch_array, metric_all_arch_array)
        rank_correlation_metric = [rank_coeff]*max_epochs

    else:
        rank_correlation_metric = []
        for j in range(metric_all_arch_array.shape[1]):
            if 'loss' in metric_name:
                metric_estimator =  - metric_all_arch_array[:, j]
            else:
                metric_estimator = metric_all_arch_array[:, j]
            rank_coeff, _ = stats.spearmanr(test_acc_all_arch_array, metric_estimator)
            rank_correlation_metric.append(rank_coeff)

    # save rank correlation performance over epochs for plotting
    if window_size is not None:
        dic_for_plot[method_name] = [range(window_size, int(window_size + len(rank_correlation_metric))), rank_correlation_metric, style]
    elif 'LcSVR' in method_name:
        dic_for_plot[method_name] = [epoch_list, rank_correlation_metric, style]
    else:
        dic_for_plot[method_name] = [range(1, len(rank_correlation_metric)), rank_correlation_metric[1:], style]

# plot the rank correlation performance of all the estimators
figure, axes = plt.subplots(1, 1, figsize=(3, 3))
fs = 11
for method in method_list:
    method_name = method['name']
    color = method['color']
    style = method['style']

    content = dic_for_plot[method_name]
    x_range, rank_corr, fmt = content
    axes.plot(x_range, rank_corr, color=color, ls=style, label=method_name)

axes.legend(prop={'size': fs-1}, loc="lower right").set_zorder(12)
axes.set_title(f'{dataset}')
axes.set_xscale('log')
axes.set_xticks(np.logspace(0.0, np.log(int(max_epochs/2)) / np.log(10), 4, base=10, endpoint=True))
axes.set_xticklabels([f'{v/(int(max_epochs/2)*2):.2f}' for v in
                              np.logspace(0.0, np.log(int(max_epochs/2)) / np.log(10), 4, base=10, endpoint=True)])
if dataset in ['cifar10-valid', 'ImageNet16-120', 'cifar100']:
    axes.set_xlim([int(max_epochs/2) * 0.04, int(max_epochs/2)])
    axes.set_ylim([0.6, 1.0])

axes.set_xlabel('Fraction of $T_{end}$', fontsize=fs)
axes.set_ylabel('Rank Correlation', fontsize=fs)
fig_name = f'./rank_corr_comparison_on_{search_space}{dataset}_for{n_arch}archs.pdf'
plt.savefig(fig_name, bbox_inches='tight')