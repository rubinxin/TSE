import pickle
import numpy as np
import matplotlib.pyplot as plt

save = True
# path_list = ['nasbench201_cifar10-valid', 'nasbench201_cifar100', 'nasbench201_ImageNet16-120']
path_list = ['nasbench201_cifar100']

n_init = 10
mean_std = True
method_list = ['regularized_evolution']
eval_metric_list = ['final_val', 'early_stop', 'tseema']
legend_label = ['Val Acc(T=200)', 'Val Acc(T=10)', 'TSE-EMA(T=10)']
fs = 12
es_budget_list = [200, 10, 10]
n_end = 200
color = ['y','g','r']

for path in path_list:
    for method in method_list:
        plt.figure(figsize=(3, 3))
        for i, eval_metric in enumerate(eval_metric_list):
            es_budget = es_budget_list[i]
            results = pickle.load(open(f'./{path}/{method}_{eval_metric}{es_budget}', 'rb'))
            if es_budget is None:
                multiplier = 1
            else:
                multiplier = int(200/es_budget)

            y_test_mono_list = []
            run_time_list = []
            for r in results:
                y_test_error = r['y_test']
                curr_y_test_error = np.min(y_test_error[:n_init])

                y_test_mono = [curr_y_test_error]
                run_time    = []
                for j, y in enumerate(y_test_error[n_init: n_init+n_end*multiplier]):
                    if y < curr_y_test_error:
                        curr_y_test_error = y

                    train_time = r['costs'][n_init+j]['train_time']
                    run_time.append(train_time)
                    y_test_mono.append(curr_y_test_error)
                run_time_list.append(run_time)
                y_test_mono_list.append(y_test_mono)

            y_test_mono_array = np.vstack(y_test_mono_list)*100
            run_time_array = np.vstack(run_time_list)
            mean_run_time_array = np.mean(run_time_array, 0)
            mean_run_time_accumulate = np.array([np.sum(mean_run_time_array[:ti]) for ti in range(len(mean_run_time_array))])/3600

            if eval_metric == 'final_val':
                mean_total_final_val_time = np.sum(mean_run_time_array)
            else:
                end_idx = np.where(mean_run_time_accumulate<=mean_total_final_val_time)[0][-1]
                mean_run_time_accumulate = mean_run_time_accumulate[:end_idx+1]
                y_test_mono_array = y_test_mono_array[:end_idx+1]

            x_length = len(mean_run_time_accumulate)
            print(f'{eval_metric}{es_budget}{len(y_test_mono)}')
            if mean_std:
                indices_range = range(0, x_length, int(5 * x_length / n_end))

                mean_y_test = np.mean(y_test_mono_array, 0)
                std_y_test = np.std(y_test_mono_array, 0)/np.sqrt(y_test_mono_array.shape[0])
                print(len(mean_y_test))
                plt.errorbar(mean_run_time_accumulate[indices_range], mean_y_test[indices_range], yerr=std_y_test[indices_range], color=color[i], label=f'{legend_label[i]}')

                if eval_metric == 'final_val':
                    plt.plot(mean_run_time_accumulate[indices_range], [mean_y_test[-1]]*len(indices_range), 'k--')
                    x_limit_upper = mean_run_time_accumulate[-1]
            else:
                indices_range = range(0, x_length)
                q25_y_test = np.quantile(y_test_mono_array, 0.25, axis=0)
                q50_y_test = np.quantile(y_test_mono_array, 0.5, axis=0)
                q75_y_test = np.quantile(y_test_mono_array, 0.75, axis=0)
                print(len(mean_run_time_accumulate))
                plt.fill_between(mean_run_time_accumulate[indices_range], q25_y_test[indices_range], q75_y_test[indices_range], color=color[i], alpha=0.5)
                plt.plot(mean_run_time_accumulate[indices_range], q50_y_test[indices_range], color[i],label=f'{legend_label[i]}')

        plt.legend(prop={'size': fs})
        plt.xlim([1, x_limit_upper])
        plt.xscale('log')
        plt.xlabel('Run time (hours)', fontsize=fs)
        plt.ylabel('Best test error (%)', fontsize=fs)
        eval_metric_str = eval_metric_list[0]+eval_metric_list[1]+eval_metric_list[2]
        fig_name = f'./{path}_{method}_{eval_metric_str}_{es_budget_list[-1]}_{mean_std}_log.pdf'
        plt.savefig(fig_name, bbox_inches='tight')