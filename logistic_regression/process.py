
from matplotlib import rc
import matplotlib.pyplot as plt


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.style.use('ggplot')

rc('text', usetex=True)

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22


# SMALL_SIZE = 10
# MEDIUM_SIZE = 12
# BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rcParams["figure.figsize"] = (3,2.7)

dpi = 300

import numpy as np

# experiments[iters][n][eps][seeds]
experiments = {
    # 1000: {
    #     100: {
    #         2.0: list(range(10)),
    #         4.0: list(range(10)),
    #         8.0: list(range(10)),
    #         'nodp': list(range(10)),
    #     },
    #     200: {
    #         2.0: list(range(10)),
    #         4.0: list(range(10)),
    #         8.0: list(range(10)),
    #         'nodp': list(range(10)),
    #     },
    #     500: {
    #         2.0: list(range(10)),
    #         4.0: list(range(10)),
    #         8.0: list(range(10)),
    #         'nodp': list(range(10)),
    #     }
    # },
    2000: {
        100: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        200: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        500: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        1000: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        }
    },
    4000: {
        100: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        200: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        500: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        1000: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        }
    },
    5000: {
        100: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        200: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        500: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        1000: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        }
    },
    10000: {
        100: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        200: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        500: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        1000: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        }
    },
    20000: {
        100: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        200: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        500: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        1000: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        }
    },
    30000: {
        100: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        200: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        500: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        1000: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        }
    },
    40000: {
        100: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        200: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        500: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        1000: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        }
    },
    60000: {
        100: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        200: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        500: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        1000: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        }
    },
    80000: {
        100: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        200: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        500: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        1000: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        }
    },
    100000: {
        100: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        200: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        500: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        },
        1000: {
            2.0: list(range(10)),
            4.0: list(range(10)),
            8.0: list(range(10)),
            'nodp': list(range(10)),
        }
    },
    # 500000: {
    #     100: {
    #         2.0: list(range(10)),
    #         4.0: list(range(10)),
    #         8.0: list(range(10)),
    #         'nodp': list(range(10)),
    #     },
    # },
    # 800000: {
    #     100: {
    #         2.0: list(range(10)),
    #         4.0: list(range(10)),
    #         8.0: list(range(10)),
    #         'nodp': list(range(10)),
    #     },
    # },
    # 1000000: {
    #     100: {
    #         2.0: list(range(10)),
    #         4.0: list(range(10)),
    #         8.0: list(range(10)),
    #         'nodp': list(range(10)),
    #     },
    # }
}

def eps_plot_label(eps):
    return f"$\\varepsilon={eps}$" if eps != "nodp" else "No DP"

baseline_aucs = {}

# which plots to create
make_AUC_iters_n_iters_plots = False
make_AUC_n_iters_plots = True
make_AUC_eps_n_iter_plots = True
make_AUC_final_iters_n_plots = True

per_final_iter_plot_data = {}
for iters, exps_for_iters in experiments.items():
    per_n_plot_data = {}
    for n, exps_for_n in exps_for_iters.items():
        if n not in baseline_aucs:
            with open(f"results/logreg_baseline_n{n}.txt", "r") as f:
                baseline_aucs[n] = float(f.readline())

        per_eps_boxplot_data = {}

        if n not in per_final_iter_plot_data:
            per_final_iter_plot_data[n] = {}

        for eps, exps_for_eps in exps_for_n.items():
            eps_str = eps if eps == "nodp" else f"eps_{eps}"
            data = None
            for seed in exps_for_eps:
                with open(f"results/logreg_{eps_str}_n{n}_i{iters}_s{seed}.npz", "rb") as f:
                    all_aucs = np.load(f)['arr_0']
                    assert(all_aucs.ndim == 2)
                    if data is None:
                        data = all_aucs
                    else:
                        assert(all_aucs.shape[0] == data.shape[0])
                        data = np.hstack((data, all_aucs[:,1,np.newaxis]))

            per_eps_boxplot_data[eps] = data[-1, 1:] # AUC for all seeds from last iteration
            if eps not in per_n_plot_data:
                per_n_plot_data[eps] = {}
            per_n_plot_data[eps][n] = data[-1, 1:]
            if eps not in per_final_iter_plot_data[n]:
                per_final_iter_plot_data[n][eps] = {}
            per_final_iter_plot_data[n][eps][iters] = data[-1, 1:]

            if make_AUC_iters_n_iters_plots:
                plt.figure(f'AUC_iters_{n}_{iters}')

                data_mean = np.mean(data[:,1:], axis=1)
                data_std = np.std(data[:,1:], axis=1)

                plt.fill_between(data[:,0], data_mean - data_std, data_mean + data_std, label=eps_plot_label(eps), alpha=.3)

        if make_AUC_iters_n_iters_plots:
            plt.figure(f'AUC_iters_{n}_{iters}')
            plt.title(f'Evolution of AUC during training')

            plt.axhline(baseline_aucs[n], label='Baseline')

            plt.ylabel('AUC')
            plt.xlabel('Training Iterations')
            plt.legend(loc='upper left')
            plt.savefig(f'AUC_iters_{n}_{iters}.pdf', bbox_inches='tight', dpi=dpi)


        if make_AUC_eps_n_iter_plots:
            plt.figure(f'AUC_eps_{n}_{iters}')
            epss, vals = zip(*per_eps_boxplot_data.items())
            plt.boxplot(vals)
            plt.xticks(np.arange(len(epss))+1, [eps if eps != "nodp" else "No DP" for eps in epss])
            plt.axhline(baseline_aucs[n], label='Baseline')
            plt.title(f'AUC after training per privacy level')
            plt.ylabel('AUC')
            plt.xlabel('$\\varepsilon$')
            plt.savefig(f'AUC_eps_{n}_{iters}.pdf', bbox_inches='tight', dpi=dpi)

    if make_AUC_n_iters_plots:
        plt.figure(f'AUC_n_{iters}')
        for eps, per_eps_data in per_n_plot_data.items():
            ns, vals = zip(*per_eps_data.items())
            # print(ns)
            # print(vals)
            vals = np.vstack(vals)
            means = np.mean(vals, axis=-1)
            stds = np.std(vals, axis=-1)

            # plt.xscale('log')
            plt.fill_between(ns, means - stds, means + stds, label=eps_plot_label(eps), alpha=.3)

        plt.title(f'AUC after training per data set size')
        plt.ylabel('AUC')
        plt.xlabel('$N$')
        plt.legend(loc='lower right')
        plt.savefig(f'AUC_n_{iters}.pdf', bbox_inches='tight', dpi=dpi)

if make_AUC_final_iters_n_plots:
    for n, per_n_data in per_final_iter_plot_data.items():
        plt.figure(f'AUC_final_iters_{n}')
        for eps, per_eps_data in per_n_data.items():
            iterss, vals = zip(*per_eps_data.items())
            vals = np.vstack(vals)
            means = np.mean(vals, axis=-1)
            # print(f'AUC_final_iters_{n} eps {eps}')
            # print(iterss)
            # print(vals)
            stds = np.std(vals, axis=-1)
            plt.fill_between(iterss, means - stds, means + stds, label=eps_plot_label(eps), alpha=.3)

        plt.axhline(baseline_aucs[n], label='Baseline')
        plt.title(f'AUC after training over total iteration count')
        plt.ylabel('AUC')
        plt.xlabel('Total training iterations')
        plt.legend(loc='lower right')
        plt.savefig(f'AUC_final_iters_{n}.pdf', bbox_inches='tight', dpi=dpi)

# plt.show()