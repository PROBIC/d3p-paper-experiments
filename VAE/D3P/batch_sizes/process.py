import numpy as np

batch_sizes = [32,64,128,256,512,1024]
batches_per_run = 100

def summarize(prefix):
    times = []
    stderrs = []

    for i in batch_sizes:
        with open(prefix + '/log{0}.o'.format(i), 'r') as f:
            lines = f.read().splitlines()
            local_times = []
            for line in lines:
                splits = line.split(' ')
                local_times.append(float(splits[5]))
            times.append(np.mean(local_times) / batches_per_run * 1000) # scale up to ms
            stderrs.append(np.std(local_times) / np.sqrt(batches_per_run) * 1000) # scale up to ms

        print(f'{prefix}: batch_size: {i} Time: {times[-1]:.2f} +/- {stderrs[-1]:.2f}')
    times = np.array(times)
    stderrs = np.array(stderrs)
    return times, stderrs

batchtimenodp, batchtimeerrnodp = summarize("no_dp")
batchtime, batchtimeerr= summarize("dp")


from matplotlib import rc

import matplotlib.pyplot as plt
import numpy as np


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.style.use('ggplot')

rc('text', usetex=True)

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.errorbar(batch_sizes, batchtime, yerr=batchtimeerr, fmt='-x', label='DP-VI', alpha=.9)
plt.errorbar(batch_sizes, batchtimenodp, yerr=batchtimeerrnodp, fmt='-o', label='Non-private', alpha=.9)
plt.fill_between(batch_sizes, batchtime-batchtimeerr, batchtime+batchtimeerr, alpha=.3)
plt.fill_between(batch_sizes, batchtimenodp-batchtimeerrnodp, batchtimenodp+batchtimeerrnodp, alpha=.3)
plt.title('Runtime per Iteration Over Minibatch Size')

plt.ylabel('Runtime / Iteration [ms]')
plt.xlabel('Minibatch Size')
plt.legend()
plt.savefig('batch_times.pdf', bbox_inches='tight')
plt.show()
