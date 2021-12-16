import numpy as np

def summarize(prefix):
    times = []
    losses = []

    for i in range(20):
        with open(prefix + '/log{0}.o'.format(i), 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
            splits = last_line.split(' ')
            times.append(np.float64(splits[5]))
            losses.append(np.float64(splits[3]))

    print(f'{prefix}: Time: {np.mean(times):.2f} +/- {np.std(times):.2f},  Loss: {np.mean(losses):.2f} +/- {np.std(losses):.2f}')

summarize("paper_logs/mnist/no_dp")
summarize("paper_logs/mnist/dp")
summarize("paper_logs/fashion/no_dp")
summarize("paper_logs/fashion/dp")
summarize("paper_logs/cifar/no_dp")
summarize("paper_logs/cifar/dp")
