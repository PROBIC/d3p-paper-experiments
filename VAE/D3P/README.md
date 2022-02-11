## Overview

Scripts for experiments regarding performance of d3p on VAE in Sec 5.1.

- `vae.py`: Implementation of the VAE model and inference for MNIST dataset.
- `vae.py`: Implementation of the VAE model and inference for Fashion-MNIST dataset.
- `vae.py`: Implementation of the VAE model and inference for CIFAR10 dataset.
- `vae_sgd.py`: Same as `vae.py` but using SGD instead of Adam optimizer; not shown in paper.
- `vae_batch_sizes.py`: Modified variant of `vae.py` that performs shorter inference; optimized for comparing effect of batch size.
- `runmultiple.sh`: Runs all required evaluation runs of `vae.py` to create measurements of d3p for Tables 2 and 3, Sec. 5.1.
- `runbatchsizes.sh`: Runs all required evaluation runs of `vae_batch_sizes.py` to create measurements of d3p for Figure 3, Sec. 5.2.
- `process.py`: Scripts for processing results obtained from runs and creating the plots and tables.

## Usage
Run `./runmultiple.sh` to get the experiment results (this will take some time). Afterwards run `python process.py`, which will print summary statistics as presented in Table 2 and 3 in the paper for the d3p runs.

For the results from Section 5.2, run `./runbatchsizes.sh` followed by `cd batch_sizes && python process.py`, which will create a PDF file containing the plot.