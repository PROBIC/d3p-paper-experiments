## Overview

Scripts for experiments regarding hierarchical logistic regression in Sec 5.3.

- `logistic_regression.py`: Scripts file containing d3p/NumPyro code for running the inference given a certain set of hyperparameters.
- `run.sh`: Bash file to run the Python code for different hyperparameter settings.
- `process.py`: Python file for parsing the results and creating plots.

## Usage

Run `./run.sh` to run the experiments. You can edit this file to change the number of data points used in the experiments (remember to also adapt the privacy bound δ accordingly).

Run `python process.py` to generate the plots. You can edit the `experiments` dictionary in this file to select which runs will be parsed. Note that this script will produce a lot of plots.