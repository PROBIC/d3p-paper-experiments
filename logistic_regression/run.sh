#!/bin/bash
set -x
set -e

# todo: adjust values for num_data, delta, num_iter and seeds as required
# NUM_DATA=500 # alternatives: 100; 200; 1000
NUM_DATA=100
# DELTA=2e-3 # alternatives: 1e-2; 5e-3; 1e-3 corresponding to NUM_DATA
DELTA=1e-2
#for NUM_ITER in 2000 4000 5000 10000 20000 30000 40000 60000 80000 100000
for NUM_ITER in 10000
do
for SEED in {0..9}
do
	python logistic_regression.py --no-dp --num_iter $NUM_ITER --num_data $NUM_DATA --delta=$DELTA -l 3 --seed $SEED
	for EPS in 2 4 8
	do
		python logistic_regression.py --epsilon $EPS --num_iter $NUM_ITER --num_data $NUM_DATA --delta=$DELTA -l 3 --seed $SEED
	done
done
done
