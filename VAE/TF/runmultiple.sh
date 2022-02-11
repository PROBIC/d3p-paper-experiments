#!/bin/bash

mkdir -p paper_logs/{cifar,fashion,mnist}/{dp,no_dp}
export CUDA_VISIBLE_DEVICES=3
for i in $(seq 0 1 20);
do
   CMD="python vae_tf.py --seed ${i}"; echo ${CMD}; ${CMD} >> paper_logs/mnist/dp/log${i}.o;
   CMD="python vae_tf.py --seed ${i} --no_dp"; echo ${CMD}; ${CMD} >> paper_logs/mnist/no_dp/log${i}.o;
   CMD="python vae_fashion.py --seed ${i}"; echo ${CMD}; ${CMD} >> paper_logs/fashion/dp/log${i}.o;
   CMD="python vae_fashion.py --seed ${i} --no_dp"; echo ${CMD}; ${CMD} >> paper_logs/fashion/no_dp/log${i}.o;
   CMD="python vae_cifar.py --seed ${i}"; echo ${CMD}; ${CMD} >> paper_logs/cifar/dp/log${i}.o;
   CMD="python vae_cifar.py --seed ${i} --no_dp"; echo ${CMD}; ${CMD} >> paper_logs/cifar/no_dp/log${i}.o;
done

