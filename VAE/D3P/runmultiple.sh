#!/bin/bash

mkdir -p paper_logs/{cifar,fashion,mnist}/{dp,no_dp,altsampler}

export CUDA_VISIBLE_DEVICES=3
for i in $(seq 0 1 20);
do
   python vae.py --seed=${i} >> paper_logs/mnist/dp/log${i}.o;
   python vae.py --seed=${i} --no_dp --sampler builtin>> paper_logs/mnist/no_dp/log${i}.o;
   python vae.py --seed=${i} --sampler builtin >> paper_logs/mnist/altsampler/log${i}.o;
   python vae_fashion.py --seed=${i} >> paper_logs/fashion/dp/log${i}.o;
   python vae_fashion.py --seed=${i} --no_dp --sampler builtin>> paper_logs/fashion/no_dp/log${i}.o;
   python vae_fashion.py --seed=${i} --sampler builtin >> paper_logs/fashion/altsampler/log${i}.o;
   python vae_cifar.py --seed=${i} >> paper_logs/cifar/dp/log${i}.o;
   python vae_cifar.py --seed=${i} --no_dp --sampler builtin>> paper_logs/cifar/no_dp/log${i}.o;
   python vae_cifar.py --seed=${i} --sampler builtin >> paper_logs/cifar/altsampler/log${i}.o;
done

