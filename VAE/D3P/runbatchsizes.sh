#!/bin/bash

mkdir -p batch_sizes/{no_,}dp

export CUDA_VISIBLE_DEVICES=3
for i in (32 64 128 256 512 1024);
do
   python vae.py --batch-size ${i} >> batch_sizes/dp/log${i}.o;
   python vae.py --batch-size ${i} --no_dp --sampler builtin>> batch_sizes/no_dp/log${i}.o;
done

