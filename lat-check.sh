#!/bin/bash

models=(
  "mobilenet_v3_small"
  "mobilenet_v3_large"
  "efficientnet_b1"
  "efficientnet_b3"
  "densenet169"
  "densenet201"
  "vit_b_16"
)
sparse_layouts=(
  "sparse_bsr"
  "sparse_bsc"
)

for model in "${models[@]}"
do
  for sl in "${sparse_layouts[@]}"
  do
    path="models/pruned/0.8-RND-${model}-Flowers102-${sl}.pth"
    python3 -m benchmark.lat-check --model "${model}" --path "${path}"
  done
done
