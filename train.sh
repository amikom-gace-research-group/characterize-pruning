#!/bin/bash


models=(
  "swin_v2_s"
  "swin_v2_b"
)

dataset="Flowers102"

for model in "${models[@]}"
do
  for i in {1..5}
  do
    args=(
      --model $model
      --dataset $dataset
      --epochs 10
      --batch-size 8
      --save-path models/${model}-${dataset}-${i}.pth
    )

    python3 train.py "${args[@]}"
  done
done
