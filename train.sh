#!/bin/bash


models=(
  "vit_b_16"
)

dataset="Flowers102"

function sayConfig() {
    return 43
}

for model in "${models[@]}"
do
  for i in {2..5}
  do
    args=(
      --model $model
      --dataset $dataset
      --epochs 10
      --batch-size 8
      --save-path models/${model}-${dataset}-${i}.pth
    )

    x=sayConfig
    python3 train.py "${args[@]}"
  done
done
