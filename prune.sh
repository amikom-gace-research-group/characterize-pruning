#!/bin/bash

#----------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------#
#                                                                                              #
#  GENERAL CONFIGURATIONS                                                                      #
#                                                                                              #
#----------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------#

models=(
    "mobilenet_v3_small"
    "mobilenet_v3_large"
    "efficientnet_b1"
    "efficientnet_b3"
    "densenet169"
    "densenet201"
    "vit_b_16"
)
layer_kinds=("Linear")
amounts=(0.5 0.8)
dim=0
sparse_layouts=(
    "sparse_coo"
    "sparse_csr"
    "sparse_csc"
    "sparse_bsc"
    "sparse_bsr"
)



#----------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------#
#                                                                                              #
#  RANDOM STRUCTURED PRUNING                                                                   #
#                                                                                              #
#----------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------#

for model in "${models[@]}"
do
    for layer_kind in "${layer_kinds[@]}"
    do
        for amount in "${amounts[@]}"
        do
            for sparse_layout in "${sparse_layouts[@]}"
            do
                args=(
                    --path "./models/${model}-Flowers102-1.pth"
                    --layer-kind "${layer_kind}"
                    --amount "${amount}"
                    --dim "${dim}"
                    --sparse-layout "${sparse_layout}"
                    --save-path "./models/pruned/${amount}-RND-${model}-Flowers102-${sparse_layout}.pth"
                )

                if [ "${sparse_layout}" = "sparse_bsc" ] || [ "${sparse_layout}" = "sparse_bsr" ]; then
                    args+=(--blocksize 1 8)
                fi

                python3 prune.py "${args[@]}"
            done
        done
    done
done



#----------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------#
#                                                                                              #
#  LN STRUCTURED PRUNING                                                                   #
#                                                                                              #
#----------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------#

ns=(1 2)

for model in "${models[@]}"
do
    for layer_kind in "${layer_kinds[@]}"
    do
        for amount in "${amounts[@]}"
        do
            for n in "${ns[@]}"
            do
                for sparse_layout in "${sparse_layouts[@]}"
                do
                    args=(
                        --path "./models/${model}-Flowers102-1.pth"
                        --layer-kind "${layer_kind}"
                        --amount "${amount}"
                        --n "${n}"
                        --dim "${dim}"
                        --sparse-layout "${sparse_layout}"
                        --save-path "./models/pruned/${amount}-N${n}-${model}-Flowers102-${sparse_layout}.pth"
                    )

                    if [ "${sparse_layout}" = "sparse_bsc" ] || [ "${sparse_layout}" = "sparse_bsr" ]; then
                        args+=(--blocksize 1 8)
                    fi

                    python3 prune.py "${args[@]}"
                done
            done
        done
    done
done



