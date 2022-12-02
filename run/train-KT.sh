#!/usr/bin/env bash
python net/train.py \
                --proj_dir logs/train_kt \
                --exp_name ${6}-${1} \
                --module KT \
                --category ${1} \
                --dataset_name ${6} \
                --batch_size ${7} \
                --lr ${5} \
                --save_frequency ${3} \
                --nr_epochs ${2} \
                --latent_dim ${4} \
                --dataset_path dataset/${6}
