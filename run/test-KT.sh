#!/usr/bin/env bash
python net/test.py \
                --proj_dir logs/test_kt/proj_log \
                --exp_name mpc-${3}-${1} \
                --module KT \
                --dataset_name ${3} \
                --category ${1} \
                --latent_dim ${2} \
                --dataset_path dataset/${3} \
                --pretrain_path pretrain/${3}/${1}.pth \