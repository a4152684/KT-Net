#!/bin/sh
latent_dim=1024
epochs=600
save_frequency=200
lr=1e-4
batch_size=32
dataset_name=3depn
# dataset_name=crn

for category in plane #cabinet car chair lamp sofa table watercraft
do
    sh run/train-KT.sh ${category} ${epochs} ${save_frequency} ${latent_dim} ${lr} ${dataset_name} ${batch_size}
    # sh run/test-KT.sh ${category} ${latent_dim} ${dataset_name}
done
