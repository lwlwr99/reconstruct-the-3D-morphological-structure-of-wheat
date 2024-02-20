#!/usr/bin/env bash
CONFIG=$1
GPUS=1
PORT=${PORT:-29500}

OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py --dist configs/softgroup_transformer_backbone.yaml --work_dir test_1_attention/backbone
OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py --dist configs/softgroup_transformer.yaml --work_dir test_1_attention