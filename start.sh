#!/usr/bin/env bash
python dataparallel.py
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port=23334 distributed.py