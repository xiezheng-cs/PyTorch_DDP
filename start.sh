#!/usr/bin/env bash
python dataparallel.py
python -m torch.distributed.launch --nproc_per_node=3 distributed.py