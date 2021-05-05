#!/usr/bin/env bash
python dataparallel.py
python -m torch.distributed.launch --nproc_per_node=4 distributed.py
python -m torch.distributed.launch --nproc_per_node=4 apex_distributed.py