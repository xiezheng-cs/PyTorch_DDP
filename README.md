# PyTorch_DDP

## 测试配置

model = resnet18，dataset = ImageNet，epoch = 5， batch_size = 1200，GPUs_num = 3 @TITAN Xp

CUDA_VISIBLE_DEVICES=0,1,2 python dataparallel.py

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port=23334 distributed.py

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port=23334 distributed_syncBN_amp.py

## 测试结果

|                 Method                 | Memory (MB) | Time (s) | ImageNet Top1 Acc(%) |
| :------------------------------------: | :---------: | :------: | :------------------: |
|              DataParallel              |    11329    |   7633   |        46.71         |
|        DistributedDataParallel         |    11329    | **4612** |      **46.83**       |
|     DistributedDataParallel + amp      |    8679     |   4680   |        46.74         |
| DistributedDataParallel + amp + SyncBN |    8679     |   8173   |        46.78         |

**注：** SyncBN: 同步BN（pytorch自带）; amp: 自动混合精度训练（pytorch自带）

## 结论

（1）SyncBN 会影响训练速度，且在图像分类中作用不大，进行目标检测和图像分割时使用

（2）amp 可大大降低模型的内存占用，但可能在小模型上加速效果不明显

（3）一般情况下，使用 DistributedDataParallel 即可



