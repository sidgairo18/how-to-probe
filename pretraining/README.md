# Pretraining Scripts

## Introduction to Pre-training

These are the pre-training scripts for SSL models: [`MoCov2`](https://github.com/open-mmlab/mmpretrain/tree/main/configs/mocov2), [`BYOL`](https://github.com/open-mmlab/mmpretrain/tree/main/configs/byol), and [`DINO`](https://github.com/facebookresearch/dino); as well as supervised models. This is done for both conventional (standard) ResNet50s as well as the inherently interpretable [`B-cos`](https://github.com/B-cos/B-cos-v2/tree/main) ResNet50s.

### Training MoCov2 and BYOL

The training of both MoCov2 and BYOL depend on the implementations from the good folks at [`open-mmlab/mmpretrain`](https://github.com/open-mmlab/mmpretrain/). It is recommended to setup the code as the `mmpretrain` does. The respective `config` files and `model` files can be replaced from the directory we provide `how-to-probe/pretraining/mmpretrain/` to get the actual parameters we used for training `conventional` and `B-cos` backbones.

Once the code is setup. Below are the training scripts for MoCov2 and BYOL respectively:

#### Training MoCov2

**B-cos ResNet50 Training on 4 GPUS, 1 Node**

```
cd /path/to/how-to-probe/pretraining/mmpretrain                                                 
                                                                                              
# run your script here                                                                        
export PYTHONPATH="/path/to/how-to-probe/pretraining/mmpretrain/:$PYTHONPATH"                   
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/mocov2/mocov2_bcosresnet50_4xb64-coslr-200e_in1k.py 4 --resume 'auto'
```

**Conventional ResNet50 Training on 4 GPUs, 1 Node**

```
cd /path/to/how-to-probe/pretraining/mmpretrain                                                 
                                                                                              
# run your script here                                                                        
export PYTHONPATH="/path/to/how-to-probe/pretraining/mmpretrain/:$PYTHONPATH"                   
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/mocov2/mocov2_resnet50_4xb64-coslr-200e_in1k.py 4 --resume 'auto'
```

#### Training BYOL

**B-cos ResNet50 Training on 4 GPUS, 1 Node**

```
cd /path/to/how-to-probe/pretraining/mmpretrain                                                 
                                                                                              
# run your script here                                                                        
export PYTHONPATH="/path/to/how-to-probe/pretraining/mmpretrain/:$PYTHONPATH"                   
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/byol/byol_bcosresnet50_4xb64-coslr-200e_in1k.py 4 --resume 'auto'
```

**Conventional ResNet50 Training on 4 GPUs, 1 Node**

```
cd /path/to/how-to-probe/pretraining/mmpretrain                                                 
                                                                                              
# run your script here                                                                        
export PYTHONPATH="/path/to/how-to-probe/pretraining/mmpretrain/:$PYTHONPATH"                   
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/byol/byol_resnet50_4xb64-coslr-200e_in1k.py 4 --resume 'auto'
```

`**Note**: Do ensure you set the proper paths for **data_root** in the respective config.py files`

### Training DINO

For training [`DINO`](https://github.com/facebookresearch/dino) we follow the official implemention, however we add a few modifications to the code for training `B-cos ResNet50`. However, it is more or less very similar to the original implementation and we would kindly refer you to it for any design choices or implementation details.

**B-cos ResNet50 Training on 4 GPUS, 1 Node**

```
cd /path/to/how-to-probe/pretraining/dino

# run your script here
export PYTHONPATH="/path/to/how-to-probe/pretraining/dino/:$PYTHONPATH"
torchrun --nproc_per_node=4 main_dino.py --arch bcosresnet50 --lr 0.003 --weight_decay 0.0 --weight_decay_end 0.0 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /path/to/imagenet/train --epochs 200 --warmup_teacher_temp_epochs 30 --optimizer adamw --use_bcos_head 1 --batch_size_per_gpu 64 --local_crops_number 8 --global_crop_size 224 --local_crop_size 96

```

**Conventional ResNet50 Training on 4 GPUs, 1 Node**

```
cd /path/to/how-to-probe/pretraining/dino

# run your script here
export PYTHONPATH="/path/to/how-to-probe/pretraining/dino/:$PYTHONPATH"
torchrun --nproc_per_node=4 main_dino.py --arch resnet50 --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /path/to/imagenet/train --epochs 200 --warmup_teacher_temp_epochs 10 --optimizer sgd --batch_size_per_gpu 64 --local_crops_number 6 --global_crop_size 224 --local_crop_size 96

```

**Additionally for multi-node, multi-gpu training, the run_with_submitit_new.py script might be used**

The sample below is for 2 Nodes, 4 GPUs on each node.

```
cd /path/to/how-to-probe/pretraining/dino

# run your script here
python run_with_submitit_new.py --nodes 2 --ngpus 4 --arch bcosresnet50 --data_path /path/to/imagenet/train --output_dir /path/to/output_directory --use_fp16 1 --epochs 200 --timeout 716 --lr 0.006 --weight_decay 0.0 --weight_decay_end 0.0 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --epochs 200 --warmup_teacher_temp_epochs 20 --optimizer adamw --use_bcos_head 1 --batch_size_per_gpu 32 --local_crops_number 8 --global_crop_size 224 --local_crop_size 96 --num_workers 8

```


`**Note**: For B-cos DINO we found the **adamw** optimizer along with **weight_decay=0** to work better rather than the default setting for conventional models.`

`Additionally feel free to modify based on your needs, and hyper-parameters.`

### Training Supervised Models

We again use the `mmpretrain` setup for supervised pre-trainings on ImageNet1K.

**B-cos Model with BCE Loss**
```
cd /path/to/how-to-probe/pretraining/mmpretrain                                                 
                                                                                              
# run your script here                                                                        
export PYTHONPATH="/path/to/how-to-probe/pretraining/mmpretrain/:$PYTHONPATH"                   
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/supervised_trainings/sup_bcosresnet50-bcoshead_bce_adamnowd_coslr_4xb64_100e_in1k.py 4 --resume 'auto'
```

```
cd /path/to/how-to-probe/pretraining/mmpretrain                                                 
                                                                                              
# run your script here                                                                        
export PYTHONPATH="/path/to/how-to-probe/pretraining/mmpretrain/:$PYTHONPATH"                   
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/supervised_trainings/sup_resnet50-stdhead_bce_adamnowd_coslr_4xb64_100e_in1k.py 4 --resume 'auto'
```

`For additional configs see respective configs/[mocov2/byol/supervised_trainings] or dino/ folders.`
