# Linear and MLP Probing

## Introduction to Pre-training

These are the scripts for linear (and MLP) probing the pre-trained SSL and supervised models. As discussed in the paper we use single linear (or B-cos) probes and MLP probes (both conventional and B-cos). The objective functions used are `BCE` and `CE` and as we see quanititatively as well as qualitatively in the paper that `BCE` trained probes tend to localise class-specific features better as compared to `CE` probes on most post-hoc attribution-based XAI methods.

The probing is performed on 3 datasets (ImageNet1K - single-label, multi-class classification, COCO and VOC - multi-label multi-class classification.

### Probing on ImageNet1K

#### Single Linear Probe (BCE)

The scripts are similar for CE. See `how-to-probe/probing/single_label_classification/`.

**B-cos ResNet50 Training on 4 GPUS, 1 Node**                                                 
                                                                                              
```                                                                                           
cd /path/to/how-to-probe/pretraining/mmpretrain                                                
                                                                                              
# run your script here                                                                        
export PYTHONPATH="/path/to/how-to-probe/pretraining/mmpretrain/:$PYTHONPATH"                  
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ../../probing/single_label_classification/bcosresnet50_std-linear-postavgpool-no-bias_bce_4xb64-coslr-100e_in1k.py 4 --resume 'auto'
```                                                                                       
                                                                                              
**Conventional ResNet50 Training on 4 GPUs, 1 Node**                                           
                                                                                              
```                                                                                           
cd /path/to/how-to-probe/pretraining/mmpretrain                                                
                                                                                              
# run your script here                                                                        
export PYTHONPATH="/path/to/how-to-probe/pretraining/mmpretrain/:$PYTHONPATH"                  
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ../../probing/single_label_classification/resnet50_std-linear-postavgpool-with-bias_bce_4xb64-coslr-100e_in1k.py 4 --resume 'auto'
```

#### MLP Probes (BCE)
The scripts are similar for CE. See `how-to-probe/probing/single_label_classification/`.

We have configurations for conventional MLPs (with 2, 3 linear probes with ReLU activations) and B-cos MLPS (with 2, 3 bcos probes).

Sample for 3 layer `B-cos` MLP.

**B-cos ResNet50 Training on 4 GPUS, 1 Node**                                                 
                                                                                              
```                                                                                           
cd /path/to/how-to-probe/pretraining/mmpretrain                                                
                                                                                              
# run your script here                                                                        
export PYTHONPATH="/path/to/how-to-probe/pretraining/mmpretrain/:$PYTHONPATH"                  
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ../../probing/single_label_classification/bcosresnet50_bcos-linear-3-postavgpool_bce_4xb64-linear-steplr-100e_in1k.py 4 --resume 'auto'
```                                                                                       
                                                                                              
**Conventional ResNet50 Training on 4 GPUs, 1 Node**                                           
                                                                                              
```                                                                                           
cd /path/to/how-to-probe/pretraining/mmpretrain                                                
                                                                                              
# run your script here                                                                        
export PYTHONPATH="/path/to/how-to-probe/pretraining/mmpretrain/:$PYTHONPATH"                  
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ../../probing/single_label_classification/resnet50_bcos-linear-3-postavgpool_bce_4xb64-linear-steplr-100e_in1k.py 4 --resume 'auto'
```


### Probing on COCO and VOC

COCO and VOC are multi-label, multi-class datasets, i.e. each image has multiple classes present in the same image (unlike ImageNet1k that only has a single class per image. Thus, only the BCE objective is used to train the different probes. Here we wish to see the impact of the probe complexity on localizing class-spefic features of frozen pre-trained SSL (and supervised) vision backbones.

Below we show an example for training a 3 layer `B-cos` MLP on COCO. Similar process is followed for `VOC`. See `how-to-probe/probing/multi_label_classification/[coco/voc]`.

**B-cos ResNet50 Training on 4 GPUS, 1 Node**                                                 
                                                                                              
```                                                                                           
cd /path/to/how-to-probe/pretraining/mmpretrain                                                
                                                                                              
# run your script here                                                                        
export PYTHONPATH="/path/to/how-to-probe/pretraining/mmpretrain/:$PYTHONPATH"                  
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ../../probing/multi_label_classification/bcosresnet50frozen-multilabelbcos-3-linearclshead_4xb16_coco14-448px.py 4 --resume 'auto'
```                                                                                       
                                                                                              
**Conventional ResNet50 Training on 4 GPUs, 1 Node**                                           
                                                                                              
```                                                                                           
cd /path/to/how-to-probe/pretraining/mmpretrain                                                
                                                                                              
# run your script here                                                                        
export PYTHONPATH="/path/to/how-to-probe/pretraining/mmpretrain/:$PYTHONPATH"                  
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ../../probing/multi_label_classification/resnet50frozen-multilabelbcos-3-linearclshead_4xb16_coco14-448px.py 4 --resume 'auto'
```

`Note: Remember to please update the data_root path and checkpoint='/path/to/pretrained-checkpoint.pth' in the respective configs, which point to the dataset and pre-trained checkpoint you wish to perform linear probing on.`
