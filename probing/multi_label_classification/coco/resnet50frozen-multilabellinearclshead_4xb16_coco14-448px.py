_base_ = ['../../../pretraining/mmpretrain/configs/_base_/datasets/coco_bs16.py',
          '../../../pretraining/mmpretrain/configs/_base_/default_runtime.py']

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(                                                                        
        type='ResNet',                                                                    
        depth=50,                                                                         
        num_stages=4,                                                                     
        out_indices=(3, ),                                                                
        frozen_stages=4,                                                                  
        style='pytorch',                                                                  
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint='/path/to/pretrained-checkpoint.pth')),
    neck=None,
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=80,
        in_channels=2048,
        with_avg_pool=True,
        with_bias=True,
        logit_bias=0.0,
        loss=dict(type='BcosBinaryCrossEntropyLoss')))

# dataset setting
data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],                                                           
    std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=448, crop_ratio_range=(0.7, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=448),
    dict(
        type='PackInputs',
        # `gt_label_difficult` is needed for VOC evaluation
        meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'gt_label_difficult')),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
# the lr of classifier.head is 10 * base_lr, which help convergence.
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=3, momentum=0.9, weight_decay=0.0),
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10)}))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-7,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(type='StepLR', by_epoch=True, step_size=6, gamma=0.1)
]

train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
val_cfg = dict()
test_cfg = dict()
