_base_ = [
    '../../pretraining/mmpretrain/configs/_base_/schedules/imagenet_sgd_steplr_100e.py',
    '../../pretraining/mmpretrain/configs/_base_/default_runtime.py',
]

# dataset settings
dataset_type = 'ImageNet'
data_root = '/path/to/imagenet/'
data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[0.0, 0.0, 0.0],
    std=[255.0, 255.0, 255.0],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/val.txt',
        data_prefix=dict(img_path='val/'),
        split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='bcosresnet50',
        frozen_stages=4,
        zero_init_residual=False,
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint='/path/to/pretrained-checkpoint.pth')
        ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        with_avg_pool=True,
        with_bias=False,
        logit_bias=0.0,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=30., momentum=0.9, weight_decay=0.))

# runtime settings
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))
