_base_ = [
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
        type='MultiLayerLinearClsHead',
        num_classes=1000,
        hid_channels=2048,
        in_channels=2048,
        num_layers=2,
        with_avg_pool=True,
        with_bias=False,
        is_bcos=True,
        loss=dict(type='UniformOffLabelsBCEWithLogitsLoss'),
        topk=(1, 5),
    ))

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR', T_max=95, by_epoch=True, begin=5, end=100)
]

# train  setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
