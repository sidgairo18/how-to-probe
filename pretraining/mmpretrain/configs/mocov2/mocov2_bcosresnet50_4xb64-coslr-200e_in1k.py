_base_ = [
    '../_base_/datasets/imagenet_bs32_bcosmocov2.py',
    '../_base_/schedules/imagenet_sgd_coslr_200e.py',
    '../_base_/default_runtime.py',
]

# update data_root location
data_root = '/path/to/imagenet/'
dataset_type='ImageNet'

# update dataset in train_dataloader, only diff is that batch_size=64
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='train',
        split='train',))

# model settings
model = dict(
    type='MoCo',
    queue_len=65536,
    feat_dim=128,
    momentum=0.001,
    backbone=dict(
        type='bcosresnet50',
        zero_init_residual=False),
    neck=dict(
        type='MoCoV2BcosNeck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.2))

# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
