_base_ = [
    '../_base_/datasets/imagenet_bs32_bcosbyol.py',
    '../_base_/schedules/imagenet_lars_coslr_200e.py',
    '../_base_/default_runtime.py',
]

data_root = '/path/to/imagenet'
train_dataloader = dict(batch_size=64,
        num_workers=8,
        dataset=dict(
            type='ImageNet',
            data_prefix='train/',
            data_root=data_root,
            split='train'))


# model settings
model = dict(
    type='BYOL',
    base_momentum=0.01,
    backbone=dict(
        type='bcosresnet50',
        #norm_cfg=dict(type='SyncBN'), # bcos has default BN without bias and Syncing
        zero_init_residual=False),
    neck=dict(
        type='BcosNonLinearNeck',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='BcosNonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False),
        loss=dict(type='CosineSimilarityLoss')),
)

# optimizer
optimizer = dict(type='LARS', lr=4.8, momentum=0.9, weight_decay=1e-6)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'bn': dict(decay_mult=0, lars_exclude=True),
            'bias': dict(decay_mult=0, lars_exclude=True),
            # bn layer in ResNet block downsample module
            'downsample.1': dict(decay_mult=0, lars_exclude=True),
        }),
)

# runtime settings
default_hooks = dict(checkpoint=dict(max_keep_ckpts=20))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096)
