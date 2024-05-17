_base_ = [                                                                                
    '../_base_/default_runtime.py'                                                     
]                                                                                         
                                                                                          
# dataset settings                                                                        
dataset_type = 'ImageNet'                                                                 
data_root = '/path/to/imagenet/'
data_preprocessor = dict(                                                                 
    num_classes=1000,                                                                     
    # RGB format normalization parameters                                                 
    mean=[123.675, 116.28, 103.53],                                                       
    std=[58.395, 57.12, 57.375],                                                          
    # convert image from BGR to RGB                                                           
    to_rgb=True,                                                                          
)                                                                                         
                                                                                          
train_pipeline = [                                                                        
    dict(type='LoadImageFromFile'),                                                       
    dict(type='RandomResizedCrop', scale=224),                                            
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),                            
    dict(type='PackInputs'),                                                              
]                                                                                         
                                                                                          
test_pipeline = [                                                                         
    dict(type='LoadImageFromFile'),                                                       
    dict(type='ResizeEdge', scale=256, edge='short'),                                     
    dict(type='CenterCrop', crop_size=224),                                               
    dict(type='PackInputs'),                                                              
]                                                                                         
                                                                                          
train_dataloader = dict(                                                                  
    batch_size=64,                                                                        
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
    batch_size=64,                                                                        
    num_workers=8,                                                                        
    dataset=dict(                                                                         
        type=dataset_type,                                                                
        data_root=data_root,                                                              
        ann_file='meta/val.txt',                                                          
        data_prefix=dict(img_path='val/'),                                                
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
        type='ResNet',                                                                    
        depth=50,                                                                         
        num_stages=4,                                                                     
        out_indices=(3, ),                                                                
        style='pytorch'),                                                                 
    neck=None,                                                                            
    head=dict(                                                                            
        type='LinearClsHead',                                                             
        num_classes=1000,                                                                 
        in_channels=2048,
        with_avg_pool=True,
        logit_bias=0.0,
        loss=dict(type='BcosBinaryCrossEntropyLoss'),
        topk=(1, 5),                                                                      
    ))                                                                                    
                                                                                          
# optimizer                                                                               
optim_wrapper = dict(                                                                     
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))
                                                                                          
# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)
                                                                                          
# train  setting                                                                          
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)                           
val_cfg = dict()                                                                          
test_cfg = dict()                                                                         
                                                                                          
# NOTE: `auto_scale_lr` is for automatically scaling LR,                                  
# based on the actual training batch size.                                                
auto_scale_lr = dict(base_batch_size=256)
