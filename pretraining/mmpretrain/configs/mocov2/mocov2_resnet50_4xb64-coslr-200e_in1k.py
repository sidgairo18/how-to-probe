_base_ = './mocov2_resnet50_8xb32-coslr-200e_in1k.py'                                         
                                                                                              
# update data_root location                                                                   
data_root = '/path/to/imagenet/'
dataset_type='ImageNet'                                                                       
                                                                                              
# update dataset in train_dataloader and batch_size=64                                        
train_dataloader = dict(                                                                      
    batch_size=64,                                                                            
    dataset=dict(                                                                             
        type=dataset_type,                                                                    
        data_root=data_root,                                                                  
        data_prefix='train',
        split='train',))
