_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/crack.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[2, 2, 2, 2]),

    decode_head=dict(num_classes=1,in_channels=[64, 128, 320, 512],

                    # loss_decode=dict(avg_non_ignore=True)
                    loss_decode=[
                                    dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                                    # dict(type='FocalLoss'),
                                    dict(type='DiceLoss', use_sigmoid=True, loss_weight=1.0, eps=1e-5),
                                ]
                ))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=1.)
        }))

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=15),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=15,
#         end=100,
#         by_epoch=False,
#     )
# ]

train_dataloader = dict(batch_size=2, num_workers=4, dataset=dict(reduce_zero_label=False))
val_dataloader = dict(batch_size=1, num_workers=4, dataset=dict(reduce_zero_label=False))
test_dataloader = val_dataloader
