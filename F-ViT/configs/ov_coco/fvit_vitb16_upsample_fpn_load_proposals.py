_base_ = './fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_original.py'
model = dict(rpn_head=None,
             roi_head=dict(bbox_head=dict(zero_shot=True)))
image_size = (640, 640)
file_client_args = dict(backend='disk')
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', pad_to_square=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    val=dict(
        proposal_file='data/coco/zero-shot/proposals_val2017.pkl',
        pipeline=test_pipeline),
    test=dict(
        proposal_file='data/coco/zero-shot/proposals_val2017.pkl',
        pipeline=test_pipeline)
)
