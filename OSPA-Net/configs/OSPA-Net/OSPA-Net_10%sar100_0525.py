_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/default_runtime.py'
]

import copy

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.triple_teacher',
        'mmdet.engine.hooks.dual_ema_hook',
    ],
    allow_failed_imports=False,
)

dior_root = '/home/zh/mmdetection/DIOR/dior_ship/'
dior_train_ann = 'train.json'
dior_train_img = 'train/images'

sar100_root = '/home/zh/mmdetection/data/sar100k/SARDet_100K/'

sar100_train_img_prefix = 'JPEGImages/train/'
sar100_val_img_prefix = 'JPEGImages/val/'
sar100_test_img_prefix = 'JPEGImages/test/'

sar100_labeled_ann = 'Annotations/instances_train_10percent.json'

sar100_unlabeled_ann = 'Annotations/instances_unlabeled_90percent.json'

sar100_val_ann = 'Annotations/val_ship.json'
sar100_test_ann = 'Annotations/test_ship.json'

detector1 = copy.deepcopy(_base_.model)

detector1.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=False,
    pad_size_divisor=32,
)

detector1.backbone = dict(
    type='ResNet',
    depth=101,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(
        type='BN',
        requires_grad=False,
    ),
    norm_eval=True,
    style='caffe',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='open-mmlab://detectron2/resnet101_caffe',
    ),
)

detector1.roi_head.bbox_head.num_classes = 1

stage1_checkpoint = (
    'work_dirs/stage1_dior_pretrain/iter_4800.pth'
)

detector1.init_cfg = dict(
    type='Pretrained',
    checkpoint=stage1_checkpoint,
)

stage2_checkpoint = (
    'work_dirs/stage2_dior_sar100_pretrain/20260807_174430/best_coco_bbox_mAP_iter_14400.pth'
)

detector2 = copy.deepcopy(detector1)

detector2.init_cfg = dict(
    type='Pretrained',
    checkpoint=stage2_checkpoint,
)

model = dict(
    _delete_=True,
    type='TripleTeacher',

    detector1=detector1,

    detector2=detector2,

    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector1.data_preprocessor
    ),

    semi_train_cfg=dict(
        freeze_teacher=True,

        sup_weight=1.0,
        unsup_weight=0.5,

        sar_sup_weight=1.0,
        sar_unsup_weight=1.0,

        opt_sup_weight=0.3,

        opt_unsup_weight=0.10,

        enable_adaptive_unsup_reg=True,
        unsup_reg_source_weights=(0.80, 0.25, 1.00, 0.00),
        unsup_reg_semantic_floor=0.50,
        unsup_reg_physical_floor=0.75,
        unsup_reg_min_reliability=0.5,
        unsup_reg_max_reliability=1.00,

        unsup_rpn_reg_reliability_power=0.5,
        unsup_roi_reg_reliability_power=0.5,

        min_pseudo_bbox_wh=(1.0, 1.0),
    ),

    semi_test_cfg=dict(

        predict_on='teacher2'
    ),

    physics_cfg=dict(

        alpha=1.2,
        beta=0.8,
        gamma=0.8,

        bg_expand=1.8,
        eps=1e-6,

        max_iters=180000,

        agree_iou_thr=0.60,

        fusion_nms_thr=0.30,
        fusion_nms_thr_by_scale=(0.30, 0.45, 0.55),
        scale_area_thr=(32.0 ** 2, 96.0 ** 2),

        tau_phys_agree=0.35,
        tau_phys_sar=0.40,
        tau_phys_opt=0.55,
        tau_phys_agree_by_scale=(0.35, 0.32, 0.30),
        tau_phys_sar_by_scale=(0.40, 0.38, 0.36),
        tau_phys_opt_by_scale=(1.01, 1.01, 0.48),

        source_weight_agree=1.15,
        source_weight_sar=1.00,
        source_weight_opt=0.60,

        sar_score_thr=0.90,
        sar_agree_score_thr=0.70,
        opt_agree_score_thr=0.70,
        opt_only_score_thr=0.95,
        sar_score_thr_by_scale=(0.90, 0.87, 0.85),
        sar_agree_score_thr_by_scale=(0.70, 0.68, 0.66),
        opt_agree_score_thr_by_scale=(0.70, 0.70, 0.70),
        opt_only_score_thr_by_scale=(1.01, 1.01, 0.95),

        keep_sar_box_on_agreement=True,

        enable_optical_only=True,
        optical_only_min_scale=2,
        optical_only_start_iter=20000,

        final_pseudo_score_thr=0.70,

        enable_mining=False,
        mining_start_iter=60000,

        tau_high_init=0.72,
        tau_high_end=0.82,

        max_rpn_mining=300,
        max_mined_boxes=0,

        mining_iou_thr=0.30,

        mining_nms_thr=0.30,

    )
)

backend_args = None
dataset_type = 'CocoDataset'

metainfo = dict(
    classes=('ship',),
    palette=[(220, 20, 60)]
)

branch_field = ['sup', 'unsup_teacher', 'unsup_student']

sup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),

    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),

    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=[
            dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                )
            )
        ]
    )
]

unsup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),

    dict(
        type='MultiBranch',
        branch_field=branch_field,

        unsup_teacher=[
            dict(type='Resize', scale=(800, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'homography_matrix',
                )
            )
        ],

        unsup_student=[
            dict(type='Resize', scale=(800, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),

            dict(type='RandomErasing', n_patches=(1, 3), ratio=(0, 0.10)),

            dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'homography_matrix',
                )
            )
        ]
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        )
    )
]

labeled_optical_dataset = dict(
    type=dataset_type,
    data_root=dior_root,
    metainfo=metainfo,
    ann_file=dior_train_ann,
    data_prefix=dict(img=dior_train_img),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=sup_pipeline,
    backend_args=backend_args
)

labeled_sar_dataset = dict(
    type=dataset_type,
    data_root=sar100_root,
    metainfo=metainfo,
    ann_file=sar100_labeled_ann,
    data_prefix=dict(img=sar100_train_img_prefix),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=sup_pipeline,
    backend_args=backend_args
)

unlabeled_sar_dataset = dict(
    type=dataset_type,
    data_root=sar100_root,
    metainfo=metainfo,
    ann_file=sar100_unlabeled_ann,
    data_prefix=dict(img=sar100_train_img_prefix),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=backend_args
)

train_dataloader = dict(
    batch_size=3,
    num_workers=2,
    persistent_workers=False,

    sampler=dict(
        type='MultiSourceSampler',
        batch_size=3,
        source_ratio=[1, 1, 1]
    ),

    dataset=dict(
        type='ConcatDataset',
        datasets=[
            labeled_optical_dataset,
            labeled_sar_dataset,
            unlabeled_sar_dataset,
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),

    dataset=dict(
        type=dataset_type,
        data_root=sar100_root,
        metainfo=metainfo,
        ann_file=sar100_val_ann,
        data_prefix=dict(img=sar100_val_img_prefix),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),

    dataset=dict(
        type=dataset_type,
        data_root=sar100_root,
        metainfo=metainfo,
        ann_file=sar100_test_ann,
        data_prefix=dict(img=sar100_test_img_prefix),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=sar100_root + sar100_val_ann,
    metric='bbox',
    backend_args=backend_args
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=sar100_root + sar100_test_ann,
    metric='bbox',
    backend_args=backend_args
)

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=180000,
    val_interval=1000
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=180000,
        by_epoch=False,
        milestones=[120000, 160000],
        gamma=0.1
    )
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        save_last=True,

        save_best='coco/bbox_mAP',
        rule='greater'
    )
)

custom_hooks = [

    dict(type='DualMeanTeacherHook')
]

log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=False
)

default_scope = 'mmdet'

randomness = dict(seed=0, deterministic=False)
