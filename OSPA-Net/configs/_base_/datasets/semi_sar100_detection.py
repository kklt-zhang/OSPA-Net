# dataset settings
dataset_type = 'CocoDataset'

# 1. 修改为你截图中的根目录绝对路径
data_root = '/home/zh/mmdetection/data/sar100k/SARDet_100K/'

# 2. [关键修改] 既然加了纯负样本，类别必须是两个，否则会报 IndexError
# 注意：如果你的 JSON 里类别名是大写，这里也要改成 ('Ship', 'Harbor')
metainfo = dict(classes=('ship'), palette=[(220, 20, 60)])

backend_args = None

# 定义颜色变换的空间（亮度、对比度、锐度、色彩等）
color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]
# 定义几何变换的空间（旋转、剪切、平移）
geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]

# SSDD/SARDet 图片包含大量小目标，设置为 608~800 左右更高效且合理
scale = [(608, 608), (800, 800)]
branch_field = ['sup', 'unsup_teacher', 'unsup_student']

# pipeline used to augment labeled data
sup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs'))
]

# pipeline used to augment unlabeled data weakly
weak_pipeline = [
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data strongly
strong_pipeline = [
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='RandAugment', aug_space=geometric, aug_num=1),
        ]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(608, 608), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

batch_size = 5
num_workers = 5

# 3. 配置有标签训练集 (Labeled) -> 指向 10% 划分文件和 train 文件夹
labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file='Annotations/instances_train_10percent.json',
    data_prefix=dict(img='JPEGImages/train/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=16),
    pipeline=sup_pipeline,
    backend_args=backend_args)

# 4. 配置无标签训练集 (Unlabeled) -> 指向 90% 划分文件和 train 文件夹
unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file='Annotations/instances_unlabeled_90percent.json',
    data_prefix=dict(img='JPEGImages/train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(
        # type='GroupMultiSourceSampler',
        # batch_size=batch_size,
        # source_ratio=[1, 4]
        type='MultiSourceSampler',   # ← 改成这个 0502
        batch_size=batch_size,
        source_ratio=[1, 4],  # 维持你原来的 1:4 比例
        shuffle=True
        ),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))

# 5. 解绑验证集 (Val) -> 指向 val.json 和 val 文件夹
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        # 备注：如果你想在验证时也只看船和港口，可以用 'Annotations/val_ship_harbor.json'
        ann_file='Annotations/val_ship.json',
        data_prefix=dict(img='JPEGImages/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

# 6. 解绑测试集 (Test) -> 指向 test.json 和 test 文件夹
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        # 备注：同样，如果你提取了测试集的特定类别，也可以换成对应的 json
        ann_file='Annotations/test_ship.json',
        data_prefix=dict(img='JPEGImages/test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

# 7. 分离验证和测试的评估器
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'Annotations/val_ship.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'Annotations/test_ship.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)