_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/default_runtime.py'
]

import copy

# =========================================================
# 0. 自定义模型导入
# =========================================================
# 如果你已经在 mmdet/models/detectors/__init__.py 里注册了 TripleTeacher，
# 这里可以不用打开。
#
# 如果你没有写入 __init__.py，就取消下面注释：
#
custom_imports = dict(
    imports=[
        'mmdet.models.detectors.triple_teacher',
        'mmdet.engine.hooks.dual_ema_hook',
    ],
    allow_failed_imports=False,
)


# =========================================================
# 1. 数据路径设置：DIOR + SAR100
# =========================================================

# -----------------------------
# 1.1 DIOR 光学船舶数据
# -----------------------------
dior_root = '/home/zh/mmdetection/DIOR/dior_ship/'
dior_train_ann = 'train.json'
dior_train_img = 'train/images'

# -----------------------------
# 1.2 SAR100 / SARDet-100K 数据
#
# 你的 SAR100 目录通常是：
# SARDet_100K/
#   ├── Annotations/
#   └── JPEGImages/
#       ├── train/
#       ├── val/
#       └── test/
#
# 所以 train / val / test 的 img_prefix 不能统一写成一个。
# -----------------------------
sar100_root = '/home/zh/mmdetection/data/sar100k/SARDet_100K/'

sar100_train_img_prefix = 'JPEGImages/train/'
sar100_val_img_prefix = 'JPEGImages/val/'
sar100_test_img_prefix = 'JPEGImages/test/'

# 少量有标注 SAR100
sar100_labeled_ann = 'Annotations/instances_train_10percent.json'

# 无标注 SAR100
# 注意：
# 这里可以是只含 images 的 COCO json；
# 也可以是原 train json。
# 因为 unsup_pipeline 使用 LoadEmptyAnnotations，不会读取 bbox。
sar100_unlabeled_ann = 'Annotations/instances_unlabeled_90percent.json'

# 验证 / 测试
sar100_val_ann = 'Annotations/val_ship.json'
sar100_test_ann = 'Annotations/test_ship.json'


# =========================================================
# 2. 模型底座：Faster R-CNN R50-FPN
# =========================================================

# =========================================================
# 2. 模型底座：与更新后的Stage1/Stage2完全一致
# =========================================================

detector1 = copy.deepcopy(_base_.model)

# ---------------------------------------------------------
# 2.1 数据预处理
# 必须与Stage1/Stage2一致：
# Caffe BGR顺序，std=[1, 1, 1]
# ---------------------------------------------------------
detector1.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=False,
    pad_size_divisor=32,
)

# ---------------------------------------------------------
# 2.2 ResNet-101 Caffe
# 必须与Stage1/Stage2一致
# ---------------------------------------------------------
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

# 单类别ship
detector1.roi_head.bbox_head.num_classes = 1

# 注意：
# 不再修改rpn_head.anchor_generator。
# 保留MMDetection标准Faster R-CNN anchors，
# 才能与Stage1/Stage2的RPN权重完全匹配。

# ---------------------------------------------------------
# 2.3 Optical分支：加载Stage1权重
# 将XXXX改成真实best checkpoint迭代数
# ---------------------------------------------------------
stage1_checkpoint = (
    'work_dirs/stage1_dior_pretrain/iter_4800.pth'
)

detector1.init_cfg = dict(
    type='Pretrained',
    checkpoint=stage1_checkpoint,
)

# ---------------------------------------------------------
# 2.4 SAR分支：加载Stage2权重
# 将XXXX改成真实best checkpoint迭代数
# ---------------------------------------------------------
stage2_checkpoint = (
    'work_dirs/stage2_dior_sar100_pretrain/20260807_174430/best_coco_bbox_mAP_iter_14400.pth'
)

detector2 = copy.deepcopy(detector1)

detector2.init_cfg = dict(
    type='Pretrained',
    checkpoint=stage2_checkpoint,
)

# =========================================================
# 3. Triple Teacher 模型配置
# =========================================================

model = dict(
    _delete_=True,
    type='TripleTeacher',

    # teacher/student 1:
    # Optical cross-domain branch
    detector1=detector1,

    # teacher/student 2:
    # SAR semi-supervised branch
    detector2=detector2,

    # 多分支数据预处理器：
    # 同时处理 sup / unsup_teacher / unsup_student 三个分支。
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector1.data_preprocessor
    ),

    semi_train_cfg=dict(
        freeze_teacher=True,

        # -------------------------------------------------
        # 总体监督 / 无监督权重
        # -------------------------------------------------
        sup_weight=1.0,
        unsup_weight=0.5,

        # -------------------------------------------------
        # 改进点 1：
        # 最终推理使用 teacher2，也就是 SAR teacher。
        # 所以 SAR branch 必须作为主分支，不能再压到 0.2。
        # -------------------------------------------------
        sar_sup_weight=1.0,
        sar_unsup_weight=1.0,

        # -------------------------------------------------
        # 改进点 2：
        # Optical branch 只作为辅助语义来源。
        # 光学 teacher 在 SAR 图像上容易产生跨域幻觉，
        # 所以 optical branch 训练损失降权。
        # -------------------------------------------------
        opt_sup_weight=0.3,
        # 降低 optical student 在 SAR 伪标签上的自训练强度，保持跨域多样性。
        opt_unsup_weight=0.10,

        # -------------------------------------------------
        # 四种标注比例统一使用的伪框回归可靠性
        # -------------------------------------------------
        #
        # 分类损失保持原权重。只有无监督 RPN/ROI 框回归根据伪框来源、
        # 语义置信度和物理可信度自适应缩放：
        #   SAR-only / optical-only / agreed / mined
        #
        # 该规则不依赖 1%、2%、5% 或 10%，所有实验完全一致。
        enable_adaptive_unsup_reg=True,
        unsup_reg_source_weights=(0.80, 0.25, 1.00, 0.00),#0.65-0.8
        unsup_reg_semantic_floor=0.50,
        unsup_reg_physical_floor=0.75,
        unsup_reg_min_reliability=0.5,#0.35-0.5
        unsup_reg_max_reliability=1.00,

        # RPN 回归使用 sqrt(reliability)，降权较温和；
        # ROI 回归直接使用 reliability，重点抑制坐标噪声。
        unsup_rpn_reg_reliability_power=0.5,
        unsup_roi_reg_reliability_power=0.5,#1-0.75-0.5

        # bbox 投影到 strong view 后的最小宽高。
        min_pseudo_bbox_wh=(1.0, 1.0),
    ),

    semi_test_cfg=dict(
        # 最终测试默认使用 SAR teacher。
        # 这样训练时虽然有多个 teacher/student，
        # 推理时只保留一个 SAR detector，不增加推理开销。
        predict_on='teacher2'
    ),

    physics_cfg=dict(
        # -------------------------------------------------
        # 物理教师公式：
        #
        # P_phys = sigmoid(alpha*(2C-1) + beta*(2S-1) - gamma*B)
        #
        # C: local contrast
        #    候选框内部均值/峰值相对于周围背景的差异。
        #
        # S: structure response
        #    边缘响应、亮斑密度、长宽比等结构信息。
        #
        # B: background complexity
        #    背景标准差、背景边缘、背景亮斑密度。
        # -------------------------------------------------
        alpha=1.2,
        beta=0.8,
        gamma=0.8,

        # 背景环区域外扩比例。
        # bg_expand=1.8 表示用 1.8 倍候选框大小构造外扩框，
        # 再去掉候选框内部，剩余区域作为背景环。
        bg_expand=1.8,
        eps=1e-6,

        # -------------------------------------------------
        # 改进点 3：
        # physics max_iters 必须和 train_cfg.max_iters 对齐。
        # 初版里 physics max_iters=40000，而训练是 180000，
        # 会导致物理阈值过早完成退火。
        # -------------------------------------------------
        max_iters=180000,

        # -------------------------------------------------
        # Source-aware fusion 设置
        # -------------------------------------------------

        # Optical box 和 SAR box 的 IoU 超过该阈值，
        # 就认为两个 teacher 对同一目标达成一致。
        agree_iou_thr=0.60,

        # source-aware fusion 之后再做一次 NMS，
        # 避免 agreed / SAR-only / optical-only 之间重复。
        fusion_nms_thr=0.30,
        fusion_nms_thr_by_scale=(0.30, 0.45, 0.55),
        scale_area_thr=(32.0 ** 2, 96.0 ** 2),

        # -------------------------------------------------
        # 不同来源候选框的物理准入阈值
        #
        # agreed:
        #   Optical teacher 和 SAR teacher 都检出，可信度最高，
        #   所以物理阈值最宽松。
        #
        # SAR-only:
        #   只有 SAR teacher 检出，作为主来源，基础物理审核。
        #
        # Optical-only:
        #   只有 optical teacher 检出，跨域风险最大，
        #   必须更严格物理审核。
        # -------------------------------------------------
        tau_phys_agree=0.35,
        tau_phys_sar=0.40,
        tau_phys_opt=0.55,
        tau_phys_agree_by_scale=(0.35, 0.32, 0.30),
        tau_phys_sar_by_scale=(0.40, 0.38, 0.36),
        tau_phys_opt_by_scale=(1.01, 1.01, 0.48),

        # -------------------------------------------------
        # 不同来源候选框的分数校准权重
        #
        # agreed:
        #   两个 teacher 都同意，增强权重。
        #
        # SAR-only:
        #   主教师来源，正常权重。
        #
        # Optical-only:
        #   辅助来源，降权，防止跨域假框主导。
        # -------------------------------------------------
        source_weight_agree=1.15,
        source_weight_sar=1.00,
        source_weight_opt=0.60,

        # -------------------------------------------------
        # 不同来源候选框的语义分阈值
        #
        # SAR-only:
        #   SAR teacher 是主教师，阈值稍低。
        #
        # Optical-only:
        #   光学 teacher 在 SAR 上不稳定，必须极高语义分。
        # -------------------------------------------------
        sar_score_thr=0.90,
        sar_agree_score_thr=0.70,
        opt_agree_score_thr=0.70,
        opt_only_score_thr=0.95,
        sar_score_thr_by_scale=(0.90, 0.87, 0.85),
        sar_agree_score_thr_by_scale=(0.70, 0.68, 0.66),
        opt_agree_score_thr_by_scale=(0.70, 0.70, 0.70),
        opt_only_score_thr_by_scale=(1.01, 1.01, 0.95),

        # Optical teacher 只提供一致性证据，默认不修改 SAR 框坐标。
        keep_sar_box_on_agreement=True,

        # 工作簿显示 DIOR 对 APl 稳定有益，但对 APm 并不稳定。
        # 因此只允许训练稳定后的 large optical-only 高置信框。
        enable_optical_only=True,
        optical_only_min_scale=2,
        optical_only_start_iter=20000,

        # 融合/NMS 后的最低伪标签分数。
        final_pseudo_score_thr=0.70,

        # -------------------------------------------------
        # Physical mining 设置
        #
        # 改进点 4：
        # 只有实现分类专用的 mined-box loss 后才应重新开启。
        # -------------------------------------------------
        # 标准 Faster R-CNN 不读取逐 GT reg_weights，因此优化版先关闭
        # mining，避免 mined box 被错误用于 RPN/ROI 回归。
        enable_mining=False,
        mining_start_iter=60000,

        # 重新开启时，物理阈值从 0.72 逐渐升到 0.82。
        # 越往后，teacher 越稳定，挖掘也越保守。
        tau_high_init=0.72,
        tau_high_end=0.82,

        max_rpn_mining=300,
        max_mined_boxes=0,

        # mined boxes 与 clean boxes 的 IoU 低于该阈值才保留，
        # 避免重复挖已经存在的伪标签。
        mining_iou_thr=0.30,

        # mined boxes 内部 NMS 阈值。
        mining_nms_thr=0.30,

    )
)


# =========================================================
# 4. Dataset / Pipeline
# =========================================================

backend_args = None
dataset_type = 'CocoDataset'

metainfo = dict(
    classes=('ship',),
    palette=[(220, 20, 60)]
)

branch_field = ['sup', 'unsup_teacher', 'unsup_student']


# ---------------------------------------------------------
# 4.1 监督分支 pipeline
#
# 用于：
#   labeled optical DIOR
#   labeled SAR100
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 4.2 无监督分支 pipeline
#
# unsup_teacher:
#   weak augmentation。
#   给 teacher 生成伪标签，增强不能太强。
#
# unsup_student:
#   strong augmentation。
#   给 student 学习伪标签。
#
# 注意：
#   SAR 图像灰度具有物理含义，
#   不建议一开始就使用很重的 PhotoMetricDistortion。
#   初版先用 resize + flip，后续可以逐步加入 CutOut / RandomErasing。
# ---------------------------------------------------------
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

            # 温和遮挡增强，不改变 SAR 灰度映射关系。
            dict(type='RandomErasing', n_patches=(1, 3), ratio=(0, 0.10)),

            # 后续可以尝试 SAR 适配强增强，例如：
            # dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
            #
            # 暂时不打开 PhotoMetricDistortion，避免破坏 SAR 灰度物理关系。
            # dict(type='PhotoMetricDistortion'),

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


# ---------------------------------------------------------
# 4.3 测试 pipeline
# ---------------------------------------------------------
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


# =========================================================
# 5. Datasets
# =========================================================

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


# =========================================================
# 6. Dataloader
# =========================================================

train_dataloader = dict(
    batch_size=3,
    num_workers=2,
    persistent_workers=False,

    # 三源数据：
    #   1 optical labeled
    #   1 SAR labeled
    #   1 SAR unlabeled
    #
    # 这里仍然保持 [1,1,1]，因为 loss 里已经对 optical branch 降权。
    # 如果后续发现光学仍然干扰大，可以改成 [1,2,1] 或 [1,3,1]。
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


# =========================================================
# 7. Evaluator
# =========================================================

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


# =========================================================
# 8. Train Schedule
# =========================================================

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=180000,
    val_interval=1000
)

# =========================================================
# 关键修复 1：
# 不再使用 TeacherStudentValLoop。
#
# 原因：
# TeacherStudentValLoop 一般只会生成 teacher/student 前缀的指标，
# 例如 teacher/coco/bbox_mAP、student/coco/bbox_mAP，
# 不一定会生成 teacher2/coco/bbox_mAP。
#
# 现在模型内部已经通过：
# semi_test_cfg=dict(predict_on='teacher2')
# 指定验证/测试时用 teacher2。
#
# 所以这里直接使用普通 ValLoop，
# 这样 CocoMetric 返回的指标名就是 coco/bbox_mAP。
# =========================================================
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


# =========================================================
# 9. Hooks
# =========================================================

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        save_last=True,

        # =================================================
        # 关键修复 2：
        # 原来写的是：
        # save_best='teacher2/coco/bbox_mAP'
        #
        # 但是普通 CocoMetric 实际返回的是：
        # coco/bbox_mAP
        #
        # teacher2 已经由 model.semi_test_cfg.predict_on 控制，
        # metric key 不需要再带 teacher2/ 前缀。
        # =================================================
        save_best='coco/bbox_mAP',
        rule='greater'
    )
)


custom_hooks = [
    # 需要确认 DualMeanTeacherHook 会调用 model.momentum_update()
    # 如果 hook 只更新 self.teacher，而不调用 model.momentum_update()，
    # 那 teacher2 不会 EMA 更新，需要改 hook。
    dict(type='DualMeanTeacherHook')
]


log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=False
)


default_scope = 'mmdet'

# 固定数据采样与初始化随机种子。不同重复实验通过命令行覆盖 seed。
randomness = dict(seed=0, deterministic=False)
