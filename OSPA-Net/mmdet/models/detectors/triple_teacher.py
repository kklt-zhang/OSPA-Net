import math
import copy
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmengine import MessageHub
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.models.detectors.semi_base import SemiBaseDetector
from mmdet.structures.bbox import bbox_project, bbox_overlaps
from mmdet.utils import OptConfigType
from mmcv.ops import nms

@MODELS.register_module()
class TripleTeacher(SemiBaseDetector):

    def __init__(self,
                 detector1: dict,
                 detector2: dict,
                 physics_cfg: dict = None,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 **kwargs):

        super().__init__(
            detector=detector1,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            **kwargs
        )

        self.student2 = MODELS.build(detector2)
        self.teacher2 = copy.deepcopy(self.student2)

        self.freeze(self.teacher2)

        self.physics_cfg = physics_cfg or dict(
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
            final_pseudo_score_thr=0.70,

            keep_sar_box_on_agreement=True,
            enable_optical_only=True,
            optical_only_min_scale=2,
            optical_only_start_iter=20000,
            enable_mining=False,

            mining_start_iter=60000,
            tau_high_init=0.72,
            tau_high_end=0.82,

            max_rpn_mining=300,
            max_mined_boxes=0,
            mining_iou_thr=0.30,
            mining_nms_thr=0.30,
        )

    def train(self, mode: bool = True):

        super().train(mode)
        self.teacher.eval()
        self.teacher2.eval()
        return self

    @torch.no_grad()
    def momentum_update(self, momentum: float) -> None:

        for src, dst in zip(self.student.parameters(), self.teacher.parameters()):
            dst.data.mul_(momentum).add_(src.data, alpha=1 - momentum)

        for src, dst in zip(self.student2.parameters(), self.teacher2.parameters()):
            dst.data.mul_(momentum).add_(src.data, alpha=1 - momentum)

        for src, dst in zip(self.student.buffers(), self.teacher.buffers()):
            dst.data.copy_(src.data)

        for src, dst in zip(self.student2.buffers(), self.teacher2.buffers()):
            dst.data.copy_(src.data)

    @torch.no_grad()
    def _to_gray_norm(self, img: Tensor) -> Tensor:

        if img.size(0) == 1:
            gray = img
        else:
            gray = img.mean(dim=0, keepdim=True)

        min_v = gray.amin()
        max_v = gray.amax()

        eps = self.physics_cfg.get('eps', 1e-6)
        gray = (gray - min_v) / (max_v - min_v + eps)

        return gray.clamp(0, 1)

    @torch.no_grad()
    def _sobel_edge(self, gray: Tensor) -> Tensor:

        device = gray.device
        dtype = gray.dtype

        kx = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            device=device,
            dtype=dtype
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            device=device,
            dtype=dtype
        ).view(1, 1, 3, 3)

        x = gray.unsqueeze(0)

        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)

        edge = torch.sqrt(gx.square() + gy.square()).squeeze(0)
        edge = edge / edge.amax().clamp_min(1e-6)

        return edge.clamp(0, 1)

    @torch.no_grad()
    def _crop_region(self,
                     img: Tensor,
                     x1: int,
                     y1: int,
                     x2: int,
                     y2: int) -> Tensor:

        H, W = img.shape[-2:]

        x1 = max(0, min(int(x1), W - 1))
        y1 = max(0, min(int(y1), H - 1))
        x2 = max(x1 + 1, min(int(x2), W))
        y2 = max(y1 + 1, min(int(y2), H))

        return img[:, y1:y2, x1:x2]

    @torch.no_grad()
    def compute_box_physical_scores(self,
                                    img: Tensor,
                                    boxes: Tensor,
                                    gray: Tensor = None,
                                    edge: Tensor = None) -> Tuple[Tensor, dict]:

        device = img.device

        if boxes.numel() == 0:
            empty = boxes.new_zeros((0,))
            return empty, dict(
                C_local=empty,
                S_structure=empty,
                B_complexity=empty
            )

        if gray is None:
            gray = self._to_gray_norm(img)
        if edge is None:
            edge = self._sobel_edge(gray)

        H, W = gray.shape[-2:]

        eps = self.physics_cfg.get('eps', 1e-6)
        expand = self.physics_cfg.get('bg_expand', 1.8)

        c_list = []
        s_list = []
        b_list = []

        boxes_cpu = boxes.detach().float().cpu().tolist()

        for x1, y1, x2, y2 in boxes_cpu:

            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)

            xi1 = int(round(x1))
            yi1 = int(round(y1))
            xi2 = int(round(x2))
            yi2 = int(round(y2))

            inside = self._crop_region(gray, xi1, yi1, xi2, yi2)
            inside_edge = self._crop_region(edge, xi1, yi1, xi2, yi2)

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            ew = bw * expand
            eh = bh * expand

            ex1 = int(round(cx - ew / 2))
            ey1 = int(round(cy - eh / 2))
            ex2 = int(round(cx + ew / 2))
            ey2 = int(round(cy + eh / 2))

            ox1 = max(0, min(ex1, W - 1))
            oy1 = max(0, min(ey1, H - 1))
            ox2 = max(ox1 + 1, min(ex2, W))
            oy2 = max(oy1 + 1, min(ey2, H))

            outer = gray[:, oy1:oy2, ox1:ox2]
            outer_edge = edge[:, oy1:oy2, ox1:ox2]

            oh, ow = outer.shape[-2:]

            mask = torch.ones((oh, ow), device=device, dtype=torch.bool)

            ix1 = max(0, xi1 - ox1)
            iy1 = max(0, yi1 - oy1)
            ix2 = min(ow, xi2 - ox1)
            iy2 = min(oh, yi2 - oy1)

            if ix2 > ix1 and iy2 > iy1:
                mask[iy1:iy2, ix1:ix2] = False

            bg_pixels = outer[0][mask]
            bg_edge_pixels = outer_edge[0][mask]

            if bg_pixels.numel() < 4:
                bg_pixels = outer.flatten()
                bg_edge_pixels = outer_edge.flatten()

            mu_in = inside.mean()
            max_in = inside.max()

            mu_bg = bg_pixels.mean()

            std_bg = bg_pixels.std(unbiased=False).clamp(min=eps)

            c_mean = (mu_in - mu_bg) / (std_bg + eps)
            c_peak = (max_in - mu_bg) / (std_bg + eps)

            c_raw = 0.6 * c_mean + 0.4 * c_peak
            c_score = torch.sigmoid(c_raw / 2.0)

            edge_in = inside_edge.mean()
            edge_bg = bg_edge_pixels.mean() if bg_edge_pixels.numel() > 0 else outer_edge.mean()
            edge_score = torch.sigmoid(5.0 * (edge_in - edge_bg))

            bright_thr = (mu_bg + 1.5 * std_bg).clamp(0, 1)
            bright_density = (inside > bright_thr).float().mean()

            aspect = max(bw / bh, bh / bw)
            aspect_tensor = torch.tensor(aspect, device=device, dtype=gray.dtype)
            shape_score = torch.clamp((aspect_tensor - 1.0) / 4.0, 0, 1)

            s_score = (
                0.45 * edge_score +
                0.35 * bright_density +
                0.20 * shape_score
            )

            bg_std_score = bg_pixels.std(unbiased=False).clamp(0, 1)
            bg_edge_score = bg_edge_pixels.mean().clamp(0, 1)
            bg_bright_density = (bg_pixels > bright_thr).float().mean()

            b_score = (
                0.50 * bg_std_score +
                0.30 * bg_edge_score +
                0.20 * bg_bright_density
            ).clamp(0, 1)

            c_list.append(c_score)
            s_list.append(s_score)
            b_list.append(b_score)

        C_local = torch.stack(c_list)
        S_structure = torch.stack(s_list)
        B_complexity = torch.stack(b_list)

        alpha = self.physics_cfg.get('alpha', 1.2)
        beta = self.physics_cfg.get('beta', 0.8)
        gamma = self.physics_cfg.get('gamma', 0.8)

        c_evidence = 2.0 * C_local - 1.0
        s_evidence = 2.0 * S_structure - 1.0

        phys_scores = torch.sigmoid(
            alpha * c_evidence +
            beta * s_evidence -
            gamma * B_complexity
        )

        return phys_scores.clamp(0, 1), dict(
            C_local=C_local,
            S_structure=S_structure,
            B_complexity=B_complexity
        )

    @torch.no_grad()
    def _box_scale_ids(self,
                       boxes: Tensor,
                       scale_factor=None) -> Tensor:

        if boxes.numel() == 0:
            return torch.empty(
                (0,), dtype=torch.long, device=boxes.device)

        wh = (boxes[:, 2:4] - boxes[:, 0:2]).clamp(min=0)
        areas = wh[:, 0] * wh[:, 1]

        if scale_factor is not None:
            sf = torch.as_tensor(
                scale_factor, dtype=areas.dtype, device=areas.device).flatten()
            if sf.numel() >= 2:
                area_scale = (sf[0] * sf[1]).clamp(min=1e-6)
                areas = areas / area_scale
            elif sf.numel() == 1:
                areas = areas / sf[0].square().clamp(min=1e-6)

        small_area, large_area = self.physics_cfg.get(
            'scale_area_thr', (32.0 ** 2, 96.0 ** 2))

        scale_ids = torch.zeros_like(areas, dtype=torch.long)
        scale_ids[areas >= float(small_area)] = 1
        scale_ids[areas >= float(large_area)] = 2
        return scale_ids

    def _scale_cfg_value(self,
                         key: str,
                         scale_id: int,
                         fallback: float) -> float:

        values = self.physics_cfg.get(key, None)
        if values is None:
            return float(fallback)
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(
                f'{key} must contain exactly three values: '
                '(small, medium, large).')
        return float(values[scale_id])

    @torch.no_grad()
    def _scale_aware_nms(self,
                         boxes: Tensor,
                         scores: Tensor,
                         scale_ids: Tensor,
                         fallback_thr: float) -> Tensor:

        if boxes.numel() == 0:
            return torch.empty(
                (0,), dtype=torch.long, device=boxes.device)

        thresholds = self.physics_cfg.get(
            'fusion_nms_thr_by_scale',
            (fallback_thr, fallback_thr, fallback_thr))
        if not isinstance(thresholds, (list, tuple)) or len(thresholds) != 3:
            raise ValueError(
                'fusion_nms_thr_by_scale must contain exactly three values.')

        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            current = order[0]
            keep.append(current)
            if order.numel() == 1:
                break

            remaining = order[1:]
            overlaps = bbox_overlaps(
                boxes[current].unsqueeze(0), boxes[remaining])[0]
            scale_id = int(scale_ids[current].item())
            order = remaining[overlaps <= float(thresholds[scale_id])]

        return torch.stack(keep)

    @torch.no_grad()
    def source_aware_fusion(self,
                            opt_boxes: Tensor,
                            opt_scores: Tensor,
                            opt_labels: Tensor,
                            sar_boxes: Tensor,
                            sar_scores: Tensor,
                            sar_labels: Tensor,
                            img: Tensor,
                            scale_factor=None,
                            gray: Tensor = None,
                            edge: Tensor = None):

        device = img.device

        def empty_result():
            dtype = sar_boxes.dtype if sar_boxes.numel() > 0 else opt_boxes.dtype
            return (
                torch.empty((0, 4), dtype=dtype, device=device),
                torch.empty((0,), dtype=dtype, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=dtype, device=device),
            )

        if opt_boxes.numel() == 0 and sar_boxes.numel() == 0:
            return empty_result()

        agree_iou_thr = self.physics_cfg.get('agree_iou_thr', 0.60)
        nms_thr = self.physics_cfg.get('fusion_nms_thr', 0.30)

        sar_score_thr = self.physics_cfg.get('sar_score_thr', 0.90)
        sar_agree_score_thr = self.physics_cfg.get(
            'sar_agree_score_thr', 0.70)
        opt_agree_score_thr = self.physics_cfg.get(
            'opt_agree_score_thr', 0.70)
        opt_score_thr = self.physics_cfg.get('opt_only_score_thr', 0.95)
        final_score_thr = self.physics_cfg.get(
            'final_pseudo_score_thr', 0.70)

        tau_phys_sar = self.physics_cfg.get('tau_phys_sar', 0.40)
        tau_phys_opt = self.physics_cfg.get('tau_phys_opt', 0.55)
        tau_phys_agree = self.physics_cfg.get('tau_phys_agree', 0.35)

        w_sar = self.physics_cfg.get('source_weight_sar', 1.00)
        w_opt = self.physics_cfg.get('source_weight_opt', 0.60)
        w_agree = self.physics_cfg.get('source_weight_agree', 1.15)

        keep_sar_box = self.physics_cfg.get(
            'keep_sar_box_on_agreement', True)
        enable_optical_only = self.physics_cfg.get(
            'enable_optical_only', False)
        optical_only_min_scale = int(self.physics_cfg.get(
            'optical_only_min_scale', 2))
        optical_only_start_iter = int(self.physics_cfg.get(
            'optical_only_start_iter', 20000))

        message_hub = MessageHub.get_current_instance()
        current_iter = message_hub.get_info('iter') if message_hub else 0
        current_iter = int(current_iter or 0)
        enable_optical_only = (
            enable_optical_only and current_iter >= optical_only_start_iter)

        sar_scale_ids = self._box_scale_ids(sar_boxes, scale_factor)
        opt_scale_ids = self._box_scale_ids(opt_boxes, scale_factor)

        fused_boxes = []
        fused_scores = []
        fused_labels = []
        fused_sources = []

        matched_opt = torch.zeros(
            len(opt_boxes), dtype=torch.bool, device=device)

        if sar_boxes.numel() > 0:
            sar_phys, _ = self.compute_box_physical_scores(
                img, sar_boxes, gray=gray, edge=edge)

            if opt_boxes.numel() > 0:
                ious = bbox_overlaps(sar_boxes, opt_boxes)
            else:
                ious = None

            sar_order = sar_scores.argsort(descending=True).detach().cpu().tolist()
            for idx in sar_order:
                s_box = sar_boxes[idx]
                s_score = sar_scores[idx]
                s_label = sar_labels[idx]
                s_phys = sar_phys[idx]
                s_scale = int(sar_scale_ids[idx].item())

                current_sar_score_thr = self._scale_cfg_value(
                    'sar_score_thr_by_scale', s_scale, sar_score_thr)
                current_sar_agree_thr = self._scale_cfg_value(
                    'sar_agree_score_thr_by_scale',
                    s_scale,
                    sar_agree_score_thr)
                current_opt_agree_thr = self._scale_cfg_value(
                    'opt_agree_score_thr_by_scale',
                    s_scale,
                    opt_agree_score_thr)
                current_tau_phys_sar = self._scale_cfg_value(
                    'tau_phys_sar_by_scale', s_scale, tau_phys_sar)
                current_tau_phys_agree = self._scale_cfg_value(
                    'tau_phys_agree_by_scale', s_scale, tau_phys_agree)

                is_agreed = False
                o_idx = None
                if ious is not None:
                    valid_opt = (~matched_opt) & (opt_labels == s_label)
                    if valid_opt.any():
                        row = ious[idx].masked_fill(~valid_opt, -1)
                        best_iou, best_idx = row.max(dim=0)
                        candidate_score = opt_scores[best_idx]
                        is_agreed = bool(
                            (best_iou >= agree_iou_thr) &
                            (s_score >= current_sar_agree_thr) &
                            (candidate_score >= current_opt_agree_thr) &
                            (s_phys >= current_tau_phys_agree)
                        )
                        if is_agreed:
                            o_idx = best_idx

                if is_agreed:
                    o_box = opt_boxes[o_idx]
                    o_score = opt_scores[o_idx]

                    if keep_sar_box:
                        new_box = s_box
                    else:
                        score_sum = s_score + o_score + 1e-6
                        new_box = (
                            s_box * s_score + o_box * o_score) / score_sum

                    new_score = torch.sqrt(
                        (s_score * o_score).clamp(min=1e-6))
                    new_score = (new_score * w_agree).clamp(max=1.0)

                    fused_boxes.append(new_box)
                    fused_scores.append(new_score)
                    fused_labels.append(s_label)
                    fused_sources.append(
                        torch.tensor(2, device=device, dtype=torch.long))
                    matched_opt[o_idx] = True
                else:
                    new_score = (s_score * w_sar).clamp(max=1.0)
                    if (s_score >= current_sar_score_thr and
                            s_phys >= current_tau_phys_sar):
                        fused_boxes.append(s_box)
                        fused_scores.append(new_score)
                        fused_labels.append(s_label)
                        fused_sources.append(
                            torch.tensor(0, device=device, dtype=torch.long))

        if enable_optical_only and opt_boxes.numel() > 0:
            opt_phys, _ = self.compute_box_physical_scores(
                img, opt_boxes, gray=gray, edge=edge)
            for idx in range(len(opt_boxes)):
                if matched_opt[idx]:
                    continue

                o_box = opt_boxes[idx]
                o_score = opt_scores[idx]
                o_label = opt_labels[idx]
                o_phys = opt_phys[idx]
                o_scale = int(opt_scale_ids[idx].item())

                if o_scale < optical_only_min_scale:
                    continue

                current_opt_score_thr = self._scale_cfg_value(
                    'opt_only_score_thr_by_scale',
                    o_scale,
                    opt_score_thr)
                current_tau_phys_opt = self._scale_cfg_value(
                    'tau_phys_opt_by_scale', o_scale, tau_phys_opt)

                new_score = (o_score * w_opt).clamp(max=1.0)

                if (o_score >= current_opt_score_thr and
                        o_phys >= current_tau_phys_opt):
                    fused_boxes.append(o_box)
                    fused_scores.append(new_score)
                    fused_labels.append(o_label)
                    fused_sources.append(
                        torch.tensor(1, device=device, dtype=torch.long))

        if len(fused_boxes) == 0:
            return empty_result()

        fused_boxes = torch.stack(fused_boxes, dim=0)
        fused_scores = torch.stack(fused_scores, dim=0)
        fused_labels = torch.stack(fused_labels, dim=0)
        fused_sources = torch.stack(fused_sources, dim=0)

        fused_scale_ids = self._box_scale_ids(fused_boxes, scale_factor)
        keep = self._scale_aware_nms(
            fused_boxes, fused_scores, fused_scale_ids, nms_thr)

        fused_boxes = fused_boxes[keep]
        fused_scores = fused_scores[keep]
        fused_labels = fused_labels[keep]
        fused_sources = fused_sources[keep]

        score_keep = (
            (fused_sources == 1) | (fused_scores >= final_score_thr))
        fused_boxes = fused_boxes[score_keep]
        fused_scores = fused_scores[score_keep]
        fused_labels = fused_labels[score_keep]
        fused_sources = fused_sources[score_keep]

        if fused_boxes.numel() == 0:
            return empty_result()

        fused_phys, _ = self.compute_box_physical_scores(
            img, fused_boxes, gray=gray, edge=edge)

        return fused_boxes, fused_scores, fused_labels, fused_sources, fused_phys

    @torch.no_grad()
    def _update_dynamic_thresholds(self):

        message_hub = MessageHub.get_current_instance()
        current_iter = message_hub.get_info('iter') if message_hub else 0

        max_iters = float(self.physics_cfg.get('max_iters', 180000))

        progress = min(float(current_iter) / max_iters, 1.0)
        alpha = 0.5 * (1 - math.cos(math.pi * progress))

        high_init = self.physics_cfg.get('tau_high_init', 0.72)
        high_end = self.physics_cfg.get('tau_high_end', 0.82)

        tau_high = high_init + (high_end - high_init) * alpha

        return tau_high

    @torch.no_grad()
    def get_pseudo_instances(self,
                             batch_inputs: Tensor,
                             batch_data_samples: list) -> Tuple[list, list]:

        self.teacher.eval()
        self.teacher2.eval()

        tau_high = self._update_dynamic_thresholds()

        x1 = self.teacher.extract_feat(batch_inputs)
        x2 = self.teacher2.extract_feat(batch_inputs)

        rpn_results1 = self.teacher.rpn_head.predict(
            x1, batch_data_samples, rescale=False)
        sem_results1 = self.teacher.roi_head.predict(
            x1, rpn_results1, batch_data_samples, rescale=False)

        rpn_results2 = self.teacher2.rpn_head.predict(
            x2, batch_data_samples, rescale=False)
        sem_results2 = self.teacher2.roi_head.predict(
            x2, rpn_results2, batch_data_samples, rescale=False)

        final_s1 = []
        final_s2 = []

        for i in range(len(batch_data_samples)):
            device = batch_inputs.device

            gray = self._to_gray_norm(batch_inputs[i])
            edge = self._sobel_edge(gray)

            opt_boxes = sem_results1[i].bboxes
            opt_scores = sem_results1[i].scores
            opt_labels = sem_results1[i].labels

            sar_boxes = sem_results2[i].bboxes
            sar_scores = sem_results2[i].scores
            sar_labels = sem_results2[i].labels

            fused_boxes, fused_scores, fused_labels, fused_sources, fused_phys = \
                self.source_aware_fusion(
                    opt_boxes=opt_boxes,
                    opt_scores=opt_scores,
                    opt_labels=opt_labels,
                    sar_boxes=sar_boxes,
                    sar_scores=sar_scores,
                    sar_labels=sar_labels,
                    img=batch_inputs[i],
                    scale_factor=batch_data_samples[i].metainfo.get(
                        'scale_factor', None),
                    gray=gray,
                    edge=edge,
                )

            base_bboxes = fused_boxes.clone()
            base_labels = fused_labels.clone()
            base_scores = fused_scores.clone()
            base_phys = fused_phys.clone()
            base_source = fused_sources.clone()

            if base_bboxes.numel() > 0:
                base_is_mining = torch.zeros(
                    len(base_bboxes), dtype=torch.bool, device=device)
            else:
                base_is_mining = torch.zeros(0, dtype=torch.bool, device=device)
                base_source = torch.zeros(0, dtype=torch.long, device=device)

            inst_s1 = InstanceData()
            inst_s1.bboxes = base_bboxes.clone()
            inst_s1.labels = base_labels.clone()
            inst_s1.scores = base_scores.clone()
            inst_s1.phys_scores = base_phys.clone()
            inst_s1.is_mining = base_is_mining.clone()
            inst_s1.source = base_source.clone()
            final_s1.append(inst_s1)

            final_bboxes = base_bboxes.clone()
            final_labels = base_labels.clone()
            final_scores = base_scores.clone()
            final_phys = base_phys.clone()
            final_is_mining = base_is_mining.clone()
            final_source = base_source.clone()

            message_hub = MessageHub.get_current_instance()
            current_iter = message_hub.get_info('iter') if message_hub else 0

            mining_start_iter = self.physics_cfg.get('mining_start_iter', 60000)
            enable_mining = (
                self.physics_cfg.get('enable_mining', False) and
                current_iter >= mining_start_iter
            )

            max_rpn_mining = self.physics_cfg.get('max_rpn_mining', 300)
            max_mined_boxes = self.physics_cfg.get('max_mined_boxes', 0)
            mining_iou_thr = self.physics_cfg.get('mining_iou_thr', 0.30)
            mining_nms_thr = self.physics_cfg.get('mining_nms_thr', 0.30)
            if enable_mining and max_mined_boxes > 0:
                rpn_boxes = rpn_results2[i].bboxes[:max_rpn_mining]

                if rpn_boxes.numel() > 0:
                    rpn_phys, _ = self.compute_box_physical_scores(
                        batch_inputs[i], rpn_boxes, gray=gray, edge=edge)

                    mining_mask = rpn_phys > tau_high
                    mine_inds = mining_mask.nonzero(as_tuple=True)[0]

                    if len(mine_inds) > 0:
                        mined_bboxes = rpn_boxes[mine_inds].clone()
                        mined_phys = rpn_phys[mine_inds].clone()

                        if final_bboxes.numel() > 0 and mined_bboxes.numel() > 0:
                            ious = bbox_overlaps(mined_bboxes, final_bboxes)
                            max_ious, _ = ious.max(dim=1)

                            new_mask = max_ious < mining_iou_thr
                            mined_bboxes = mined_bboxes[new_mask]
                            mined_phys = mined_phys[new_mask]

                        if mined_bboxes.numel() > 0:
                            _, keep_mine = nms(
                                mined_bboxes,
                                mined_phys,
                                iou_threshold=mining_nms_thr
                            )

                            keep_mine = keep_mine[:max_mined_boxes]

                            mined_bboxes = mined_bboxes[keep_mine]
                            mined_phys = mined_phys[keep_mine]

                        if mined_bboxes.numel() > 0:

                            mined_labels = torch.zeros(
                                len(mined_bboxes),
                                dtype=torch.long,
                                device=device
                            )

                            mined_scores = mined_phys.clamp(min=0.50, max=1.0)

                            mined_flags = torch.ones(
                                len(mined_bboxes),
                                dtype=torch.bool,
                                device=device
                            )

                            mined_source = torch.full(
                                (len(mined_bboxes),),
                                3,
                                dtype=torch.long,
                                device=device
                            )

                            final_bboxes = torch.cat(
                                [final_bboxes, mined_bboxes], dim=0)
                            final_labels = torch.cat(
                                [final_labels, mined_labels], dim=0)
                            final_scores = torch.cat(
                                [final_scores, mined_scores], dim=0)
                            final_phys = torch.cat(
                                [final_phys, mined_phys], dim=0)
                            final_is_mining = torch.cat(
                                [final_is_mining, mined_flags], dim=0)
                            final_source = torch.cat(
                                [final_source, mined_source], dim=0)

            inst_s2 = InstanceData()
            inst_s2.bboxes = final_bboxes
            inst_s2.labels = final_labels
            inst_s2.scores = final_scores
            inst_s2.phys_scores = final_phys
            inst_s2.is_mining = final_is_mining
            inst_s2.source = final_source

            final_s2.append(inst_s2)

        s1_samples = []
        s2_samples = []

        for i, data_sample in enumerate(batch_data_samples):
            teacher_matrix = torch.as_tensor(
                data_sample.homography_matrix,
                device=batch_inputs.device,
                dtype=torch.float32
            )

            inverse_matrix = teacher_matrix.inverse()

            inst1 = final_s1[i]
            if inst1.bboxes.numel() > 0:
                inst1.bboxes = bbox_project(
                    inst1.bboxes,
                    inverse_matrix,
                    data_sample.ori_shape
                )

            ds1 = copy.deepcopy(data_sample)
            ds1.gt_instances = inst1
            s1_samples.append(ds1)

            inst2 = final_s2[i]
            if inst2.bboxes.numel() > 0:
                inst2.bboxes = bbox_project(
                    inst2.bboxes,
                    inverse_matrix,
                    data_sample.ori_shape
                )

            ds2 = copy.deepcopy(data_sample)
            ds2.gt_instances = inst2
            s2_samples.append(ds2)

        return s1_samples, s2_samples

    def _split_sup_by_domain(self, sup_inputs: Tensor, sup_samples: list):

        optical_inputs, optical_samples = [], []
        sar_inputs, sar_samples = [], []

        for img, sample in zip(sup_inputs, sup_samples):
            img_path = getattr(sample, 'img_path', '')
            img_path_low = img_path.lower()

            if 'dior' in img_path_low or 'optical' in img_path_low:
                optical_inputs.append(img)
                optical_samples.append(sample)
            else:
                sar_inputs.append(img)
                sar_samples.append(sample)

        if len(optical_inputs) > 0:
            optical_inputs = torch.stack(optical_inputs)
        else:
            optical_inputs = None

        if len(sar_inputs) > 0:
            sar_inputs = torch.stack(sar_inputs)
        else:
            sar_inputs = None

        return optical_inputs, optical_samples, sar_inputs, sar_samples

    @torch.no_grad()
    def _pseudo_reg_reliability(self, pseudo_samples: list) -> Tensor:

        if not self.semi_train_cfg.get(
                'enable_adaptive_unsup_reg', False):
            return next(self.parameters()).new_tensor(1.0)

        source_weights = self.semi_train_cfg.get(
            'unsup_reg_source_weights', (0.65, 0.25, 1.00, 0.00))
        if (not isinstance(source_weights, (list, tuple)) or
                len(source_weights) != 4):
            raise ValueError(
                'unsup_reg_source_weights must contain four values for '
                '(SAR-only, optical-only, agreed, mined).')

        semantic_floor = float(self.semi_train_cfg.get(
            'unsup_reg_semantic_floor', 0.50))
        physical_floor = float(self.semi_train_cfg.get(
            'unsup_reg_physical_floor', 0.75))
        min_reliability = float(self.semi_train_cfg.get(
            'unsup_reg_min_reliability', 0.35))
        max_reliability = float(self.semi_train_cfg.get(
            'unsup_reg_max_reliability', 1.00))

        quality_list = []
        for data_sample in pseudo_samples:
            instances = data_sample.gt_instances
            if len(instances) == 0:
                continue

            device = instances.bboxes.device
            dtype = instances.bboxes.dtype
            num_boxes = len(instances)

            if hasattr(instances, 'source'):
                source = instances.source.long().clamp(min=0, max=3)
            else:

                source = torch.zeros(
                    num_boxes, dtype=torch.long, device=device)

            source_table = torch.as_tensor(
                source_weights, dtype=dtype, device=device)
            source_quality = source_table[source]

            if hasattr(instances, 'scores'):
                semantic = instances.scores.detach().to(dtype).clamp(0, 1)
            else:
                semantic = torch.ones(
                    num_boxes, dtype=dtype, device=device)

            if hasattr(instances, 'phys_scores'):
                physical = instances.phys_scores.detach().to(dtype).clamp(0, 1)
            else:
                physical = torch.ones(
                    num_boxes, dtype=dtype, device=device)

            semantic_quality = (
                semantic_floor + (1.0 - semantic_floor) * semantic)
            physical_quality = (
                physical_floor + (1.0 - physical_floor) * physical)

            quality_list.append(
                source_quality * semantic_quality * physical_quality)

        if not quality_list:
            return next(self.parameters()).new_tensor(1.0)

        reliability = torch.cat(quality_list).mean().detach()
        return reliability.clamp(
            min=min_reliability, max=max_reliability)

    def _get_unsup_component_scale(self,
                                   loss_name: str,
                                   base_scale: float,
                                   reg_reliability: Tensor):

        loss_name = loss_name.lower()

        if 'acc' in loss_name:
            return 1.0

        if 'loss_rpn_bbox' in loss_name:
            rpn_power = float(self.semi_train_cfg.get(
                'unsup_rpn_reg_reliability_power', 0.5))
            return base_scale * reg_reliability.pow(rpn_power)

        if loss_name == 'loss_bbox' or loss_name.endswith('_loss_bbox'):
            roi_power = float(self.semi_train_cfg.get(
                'unsup_roi_reg_reliability_power', 1.0))
            return base_scale * reg_reliability.pow(roi_power)

        return base_scale

    def loss(self,
             multi_batch_inputs: dict,
             multi_batch_data_samples: dict) -> dict:

        unsup_teacher_inputs = multi_batch_inputs['unsup_teacher']
        unsup_teacher_samples = multi_batch_data_samples['unsup_teacher']

        unsup_student_inputs = multi_batch_inputs['unsup_student']
        unsup_student_samples = multi_batch_data_samples['unsup_student']

        pseudo_s1, pseudo_s2 = self.get_pseudo_instances(
            unsup_teacher_inputs,
            unsup_teacher_samples
        )

        for ps1, ps2, student_sample in zip(
                pseudo_s1, pseudo_s2, unsup_student_samples):

            student_matrix = torch.as_tensor(
                student_sample.homography_matrix,
                device=unsup_student_inputs.device,
                dtype=torch.float32
            )

            if ps1.gt_instances.bboxes.numel() > 0:
                ps1.gt_instances.bboxes = bbox_project(
                    ps1.gt_instances.bboxes,
                    student_matrix,
                    student_sample.img_shape
                )

            if ps2.gt_instances.bboxes.numel() > 0:
                ps2.gt_instances.bboxes = bbox_project(
                    ps2.gt_instances.bboxes,
                    student_matrix,
                    student_sample.img_shape
                )

            meta_updates = dict(img_shape=student_sample.img_shape)

            for key in ['ori_shape', 'scale_factor', 'flip', 'flip_direction']:
                if key in student_sample.metainfo:
                    meta_updates[key] = student_sample.metainfo[key]

            ps1.set_metainfo(meta_updates)
            ps2.set_metainfo(meta_updates)

            min_w, min_h = self.semi_train_cfg.get(
                'min_pseudo_bbox_wh', (1.0, 1.0))
            for pseudo_sample in (ps1, ps2):
                instances = pseudo_sample.gt_instances
                if instances.bboxes.numel() == 0:
                    continue
                boxes = instances.bboxes
                wh = boxes[:, 2:4] - boxes[:, 0:2]
                keep = (
                    torch.isfinite(boxes).all(dim=1) &
                    (wh[:, 0] >= min_w) &
                    (wh[:, 1] >= min_h)
                )
                pseudo_sample.gt_instances = instances[keep]

        sup_inputs = multi_batch_inputs['sup']
        sup_samples = multi_batch_data_samples['sup']

        optical_inputs, optical_samples, sar_inputs, sar_samples = \
            self._split_sup_by_domain(sup_inputs, sup_samples)

        total_loss = {}

        opt_sup_weight = self.semi_train_cfg.get('opt_sup_weight', 0.3)
        opt_unsup_weight = self.semi_train_cfg.get('opt_unsup_weight', 0.25)

        sar_sup_weight = self.semi_train_cfg.get('sar_sup_weight', 1.0)
        sar_unsup_weight = self.semi_train_cfg.get('sar_unsup_weight', 1.0)
        sup_weight = self.semi_train_cfg.get('sup_weight', 1.0)
        unsup_weight = self.semi_train_cfg.get('unsup_weight', 0.5)

        loss_s1 = {}

        if optical_inputs is not None and len(optical_samples) > 0:
            loss_sup1 = self.student.loss(optical_inputs, optical_samples)
            for k, v in loss_sup1.items():
                if isinstance(v, (list, tuple)):
                    loss_s1[f'student1_sup_{k}'] = [
                        loss * sup_weight * opt_sup_weight for loss in v]
                else:
                    loss_s1[f'student1_sup_{k}'] = \
                        v * sup_weight * opt_sup_weight

        loss_unsup1 = self.student.loss(unsup_student_inputs, pseudo_s1)
        pseudo_count1 = sum(len(s.gt_instances) for s in pseudo_s1)
        unsup_scale1 = (
            unsup_weight * opt_unsup_weight if pseudo_count1 > 0 else 0.0)
        reg_reliability1 = self._pseudo_reg_reliability(pseudo_s1)
        for k, v in loss_unsup1.items():
            component_scale = self._get_unsup_component_scale(
                k, unsup_scale1, reg_reliability1)
            if isinstance(v, (list, tuple)):
                loss_s1[f'student1_unsup_{k}'] = [
                    loss * component_scale for loss in v]
            else:
                loss_s1[f'student1_unsup_{k}'] = v * component_scale
        loss_s1['student1_unsup_reg_reliability'] = reg_reliability1

        loss_s2 = {}

        if sar_inputs is not None and len(sar_samples) > 0:
            loss_sup2 = self.student2.loss(sar_inputs, sar_samples)
            for k, v in loss_sup2.items():
                if isinstance(v, (list, tuple)):
                    loss_s2[f'student2_sup_{k}'] = [
                        loss * sup_weight * sar_sup_weight for loss in v]
                else:
                    loss_s2[f'student2_sup_{k}'] = \
                        v * sup_weight * sar_sup_weight

        loss_unsup2 = self.student2.loss(unsup_student_inputs, pseudo_s2)
        pseudo_count2 = sum(len(s.gt_instances) for s in pseudo_s2)
        unsup_scale2 = (
            unsup_weight * sar_unsup_weight if pseudo_count2 > 0 else 0.0)
        reg_reliability2 = self._pseudo_reg_reliability(pseudo_s2)
        for k, v in loss_unsup2.items():
            component_scale = self._get_unsup_component_scale(
                k, unsup_scale2, reg_reliability2)
            if isinstance(v, (list, tuple)):
                loss_s2[f'student2_unsup_{k}'] = [
                    loss * component_scale for loss in v]
            else:
                loss_s2[f'student2_unsup_{k}'] = v * component_scale
        loss_s2['student2_unsup_reg_reliability'] = reg_reliability2

        total_loss.update(loss_s1)
        total_loss.update(loss_s2)

        return total_loss

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: list,
                rescale: bool = True):

        if self.semi_test_cfg is None:
            predict_on = 'teacher2'
        else:
            predict_on = self.semi_test_cfg.get('predict_on', 'teacher2')

        if predict_on == 'teacher2':
            return self.teacher2.predict(
                batch_inputs, batch_data_samples, rescale=rescale)

        if predict_on == 'student2':
            return self.student2.predict(
                batch_inputs, batch_data_samples, rescale=rescale)

        if predict_on == 'teacher':
            return self.teacher.predict(
                batch_inputs, batch_data_samples, rescale=rescale)

        if predict_on == 'student':
            return self.student.predict(
                batch_inputs, batch_data_samples, rescale=rescale)

        return self.teacher2.predict(
            batch_inputs, batch_data_samples, rescale=rescale)
