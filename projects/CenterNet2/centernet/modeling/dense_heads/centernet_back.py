import math
import json
import copy
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.layers import ShapeSpec, cat
from detectron2.structures import Instances, Boxes
from detectron2.utils.comm import get_world_size
from ..layers.iou_loss import IOULoss
from ..layers.ml_nms import ml_nms
from ..debug import debug_train, debug_test
from .centernet_utils import reduce_sum, _transpose
from .centernet_head import CenterNetHead

INF = 1e8

@PROPOSAL_GENERATOR_REGISTRY.register()
class CenterNet(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.in_features          = cfg.MODEL.CENTERNET.IN_FEATURES
        self.strides              = cfg.MODEL.CENTERNET.FPN_STRIDES
        self.score_thresh         = cfg.MODEL.CENTERNET.INFERENCE_TH
        self.num_classes          = cfg.MODEL.CENTERNET.NUM_CLASSES
        self.reg_weight           = cfg.MODEL.CENTERNET.REG_WEIGHT
        self.hm_focal_alpha       = cfg.MODEL.CENTERNET.HM_FOCAL_ALPHA
        self.hm_focal_beta        = cfg.MODEL.CENTERNET.HM_FOCAL_BETA
        self.loss_gamma           = cfg.MODEL.CENTERNET.LOSS_GAMMA
        self.ignore_high_fp       = cfg.MODEL.CENTERNET.IGNORE_HIGH_FP
        self.with_agn_hm          = cfg.MODEL.CENTERNET.WITH_AGN_HM
        self.sigmoid_clamp        = cfg.MODEL.CENTERNET.SIGMOID_CLAMP
        self.more_pos             = cfg.MODEL.CENTERNET.MORE_POS
        self.more_pos_thresh      = cfg.MODEL.CENTERNET.MORE_POS_THRESH
        self.only_proposal        = cfg.MODEL.CENTERNET.ONLY_PROPOSAL
        self.as_proposal          = cfg.MODEL.CENTERNET.AS_PROPOSAL
        self.pos_weight           = cfg.MODEL.CENTERNET.POS_WEIGHT
        self.neg_weight           = cfg.MODEL.CENTERNET.NEG_WEIGHT
        self.vis_thresh           = cfg.VIS_THRESH
        self.debug                = cfg.DEBUG
        self.iou_loss = IOULoss(cfg.MODEL.CENTERNET.LOC_LOSS_TYPE)
        self.sizes_of_interest = cfg.MODEL.CENTERNET.SOI
        self.delta = (1 - cfg.MODEL.CENTERNET.HM_MIN_OVERLAP) \
                    / (1 + cfg.MODEL.CENTERNET.HM_MIN_OVERLAP)
        self.min_radius = cfg.MODEL.CENTERNET.MIN_RADIUS
        self.pre_nms_topk_train = cfg.MODEL.CENTERNET.PRE_NMS_TOPK_TRAIN
        self.pre_nms_topk_test = cfg.MODEL.CENTERNET.PRE_NMS_TOPK_TEST
        self.post_nms_topk_train = cfg.MODEL.CENTERNET.POST_NMS_TOPK_TRAIN
        self.post_nms_topk_test = cfg.MODEL.CENTERNET.POST_NMS_TOPK_TEST
        self.nms_thresh_train = cfg.MODEL.CENTERNET.NMS_TH_TRAIN
        self.nms_thresh_test = cfg.MODEL.CENTERNET.NMS_TH_TEST
        assert (not self.only_proposal) or self.with_agn_hm

        input_shape_head = [input_shape[f] for f in self.in_features]
        self.centernet_head = CenterNetHead(cfg, input_shape_head)

        if self.debug:
            pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.denormalizer = lambda x: x * pixel_std + pixel_mean


    def forward(self, images, features, gt_instances):
        features = [features[f] for f in self.in_features]
        clss_per_level, reg_pred_per_level, agn_hm_pred_per_level = \
            self.centernet_head(features)
        grids = self.compute_grids(features)
        shapes_per_level = grids[0].new_tensor(
                    [(x.shape[2], x.shape[3]) for x in reg_pred_per_level])
        
        if not self.training:
            return self.inference(
                images, clss_per_level, reg_pred_per_level, 
                agn_hm_pred_per_level, grids)
        else:
            center_inds, labels, masks, regs, c33_inds, c33_mask, \
                c33_regs, flattened_hms = self._get_ground_truth(
                    grids, shapes_per_level, gt_instances)
            
            clss, reg_pred, agn_hm_pred = self._flatten_outputs(
                clss_per_level, reg_pred_per_level, agn_hm_pred_per_level)
            logits_pred = clss if not self.only_proposal \
                else agn_hm_pred.view(-1, 1)

            losses, assigned_center_inds, c33_inds, c33_regs = self.losses(
                center_inds, labels, masks, regs, c33_inds, c33_mask, c33_regs, 
                flattened_hms, logits_pred, reg_pred)

            proposals = None
            if self.only_proposal or self.as_proposal:
                clss_per_level = [x.sigmoid() for x in \
                    (clss_per_level if self.as_proposal else agn_hm_pred_per_level)]
                agn_hm_pred_per_level = [None for _ in agn_hm_pred_per_level]
                proposals = self.predict_instances(
                    grids, clss_per_level, reg_pred_per_level, 
                    images.image_sizes, agn_hm_pred_per_level)
                for p in range(len(proposals)):
                    proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
                    proposals[p].objectness_logits = proposals[p].get('scores')
                    proposals[p].remove('pred_boxes')
                    proposals[p].remove('scores')
                    proposals[p].remove('pred_classes')

            if self.debug:
                reg_targets = flattened_hms.new_zeros(
                    (flattened_hms.shape[0], 4)) - INF
                reg_targets[c33_inds] = c33_regs
                # import pdb; pdb.set_trace()
                debug_train(
                    [self.denormalizer(x) for x in images], 
                    gt_instances, flattened_hms, reg_targets, 
                    labels, assigned_center_inds, shapes_per_level, grids, self.strides)
            return proposals, losses
    

    def inference(self, images, clss_per_level, reg_pred_per_level, 
        agn_hm_pred_per_level, grids):
        logits_pred = [x.sigmoid() for x in \
            (agn_hm_pred_per_level if self.only_proposal else clss_per_level)]
        agn_hm_pred_per_level = [None for _ in agn_hm_pred_per_level]
        # agn_hm_pred_per_level = [x.sigmoid_() if x is not None else None \
        #     for x in agn_hm_pred_per_level]

        proposals = self.predict_instances(
            grids, logits_pred, reg_pred_per_level, 
            images.image_sizes, agn_hm_pred_per_level)

        if self.as_proposal or self.only_proposal:
            for p in range(len(proposals)):
                proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
                proposals[p].objectness_logits = proposals[p].get('scores')
                proposals[p].remove('pred_boxes')

        if self.debug:
            debug_test(
                [self.denormalizer(x) for x in images], 
                logits_pred, reg_pred_per_level, 
                agn_hm_pred_per_level, preds=proposals,
                vis_thresh=self.vis_thresh, 
                debug_show_name=False)
        return proposals, {}


    def losses(
        self, center_inds, labels, masks, regs, c33_inds, c33_mask, c33_regs, flattened_hms,
        logits_pred, reg_pred):
        '''
        center_inds: N x L
        labels: N
        masks: N x L
        regs: N x L x 4
        c33_inds: N x L x K
        c33_mask: N x L x K
        c33_regs: N x L x K x 4
        flattened_hms: M x C
        logits_pred: M x C
        reg_pred: M x 4
        '''
        assert (torch.isfinite(logits_pred).all().item())
        assert (torch.isfinite(reg_pred).all().item())
        hm_pred = logits_pred.sigmoid()

        num_pos_local = center_inds.shape[0]
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(
            center_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(1.0 * total_num_pos / num_gpus, 1.0)

        losses = {}
        N, L = center_inds.shape
        center_inds = center_inds.view(N * L)
        labels_expand = labels.view(N, 1).expand(N, L).clone().view(N * L)

        assigned_center_inds = center_inds[masks.view(N * L)]
        labels = labels_expand[masks.view(N * L)]

        K = c33_inds.shape[2]
        if self.more_pos >= 0:
            c33_inds[c33_mask == 0] = 0
            reg_pred_c33 = reg_pred[c33_inds].detach() # N x L x K
            invalid_reg = c33_mask == 0
            c33_regs_tmp = c33_regs.view(N * L * K, 4).clamp(min=0)
            if N > 0:
                with torch.no_grad():
                    c33_reg_loss = self.iou_loss(
                        reg_pred_c33.view(N * L * K, 4), 
                        c33_regs_tmp, None,
                        reduction='none').view(N, L, K).detach() # N x L x K
            else:
                c33_reg_loss = reg_pred_c33.new_zeros((N, L, K)).detach()
            c33_reg_loss[invalid_reg] = INF # N x L x K
            c33_reg_loss.view(N * L, K)[masks.view(N * L), 4] = 0
            c33_reg_loss = c33_reg_loss.view(N, L * K)
            if self.more_pos == 0 or N == 0:
                loss_thresh = c33_reg_loss.new_ones((N)).float()
            else:
                loss_thresh = torch.kthvalue(
                    c33_reg_loss,
                    self.more_pos, dim=1)[0] # N
            loss_thresh[loss_thresh > self.more_pos_thresh] = self.more_pos_thresh
            new_pos = c33_reg_loss.view(N, L, K) < \
                loss_thresh.view(N, 1, 1).expand(N, L, K)
            new_inds = c33_inds[new_pos].view(-1) # P
            assigned_center_inds = new_inds
            labels = labels_expand.view(N, L, 1).expand(N, L, K).clone()[new_pos].view(-1)
            num_pos_avg = max(1.0 * assigned_center_inds.shape[0], 1.0)

        hm_pred_pos = hm_pred[assigned_center_inds, labels] # P

        pos_pred = torch.clamp(
            hm_pred_pos, 
            min=self.sigmoid_clamp, max=1-self.sigmoid_clamp)
        pos_loss = - torch.log(pos_pred) * torch.pow(
            1. - pos_pred, self.loss_gamma)
        neg_weights = torch.pow(1. - flattened_hms, self.hm_focal_beta)
        neg_loss = - torch.log(1. - hm_pred) * torch.pow(
            hm_pred, self.loss_gamma) * neg_weights
        if self.ignore_high_fp > 0:
            not_high_fp = (hm_pred < self.ignore_high_fp).float()
            neg_loss = not_high_fp * neg_loss
        pos_loss = self.pos_weight * self.hm_focal_alpha * pos_loss.sum() / num_pos_avg
        neg_loss = self.neg_weight * \
            (1. - self.hm_focal_alpha) * neg_loss.sum() / num_pos_avg
        losses = {'loss_pos': pos_loss, 'loss_neg': neg_loss}

        c33_mask = c33_mask & masks.view(N, L, 1).expand(N, L, K)
        c33_inds = c33_inds[c33_mask] # P
        c33_regs = c33_regs[c33_mask] # P x 4

        if c33_mask.sum() > 0:
            reg_weight_map = hm_pred.detach()[c33_inds].max(dim=1)[0]
            reg_norm = max(1.0 * reg_weight_map.sum().item(), 1)
            reg_loss = self.reg_weight * self.iou_loss(
                reg_pred[c33_inds], c33_regs, reg_weight_map,
                reduction='sum') / reg_norm
            losses['loss_loc'] = reg_loss
        else:
            losses['loss_loc'] = reg_pred.sum() * 0

        if self.debug:
            print('losses', losses)
            print('reg_norm', reg_norm)
            print('total_num_pos', total_num_pos)
        return losses, assigned_center_inds, c33_inds, c33_regs


    def compute_grids(self, features):
        grids = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            shifts_x = torch.arange(
                0, w * self.strides[level], 
                step=self.strides[level],
                dtype=torch.float32, device=feature.device)
            shifts_y = torch.arange(
                0, h * self.strides[level], 
                step=self.strides[level],
                dtype=torch.float32, device=feature.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            grids_per_level = torch.stack((shift_x, shift_y), dim=1) + \
                self.strides[level] // 2
            grids.append(grids_per_level)
        return grids


    def _get_ground_truth(self, grids, shapes_per_level, gt_instances):
        '''
        Input:
            grids: list of tensors [(hl x wl, 2)]_l
            shapes_per_level: list of tuples L x 2:
            gt_instances: gt instances
        Retuen:
            pos_inds: N
            labels: N
            reg_targets: M x 4
            flattened_hms: M x c'
        '''
        center_inds, labels, masks, regs, c33_inds, c33_mask, c33_regs = \
            self._get_label_inds(gt_instances, shapes_per_level)
        L = len(grids)
        num_loc_list = [len(loc) for loc in grids]
        strides = torch.cat([
            grids[l].new_ones(num_loc_list[l]) * self.strides[l] \
            for l in range(L)]).float() # M
        grids = torch.cat(grids, dim=0) # M x 2
        M = grids.shape[0]

        flattened_hms = []
        for i in range(len(gt_instances)): # images
            boxes = gt_instances[i].gt_boxes.tensor # N x 4
            area = gt_instances[i].gt_boxes.area()
            gt_classes = gt_instances[i].gt_classes # N in [0, self.num_classes]
            if self.only_proposal:
                gt_classes = gt_classes * 0
            N = boxes.shape[0]
            if N == 0:
                flattened_hms.append(grids.new_zeros((M, self.num_classes)))
                continue

            centers = ((boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2) # N x 2
            centers_expanded = centers.view(1, N, 2).expand(M, N, 2) # M x N x 2
            strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)
            centers_discret = ((centers_expanded / strides_expanded).int() * \
                            strides_expanded).float() + strides_expanded / 2 # M x N x 2
            is_peak = (((grids.view(M, 1, 2).expand(M, N, 2) - \
                            centers_discret) ** 2).sum(dim=2) == 0) # M x N
            dist2 = ((grids.view(M, 1, 2).expand(M, N, 2) - \
                centers_expanded) ** 2).sum(dim=2) # M x N
            dist2[is_peak] = 0
            radius2 = self.delta ** 2 * 2 * area # N
            radius2 = torch.clamp(
                radius2, min=self.min_radius ** 2)
            weighted_dist2 = dist2 / radius2.view(1, N).expand(M, N) # M x N            
            flattened_hm = self._create_heatmaps_from_dist(
                weighted_dist2.clone(), gt_classes, channels=self.num_classes) # M x C

            flattened_hms.append(flattened_hm)
        flattened_hms = _transpose(flattened_hms, num_loc_list)
        flattened_hms = cat([x for x in flattened_hms], dim=0) # MB x C
        
        return center_inds, labels, masks, regs, c33_inds, c33_mask, c33_regs, flattened_hms


    def _get_label_inds(self, gt_instances, shapes_per_level):
        '''
        Get the center (and the 3x3 region near center) locations of each objects
        Inputs:
            gt_instances: [n_i], sum n_i = N
            shapes_per_level: L x 2 [(h_l, w_l)]_L
        Returns:
            pos_inds: N'
            labels: N'
        '''
        center_inds = []
        labels = []
        masks = []
        regs = []
        c33_inds = []
        c33_masks = []
        c33_regs = []
        L = len(self.strides)
        B = len(gt_instances)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1]).long() # L
        level_bases = []
        s = 0
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l]
        level_bases = shapes_per_level.new_tensor(level_bases).long() # L
        strides_default = shapes_per_level.new_tensor(self.strides).float() # L
        K = 9
        dx = shapes_per_level.new_tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1]).long()
        dy = shapes_per_level.new_tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1]).long()
        for im_i in range(B):
            targets_per_im = gt_instances[im_i]
            bboxes = targets_per_im.gt_boxes.tensor # n x 4
            n = bboxes.shape[0]
            if n == 0:
                continue
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2) # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2)

            strides = strides_default.view(1, L, 1).expand(n, L, 2) # 
            centers_inds = (centers / strides).long() # n x L x 2

            center_grids = centers_inds * strides + strides // 2# n x L x 2
            l = center_grids[:, :, 0] - bboxes[:, 0].view(n, 1).expand(n, L)
            t = center_grids[:, :, 1] - bboxes[:, 1].view(n, 1).expand(n, L)
            r = bboxes[:, 2].view(n, 1).expand(n, L) - center_grids[:, :, 0]
            b = bboxes[:, 3].view(n, 1).expand(n, L) - center_grids[:, :, 1] # n x L
            reg = torch.stack([l, t, r, b], dim=2) # n x L x 4
            reg = reg / strides_default.view(1, L, 1).expand(n, L, 4).float()
            
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            Hs = shapes_per_level[:, 0].view(1, L).expand(n, L)
            pos_ind = level_bases.view(1, L).expand(n, L) + \
                       im_i * loc_per_level.view(1, L).expand(n, L) + \
                       centers_inds[:, :, 1] * Ws + \
                       centers_inds[:, :, 0] # n x L
            expand_Ws = Ws.view(n, L, 1).expand(n, L, K)
            expand_Hs = Hs.view(n, L, 1).expand(n, L, K)
            label = targets_per_im.gt_classes.view(n).clone()
            if self.only_proposal:
                label = label * 0
            mask = reg.min(dim=2)[0] >= 0 # n x L
            mask = mask & self._assign_fpn_level(bboxes)
            center_inds.append(pos_ind) # n x L
            labels.append(label) # n
            masks.append(mask) # n x L
            regs.append(reg) # n x L x 4

            Dy = dy.view(1, 1, K).expand(n, L, K)
            Dx = dx.view(1, 1, K).expand(n, L, K)
            c33_ind = level_bases.view(1, L, 1).expand(n, L, K) + \
                       im_i * loc_per_level.view(1, L, 1).expand(n, L, K) + \
                       (centers_inds[:, :, 1:2].expand(n, L, K) + Dy) * expand_Ws + \
                       (centers_inds[:, :, 0:1].expand(n, L, K) + Dx) # n x L x K
            
            c33_mask = \
                ((centers_inds[:, :, 1:2].expand(n, L, K) + dy) < expand_Hs) & \
                ((centers_inds[:, :, 1:2].expand(n, L, K) + dy) >= 0) & \
                ((centers_inds[:, :, 0:1].expand(n, L, K) + dx) < expand_Ws) & \
                ((centers_inds[:, :, 0:1].expand(n, L, K) + dx) >= 0)
            # TODO: think about better way to implement this
            # Currently it hard codes the 3x3 region
            c33_reg = reg.view(n, L, 1, 4).expand(n, L, K, 4).clone()
            c33_reg[:, :, [0, 3, 6], 0] -= 1
            c33_reg[:, :, [0, 3, 6], 2] += 1
            c33_reg[:, :, [2, 5, 8], 0] += 1
            c33_reg[:, :, [2, 5, 8], 2] -= 1
            c33_reg[:, :, [0, 1, 2], 1] -= 1
            c33_reg[:, :, [0, 1, 2], 3] += 1
            c33_reg[:, :, [6, 7, 8], 1] += 1
            c33_reg[:, :, [6, 7, 8], 3] -= 1
            c33_mask = c33_mask & (c33_reg.min(dim=3)[0] >= 0) # n x L x K
            c33_inds.append(c33_ind)
            c33_masks.append(c33_mask)
            c33_regs.append(c33_reg)
        
        if len(center_inds) > 0:
            center_inds = torch.cat(center_inds, dim=0).long()
            labels = torch.cat(labels, dim=0)
            masks = torch.cat(masks, dim=0)
            regs = torch.cat(regs, dim=0)
            c33_inds = torch.cat(c33_inds, dim=0).long()
            c33_regs = torch.cat(c33_regs, dim=0)
            c33_masks = torch.cat(c33_masks, dim=0)
        else:
            center_inds = shapes_per_level.new_zeros((0, L)).long()
            labels = shapes_per_level.new_zeros((0)).long()
            masks = shapes_per_level.new_zeros((0, L)).bool()
            regs = shapes_per_level.new_zeros((0, L, 4)).float()
            c33_inds = shapes_per_level.new_zeros((0, L, K)).long()
            c33_regs = shapes_per_level.new_zeros((0, L, K, 4)).float()
            c33_masks = shapes_per_level.new_zeros((0, L, K)).bool()
        return center_inds, labels, masks, regs, c33_inds, c33_masks, c33_regs # N x L, N, N x L x K


    def _assign_fpn_level(self, boxes):
        # we assign objects to FPN levels by their diagnal length
        crit = ((boxes[:, 2:] - boxes[:, :2]) **2).sum(dim=1) ** 0.5 / 2 # n
        n, L = crit.shape[0], len(self.sizes_of_interest)
        crit = crit.view(n, 1).expand(n, L)
        size_ranges = boxes.new_tensor(self.sizes_of_interest)
        size_ranges_expand = size_ranges.view(1, L, 2).expand(n, L, 2)
        is_cared_in_the_level = (crit >= size_ranges_expand[:, :, 0]) & \
            (crit <= size_ranges_expand[:, :, 1])
        return is_cared_in_the_level


    def _flatten_outputs(self, clss, reg_pred, agn_hm_pred):
        # Reshape: (N, F, Hl, Wl) -> (N, Hl, Wl, F) -> (sum_l N*Hl*Wl, F)
        clss = cat([x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]) \
            for x in clss], dim=0) if clss[0] is not None else None
        reg_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred], dim=0)            
        agn_hm_pred = cat([x.permute(0, 2, 3, 1).reshape(-1) \
            for x in agn_hm_pred], dim=0) if self.with_agn_hm else None
        return clss, reg_pred, agn_hm_pred


    def _create_heatmaps_from_dist(self, dist, labels, channels):
        '''
        dist: M x N
        labels: N
        return:
          heatmaps: M x C
        '''
        heatmaps = dist.new_zeros((dist.shape[0], channels))
        for c in range(channels):
            inds = (labels == c) # N
            if inds.int().sum() == 0:
                continue
            # TODO: check if ignoring INF can be faster
            heatmaps[:, c] = torch.exp(-dist[:, inds].min(dim=1)[0])
            zeros = heatmaps[:, c] < 1e-4
            heatmaps[zeros, c] = 0
        return heatmaps


    def predict_instances(
        self, grids, logits_pred, reg_pred, image_sizes, agn_hm_pred):
        sampled_boxes = []
        for l in range(len(grids)):
            sampled_boxes.append(self.predict_single_level(
                grids[l], logits_pred[l], reg_pred[l] * self.strides[l],
                image_sizes, agn_hm_pred[l], l))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.nms_and_topK(boxlists)
        return boxlists


    def predict_single_level(
        self, grids, heatmap, reg_pred, image_sizes, agn_hm, level):
        B, C, H, W = heatmap.shape
        # put in the same format as grids
        heatmap = heatmap.permute(0, 2, 3, 1) # B x H x W x C
        heatmap = heatmap.reshape(B, -1, C) # B x HW x C
        box_regression = reg_pred.view(B, 4, H, W).permute(0, 2, 3, 1) # B x H x W x 4 
        box_regression = box_regression.reshape(B, -1, 4)

        candidate_inds = heatmap > self.score_thresh
        pre_nms_top_n = candidate_inds.view(B, -1).long().sum(dim=1) # B
        pre_nms_topk = self.pre_nms_topk_train \
            if self.training else self.pre_nms_topk_test
        pre_nms_top_n = pre_nms_top_n.clamp(max=pre_nms_topk) # B

        if agn_hm is not None:
            agn_hm = agn_hm.view(B, 1, H, W).permute(0, 2, 3, 1)
            agn_hm = agn_hm.reshape(B, -1)
            heatmap = heatmap * agn_hm[:, :, None]

        results = []
        for i in range(B):
            per_box_cls = heatmap[i] # HW x C
            per_candidate_inds = candidate_inds[i] # n
            per_box_cls = per_box_cls[per_candidate_inds] # n

            per_candidate_nonzeros = per_candidate_inds.nonzero() # n
            per_box_loc = per_candidate_nonzeros[:, 0] # n
            per_class = per_candidate_nonzeros[:, 1] # n
            per_box_regression = box_regression[i] # HW x 4
            per_box_regression = per_box_regression[per_box_loc] # n x 4
            per_grids = grids[per_box_loc] # n x 2
            per_pre_nms_top_n = pre_nms_top_n[i] # 1

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_grids = per_grids[top_k_indices]

            detections = torch.stack([
                per_grids[:, 0] - per_box_regression[:, 0],
                per_grids[:, 1] - per_box_regression[:, 1],
                per_grids[:, 0] + per_box_regression[:, 2],
                per_grids[:, 1] + per_box_regression[:, 3],
            ], dim=1) # n x 4
            detections[:, 2] = torch.max(detections[:, 2], detections[:, 0] + 0.01)
            detections[:, 3] = torch.max(detections[:, 3], detections[:, 1] + 0.01)
            boxlist = Instances(image_sizes[i])
            boxlist.scores = torch.sqrt(per_box_cls) \
                if self.with_agn_hm else per_box_cls # n
            boxlist.pred_boxes = Boxes(detections)
            boxlist.pred_classes = per_class
            results.append(boxlist)
        return results


    def nms_and_topK(self, boxlists, nms=True):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            nms_thresh = self.nms_thresh_train if self.training else \
                self.nms_thresh_test
            result = ml_nms(boxlists[i], nms_thresh) \
                if nms else boxlists[i]
            if self.debug:
                print('#proposals before nms', len(boxlists[i]))
                print('#proposals after nms', len(result))
            number_of_detections = len(result)

            post_nms_topk = self.post_nms_topk_train if self.training else self.post_nms_topk_test
            if number_of_detections > post_nms_topk:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            if self.debug:
                print('#proposals after filter', len(result))
            results.append(result)
        return results
