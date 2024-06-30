"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
This file is derived from [DETR](https://github.com/facebookresearch/detr/blob/main/models/matcher.py).
Modified for [PARQ] by Yiming Xie.

Original header:
Copyright 2020 - present, Facebook, Inc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcherModified(nn.Module):
    """This is modified from the HungarianMatcher and Aside from Hungarian matching, we also match the GT box and
    the predictions whose corresponding reference points are in close proximity to this GT box, since
    for two adjacent reference points which have the similar queries, they should both detect nearby objects.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        ratio=0.5,
        max_padding=10,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.ratio = ratio
        self.max_padding = max_padding
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    def forward(self, preds, targets):
        with torch.no_grad():
            bs, num_queries = preds["pred_logits"].shape[:2]

            out_prob = preds["pred_logits"].softmax(
                -1
            )  # [batch_size * num_queries, num_classes]

            indices = []
            punish_mask_list = []
            assert bs == len(targets) #this is actually volume length
            for batch_idx in range(bs):
                pred_center = preds["coord_pos"][batch_idx]
                pred_cls_prob = out_prob[batch_idx]
                tgt_cls = targets[batch_idx]["labels"]
                tgt_center = targets[batch_idx]["center"]
                tgt_corners = targets[batch_idx]["corners"]

                num_objects = len(tgt_cls)
                if num_objects == 0:  # empty object in key frame
                    indices.append([])
                    punish_mask_list.append([])
                    continue

                # ---hungarian matching---
                # Compute the classification cost.
                cost_class = -pred_cls_prob[:, tgt_cls.long()]
                cost_center = torch.cdist(pred_center, tgt_center, p=1)
                cost = (
                    self.cost_bbox * cost_center + self.cost_class * cost_class
                )  # + 100.0 * (~is_in_boxes_and_center)
                indices_batchi = linear_sum_assignment(cost.cpu())
                indices_batchi = list(indices_batchi)
                # -----------------------

                # ---match the GT box and the predictions whose corresponding reference points
                # are in close proximity to this GT box---
                pred_indices = []
                gt_indices = []
                for j, box_j in enumerate(tgt_corners):
                    inside_sph = cost_center[..., j] < self.ratio
                    pred_ind = torch.nonzero(inside_sph).squeeze(1).data.cpu().numpy()
                    punish_mask = torch.ones_like(inside_sph).bool()
                    punish_mask[pred_ind] = False
                    # filter out based on max padding
                    if pred_ind.shape[0] > self.max_padding:
                        choose = np.random.choice(
                            pred_ind.shape[0], self.max_padding, replace=False
                        )
                        pred_ind = pred_ind[choose]
                    punish_mask[pred_ind] = True
                    pred_indices.append(pred_ind)
                    gt_indices.append(np.ones_like(pred_ind) * j)
                pred_indices = np.concatenate(pred_indices)
                gt_indices = np.concatenate(gt_indices)
                # ----------------------

                indices_batchi[0] = np.concatenate([indices_batchi[0], pred_indices])
                indices_batchi[1] = np.concatenate([indices_batchi[1], gt_indices])
                # TODO: review here!!
                # remove the redundant
                _, inverse_indices = np.unique(indices_batchi[0], return_index=True)
                indices_batchi[0] = indices_batchi[0][inverse_indices]
                indices_batchi[1] = indices_batchi[1][inverse_indices]

                indices.append(indices_batchi)
                punish_mask_list.append(punish_mask)
        return indices, punish_mask_list
