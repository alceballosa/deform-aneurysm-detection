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
    """Modified Hungarian matcher that combines two matching strategies:

    1. **Hungarian matching**: Standard LSAP (linear sum assignment problem) to find
       optimal 1-to-1 prediction-to-GT assignment based on classification + L1 center cost.

    2. **Proximity matching**: Additionally matches predictions whose reference points
       fall within L1 distance < `match_dist_threshold` of a GT center. This provides extra supervision
       for queries near GT objects (since adjacent reference points with similar features
       should both learn to detect nearby objects).

    Returns:
        indices: List[Tuple[np.ndarray, np.ndarray]] — per-sample matched (pred, gt) index pairs.
            Combines Hungarian + proximity matches, deduplicated by pred index.
        punish_mask_list: List[torch.Tensor] — per-sample boolean mask over all queries.
            Used in compute_losses to weight the classification loss:
            - True  → query contributes to classification loss (punished)
            - False → query is excluded from classification loss (silenced)

    Punish mask behavior:
        The mask controls which queries participate in the classification loss.
        Two separate boolean masks are tracked across all GTs and combined:
        - nearby_any_gt: accumulates queries within L1 distance < match_dist_threshold of ANY GT
        - selected_for_supervision: accumulates queries picked for supervision by ANY GT
        Final mask: punish_mask = ~nearby_any_gt | selected_for_supervision
        - Far-from-all-GTs queries → True (punished toward background)
        - Nearby + selected for any GT → True (supervised with correct class target)
        - Nearby + never selected → False (silenced — ambiguous zone near GT)
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        match_dist_threshold=0.5,
        max_nearby_per_gt=8,
    ):
        """Creates the matcher

        Params:
            cost_class: Relative weight of classification error in Hungarian matching cost.
            cost_bbox: Relative weight of L1 center distance in Hungarian matching cost.
            cost_giou: Unused (vestigial from DETR's GIoU matching cost).
            match_dist_threshold: L1 distance threshold for proximity matching. Predictions with
                L1 distance to a GT center < match_dist_threshold are considered "nearby".
            max_nearby_per_gt: Maximum number of proximity-matched predictions per GT box.
                If more predictions are nearby, a random subset of this size is selected.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.match_dist_threshold = match_dist_threshold
        self.max_nearby_per_gt = max_nearby_per_gt
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    def forward(self, preds, targets):
        """Match predictions to ground truth using Hungarian + proximity matching.

        Args:
            preds: Dict with keys:
                - "class_logits": (bs, num_queries, num_classes)
                - "pre_refinement_center": (bs, num_queries, 3) — reference point locations
            targets: List[Dict] of length bs, each with keys:
                - "labels": (num_objects,) — class indices
                - "center": (num_objects, 3) — GT center coordinates
                - "corners": (num_objects, ...) — GT box corners (unused in proximity check)

        Returns:
            indices: List of [pred_indices, gt_indices] arrays per sample
            penalize_mask_list: List of boolean tensors per sample (see class docstring)
        """
        with torch.no_grad():
            bs, num_queries = preds["class_logits"].shape[:2]

            out_prob = preds["class_logits"].softmax(
                -1
            )  # [batch_size, num_queries, num_classes]

            indices = []
            penalize_mask_list = []
            assert bs == len(targets)  # this is actually volume length
            for batch_idx in range(bs):
                pred_center = preds["pre_refinement_center"][batch_idx]
                pred_cls_prob = out_prob[batch_idx]
                tgt_cls = targets[batch_idx]["labels"]
                tgt_center = targets[batch_idx]["center"]
                tgt_corners = targets[batch_idx]["corners"]

                num_objects = len(tgt_cls)
                if num_objects == 0:  # empty object in key frame
                    indices.append([])
                    penalize_mask_list.append([])
                    continue

                # ---hungarian matching---
                # Compute the classification cost.
                cost_class = -pred_cls_prob[:, tgt_cls.long()]
                cost_center = torch.cdist(pred_center, tgt_center, p=1)
                cost = (
                    self.cost_bbox * cost_center + self.cost_class * cost_class
                )  # + 100.0 * (~is_in_boxes_and_center)
                matched_pairs = linear_sum_assignment(cost.cpu())
                matched_pairs = list(matched_pairs)
                # -----------------------

                # --- Proximity matching ---
                # For each GT, find predictions within L1 distance < match_dist_threshold and add
                # them as extra matched pairs (on top of Hungarian matches).
                # Also build a punish_mask to control classification loss weighting.
                #
                # Two separate masks track independent concerns:
                #   nearby_any_gt: True if query is within match_dist_threshold of ANY GT
                #   selected_for_supervision: True if query was picked for supervision by ANY GT
                # Final mask: silence only queries that are nearby but never selected.
                pred_indices = []
                gt_indices = []
                nearby_any_gt = torch.zeros(num_queries, dtype=torch.bool, device=cost_center.device)
                selected_for_supervision = torch.zeros(num_queries, dtype=torch.bool, device=cost_center.device)
                for j, box_j in enumerate(tgt_corners):
                    inside_sph = cost_center[..., j] < self.match_dist_threshold
                    pred_ind = torch.nonzero(inside_sph).squeeze(1).data.cpu().numpy()

                    nearby_any_gt[pred_ind] = True

                    # Cap the number of proximity matches per GT box.
                    # If more nearby predictions than max_nearby_per_gt, randomly subsample.
                    if pred_ind.shape[0] > self.max_nearby_per_gt:
                        choose = np.random.choice(
                            pred_ind.shape[0], self.max_nearby_per_gt, replace=False
                        )
                        pred_ind = pred_ind[choose]

                    selected_for_supervision[pred_ind] = True
                    pred_indices.append(pred_ind)
                    gt_indices.append(np.ones_like(pred_ind) * j)

                # Silence only queries that are nearby some GT but never selected.
                # Far queries: ~nearby(F) | selected(F) = T (punished toward background)
                # Nearby + selected: ~nearby(T) | selected(T) = T (supervised with GT target)
                # Nearby + not selected: ~nearby(T) | selected(F) = F (silenced)
                penalize_mask = ~nearby_any_gt | selected_for_supervision
                pred_indices = np.concatenate(pred_indices)
                gt_indices = np.concatenate(gt_indices)
                # ----------------------

                # Merge Hungarian and proximity matches
                matched_pairs[0] = np.concatenate([matched_pairs[0], pred_indices])
                matched_pairs[1] = np.concatenate([matched_pairs[1], gt_indices])

                # Deduplicate by pred index: if a prediction was matched by both
                # Hungarian and proximity, keep the first occurrence (Hungarian takes
                # priority since it's concatenated first).
                _, inverse_indices = np.unique(matched_pairs[0], return_index=True)
                matched_pairs[0] = matched_pairs[0][inverse_indices]
                matched_pairs[1] = matched_pairs[1][inverse_indices]

                indices.append(matched_pairs)
                penalize_mask_list.append(penalize_mask)
        return indices, penalize_mask_list
