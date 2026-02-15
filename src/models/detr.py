import copy
import math
import os
import pdb
import pickle
import time

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils.events import get_event_storage

from src.dataset.split_comb import SplitComb
from src.models.backbones import cnn_backbone
from src.models.box_utils import get_3d_corners, nms_3D
from src.models.hungarian_matcher import HungarianMatcherModified
from src.models.trx_deformable.def_trx import build_deformable_transformer
from src.models.trx_deformable.nms import nms
from src.models.trx_efficient.trx import build_efficient_transformer
from src.utils.general import inverse_sigmoid
from src.utils.losses import (
    bbox_iou_loss,
    focal_loss,
    no_targets_cross_entropy_loss,
    no_targets_focal_loss,
)

total_samples = 0
total_pos = 0


build_backbone = {
    "CNN": cnn_backbone.build_backbone,
}


@META_ARCH_REGISTRY.register()
class PARQ_Deformable_R(nn.Module):
    @configurable
    def __init__(
        self,
        cfg,
        backbone_type="CNN",
        patch_size=(64, 64, 64),
        num_queries=8,
        d_model=768,
        device=None,
        loss_weights=dict(
            cls_w=1.0,
            shape_w=1.0,
            offset_w=1.0,
            iou_w=1.0,
        ),
        use_pretrained_unet_encoder=False,
        path_unet_weights="",
        frozen_pretrained_encoder=False,
        use_checkpoint=False,
    ):
        super(PARQ_Deformable_R, self).__init__()
        self.cfg = cfg
        self.backbone_type = backbone_type
        self.use_checkpoint = use_checkpoint

        self.device = device
        self._split_comb = None

        # losses
        self.num_semcls = 1
        self.loss_weights = loss_weights
        self.iou_loss = bbox_iou_loss

        # NOTE following the official DETR rep0, bg_cls_weight means relative classification weight of the no-object class.
        # TODO put class weight this in yaml
        self.class_weight = torch.ones(self.num_semcls + 1)  # * class_weight
        # set background class as the last indice
        # TODO: check if giving less weight to main class helps
        # self.class_weight[self.num_semcls] = 0.1

        # pretrained encoder
        self.use_pretrained_encoder = use_pretrained_unet_encoder
        self.path_encoder_weights = path_unet_weights
        self.frozen_pretrained_encoder = frozen_pretrained_encoder

        self.use_vessel_info = self.cfg.MODEL.USE_VESSEL_INFO
        self.use_cvs_info = self.cfg.MODEL.USE_CVS_INFO
        if self.use_vessel_info in ["pos_emb", "start"]:
            assert (
                self.cfg.DATA.DIR.TRAIN.VESSEL_DIR != ""
            ), "Vessel path must be defined"

        # self.frozen_parameters_list = []

        self.patch_size = torch.Tensor(patch_size).to(device)
        self.backbone = build_backbone[backbone_type](cfg)
        # first half of the weights used as learnable pos embedding
        # second half used as the actual query embeddings

        # TODO: do this in a better way
        self.query_pos_embed_plus_query = nn.Embedding(num_queries, d_model * 2)
        if cfg.MODEL.DEFORMABLE.EFFICIENT_MASK_V2:
            self.transformer = build_efficient_transformer(cfg)
        else:
            self.transformer = build_deformable_transformer(cfg)
        matcher_cfg = cfg.MODEL.PARQ_MODEL.MATCHER
        self.matcher = HungarianMatcherModified(
            cost_class=matcher_cfg.COST_CLASS,
            cost_bbox=matcher_cfg.COST_BBOX,
            cost_giou=matcher_cfg.COST_GIOU,
            match_dist_threshold=matcher_cfg.MATCH_DIST_THRESHOLD,
            max_nearby_per_gt=matcher_cfg.MAX_NEARBY_PER_GT,
        )
        self.__init_weight()

    @property
    def split_com(self) -> SplitComb:
        if self._split_comb is None:
            self._split_comb = SplitComb(
                crop_size=self.cfg.DATA.PATCH_SIZE,
                overlap=self.cfg.DATA.OVERLAP,
                pad_value=-1,  # normalized minimum
            )
        return self._split_comb

    @classmethod
    def from_config(cls, cfg):
        conv_cfg = cfg.MODEL.CONV_MODEL
        parq_loss_cfg = cfg.MODEL.PARQ_MODEL.PARQ_LOSS
        return {
            "cfg": cfg,
            "device": cfg.MODEL.DEVICE,
            "backbone_type": cfg.MODEL.CONV_MODEL.BACKBONE_TYPE,
            "patch_size": cfg.DATA.PATCH_SIZE,
            "d_model": cfg.MODEL.D_MODEL,
            "num_queries": cfg.MODEL.PARQ_MODEL.NUM_QUERIES,
            "use_pretrained_unet_encoder": conv_cfg.USE_PRETRAINED_UNET_ENCODER,
            "path_unet_weights": conv_cfg.PRETRAINED_UNET_ENCODER_PATH,
            "frozen_pretrained_encoder": conv_cfg.FROZEN_PRETRAINED_ENCODER,
            "use_checkpoint": cfg.MODEL.DEFORMABLE.USE_CHECKPOINT,
            "loss_weights": {
                "cls_w": parq_loss_cfg.CLS_W,
                "shape_w": parq_loss_cfg.SHAPE_W,
                "offset_w": parq_loss_cfg.OFFSET_W,
                "iou_w": parq_loss_cfg.IOU_W,
            },
        }

    def __init_weight(self):
        # TODO: add inits for new modules
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # prior = 0.01
        # nn.init.constant_(self.head.cls_output.weight, 0)
        # nn.init.constant_(self.head.cls_output.bias, -math.log((1.0 - prior) / prior))

        # nn.init.constant_(self.head.shape_output.weight, 0)
        # nn.init.constant_(self.head.shape_output.bias, 0.5)

        # nn.init.constant_(self.head.offset_output.weight, 0)
        # nn.init.constant_(self.head.offset_output.bias, 0.05)

    def freeze_parameters(self):
        """
        Freezes the parameters defined in `self.frozen_parameters_list`.
        """
        for name, param in self.named_parameters():
            if name in self.frozen_parameters_list:
                param.requires_grad = False
        print("\nParameters frozen successfully")
        return

    def forward(self, input_batch):
        if self.training:
            torch.cuda.empty_cache()
            return self._forward_train(input_batch)
        try:
            return self._forward_eval(input_batch)
        except MemoryError as e:
            print(f"Error {e}. CUDA out of memory.Trying to clear cache")
            torch.cuda.empty_cache()
            return self._forward_eval(input_batch)

    def _forward_train(self, input_batch):
        # if self.cfg.CUSTOM.TRACKING_GRADIENT_NORM:
        #    get_event_storage().put_scalar("grad_norm", get_gradient_norm(self))
        x, vessel_dists, cvs_dists = self.preprocess_train_input(input_batch)

        targets = self.preprocess_train_labels(input_batch)
        box_prediction_list, _ = self._forward_network(x, vessel_dists, cvs_dists)
        loss_dict = self.compute_losses(box_prediction_list, targets)
        return loss_dict

    # NOTE: missing implementation of vessel  dists for this
    def _forward_eval(self, input_batch):
        """
        ! assume batch_size is alway 1
        """
        assert len(input_batch) == 1
        scan_id = input_batch[0]["scan_id"]
        bs = self.cfg.TEST.PATCHES_PER_ITER
        patches, nzhw, splits_boxes = self.split_com.split(input_batch[0]["image"])
        patches = np.concatenate(patches, axis=0)
        if self.use_vessel_info != "no":
            patches_vessel_edt, _, _ = self.split_com.split(
                input_batch[0]["vessel_edt"]
            )
            patches_vessel_edt = np.concatenate(patches_vessel_edt, axis=0)
        if self.use_cvs_info != "no":
            patches_cvs, _, _ = self.split_com.split(input_batch[0]["cvs_mask"])
            patches_cvs = np.concatenate(patches_cvs, axis=0)
        outputs = []
        list_viz_outputs = []

        for i in range(int(math.ceil(len(patches) / bs))):
            end = (i + 1) * bs
            if end > len(patches):
                end = len(patches)

            batch_data = torch.tensor(patches[i * bs : end], device=self.device)
            vessel_data = None
            cvs_data = None
            if self.use_vessel_info != "no":
                vessel_data = torch.tensor(
                    patches_vessel_edt[i * bs : end], device=self.device
                )
            if self.use_cvs_info != "no":
                cvs_data = torch.tensor(patches_cvs[i * bs : end], device=self.device)
            prediction_dicts, viz_outputs = self._forward_network(
                batch_data, vessel_data, cvs_data
            )

            for prediction_dict in prediction_dicts:
                for key in prediction_dict:
                    prediction_dict[key] = prediction_dict[key].detach().to("cpu")
            dets = self.parse_pred(prediction_dicts[-1]).detach().to("cpu")

            del batch_data
            del vessel_data
            del cvs_data
            torch.cuda.empty_cache()

            outputs.append(dets)
            list_viz_outputs.append(viz_outputs)

        # post process outputs
        outputs = torch.cat(outputs, dim=0)
        all_outputs = outputs = self.split_com.combine(outputs, nzhw)
        outputs = outputs.view(-1, 8)

        object_ids = outputs[:, 0] != -1
        outputs = outputs[object_ids]
        if len(outputs) > 0:
            keep = nms_3D(outputs[:, 1:], overlap=0.05, top_k=self.cfg.TEST.NMS_TOPK)
            # keep = nms_3D(outputs[:, 1:], overlap=0.5, top_k=120)
            outputs = outputs[keep]

        vizmode = self.cfg.MODEL.EVAL_VIZ_MODE
        if vizmode:
            print("EN MODO VISUALIZACION")
            filename_viz = f"{scan_id}_viz.pkl"
            filename_res = f"{scan_id}_res.pkl"
            filename_splits = f"{scan_id}_slits.pkl"
            filename_all = f"{scan_id}_all.pkl"
            path_viz = os.path.join(self.cfg.OUTPUT_DIR, filename_viz)
            path_res = os.path.join(self.cfg.OUTPUT_DIR, filename_res)
            path_splits = os.path.join(self.cfg.OUTPUT_DIR, filename_splits)
            path_all = os.path.join(self.cfg.OUTPUT_DIR, filename_all)

            with open(path_viz, "wb") as f:
                pickle.dump(list_viz_outputs, f)
            with open(path_res, "wb") as f:
                pickle.dump(outputs, f)
            with open(path_splits, "wb") as f:
                pickle.dump(splits_boxes, f)
            with open(path_all, "wb") as f:
                pickle.dump(all_outputs, f)

        return outputs

    def _forward_network(self, x, vessel_dists=None, cvs_dists=None):
        """
        This function receives a tensor x with various volumes to be
        classified and outputs a list of predictions produced by the PARQ
        module using the output of the UNET given x.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, channels, depth, height, width) containing
            the input volumes.

        Returns
        _______
        box_prediction_list : list
            List of dicts where each entry is a dictionary containing the
            predictions made for each of the `batch` volumes in `x`. There
            are `dec_layers` dicts, meaning one dict for each iteration of
            the PARQ framework.

            The key, value pairs of each dict are as follows:

            - `pred_logits` : torch.Tensor of shape (batch, num_queries, num_classes)
                Contains the logits for each of the classes for each of the
                `num_queries` queries.
            - `center_unnormalized` : torch.Tensor of shape (batch, num_queries, 3)
                Contains the predicted center coordinates from each of the
                `num_queries` queries.
            - `size_unnormalized` : torch.Tensor of shape (batch, num_queries, 3)
                Contains the predicted size from each of the `num_queries` queries.
            - `sem_cls_prob` : torch.Tensor of shape (batch, num_queries, num_classes)
                Contains the predicted class probabilities from each of the
                `num_queries` queries.
            - `coord_pos`: torch.Tensor of shape (batch, num_queries, 3)
                Contains the coordinates of each query prior to computing the new
                centers during each iteration

        """
        vessel_segs = None
        if self.cfg.MODEL.DEFORMABLE.EFFICIENT_MASK_V2 and vessel_dists is not None:
            vessel_segs = (vessel_dists > 0).float()
        if self.use_vessel_info == "start":
            x = torch.cat((x, vessel_dists / self.cfg.DATA.PATCH_SIZE[0]), dim=1)
            vessel_dists = None  # no need to keep using this
            if self.use_cvs_info == "start":
                x = torch.cat((x, cvs_dists / self.cfg.DATA.PATCH_SIZE[0]), dim=1)

        elif self.use_vessel_info == "no":
            vessel_dists = None  # shouldn't use vessel info here

        # Checkpoint the entire backbone forward pass to save memory
        if self.use_checkpoint and self.training:
            multiscale_feats, multiscale_pos_embs, key_padding_mask = (
                torch.utils.checkpoint.checkpoint(
                    self.backbone,
                    x,
                    vessel_dists,
                    vessel_segs,
                    self.transformer.level_embed,
                    use_reentrant=False,
                )
            )
        else:
            multiscale_feats, multiscale_pos_embs, key_padding_mask = self.backbone(
                x, vessel_dists, vessel_segs, self.transformer.level_embed
            )

        box_prediction_list, init_reference_out, viz_outputs, attn_list = (
            self.transformer.forward(
                multiscale_feats,
                multiscale_pos_embs,
                self.query_pos_embed_plus_query.weight,
                key_padding_mask,
            )
        )
        return box_prediction_list, viz_outputs

    def compute_losses(self, output_dict, targets):
        """
        input:
            out_dict_list:  predicted box3d parameters
            obbs_padded:    target box3d parameters
        output:
            loss
        """

        # Use graph-connected zeros so DDP sees gradients for all parameters
        # even when no targets are matched (avoids AllReduce deadlock).
        _dummy = output_dict[0]
        _zero = (
            _dummy["center"].sum() + _dummy["size"].sum() + _dummy["class_logits"].sum()
        ) * 0.0
        loss_dict = {
            "center_loss": _zero.clone(),
            "size_loss": _zero.clone(),
            "cat_loss": _zero.clone(),
            "iou_loss": _zero.clone(),
        }

        valid_bs_loc_shape = 0
        valid_bs_cls = 0
        # compute loss for every layer
        for out_dict in output_dict:
            # apply matching
            matched_indices, penalize_mask = self.matcher(out_dict, targets)
            # TODO: count samples instead of batches
            bs = len(matched_indices)
            # compute loss for every sample
            for i in range(bs):
                # category loss for the case with no target objects
                valid_bs_cls += 1
                if len(matched_indices[i]) == 0:

                    if self.cfg.MODEL.PARQ_MODEL.PARQ_LOSS.DO_CLF_FOCAL:
                        cat_loss = (
                            no_targets_focal_loss(
                                classes_pred=out_dict["class_logits"][i],
                                alpha=self.cfg.MODEL.PARQ_MODEL.PARQ_LOSS.FOCAL_ALPHA,
                                gamma=self.cfg.MODEL.PARQ_MODEL.PARQ_LOSS.FOCAL_GAMMA,
                            )
                            * self.loss_weights["cls_w"]
                        )
                    else:
                        cat_loss = (
                            no_targets_cross_entropy_loss(
                                out_dict["class_logits"][i], self.class_weight
                            )
                            * self.loss_weights["cls_w"]
                        )
                    loss_dict["cat_loss"] += cat_loss
                elif len(matched_indices[i][0]) != 0:
                    valid_bs_loc_shape += 1
                    # center loss: l1, applied only when matched to something
                    center_predict = out_dict["center"][i][matched_indices[i][0]]
                    center_target = targets[i]["center"][matched_indices[i][1]]
                    center_loss = (center_predict - center_target).abs().mean()
                    center_loss *= self.loss_weights["offset_w"]
                    loss_dict["center_loss"] += center_loss

                    # size loss: l1, applied only when matched to something
                    size_predict = out_dict["size"][i][matched_indices[i][0]]
                    size_target = targets[i]["size"][matched_indices[i][1]]
                    size_loss = (size_predict - size_target).abs().mean()
                    size_loss *= self.loss_weights["shape_w"]
                    loss_dict["size_loss"] += size_loss

                    # category loss
                    if self.cfg.MODEL.PARQ_MODEL.PARQ_LOSS.DO_CLF_FOCAL:
                        cat_loss = focal_loss(
                            out_dict["class_logits"][i],
                            targets[i],
                            matched_indices[i],
                            self.cfg.MODEL.PARQ_MODEL.PARQ_LOSS.FOCAL_ALPHA,
                            self.cfg.MODEL.PARQ_MODEL.PARQ_LOSS.FOCAL_GAMMA,
                            penalize_mask=penalize_mask[i],
                        )
                    else:
                        # TODO: modularize into losses.py
                        matched_classes_target = targets[i]["labels"][
                            matched_indices[i][1]
                        ].long()
                        classes_target = torch.full(
                            out_dict["class_logits"].shape[1:2],
                            self.num_semcls,
                            dtype=torch.int64,
                            device=out_dict["class_logits"].device,
                        )

                        classes_target[matched_indices[i][0]] = matched_classes_target
                        if penalize_mask is not None:
                            cross_entropy = torch.nn.CrossEntropyLoss(
                                self.class_weight.to(matched_classes_target.device),
                                reduction="none",
                            )

                            cat_loss = cross_entropy(
                                out_dict["class_logits"][i], classes_target
                            )

                            cat_loss = (
                                cat_loss * penalize_mask[i]
                            ).sum() / penalize_mask[i].sum()

                        else:
                            cross_entropy = torch.nn.CrossEntropyLoss(
                                self.class_weight.to(matched_classes_target.device)
                            )

                            cat_loss = cross_entropy(
                                out_dict["class_logits"][i], classes_target
                            )

                    cat_loss *= self.loss_weights["cls_w"]
                    loss_dict["cat_loss"] += cat_loss

                    # iou loss
                    center_size_predict = torch.cat(
                        [center_predict, size_predict], dim=-1
                    )
                    center_size_target = torch.cat([center_target, size_target], dim=-1)
                    iou_loss = bbox_iou_loss(center_size_predict, center_size_target)
                    iou_loss *= self.loss_weights["iou_w"]

                    loss_dict["iou_loss"] += iou_loss

        # average losses
        if (valid_bs_loc_shape + valid_bs_cls) != 0:
            for key, value in loss_dict.items():
                if (
                    key in ["center_loss", "size_loss", "iou_loss"]
                    and valid_bs_loc_shape != 0
                ):
                    loss_dict[key] = value / valid_bs_loc_shape
                elif key in ["cat_loss"] and valid_bs_cls != 0:
                    loss_dict[key] = value / valid_bs_cls
        return loss_dict

    def parse_pred(self, pred_dict):
        """
        reorganize the predicitions into OBB class.
        Also aplly some filters here, e.g. remove the predictions out the scope, nms
        """
        # only use the prediciton in the last iteration
        size_predict = pred_dict["size"]
        center_predict = pred_dict["center"]
        # TODO: consider splitting class logits into objectness + semantic class
        # probabilities (two-stage approach, see legacy BoxProcessor)
        logits = pred_dict["class_probs"]
        labels = torch.argmax(logits, dim=-1)
        bs = logits.shape[0]
        # print(pred_dict)
        n_queries = logits.shape[1]
        center_predict_flat = center_predict.reshape(bs * n_queries, 3)
        size_predict_flat = size_predict.reshape(bs * n_queries, 3)
        corners = get_3d_corners(center_predict_flat, size_predict_flat)
        corners = corners.reshape(bs, n_queries, 8, 3)
        # TODO: filter out of bounds
        # valid = torch.ones_like(center_predict[..., 0]).bool()
        # pred_mask = nms(corners, labels, logits, self.num_semcls, 0.1, "nms_3d_faster")
        # pred_mask = torch.tensor(pred_mask).to(valid.device) & valid
        dets = torch.ones((bs, n_queries, 8)) * -1
        for j in range(bs):
            for i in range(n_queries):
                # if pred_mask[j, i]:
                dets[j, i, 0] = 1
                dets[j, i, 1] = logits[j, i, 0]  # score for tgt class
                dets[j, i, 2:5] = center_predict[j, i] * self.patch_size.to("cpu")
                dets[j, i, 5:8] = size_predict[j, i] * self.patch_size.to("cpu")

        return dets

    def preprocess_train_input(self, input_batch):
        all_samples = sum([x["samples"] for x in input_batch], [])
        imgs = [s["image"] for s in all_samples]
        imgs = torch.tensor(np.stack(imgs, axis=0))
        imgs = imgs.to(self.device)

        vessel_dists = None
        cvs_dists = None
        if self.use_vessel_info in ["pos_emb", "start"]:
            vessel_dists = [s["vessel_edt"] for s in all_samples]
            vessel_dists = torch.tensor(np.stack(vessel_dists, axis=0))
            vessel_dists = vessel_dists.to(self.device)

            # vessel_dists = [s["vessel_edt"] for s in all_samples]
            # vessel_dists = torch.stack(vessel_dists, dim=0)
            # vessel_dists = vessel_dists.to(self.device)
        if self.use_cvs_info in ["start"]:
            cvs_dists = [s["cvs_mask"] for s in all_samples]
            cvs_dists = torch.tensor(np.stack(cvs_dists, axis=0))
            cvs_dists = cvs_dists.to(self.device)

        return imgs, vessel_dists, cvs_dists

    def preprocess_train_labels(self, input_batches: list):
        """
        Preprocesses labels in the positive only setting in which
        we only use the positive aneurysm labels for training.
        """

        valid_labels = [0, 1]
        all_samples = sum([x["samples"] for x in input_batches], [])
        target_list = []

        # TODO: verify scenario where there are no detected items at all
        for sample in all_samples:
            labels = []
            center = []
            size = []

            n_objects = sample["annot"].shape[0]

            for i in range(n_objects):
                label = int(sample["annot"][i][-1])  # + 1
                if label in valid_labels:
                    labels.append(label)
                    center.append(torch.tensor(sample["annot"])[i][:3].unsqueeze(0))
                    size.append(torch.tensor(sample["annot"])[i][3:6].unsqueeze(0))
            labels = torch.Tensor(labels).to(self.device) if len(labels) > 0 else []

            center = (
                torch.cat(center, dim=0).to(self.device) / self.patch_size
                if len(center) > 0
                else []
            )

            size = (
                torch.cat(size, dim=0).to(self.device) / self.patch_size
                if len(size) > 0
                else []
            )

            corners = (
                get_3d_corners(center, size).to(self.device) if len(labels) > 0 else []
            )
            target_dict = {
                "labels": labels,
                "center": center,
                "size": size,
                "corners": corners,
            }
            target_list.append(target_dict)

        return target_list


def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            total_norm += p.grad.detach().data.norm(2).item() ** 2
    total_norm = total_norm**0.5
    return total_norm
