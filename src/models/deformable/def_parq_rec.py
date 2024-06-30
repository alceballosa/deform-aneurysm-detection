import copy
import math
import os
import pdb
import pickle
from functools import partial

import edt
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils.events import get_event_storage
from src.dataset.split_comb import SplitComb
from src.models.box_utils import nms_3D
from src.models.deformable import cnn_backbone, unet_backbone_4l, unet_backbone, cnn_backbone_1l, cnn_backbone_2l
from src.models.deformable.def_trx_rec import build_deformable_transformer
from src.models.parq.box_processor import BoxProcessor
from src.models.parq.generic_mlp import GenericMLP
from src.models.parq.nms import nms
from src.models.parq.parq_matcher_rec import HungarianMatcherModified
from src.models.parq.parq_utils import get_3d_corners
from src.utils.general import inverse_sigmoid
from src.utils.losses import (
    bbox_iou_loss,
    focal_loss,
    no_targets_cross_entropy_loss,
    no_targets_focal_loss,
)
from tqdm import tqdm

total_samples = 0
total_pos = 0


build_backbone = {
    "CNN": cnn_backbone.build_backbone,
    "CNN_1L": cnn_backbone_1l.build_backbone,
    "CNN_2L": cnn_backbone_2l.build_backbone,
    "UNET_4L": unet_backbone_4l.build_backbone,
    "UNET": unet_backbone.build_backbone,
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
    ):
        super(PARQ_Deformable_R, self).__init__()
        self.cfg = cfg
        self.backbone_type = backbone_type

        self.device = device
        self._split_comb = None

        # losses
        self.num_semcls = 1
        self.loss_weights = loss_weights
        self.class_weight = torch.ones(self.num_semcls + 1).to(
            self.device
        )  # * class_weight
        self.iou_loss = bbox_iou_loss
        self.clf_loss = torch.nn.CrossEntropyLoss(self.class_weight, reduction="mean")

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
        self.transformer = build_deformable_transformer(cfg)
        self.matcher = HungarianMatcherModified(cost_class=2, cost_bbox=0.25)
        self.__init_weight()

    @property
    def split_com(self) -> SplitComb:
        if self._split_comb is None:
            self._split_comb = SplitComb(
                crop_size=self.cfg.DATA.PATCH_SIZE,
                overlap=self.cfg.DATA.OVERLAP,
                pad_value=self.cfg.DATA.WINDOW[0],  # padding min value of window
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
        if self.cfg.CUSTOM.USE_SINGLE_BATCH:
            path_pickle = (
                "/home/ceballosarroyo.a/workspace/medical/cta-det2/src/input_batch.pkl"
            )
            # save file with input_batch to disk
            # with open(path_pickle, 'wb') as f:
            #    pickle.dump(input_batch, f)
            # load into input_batch
            with open(path_pickle, "rb") as f:
                input_batch = pickle.load(f)
            # input_batch = self.read_pickled_batch()

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
        if self.cfg.CUSTOM.TRACKING_GRADIENT_NORM:
            get_event_storage().put_scalar("grad_norm", get_gradient_norm(self))
        x, vessel_dists = self.preprocess_train_input(input_batch)
        targets = self.preprocess_train_labels(input_batch)
        box_prediction_list, _ = self._forward_network(x, vessel_dists)
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
        if self.use_vessel_info != "no":
            patches_vessel, _, _ = self.split_com.split(input_batch[0]["mask"])
        outputs = []
        list_viz_outputs = []
        for i in range(int(math.ceil(len(patches) / bs))):
            # preprocess batch
            end = (i + 1) * bs
            if end > len(patches):
                end = len(patches)
            # batch_data = np.concatenate(patches[i * bs : end], axis=0)
            # batch_data = torch.tensor(batch_data, device=self.device)
            batch_data = torch.cat(patches[i * bs : end], dim=0).to(self.device)
            vessel_data = None
            if self.use_vessel_info != "no":
                vessel_data = torch.cat(patches_vessel[i * bs : end], dim=0).to(
                    self.device
                )
            # NOTE may need to normalize other things
            batch_data = self.normalize_input_values(batch_data)

            prediction_dicts, viz_outputs = self._forward_network(
                batch_data, vessel_data
            )
            # TODO: do an alternatve version of parse pred with viz prep 239
            for prediction_dict in prediction_dicts:
                for key in prediction_dict:
                    prediction_dict[key] = prediction_dict[key].detach().to("cpu")
            dets = self.parse_pred(prediction_dicts[-1]).detach().to("cpu")
            del batch_data
            del vessel_data
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

    def _forward_network(self, x, vessel_dists=None):
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

        if self.use_vessel_info == "start":
            x = torch.cat((x, vessel_dists / self.cfg.DATA.PATCH_SIZE[0]), dim=1)
            vessel_dists = None  # no need to keep using this
        elif self.use_vessel_info == "no":
            vessel_dists = None  # shouldn't use vessel info here

        multiscale_feats, multiscale_pos_embs = self.backbone(x, vessel_dists)

        box_prediction_list, init_reference_out, viz_outputs, attn_list = (
            self.transformer.forward(
                multiscale_feats,
                multiscale_pos_embs,
                self.query_pos_embed_plus_query.weight,
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
        # assert targets.ndim == 3, f"{targets.shape}"
        loss_total = (
            output_dict[-1]["size"].sum()
            * output_dict[-1]["center"].sum()
            * output_dict[-1]["class_logits"].sum()
            * 0
        )

        loss_dict = {
            "center_loss": torch.tensor(0.0).to(self.device),
            "size_loss": torch.tensor(0.0).to(self.device),
            "cat_loss": torch.tensor(0.0).to(self.device),
            "iou_loss": torch.tensor(0.0).to(self.device),
        }

        valid_bs_loc_shape = 0
        valid_bs_cls = 0
        # compute loss for every layer
        for out_dict in output_dict:
            # apply matching
            matched_indices, punish_mask = self.matcher(out_dict, targets)
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
                    loss_total += center_loss
                    loss_dict["center_loss"] += center_loss

                    # size loss: l1, applied only when matched to something
                    size_predict = out_dict["size"][i][matched_indices[i][0]]
                    size_target = targets[i]["size"][matched_indices[i][1]]
                    size_loss = (size_predict - size_target).abs().mean()
                    size_loss *= self.loss_weights["shape_w"]
                    loss_total += size_loss
                    loss_dict["size_loss"] += size_loss

                    # category loss
                    if self.cfg.MODEL.PARQ_MODEL.PARQ_LOSS.DO_CLF_FOCAL:
                        matched_classes_target = targets[i]["labels"][
                            matched_indices[i][1]
                        ].long()
                        cat_loss = focal_loss(
                            out_dict["class_logits"][i],
                            targets[i],
                            matched_indices[i],
                            self.cfg.MODEL.PARQ_MODEL.PARQ_LOSS.FOCAL_ALPHA,
                            self.cfg.MODEL.PARQ_MODEL.PARQ_LOSS.FOCAL_GAMMA,
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
                        # TODO: review punish mask how it works and looks
                        if punish_mask is not None:
                            cross_entropy = torch.nn.CrossEntropyLoss(
                                self.class_weight.to(matched_classes_target.device),
                                reduction="none",
                            )
                            cat_loss = cross_entropy(
                                out_dict["class_logits"][i], classes_target
                            )

                            cat_loss = (cat_loss * punish_mask[i]).sum() / punish_mask[
                                i
                            ].sum()
                        else:
                            cross_entropy = torch.nn.CrossEntropyLoss(
                                self.class_weight.to(matched_classes_target.device)
                            )
                            cat_loss = cross_entropy(
                                out_dict["class_logits"][i], classes_target
                            )

                    cat_loss *= self.loss_weights["cls_w"]
                    loss_total += cat_loss
                    loss_dict["cat_loss"] += cat_loss

                    # iou loss
                    center_size_predict = torch.cat(
                        [center_predict, size_predict], dim=-1
                    )
                    center_size_target = torch.cat([center_target, size_target], dim=-1)
                    iou_loss = bbox_iou_loss(center_size_predict, center_size_target)
                    iou_loss *= self.loss_weights["iou_w"]

                    loss_total += iou_loss
                    loss_dict["iou_loss"] += iou_loss

        # average losses
        if (valid_bs_loc_shape + valid_bs_cls) != 0:
            loss_total = loss_total / (valid_bs_loc_shape + valid_bs_cls)
            for key, value in loss_dict.items():
                if (
                    key in ["center_loss", "size_loss", "iou_loss"]
                    and valid_bs_loc_shape != 0
                ):
                    loss_dict[key] = value / valid_bs_loc_shape
                elif key in ["cat_loss"] and valid_bs_cls != 0:
                    loss_dict[key] = value / valid_bs_cls

        loss_dict["total_loss"] = loss_total
        return loss_dict

    def parse_pred(self, pred_dict):
        """
        reorganize the predicitions into OBB class.
        Also aplly some filters here, e.g. remove the predictions out the scope, nms
        """
        # only use the prediciton in the last iteration
        size_predict = pred_dict["size"]
        center_predict = pred_dict["center"]
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
        valid = torch.ones_like(center_predict[..., 0]).bool()
        pred_mask = nms(corners, labels, logits, self.num_semcls, 0.1, "nms_3d_faster")
        pred_mask = torch.tensor(pred_mask).to(valid.device) & valid
        dets = torch.ones((bs, n_queries, 8)) * -1
        for j in range(bs):
            for i in range(n_queries):
                #if pred_mask[j, i]:
                dets[j, i, 0] = 1
                dets[j, i, 1] = logits[j, i, 0]  # score for tgt class
                dets[j, i, 2:5] = center_predict[j, i] * self.patch_size.to("cpu")
                dets[j, i, 5:8] = size_predict[j, i] * self.patch_size.to("cpu")

        return dets

    def preprocess_train_input(self, input_batch):
        all_samples = sum([x["samples"] for x in input_batch], [])
        imgs = [s["image"] for s in all_samples]

        imgs = torch.stack(imgs, dim=0)
        imgs = imgs.to(self.device)

        vessel_dists = None
        if self.use_vessel_info in ["pos_emb", "start"]:
            vessel_dists = [s["mask"] for s in all_samples]
            vessel_dists = torch.stack(vessel_dists, dim=0)
            vessel_dists = vessel_dists.to(self.device)
        # imgs = np.stack(imgs)
        # imgs = torch.tensor(imgs, device=self.device)
        imgs = self.normalize_input_values(imgs)
        return imgs, vessel_dists

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
                    center.append(sample["annot"][i][:3].unsqueeze(0))
                    size.append(sample["annot"][i][3:6].unsqueeze(0))
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

    def normalize_input_values(self, x: torch.Tensor):
        if self.backbone_type in ["UNET", "UNET2D", "CNN","CNN_1L", "CNN_2L"]:
            min_value, max_value = self.cfg.DATA.WINDOW
            x.clamp_(min=min_value, max=max_value)
            x -= (min_value + max_value) / 2
            x /= (max_value - min_value) / 2
        elif self.backbone_type in ["SAM3D", "SAM2D"]:
            if self.backbone_type == "SAM2D":
                # replicate across channels axis

                x = einops.repeat(x, "b c d h w -> b (c rep) d h w", rep=3)

            x = (x - self.pixel_mean) / self.pixel_std
        else:
            raise NotImplementedError(f"no encoder type {self.backbone_type}")

        return x

    @staticmethod
    def target_preprocess(annotations, device, input_size, mask_ignore):
        batch_size = annotations.shape[0]
        annotations_new = -1 * torch.ones_like(annotations).to(device)
        for j in range(batch_size):
            bbox_annotation = annotations[j]
            bbox_annotation_boxes = bbox_annotation[bbox_annotation[:, -1] > -1]
            bbox_annotation_target = []
            # z_ctr, y_ctr, x_ctr, d, h, w
            crop_box = torch.tensor(
                [0.0, 0.0, 0.0, input_size[0], input_size[1], input_size[2]]
            ).to(device)
            for s, _ in enumerate(bbox_annotation_boxes):
                # coordinate z_ctr, y_ctr, x_ctr, d, h, w
                each_label = bbox_annotation_boxes[s]
                # coordinate convert zmin, ymin, xmin, d, h, w
                z1 = torch.max(each_label[0] - each_label[3] / 2.0, crop_box[0])
                y1 = torch.max(each_label[1] - each_label[4] / 2.0, crop_box[1])
                x1 = torch.max(each_label[2] - each_label[5] / 2.0, crop_box[2])

                z2 = torch.min(each_label[0] + each_label[3] / 2.0, crop_box[3])
                y2 = torch.min(each_label[1] + each_label[4] / 2.0, crop_box[4])
                x2 = torch.min(each_label[2] + each_label[5] / 2.0, crop_box[5])

                nd = torch.clamp(z2 - z1, min=0.0)
                nh = torch.clamp(y2 - y1, min=0.0)
                nw = torch.clamp(x2 - x1, min=0.0)
                if nd * nh * nw == 0:
                    continue
                percent = nw * nh * nd / (each_label[3] * each_label[4] * each_label[5])
                if (percent > 0.1) and (nw * nh * nd >= 15):
                    bbox = torch.from_numpy(
                        np.array(
                            [
                                float(z1 + 0.5 * nd),
                                float(y1 + 0.5 * nh),
                                float(x1 + 0.5 * nw),
                                float(nd),
                                float(nh),
                                float(nw),
                                0,
                            ]
                        )
                    ).to(device)
                    bbox_annotation_target.append(bbox.view(1, 7))
                else:
                    mask_ignore[
                        j,
                        0,
                        int(z1) : int(torch.ceil(z2)),
                        int(y1) : int(torch.ceil(y2)),
                        int(x1) : int(torch.ceil(x2)),
                    ] = -1
            if len(bbox_annotation_target) > 0:
                bbox_annotation_target = torch.cat(bbox_annotation_target, 0)
                annotations_new[j, : len(bbox_annotation_target)] = (
                    bbox_annotation_target
                )
        # ctr_z, ctr_y, ctr_x, d, h, w, (0 or -1)
        return annotations_new, mask_ignore

    @staticmethod
    def bbox_iou(box1, box2, DIoU=True, eps=1e-7):
        def zyxdhw2zyxzyx(box, dim=-1):
            ctr_zyx, dhw = torch.split(box, 3, dim)
            z1y1x1 = ctr_zyx - dhw / 2
            z2y2x2 = ctr_zyx + dhw / 2
            return torch.cat((z1y1x1, z2y2x2), dim)  # zyxzyx bbox

        box1 = zyxdhw2zyxzyx(box1)
        box2 = zyxdhw2zyxzyx(box2)
        # Get the coordinates of bounding boxes
        b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = box1.chunk(6, -1)
        b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = box2.chunk(6, -1)
        w1, h1, d1 = b1_x2 - b1_x1, b1_y2 - b1_y1, b1_z2 - b1_z1
        w2, h2, d2 = b2_x2 - b2_x1, b2_y2 - b2_y1, b2_z2 - b2_z1

        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
            b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        ).clamp(0) * (b1_z2.minimum(b2_z2) - b1_z1.maximum(b2_z1)).clamp(0) + eps

        # Union Area
        union = w1 * h1 * d1 + w2 * h2 * d2 - inter

        # IoU
        iou = inter / union
        if DIoU:
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(
                b2_x1
            )  # convex (smallest enclosing box) width
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
            cd = b1_z2.maximum(b2_z2) - b1_z1.minimum(b2_z1)  # convex depth
            c2 = cw**2 + ch**2 + cd**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
                + +((b2_z1 + b2_z2 - b1_z1 - b1_z2) ** 2)
            ) / 4  # center dist ** 2
            return iou - rho2 / c2  # DIoU
        return iou  # IoU

    @staticmethod
    def bbox_decode(anchor_points, pred_offsets, pred_shapes, stride_tensor, dim=-1):
        c_zyx = (anchor_points + pred_offsets) * stride_tensor
        return torch.cat((c_zyx, 2 * pred_shapes), dim)  # zyxdhw bbox

    @staticmethod
    def get_pos_target(
        annotations, anchor_points, stride, spacing, topk=7, ignore_ratio=26
    ):
        batchsize, num, _ = annotations.size()
        mask_gt = annotations[:, :, -1].clone().gt_(-1)
        ctr_gt_boxes = annotations[:, :, :3] / stride  # z0, y0, x0
        shape = annotations[:, :, 3:6] / 2  # half d h w
        sp = torch.from_numpy(spacing).to(ctr_gt_boxes.device).view(1, 1, 1, 3)
        # distance (b, n_max_object, anchors)
        distance = -(
            ((ctr_gt_boxes.unsqueeze(2) - anchor_points.unsqueeze(0)) * sp)
            .pow(2)
            .sum(-1)
        )
        _, topk_inds = torch.topk(
            distance, (ignore_ratio + 1) * topk, dim=-1, largest=True, sorted=True
        )
        mask_topk = F.one_hot(topk_inds[:, :, :topk], distance.size()[-1]).sum(-2)
        mask_ignore = -1 * F.one_hot(topk_inds[:, :, topk:], distance.size()[-1]).sum(
            -2
        )
        mask_pos = mask_topk * mask_gt.unsqueeze(-1)
        mask_ignore = mask_ignore * mask_gt.unsqueeze(-1)
        gt_idx = mask_pos.argmax(-2)
        batch_ind = torch.arange(
            end=batchsize, dtype=torch.int64, device=ctr_gt_boxes.device
        )[..., None]
        gt_idx = gt_idx + batch_ind * num
        target_ctr = ctr_gt_boxes.view(-1, 3)[gt_idx]
        target_offset = target_ctr - anchor_points
        target_shape = shape.view(-1, 3)[gt_idx]
        target_bboxes = annotations[:, :, :-1].view(-1, 6)[gt_idx]
        target_scores, _ = torch.max(mask_pos, 1)
        mask_ignore, _ = torch.min(mask_ignore, 1)
        del target_ctr, distance, mask_topk
        return (
            target_offset,
            target_shape,
            target_bboxes,
            target_scores.unsqueeze(-1),
            mask_ignore.unsqueeze(-1),
        )


def make_anchors(feat, input_size, grid_cell_offset=0):
    """Generate anchors from a feature."""
    assert feat is not None
    dtype, device = feat.dtype, feat.device
    _, _, d, h, w = feat.shape
    strides = (
        torch.tensor([input_size[0] / d, input_size[1] / h, input_size[2] / w])
        .type(dtype)
        .to(device)
    )
    sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
    sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
    sz = torch.arange(end=d, device=device, dtype=dtype) + grid_cell_offset  # shift z
    anchor_points = torch.cartesian_prod(sz, sy, sx)
    stride_tensor = strides.repeat(d * h * w, 1)
    return anchor_points, stride_tensor


def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            total_norm += p.grad.detach().data.norm(2).item() ** 2
    total_norm = total_norm**0.5
    return total_norm
