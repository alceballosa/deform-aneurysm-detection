import copy
import math
import pickle
import random
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils.events import get_event_storage
from src.dataset.split_comb import SplitComb
from src.models.box_utils import nms_3D
from src.models.layers.conv_layers import (
    ClsRegHead,
    ConvBlock,
    DownsamplingConvBlock,
    LayerBasic,
    UpsamplingDeconvBlock,
)
from src.models.parq.box_processor import BoxProcessor
from src.models.parq.generic_mlp import GenericMLP
from src.models.parq.nms import nms
from src.models.parq.parq_matcher import HungarianMatcherModified
from src.models.parq.parq_utils import get_3d_corners
from src.models.parq.tokenizer import ImageSeqTokenizer
from src.models.parq.transformer_parq import Transformer
from src.utils.losses import (
    bbox_iou_loss,
    focal_loss,
    no_targets_cross_entropy_loss,
    no_targets_focal_loss,
)
from tqdm import tqdm


def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@META_ARCH_REGISTRY.register()
class CPM_PARQ(nn.Module):
    @configurable
    def __init__(
        self,
        cfg,
        n_channels=1,
        n_blocks=[2, 3, 3, 3],
        n_filters=[64, 96, 128, 160],
        stem_filters=32,
        norm_type="BN",
        head_norm="BN",
        act_type="ReLU",
        post_unet_scale_ratio=[1 / 4, 1 / 4, 1 / 4],
        se=False,
        first_stride=(2, 2, 2),
        patch_size_3d=[2, 2, 2],
        embed_dim=256,
        detection_loss=None,
        detection_postprocess=None,
        device=None,
        loss_weights=dict(
            cls_w=1.0,
            shape_w=1.0,
            offset_w=1.0,
            iou_w=1.0,
        ),
        use_pretrained_unet_encoder=False,
        path_unet_weights="",
    ):
        super(CPM_PARQ, self).__init__()
        self.cfg = cfg

        self.detection_loss = detection_loss
        self.postprocess = detection_postprocess
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

        # pretrained UNET
        self.use_pretrained_unet_encoder = use_pretrained_unet_encoder
        self.path_unet_weights = path_unet_weights
        self.frozen_parameters_list = []

        # fmt: off
        self.in_conv = ConvBlock(n_channels, stem_filters, stride=1, norm_type=norm_type, act_type=act_type)
        self.in_dw = ConvBlock(stem_filters, n_filters[0], stride=first_stride, norm_type=norm_type, act_type=act_type)

        self.block1 = LayerBasic(n_blocks[0], n_filters[0], n_filters[0], norm_type=norm_type, act_type=act_type, se=se)
        self.block1_dw = DownsamplingConvBlock(n_filters[0], n_filters[1], norm_type=norm_type, act_type=act_type)

        self.block2 = LayerBasic(n_blocks[1], n_filters[1], n_filters[1], norm_type=norm_type, act_type=act_type, se=se)
        self.block2_dw = DownsamplingConvBlock(n_filters[1], n_filters[2], norm_type=norm_type, act_type=act_type)

        self.block3 = LayerBasic(n_blocks[2], n_filters[2], n_filters[2], norm_type=norm_type, act_type=act_type, se=se)
        self.block3_dw = DownsamplingConvBlock(n_filters[2], n_filters[3], norm_type=norm_type, act_type=act_type)

        self.block4 = LayerBasic(n_blocks[3], n_filters[3], n_filters[3], norm_type=norm_type, act_type=act_type, se=se)

        self.block33_up = UpsamplingDeconvBlock(n_filters[3], n_filters[2], norm_type=norm_type, act_type=act_type)
        self.block33_res = LayerBasic(1, n_filters[2], n_filters[2], norm_type=norm_type, act_type=act_type, se=se)
        self.block33 = LayerBasic(2, n_filters[2] * 2, n_filters[2], norm_type=norm_type, act_type=act_type, se=se)

        self.block22_up = UpsamplingDeconvBlock(n_filters[2], n_filters[1], norm_type=norm_type, act_type=act_type)
        self.block22_res = LayerBasic(1, n_filters[1], n_filters[1], norm_type=norm_type, act_type=act_type, se=se)
        self.block22 = LayerBasic(2, n_filters[1] * 2, n_filters[1], norm_type=norm_type, act_type=act_type, se=se)
        # fmt: on

        # scale factor from original label scale to feature map volume scale for eval/loss calc
        self.post_unet_scale_ratio = torch.Tensor(post_unet_scale_ratio).to(device)
        self.patch_size_3d = torch.Tensor(patch_size_3d).to(device)

        self.feature_to_volume_scale_ratio = (
            self.post_unet_scale_ratio / self.patch_size_3d
        )
        size_t, size_h, size_w = patch_size_3d

        self.tokenizer = ImageSeqTokenizer(
            in_channels=n_filters[1] * size_t * size_h * size_w,
            out_channels=embed_dim,
            patch_size=patch_size_3d,
        )

        self.query_ref_pos = torch.nn.Embedding(
            self.cfg.MODEL.PARQ_MODEL.NUM_QUERIES, 3
        )

        self.transformer = Transformer(
            dec_dim=cfg.MODEL.PARQ_MODEL.DECODER.DEC_DIM,
            queries_dim=cfg.MODEL.PARQ_MODEL.DECODER.QUERIES_DIM,
            dec_heads=cfg.MODEL.PARQ_MODEL.DECODER.DEC_HEADS,
            dec_ffn_dim=cfg.MODEL.PARQ_MODEL.DECODER.DEC_FFN_DIM,
            dec_layers=cfg.MODEL.PARQ_MODEL.DECODER.DEC_LAYERS,
            dropout_rate=cfg.MODEL.PARQ_MODEL.DECODER.DROPOUT_RATE,
            share_weights=cfg.MODEL.PARQ_MODEL.DECODER.SHARE_WEIGHTS,
        )

        self.build_mlp_heads(
            self.num_semcls,
            cfg.MODEL.PARQ_MODEL.DECODER.DEC_DIM,
            cfg.MODEL.PARQ_MODEL.HEADS.MEAN_SIZE_PATH,
            mlp_dropout=0.3,
        )

        self.box_processor = BoxProcessor(
            self.num_semcls, cfg.MODEL.PARQ_MODEL.HEADS.MEAN_SIZE_PATH
        )
        self.matcher = HungarianMatcherModified(cost_class=2, cost_bbox=0.25)
        self.transformer.decoder.mlp_heads = self.mlp_heads
        self.transformer.decoder.box_processor = self.box_processor

        if self.use_pretrained_unet_encoder and self.path_unet_weights != "":
            self.load_pretrained_unet_weights()
            self.freeze_parameters()

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
        parq_cfg = cfg.MODEL.PARQ_MODEL
        loss_cfg = cfg.MODEL.CONV_MODEL.DET_LOSS
        parq_loss_cfg = cfg.MODEL.PARQ_MODEL.PARQ_LOSS
        post_process_cfg = cfg.MODEL.CONV_MODEL.DET_POSTPROCESS
        return {
            "cfg": cfg,
            "device": cfg.MODEL.DEVICE,
            "n_channels": cfg.DATA.N_CHANNELS,
            "n_blocks": conv_cfg.N_BLOCKS,
            "n_filters": conv_cfg.N_FILTERS,
            "stem_filters": conv_cfg.STEM_FILTERS,
            "norm_type": conv_cfg.NORM,
            "head_norm": conv_cfg.HEAD_NORM,
            "act_type": conv_cfg.ACT,
            "se": conv_cfg.SE,
            "first_stride": conv_cfg.FIRST_STRIDE,
            "post_unet_scale_ratio": conv_cfg.POST_UNET_SCALE_RATIO,
            "use_pretrained_unet_encoder": conv_cfg.USE_PRETRAINED_UNET_ENCODER,
            "path_unet_weights": conv_cfg.PRETRAINED_UNET_ENCODER_PATH,
            "patch_size_3d": parq_cfg.PATCH_SIZE_3D,
            "embed_dim": parq_cfg.EMBED_DIM,
            "detection_loss": DetectionLoss(
                crop_size=cfg.DATA.PATCH_SIZE,
                topk=loss_cfg.TOPK,
                spacing=cfg.DATA.SPACING,
                ignore_ratio=loss_cfg.IGNORE_RATIO,
                alpha=loss_cfg.FOCAL_ALPHA,
                gamma=loss_cfg.FOCAL_GAMMA,
                num_neg=loss_cfg.NUM_NEG,
                num_hard=loss_cfg.NUM_HARD_NEG,
                ratio=loss_cfg.NEG_RATIO,
            ),
            "detection_postprocess": DetectionPostprocess(
                topk=post_process_cfg.TOPK,
                threshold=post_process_cfg.SCORE_THRESHOLD,
                nms_threshold=post_process_cfg.NMS_THRESHOLD,
                num_topk=post_process_cfg.NMS_TOPK,
                crop_size=cfg.DATA.PATCH_SIZE,
            ),
            "loss_weights": {
                "cls_w": parq_loss_cfg.CLS_W,
                "shape_w": parq_loss_cfg.SHAPE_W,
                "offset_w": parq_loss_cfg.OFFSET_W,
                "iou_w": loss_cfg.IOU_W,
            },
        }

    def build_mlp_heads(
        self,
        num_semcls,
        decoder_dim,
        mean_size_path=None,
        mlp_dropout=None,
    ):
        """
        build mlp head to regress the parameters of the boxes
        """
        # TODO: check whether it is necessary to have the small mlp funcs.
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="ln",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=0.0,
            input_dim=decoder_dim,
        )

        # semantic class of the box, add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=num_semcls + 1)
        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)

        # generate one head for every layer if required
        if not self.cfg.MODEL.PARQ_MODEL.HEADS.SHARED:
            semcls_head = _get_clones(
                semcls_head, self.cfg.MODEL.PARQ_MODEL.DECODER.DEC_LAYERS
            )
            center_head = _get_clones(
                center_head, self.cfg.MODEL.PARQ_MODEL.DECODER.DEC_LAYERS
            )
            size_head = _get_clones(
                size_head, self.cfg.MODEL.PARQ_MODEL.DECODER.DEC_LAYERS
            )

        self.mlp_heads = torch.nn.ModuleDict(
            [
                ("sem_cls_head", semcls_head),
                ("center_head", center_head),
                ("size_head", size_head),
            ]  # type: ignore
        )
        self.box_processor = BoxProcessor(num_semcls, mean_size_path)

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

    def load_pretrained_unet_weights(self):
        """
        Loads weights from a pretrained UNET model into the encoder
        to make the training more stable.
        """
        unet_weights = torch.load(
            self.path_unet_weights, map_location=torch.device("cpu")
        )["model"]
        unet_keys = list(unet_weights.keys())
        remapped_unet_keys = {
            unet_key.replace("model.", ""): unet_weights[unet_key]
            for unet_key in unet_keys
        }
        remapped_unet_keys = OrderedDict(remapped_unet_keys)
        model_keys = list(self.state_dict().keys())
        # remove keys not in model_keys
        remapped_unet_keys = {
            model_key: remapped_unet_keys[model_key]
            for model_key in model_keys
            if model_key in remapped_unet_keys
        }
        list_unet_keys = list(remapped_unet_keys.keys())
        self.frozen_parameters_list.extend(list_unet_keys)
        self.load_state_dict(remapped_unet_keys, strict=False)
        print(
            "\nLoaded pretrained UNET weights"
            + f"from {self.path_unet_weights} successfully"
        )
        return

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
            return self._forward_train(input_batch)
        try:
            return self._forward_eval(input_batch)
        except MemoryError as e:
            print("CUDA out of memory.Trying to clear cache")
            torch.cuda.empty_cache()
            return self._forward_eval(input_batch)

    def _forward_train(self, input_batch):
        if self.cfg.CUSTOM.TRACKING_GRADIENT_NORM:
            get_event_storage().put_scalar("grad_norm", get_gradient_norm(self))
        x = self.preprocess_train_input(input_batch)
        targets = self.preprocess_train_labels(input_batch)
        box_prediction_list, attn_list = self._forward_network(x)
        loss_dict = self.compute_losses(box_prediction_list, targets)
        return loss_dict

    def _forward_eval(self, input_batch):
        """
        ! assume batch_size is alway 1
        """

        assert len(input_batch) == 1
        bs = self.cfg.TEST.PATCHES_PER_ITER
        patches, nzhw, splits_boxes = self.split_com.split(input_batch[0]["image"])

        outputs = []
        attn_outputs = []
        # query_locs = []
        for i in tqdm(range(int(math.ceil(len(patches) / bs)))):
            # preprocess batch
            end = (i + 1) * bs
            if end > len(patches):
                end = len(patches)
            # batch_data = np.concatenate(patches[i * bs : end], axis=0)
            # batch_data = torch.tensor(batch_data, device=self.device)
            batch_data = torch.cat(patches[i * bs : end], dim=0).to(self.device)
            batch_data = self.normalize(batch_data)
            prediction_dicts, attn_list = self._forward_network(batch_data)

            for prediction_dict in prediction_dicts:
                for key in prediction_dict:
                    prediction_dict[key] = prediction_dict[key].detach().to("cpu")
            dets = self.parse_pred(prediction_dicts[-1]).detach().to("cpu")
            del batch_data
            torch.cuda.empty_cache()
            outputs.append(dets)
            attn_outputs.append(attn_list)
            # TODO: use query locs!!
            # query_locs.append(
            #     [
            #         {
            #             "center_unnormalized": prediction_dicts[i][
            #                 "center_unnormalized"
            #             ]
            #             .detach()
            #             .to("cpu")
            #         }
            #         for i in range(len(prediction_dicts))
            #     ]
            # )

        print("Finished forward pass")
        # post process outputs
        outputs = torch.cat(outputs, dim=0)
        outputs = self.split_com.combine(outputs, nzhw)
        outputs = outputs.view(-1, 8)

        object_ids = outputs[:, 0] != -1
        outputs = outputs[object_ids]
        if len(outputs) > 0:
            keep = nms_3D(outputs[:, 1:], overlap=0.05, top_k=self.cfg.TEST.NMS_TOPK)
            outputs = outputs[keep]

        return outputs

    def _forward_network(self, x):
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
        x = self.in_conv(x)
        x = self.in_dw(x)

        x1 = self.block1(x)
        x = self.block1_dw(x1)

        x2 = self.block2(x)
        x = self.block2_dw(x2)

        x3 = self.block3(x)
        x = self.block3_dw(x3)

        x = self.block4(x)

        # "decode"
        x = self.block33_up(x)
        x3 = self.block33_res(x3)

        x = torch.cat([x, x3], dim=1)
        x = self.block33(x)

        x = self.block22_up(x)
        x2 = self.block22_res(x2)

        x = torch.cat([x, x2], dim=1)
        x = self.block22(x)
        global_tokens = self.tokenizer(x)
        pre_tokenization_shape = list(x.size())

        box_prediction_list, attn_list = self.transformer.forward(
            global_tokens, self.query_ref_pos.weight, pre_tokenization_shape
        )

        return box_prediction_list, attn_list

    def compute_losses(self, out_dict_list, targets):
        """
        input:
            out_dict_list:  predicted box3d parameters
            obbs_padded:    target box3d parameters
        output:
            loss
        """
        # assert targets.ndim == 3, f"{targets.shape}"
        loss_total = (
            out_dict_list[-1]["size_unnormalized"].sum()
            * out_dict_list[-1]["center_unnormalized"].sum()
            * out_dict_list[-1]["pred_logits"].sum()
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
        for out_dict in out_dict_list:
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
                                classes_pred=out_dict["pred_logits"][i],
                                alpha=self.cfg.MODEL.PARQ_MODEL.PARQ_LOSS.FOCAL_ALPHA,
                                gamma=self.cfg.MODEL.PARQ_MODEL.PARQ_LOSS.FOCAL_GAMMA,
                            )
                            * self.loss_weights["cls_w"]
                        )
                    else:
                        cat_loss = (
                            no_targets_cross_entropy_loss(
                                out_dict["pred_logits"][i], self.class_weight
                            )
                            * self.loss_weights["cls_w"]
                        )
                    loss_dict["cat_loss"] += cat_loss
                elif len(matched_indices[i][0]) != 0:
                    valid_bs_loc_shape += 1
                    # center loss: l1, applied only when matched to something
                    center_predict = out_dict["center_unnormalized"][i][
                        matched_indices[i][0]
                    ]
                    center_target = targets[i]["center"][matched_indices[i][1]]
                    center_loss = (center_predict - center_target).abs().mean()
                    center_loss *= self.loss_weights["offset_w"]
                    loss_total += center_loss
                    loss_dict["center_loss"] += center_loss

                    # size loss: l1, applied only when matched to something
                    size_predict = out_dict["size_unnormalized"][i][
                        matched_indices[i][0]
                    ]
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
                            out_dict["pred_logits"][i],
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
                            out_dict["pred_logits"].shape[1:2],
                            self.num_semcls,
                            dtype=torch.int64,
                            device=out_dict["pred_logits"].device,
                        )
                        classes_target[matched_indices[i][0]] = matched_classes_target
                        # TODO: review punish mask how it works and looks
                        if punish_mask is not None:
                            cross_entropy = torch.nn.CrossEntropyLoss(
                                self.class_weight.to(matched_classes_target.device),
                                reduction="none",
                            )
                            cat_loss = cross_entropy(
                                out_dict["pred_logits"][i], classes_target
                            )

                            cat_loss = (cat_loss * punish_mask[i]).sum() / punish_mask[
                                i
                            ].sum()
                        else:
                            cross_entropy = torch.nn.CrossEntropyLoss(
                                self.class_weight.to(matched_classes_target.device)
                            )
                            cat_loss = cross_entropy(
                                out_dict["pred_logits"][i], classes_target
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
        size_predict = pred_dict["size_unnormalized"]
        center_predict = pred_dict["center_unnormalized"]
        logits = pred_dict["sem_cls_prob"]
        labels = torch.argmax(logits, dim=-1)
        bs = logits.shape[0]

        n_queries = logits.shape[1]
        center_predict_flat = center_predict.view(bs * n_queries, 3)
        size_predict_flat = size_predict.view(bs * n_queries, 3)
        corners = get_3d_corners(center_predict_flat, size_predict_flat)
        corners = corners.view(bs, n_queries, 8, 3)
        # TODO: filter out of bounds
        valid = torch.ones_like(center_predict[..., 0]).bool()
        pred_mask = nms(corners, labels, logits, self.num_semcls, 0.1, "nms_3d_faster")
        pred_mask = torch.tensor(pred_mask).to(valid.device) & valid
        dets = torch.ones((bs, n_queries, 8)) * -1
        for j in range(bs):
            for i in range(n_queries):
                if pred_mask[j, i]:
                    dets[j, i, 0] = 1
                    dets[j, i, 1] = logits[j, i, 0]  # score for tgt class
                    dets[j, i, 2:5] = center_predict[
                        j, i
                    ] / self.feature_to_volume_scale_ratio.to("cpu")
                    dets[j, i, 5:8] = size_predict[
                        j, i
                    ] / self.feature_to_volume_scale_ratio.to("cpu")

        return dets

    def preprocess_train_input(self, input_batch):
        all_samples = sum([x["samples"] for x in input_batch], [])
        imgs = [s["image"] for s in all_samples]
        imgs = torch.stack(imgs, dim=0)
        imgs = imgs.to(self.device)
        # imgs = np.stack(imgs)
        # imgs = torch.tensor(imgs, device=self.device)
        imgs = self.normalize(imgs)
        return imgs

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
                torch.cat(center, dim=0).to(self.device)
                * self.feature_to_volume_scale_ratio
                if len(center) > 0
                else []
            )

            size = (
                torch.cat(size, dim=0).to(self.device)
                * self.feature_to_volume_scale_ratio
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

    def normalize(self, x: torch.Tensor):
        min_value, max_value = self.cfg.DATA.WINDOW
        x.clamp_(min=min_value, max=max_value)
        x -= (min_value + max_value) / 2
        x /= (max_value - min_value) / 2
        return x


class DetectionLoss(nn.Module):
    def __init__(
        self,
        crop_size=[64, 128, 128],
        topk=7,
        spacing=[1.0, 1.0, 1.0],
        ignore_ratio=26,
        alpha=0.75,
        gamma=2.0,
        num_neg=10000,
        num_hard=100,
        ratio=100,
    ):
        super(DetectionLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_hard = num_hard
        self.ratio = ratio
        self.num_neg = num_neg
        self.crop_size = crop_size
        self.topk = topk
        self.spacing = np.array(spacing)
        self.ignore_ratio = ignore_ratio

    def forward(self, output, annotations, device):
        Cls = output["Cls"]
        Shape = output["Shape"]
        Offset = output["Offset"]
        batch_size = Cls.size()[0]
        target_mask_ignore = torch.zeros(Cls.size()).to(device)
        # view shape
        pred_scores = Cls.view(batch_size, 1, -1)
        pred_shapes = Shape.view(batch_size, 3, -1)
        pred_offsets = Offset.view(batch_size, 3, -1)
        # (b, num_points, 1|3)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_shapes = pred_shapes.permute(0, 2, 1).contiguous()
        pred_offsets = pred_offsets.permute(0, 2, 1).contiguous()
        # process annotations
        process_annotations, target_mask_ignore = self.target_preprocess(
            annotations, device, self.crop_size, target_mask_ignore
        )
        target_mask_ignore = target_mask_ignore.view(batch_size, 1, -1)
        target_mask_ignore = target_mask_ignore.permute(0, 2, 1).contiguous()
        # generate center points. Only support single scale feature
        anchor_points, stride_tensor = make_anchors(Cls, self.crop_size, 0)  # z, y, x
        # predict bboxes (zyxdhw)
        pred_bboxes = self.bbox_decode(
            anchor_points, pred_offsets, pred_shapes, stride_tensor
        )
        # assigned points and targets (target bboxes zyxdhw)
        (
            target_offset,
            target_shape,
            target_bboxes,
            target_scores,
            mask_ignore,
        ) = self.get_pos_target(
            annotations=process_annotations,
            anchor_points=anchor_points,
            stride=stride_tensor[0].view(1, 1, 3),
            spacing=self.spacing,
            topk=self.topk,
            ignore_ratio=self.ignore_ratio,
        )
        # merge mask ignore
        mask_ignore = mask_ignore.bool() | target_mask_ignore.bool()
        fg_mask = target_scores.squeeze(-1).bool()
        classification_losses = self.cls_loss(
            pred=pred_scores,
            target=target_scores,
            mask_ignore=mask_ignore.int(),
            alpha=self.alpha,
            gamma=self.gamma,
            num_neg=self.num_neg,
            num_hard=self.num_hard,
            ratio=self.ratio,
        )
        if fg_mask.sum() == 0:
            # the network still has zero gradient instead of None
            reg_losses = 0.0 * pred_shapes.sum()
            offset_losses = 0.0 * pred_offsets.sum()
            iou_losses = 0.0 * pred_bboxes.sum()
        else:
            reg_losses = torch.abs(pred_shapes[fg_mask] - target_shape[fg_mask]).mean()
            offset_losses = torch.abs(
                pred_offsets[fg_mask] - target_offset[fg_mask]
            ).mean()
            iou_losses = -(
                self.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            ).mean()

        return classification_losses, reg_losses, offset_losses, iou_losses

    @staticmethod
    def cls_loss(
        pred,
        target,
        mask_ignore,
        alpha=0.75,
        gamma=2.0,
        num_neg=10000,
        num_hard=100,
        ratio=100,
    ):
        classification_losses = []
        batch_size = pred.shape[0]
        for j in range(batch_size):
            pred_b = pred[j]
            target_b = target[j]
            mask_ignore_b = mask_ignore[j]
            cls_prob = torch.sigmoid(pred_b.detach())
            cls_prob = torch.clamp(cls_prob, 1e-4, 1.0 - 1e-4)
            alpha_factor = torch.ones(pred_b.shape).to(pred_b.device) * alpha
            alpha_factor = torch.where(
                torch.eq(target_b, 1.0), alpha_factor, 1.0 - alpha_factor
            )
            focal_weight = torch.where(
                torch.eq(target_b, 1.0), 1.0 - cls_prob, cls_prob
            )
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = F.binary_cross_entropy_with_logits(pred_b, target_b, reduction="none")
            num_positive_pixels = torch.sum(target_b == 1)
            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.eq(mask_ignore_b, 0), cls_loss, 0)
            record_targets = target_b.clone()
            if num_positive_pixels > 0:
                FN_weights = 4.0  # 10.0  for ablation study
                FN_index = torch.lt(cls_prob, 0.8) & (record_targets == 1)  # 0.9
                cls_loss[FN_index == 1] = FN_weights * cls_loss[FN_index == 1]
                Negative_loss = cls_loss[record_targets == 0]
                Positive_loss = cls_loss[record_targets == 1]
                neg_idcs = random.sample(
                    range(len(Negative_loss)), min(num_neg, len(Negative_loss))
                )
                Negative_loss = Negative_loss[neg_idcs]
                _, keep_idx = torch.topk(Negative_loss, ratio * num_positive_pixels)
                Negative_loss = Negative_loss[keep_idx]
                Positive_loss = Positive_loss.sum()
                Negative_loss = Negative_loss.sum()
                cls_loss = Positive_loss + Negative_loss

            else:
                Negative_loss = cls_loss[record_targets == 0]
                neg_idcs = random.sample(
                    range(len(Negative_loss)), min(num_neg, len(Negative_loss))
                )
                Negative_loss = Negative_loss[neg_idcs]
                assert len(Negative_loss) > num_hard
                _, keep_idx = torch.topk(Negative_loss, num_hard)
                Negative_loss = Negative_loss[keep_idx]
                Negative_loss = Negative_loss.sum()
                cls_loss = Negative_loss
            classification_losses.append(
                cls_loss / torch.clamp(num_positive_pixels.float(), min=1.0)
            )
        return torch.mean(torch.stack(classification_losses))

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
            for s in range(len(bbox_annotation_boxes)):
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
                annotations_new[
                    j, : len(bbox_annotation_target)
                ] = bbox_annotation_target
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


class DetectionPostprocess(nn.Module):
    def __init__(
        self,
        topk=60,
        threshold=0.15,
        nms_threshold=0.05,
        num_topk=20,
        crop_size=[64, 96, 96],
    ):
        super(DetectionPostprocess, self).__init__()
        self.topk = topk
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.nms_topk = num_topk
        self.crop_size = crop_size

    @staticmethod
    def bbox_decode(anchor_points, pred_offsets, pred_shapes, stride_tensor, dim=-1):
        c_zyx = (anchor_points + pred_offsets) * stride_tensor
        return torch.cat((c_zyx, 2 * pred_shapes), dim)  # zyxdhw bbox

    def forward(self, output, device):
        Cls = output["Cls"]
        Shape = output["Shape"]
        Offset = output["Offset"]
        batch_size = Cls.size()[0]
        dets = (-torch.ones((batch_size, self.topk, 8))).to(device)
        anchor_points, stride_tensor = make_anchors(Cls, self.crop_size, 0)
        # view shape
        pred_scores = Cls.view(batch_size, 1, -1)
        pred_shapes = Shape.view(batch_size, 3, -1)
        pred_offsets = Offset.view(batch_size, 3, -1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous().sigmoid()
        pred_shapes = pred_shapes.permute(0, 2, 1).contiguous()
        pred_offsets = pred_offsets.permute(0, 2, 1).contiguous()

        # recale to input_size
        pred_bboxes = self.bbox_decode(
            anchor_points, pred_offsets, pred_shapes, stride_tensor
        )
        topk_scores, topk_idxs = torch.topk(
            pred_scores.squeeze(), self.topk, dim=-1, largest=True
        )
        dets = (-torch.ones((batch_size, self.topk, 8))).to(device)
        for j in range(batch_size):
            topk_score = topk_scores[j]
            topk_idx = topk_idxs[j]
            keep_box_mask = topk_score > self.threshold
            keep_box_n = keep_box_mask.sum()
            if keep_box_n > 0:
                det = (-torch.ones((torch.sum(keep_box_n), 8))).to(device)
                keep_topk_score = topk_score[keep_box_mask]
                keep_topk_idx = topk_idx[keep_box_mask]
                for k, idx, score in zip(
                    range(keep_box_n), keep_topk_idx, keep_topk_score
                ):
                    det[k, 0] = 1
                    det[k, 1] = score
                    det[k, 2:] = pred_bboxes[j][idx]
                # 1, prob, ctr_z, ctr_y, ctr_x, d, h, w
                keep = nms_3D(
                    det[:, 1:], overlap=self.nms_threshold, top_k=self.nms_topk
                )
                dets[j][: len(keep)] = det[keep.long()]
        return dets


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
