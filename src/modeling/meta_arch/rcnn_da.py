import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
#from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, build_roi_heads

from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n

import logging
import math

from ..roi_heads import build_roi_heads 
from ..domain_alignment import build_da_head

__all__ = ["DARCNN"]


@META_ARCH_REGISTRY.register()
class DARCNN(nn.Module):
    """
    Cross-domain detection
    """

    def __init__(self, cfg):
        super().__init__()
        # pylint: disable=no-member
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.from_config(cfg)

        # print(ROI_HEADS_REGISTRY)
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        self.da_head = build_da_head(cfg, self.backbone.output_shape())

        self.to(self.device)

    def from_config(self, cfg):
        # only train/eval the da branch for debugging.
        # self.da_only = cfg.MODEL.DA.DA_ONLY
        self.da_name = cfg.MODEL.DA.NAME
        self.pix_feat_level = cfg.MODEL.DA.PIX_FEAT_LEVEL  # res2
        self.img_feat_level = cfg.MODEL.DA.IMG_FEAT_LEVEL  # use res4 for now

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def forward(self, batched_inputs, domain='source'):
        """
        Args:
            batched_inputs
            labeled (string, optional): whether has ground-truth label. Default: 'source'
        Returns:
            losses
        """
        if not self.training:
            return self.inference(batched_inputs)
        
        losses = {}

        # start detection
        images = self.preprocess_image(batched_inputs)
        # only load gt instances for source domain data
        if "instances" in batched_inputs[0] and domain == 'source':
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        # print(images.tensor.size(), images.image_sizes)

        features = self.backbone(images.tensor) 
        # print(features['res2'].size()) # torch.Size([1, 256, 164, 334])
        # print(features['res4'].size()) # torch.Size([1, 1024, 41, 84])

        # img features to img-level domain discriminator
        if self.da_name == "build_sw_da_head":
            da_losses = self.da_head(features, self.pix_feat_level, self.img_feat_level, domain)
        else:
            da_losses = self.da_head(features, self.img_feat_level, domain)
        losses.update(da_losses)
        
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances, domain =='source'
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}
        # print(len(proposals), proposals[0])

        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances, domain =='source'
        )

        
        losses.update(detector_losses)
        losses.update(proposal_losses)


        for k, v in losses.items():
            try:
                assert math.isnan(v) == False, batched_inputs 
            except AssertionError as msg:
                print(f'{k} is nan? {math.isnan(v)}')


        return losses

    def det_inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]

            results, others = self.roi_heads(images, features, proposals, None)
            if isinstance(others, tuple):
                others, box_features = others

            else:
                box_features = None
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )
            box_features = None

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results, box_features
        else:
            return results, box_features

    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """ used for standard detectron2 test method"""
        results, _ = self.det_inference(
            batched_inputs, detected_instances, do_postprocess
        )
        return results

    def preprocess_image(self, batched_inputs):
        """normalize, pad and batch the input images"""
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images
