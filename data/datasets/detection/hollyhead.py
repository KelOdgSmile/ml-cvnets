#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from pycocotools.coco import COCO
import os
from typing import Optional, Tuple, Dict
import numpy as np
import math

from utils import logger
from cvnets.misc.anchor_generator import SSDAnchorGenerator
from cvnets.misc.match_prior import SSDMatcher

from ...transforms import image as tf
from ...datasets import BaseImageDataset, register_dataset


COCO_CLASS_LIST = ['head','not'
                   ]


@register_dataset(name="hollyhead", task="detection")
class HollyHeadDetection(BaseImageDataset):
    """
        Dataset class for the COCO Object detection

        Dataset structure should be something like this
        + coco
        + --- annotations
        + ------ *.json
        + --- images
        + ------ train2017
        + ---------- *.jpg
        + ------ val2017
        + ---------- *.jpg

    """
    def __init__(self, opts, is_training: Optional[bool] = True, is_evaluation: Optional[bool] = False):
        super(HollyHeadDetection, self).__init__(opts=opts, is_training=is_training, is_evaluation=is_evaluation)

        split = 'train' if is_training else 'test'
        year = 2017
        path = self.root

        img_path = os.path.join(os.path.join(path, 'images'), split)
        label_path = os.path.join(os.path.join(path, 'labels'), split)

        self.imgs = [item for item in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, item))]


        self.img_dir = img_path
        self.ann_dir = label_path
        self.ids = [i for i in range(len(self.img_dir))]

        self.num_classes = 2

        setattr(opts, "model.detection.n_classes", self.num_classes)


    def training_transforms(self, size: tuple, ignore_idx: Optional[int] = 255):
        # implement these functions in sub classes
        raise NotImplementedError

    def validation_transforms(self, size: tuple, *args, **kwargs):
        raise NotImplementedError

    def evaluation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = []
        if getattr(self.opts, "evaluation.detection.resize_input_images", False):
            aug_list.append(tf.Resize(opts=self.opts, size=size))

        aug_list.append(tf.NumpyToTensor(opts=self.opts))
        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        crop_size_h, crop_size_w, img_index = batch_indexes_tup

        if self.is_training:
            transform_fn = self.training_transforms(size=(crop_size_h, crop_size_w))
        elif self.is_evaluation:
            transform_fn = self.evaluation_transforms(size=(crop_size_h, crop_size_w))
        else: # same for validation and evaluation
            transform_fn = self.validation_transforms(size=(crop_size_h, crop_size_w))

        image_id = self.ids[img_index]

        image, img_name = self._get_image(image_id=image_id)
        boxes, labels = self._get_annotation(image_id=image_id)

        im_height, im_width = image.shape[:2]

        data = {
            "image": image,
            "box_labels": labels,
            "box_coordinates": boxes
        }

        if transform_fn is not None:
            data = transform_fn(data)

        new_data = {
            "image": data["image"],
            "label": {
                "box_labels": data["box_labels"],
                "box_coordinates": data["box_coordinates"],
                "image_id": image_id
            }
        }

        del data

        if self.is_evaluation:
            new_data["file_name"] = img_name
            new_data["im_width"] = im_width
            new_data["im_height"] = im_height

        return new_data

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        ann_file = self.imgs[image_id][:-4]+'.txt'
        # print(ann_file)
        ann_path = os.path.join(self.ann_dir, ann_file)
        txt_dat = np.loadtxt(ann_path,skiprows=1)
        # print(txt_dat, txt_dat.shape)
        if len(txt_dat.shape) <= 1:
            txt_dat = np.expand_dims(txt_dat, axis=0)
        # print(txt_dat, txt_dat.shape)
        labels = txt_dat[:, 0]
        boxes = txt_dat[:, 1:]
        # print(labels, boxes)
        # print(txt_dat)
        # print(boxes[0, :].tolist()[0])
        # print(self._xywh2xyxy(boxes[0, :].tolist()))
        boxes = np.array([self._xywh2xyxy(boxes[i, :].tolist()) for i in range(boxes.shape[0])],
                         np.float32).reshape((-1, 4))

        # else:
        #     labels = txt_dat[0]
        #     boxes = txt_dat[1:]
        #     boxes = np.array(self._xywh2xyxy(boxes.tolist()))
        # filter crowd annotations

        return boxes, np.array(labels,np.int64)

    def _xywh2xyxy(self, box):
        # print(box)
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def _get_image(self, image_id):
        ann_file = self.imgs[image_id]
        # print(ann_file)
        ann_path = os.path.join(self.img_dir, ann_file)

        image = self.read_image(ann_path)
        return image, ann_path


@register_dataset(name="hollyhead_ssd", task="detection")
class HollyHeadDetectionSSD(HollyHeadDetection):
    """
        Dataset class for the COCO Object detection using SSD
    """
    def __init__(self, opts, is_training: Optional[bool] = True, is_evaluation: Optional[bool] = False):
        super(HollyHeadDetectionSSD, self).__init__(
            opts=opts,
            is_training=is_training,
            is_evaluation=is_evaluation
        )

        anchors_aspect_ratio = getattr(opts, "model.detection.ssd.anchors_aspect_ratio", [[2, 3], [2, 3], [2]])
        output_strides = getattr(opts, "model.detection.ssd.output_strides", [8, 16, 32])

        if len(anchors_aspect_ratio) != len(output_strides):
            logger.error(
                "SSD model requires anchors to be defined for feature maps from each output stride. So,"
                "len(anchors_per_location) == len(output_strides). "
                "Got len(output_strides)={} and len(anchors_aspect_ratio)={}. "
                "Please specify correct arguments using following arguments: "
                "\n--model.detection.ssd.anchors-aspect-ratio "
                "\n--model.detection.ssd.output-strides".format(
                    len(output_strides),
                    len(anchors_aspect_ratio),
                )
            )

        self.output_strides = output_strides

        self.anchor_box_generator = SSDAnchorGenerator(
            output_strides=output_strides,
            aspect_ratios=anchors_aspect_ratio,
            min_ratio=getattr(opts, "model.detection.ssd.min_box_size", 0.1),
            max_ratio=getattr(opts, "model.detection.ssd.max_box_size", 1.05)
        )

        self.match_prior = SSDMatcher(
            center_variance=getattr(opts, "model.detection.ssd.center_variance", 0.1),
            size_variance=getattr(opts, "model.detection.ssd.size_variance", 0.2),
            iou_threshold=getattr(opts, "model.detection.ssd.iou_threshold", 0.5) # we use nms_iou_threshold during inference
        )

    def training_transforms(self, size: tuple, ignore_idx: Optional[int] = 255):
        aug_list = [
            #tf.RandomZoomOut(opts=self.opts),
            tf.SSDCroping(opts=self.opts),
            tf.PhotometricDistort(opts=self.opts),
            tf.RandomHorizontalFlip(opts=self.opts),
            tf.BoxPercentCoords(opts=self.opts),
            tf.Resize(opts=self.opts, size=size),
            tf.NumpyToTensor(opts=self.opts)
        ]

        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def validation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = [
            tf.BoxPercentCoords(opts=self.opts),
            tf.Resize(opts=self.opts, size=size),
            tf.NumpyToTensor(opts=self.opts)
        ]
        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def evaluation_transforms(self, size: tuple, *args, **kwargs):
        return self.validation_transforms(size=size)

    def get_anchors(self, crop_size_h, crop_size_w):
        anchors = []
        for output_stride in self.output_strides:
            if output_stride == -1:
                fm_width = fm_height = 1
            else:
                fm_width = int(math.ceil(crop_size_w / output_stride))
                fm_height = int(math.ceil(crop_size_h / output_stride))
            fm_anchor = (
                self.anchor_box_generator(
                    fm_height=fm_height,
                    fm_width=fm_width,
                    fm_output_stride=output_stride
                )
            )
            anchors.append(fm_anchor)
        anchors = torch.cat(anchors, dim=0)
        return anchors

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        crop_size_h, crop_size_w, img_index = batch_indexes_tup

        if self.is_training:
            transform_fn = self.training_transforms(size=(crop_size_h, crop_size_w))
        else: # same for validation and evaluation
            transform_fn = self.validation_transforms(size=(crop_size_h, crop_size_w))

        image_id = self.ids[img_index]

        image, img_fname = self._get_image(image_id=image_id)
        boxes, labels = self._get_annotation(image_id=image_id)

        data = {
            "image": image,
            "box_labels": labels,
            "box_coordinates": boxes
        }
        data = transform_fn(data)

        # convert to priors
        anchors = self.get_anchors(crop_size_h=crop_size_h, crop_size_w=crop_size_w)

        gt_coordinates, gt_labels = self.match_prior(
            gt_boxes_cor=data["box_coordinates"],
            gt_labels=data["box_labels"],
            reference_boxes_ctr=anchors
        )

        return {
            "image": data["image"],
            "label": {
                "box_labels": gt_labels,
                "box_coordinates": gt_coordinates
            }
        }

    def __repr__(self):
        from utils.tensor_utils import tensor_size_from_opts
        im_h, im_w = tensor_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self.training_transforms(size=(im_h, im_w))
        elif self.is_evaluation:
            transforms_str = self.evaluation_transforms(size=(im_h, im_w))
        else:
            transforms_str = self.validation_transforms(size=(im_h, im_w))

        return "{}(\n\troot={}\n\t is_training={}\n\tsamples={}\n\ttransforms={}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            len(self.ids),
            transforms_str
        )
