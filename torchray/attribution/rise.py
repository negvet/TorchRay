# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
This module provides an implementation of the *RISE* method of [RISE]_ for
saliency visualization. This is given by the :func:`rise` function, which
can be used as follows:

.. literalinclude:: ../examples/rise.py
    :language: python
    :linenos:

References:

    .. [RISE] V. Petsiuk, A. Das and K. Saenko
              *RISE: Randomized Input Sampling for Explanation of Black-box
              Models,*
              BMVC 2018,
              `<https://arxiv.org/pdf/1806.07421.pdf>`__.
"""

__all__ = ['rise', 'rise_class']

import math
import os
from typing import Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt

from dlib import find_min_global, find_max_global

from timeit import default_timer

import torch
import torch.nn.functional as F
import torchvision
from .common import resize_saliency

from torchray.benchmark.insertion_deletion import gkern
from torchray.benchmark.insertion_deletion import CausalMetric
from torchray.benchmark.insertion_deletion import auc


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
    'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]


def get_stats(a):
    print(a.min(), a.max(), a.mean())


def _upsample_reflect(x, size, interpolate_mode="bilinear"):
    r"""Upsample 4D :class:`torch.Tensor` with reflection padding.

    Args:
        x (:class:`torch.Tensor`): 4D tensor to interpolate.
        size (int or list or tuple of ints): target size
        interpolate_mode (str): mode to pass to
            :function:`torch.nn.functional.interpolate` function call
            (default: "bilinear").

    Returns:
        :class:`torch.Tensor`: upsampled tensor.
    """
    # Check and get input size.
    assert len(x.shape) == 4
    orig_size = x.shape[2:]

    # Check target size.
    if not isinstance(size, tuple) and not isinstance(size, list):
        assert isinstance(size, int)
        size = (size, size)
    assert len(size) == 2

    # Ensure upsampling.
    for i, o_s in enumerate(orig_size):
        assert o_s <= size[i]

    # Get size of input cell when interpolated.
    cell_size = [int(np.ceil(s / orig_size[i])) for i, s in enumerate(size)]

    # Get size of interpolated input with padding.
    pad_size = [int(cell_size[i] * (orig_size[i] + 2))
                for i in range(len(orig_size))]

    # Pad input with reflection padding.
    x_padded = F.pad(x, (1, 1, 1, 1), mode="reflect")

    # Interpolated padded input.
    x_up = F.interpolate(x_padded,
                         pad_size,
                         mode=interpolate_mode,
                         align_corners=False)

    # Slice interpolated input to size.
    x_new = x_up[:,
                 :,
                 cell_size[0]:cell_size[0] + size[0],
                 cell_size[1]:cell_size[1] + size[1]]

    return x_new


def rise_class(*args, target, **kwargs):
    r"""Class-specific RISE.

    This function has the all the arguments of :func:`rise` with the following
    additional argument and returns a class-specific saliency map for the
    given :attr:`target` class(es).

    Args:
        target (int, :class:`torch.Tensor`, list, or :class:`np.ndarray`):
            target label(s) that can be cast to :class:`torch.long`.
    """
    saliency = rise(*args, **kwargs)
    assert len(saliency.shape) == 4
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.long, device=saliency.device)
    assert isinstance(target, torch.Tensor)
    assert target.dtype == torch.long
    assert len(target) == len(saliency)

    class_saliency = torch.cat([saliency[i, t].unsqueeze(0).unsqueeze(1)
                                for i, t in enumerate(target)], dim=0)
    output_shape = list(saliency.shape)
    output_shape[1] = 1
    assert list(class_saliency.shape) == output_shape

    return class_saliency


class ARISE:
    def __init__(self,
                 model,
                 loc_det_method,
                 p,
                 dev,
                 num_cells,
                 class_id,
                 input_size: tuple = (224, 224),
                 dataset=None
                 ):
        self.model = model
        self.loc_det_method = loc_det_method
        self.p = p
        self.dev = dev
        self.num_cells = num_cells
        self.grid_size = (num_cells, num_cells)
        self.class_id = class_id
        self.input_size = input_size

        if dataset == 'voc_2007':
            self.n_classes = 20
        elif dataset == 'coco':
            self.n_classes = 80

        self.window_sizes = [(224 // 2, 224 // 2), (224 // 4, 224 // 4),
                             (26, 26)]  # 3*3, 5*5, 10*10 (134 inferences)
        self.step_sizes = [56, 42, 22]

        if self.loc_det_method == 'entropy_abs':
            # resnet voc 85.2
            self.e_threshold = [0.08391669513844599, 0.3926695687128092, 0.4485679560147107, 0.837019970367532]
            self.hide_prob_vs_level = [0.02404863575342404, 0.05835518311472475, 0.1503570413649637, 0.2762894266319287]
            self.global_hide_prob_scale = 1.0 # hide_prob_vs_level already scaled
        if self.loc_det_method == 'entropy_norm_rel':
            # resnet voc 84
            self.global_hide_prob_scale = 0.3
            self.scale_scores = (0.6, 0.75, 1)
        if self.loc_det_method == 'entropy_norm_rel_v2':
            self.entropy_max = -np.log(1/self.n_classes) # 2.995732273553991 for VOC
            # resnet voc
            self.global_hide_prob_scale = 0.3
            self.scale_scores = (0.6, 0.75, 1)
        if self.loc_det_method == 'target_class_score':
            # resnet voc 85.4
            self.global_hide_prob_scale = 0.3
            self.scale_scores = (0.5, 0.7, 1)
        if self.loc_det_method == 'target_class_score_nms':
            # resnet voc 82.4
            self.prob_thresh = 0.1
            self.iou_threshold = 0.7
            self.global_hide_prob_scale = 0.6
            self.scale_scores = (0.6, 0.75, 1)
        if self.loc_det_method == 'el2n':
            # resnet voc 84.7
            self.global_hide_prob_scale = 0.3
            self.scale_scores = (0.6, 0.8, 1)

    @staticmethod
    def _sliding_window(image, stepSize, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[2], stepSize):
            for x in range(0, image.shape[3], stepSize):
                # yield the current window
                yield (x, y, image[:, :, y:y + windowSize[1], x:x + windowSize[0]])

    @staticmethod
    def _update_default_hide_probability_map(hide_probability_map, object_occurrence_probability_map):
        map_to_update_with = object_occurrence_probability_map - object_occurrence_probability_map.mean()  # zero mean
        return hide_probability_map + map_to_update_with

    def _infer(self, frame):
        frame = torch.from_numpy(frame).cuda()
        y = self.model(frame)
        logits = torch.sigmoid(y)
        return logits.cpu().numpy().squeeze()

    def _go_with_sliding_window(self, input_img):
        # TODO: optimize
        window_history = {}
        input_img = input_img.cpu().numpy()

        step = 1
        for (winW, winH), STEP_SIZE in zip(self.window_sizes, self.step_sizes):
            windows = []
            window_params_history = []
            for (x, y, window) in self._sliding_window(input_img, stepSize=STEP_SIZE, windowSize=(winW, winH)):
                if window.shape[2] != winH or window.shape[3] != winW:
                    continue
                # if (not np.any(updated_mask_prob[y: y + winH, x: x + winW])) and (not self.offline_method):
                #     # if mask was not updated last iteration - skip further analysis of this region of the image
                #     # print('SKIIIIIP')
                #     continue

                window = np.stack([cv2.resize(frm, dsize=self.input_size) for frm in window[0]], axis=0)
                window = window[np.newaxis, ...]
                windows.append(window)

                window_params = x, y, winW, winH
                window_params_history.append(window_params)

            window_vs_logit = self._infer(np.concatenate(windows))
            for i in range(window_vs_logit.shape[0]):
                logits = window_vs_logit[i]
                window_params = window_params_history[i]

                if step not in window_history:
                    window_history[step] = []
                window_history[step].append((window_params, logits))
            step += 1

        return window_history

    def _calculate_entropy(self, logits):
        return - np.sum(logits * np.log(logits)) / np.log(self.n_classes)

    def _calculate_entropy_wo_norm(self, logits):
        return - np.sum(logits * np.log(logits))

    def _get_object_occurrence_probability_map(self, window_history):
        if self.loc_det_method == 'entropy_abs':
            mask_prob = np.zeros((224, 224))

            for step in list(window_history.keys()):
                for window_params, logits in window_history[step]:
                    x, y, winW, winH = window_params
                    entropy = self._calculate_entropy(logits)
                    target_class_score = logits[self.class_id]
                    if entropy < self.e_threshold[step]:
                        sub_m = mask_prob[y: y + winH, x: x + winW]
                        hide_prob_scaled = target_class_score * self.hide_prob_vs_level[step]
                        sub_m[sub_m < hide_prob_scaled] = hide_prob_scaled
                        mask_prob[y: y + winH, x: x + winW] = sub_m
            return mask_prob
        elif self.loc_det_method == 'entropy_norm_rel':
            # TODO: scale with target_class_score
            # TODO: add more weight to the later steps (small scale), yes yes
            mask_prob = np.zeros((224, 224))

            all_window_history = window_history[1] + window_history[2] + window_history[3]
            all_entropies = np.array([self._calculate_entropy(logits) for _, logits in all_window_history])
            entropy_max = all_entropies.max()

            for step in list(window_history.keys()):
                for window_params, logits in window_history[step]:
                    x, y, winW, winH = window_params
                    entropy = self._calculate_entropy(logits)
                    target_class_score = logits[self.class_id]
                    entropy_norm = entropy / entropy_max # normalization
                    hide_prob = - entropy_norm + 1 # just some linear relationship

                    sub_m = mask_prob[y: y + winH, x: x + winW]
                    hide_prob_scaled = hide_prob * target_class_score * self.scale_scores[step - 1]
                    sub_m[sub_m < hide_prob_scaled] = hide_prob_scaled
                    mask_prob[y: y + winH, x: x + winW] = sub_m
            return mask_prob
        if self.loc_det_method == 'entropy_norm_rel_v2':
            mask_prob = np.zeros((224, 224))
            for step in list(window_history.keys()):
                for window_params, logits in window_history[step]:
                    x, y, winW, winH = window_params
                    entropy = - np.sum(logits * np.log(logits))
                    entropy_norm = entropy / self.entropy_max
                    hide_prob = 1 - entropy_norm

                    sub_m = mask_prob[y: y + winH, x: x + winW]
                    hide_prob_scaled = hide_prob * self.scale_scores[step - 1]
                    sub_m[sub_m < hide_prob_scaled] = hide_prob_scaled
                    mask_prob[y: y + winH, x: x + winW] = sub_m
            return mask_prob
        elif self.loc_det_method == 'target_class_score':
            # TODO: scale with target_class_score
            # TODO: add more weight to the later steps (small scale), yes yes
            mask_prob = np.zeros((224, 224))
            # certainty_map = np.zeros((224, 224))
            # all_window_history = self.window_history[1] + self.window_history[2] + self.window_history[3]
            # all_target_class_score = [logits[target_class] for window_params, logits, target_class in all_window_history]
            # target_class_score_max = max(all_target_class_score)
            # print('\ntarget_class_score_max global', target_class_score_max)

            # el2n_all = []
            # for window_params, logits, target_class in all_window_history:
            #     y = np.zeros(self.n_classes)
            #     y[target_class] = 1
            #     el2n = np.linalg.norm(y - logits)
            #     el2n_all.append(el2n)
            # el2n_max = max(el2n_all)

            # e_to_use_all = []
            # for window_params, logits, target_class in all_window_history:
            #     y_eps = np.zeros(self.n_classes) + 1e-7
            #     y_eps[target_class] = 1
            #     e_to_use = np.sum(logits * np.log(logits/y_eps))
            #     e_to_use_all.append(e_to_use)
            # e_to_use_max = max(e_to_use_all)

            # e_ce_all = []
            # ce_all = []
            # for window_params, logits, target_class in all_window_history:
            #     e = - np.sum(logits * np.log(logits))
            #     y = np.zeros(self.n_classes)
            #     y[target_class] = 1
            #     ce = - np.sum(y * np.log(logits))
            #     e_ce_all.append(e / ce)
            #     ce_all.append(ce)
            # e_ce_max = max(e_ce_all)
            for step in list(window_history.keys()):
                # target_class_scores = [logits[target_class] for window_params, logits, target_class in self.window_history[step]]
                # target_class_score_max_within_step = max(target_class_scores)
                # print('step', step, 'target_class_score_max_within_step', target_class_score_max_within_step)

                # el2n_all = []
                # for window_params, logits, target_class in self.window_history[step]:
                #     y = np.zeros(self.n_classes)
                #     y[target_class] = 1
                #     el2n = np.linalg.norm(y - logits)
                #     el2n_all.append(el2n)
                # el2n_max = max(el2n_all)

                # e_to_use_all = []
                # for window_params, logits, target_class in self.window_history[step]:
                #     y_eps = np.zeros(self.n_classes) + 1e-7
                #     y_eps[target_class] = 1
                #     e_to_use = np.sum(logits * np.log(logits/y_eps))
                #     e_to_use_all.append(e_to_use)
                # e_to_use_max = max(e_to_use_all)

                # e_ce_all = []
                # for window_params, logits, target_class in self.window_history[step]:
                #     e = - np.sum(logits * np.log(logits))
                #     y = np.zeros(self.n_classes)
                #     y[target_class] = 1
                #     ce = - np.sum(y * np.log(logits))
                #     e_ce_all.append(e/ce)
                # e_ce_max = max(e_ce_all)
                for window_params, logits in window_history[step]:
                    target_class_score = logits[self.class_id]
                    # e = - np.sum(logits * np.log(logits))
                    # y = np.zeros(self.n_classes)
                    # y[target_class] = 1
                    # ce = - np.sum(y * np.log(logits))
                    # print(f'\nstep {step}, window_params {window_params}, target_class_score {target_class_score:.3f}, target_class_is_max {np.argmax(logits) == target_class}')
                    # print(f'e {e}, ce {ce}')
                    # print(f'e/ce {e / ce}, ce/e {ce / e}')
                    #
                    # y_eps = np.zeros(self.n_classes) + 1e-7
                    # y_eps[target_class] = 1
                    # print(f'Dkl(logits||y_eps) {np.sum(logits * np.log(logits/y_eps))}, H(logits, y_eps) {- np.sum(logits * np.log(y_eps))}, H(logits) {- np.sum(logits * np.log(logits))}')
                    # print(f'Dkl(y_eps||logits) {np.sum(y_eps * np.log(y_eps/logits))}, H(y_eps, logits) {- np.sum(y_eps * np.log(logits))}, H(y_eps) {- np.sum(y_eps * np.log(y_eps))}')
                    # # el2n = np.linalg.norm(y - logits)
                    # # print(f'el2n {el2n:.3f}')



                    # el2n = el2n / el2n_max
                    # hide_prob = - el2n + 1
                    # e_to_use = np.sum(logits * np.log(logits/y_eps))
                    # e_to_use = e_to_use / e_to_use_max
                    # hide_prob = - e_to_use + 1
                    # hide_prob = (e/ce) / e_ce_max
                    # x, y, winW, winH = window_params
                    # sub_m = mask_prob[y: y + winH, x: x + winW]
                    # hide_prob_scaled = hide_prob * self.scale_scores[step - 1] * self.global_hide_prob_scale
                    # sub_m[sub_m < hide_prob_scaled] = hide_prob_scaled
                    # mask_prob[y: y + winH, x: x + winW] = sub_m


                    # if target_class_score > 0.5:
                    x, y, winW, winH = window_params
                    # sub_certainty_map = certainty_map[y: y + winH, x: x + winW]
                    # # sub_certainty_map_to_update = np.logical_and(((1/e) * target_class_score) > sub_certainty_map, sub_m_to_update)
                    # sub_certainty_map_to_update = ((1 / e) * target_class_score) > sub_certainty_map
                    # sub_certainty_map[sub_certainty_map_to_update] = (1/e) * target_class_score
                    # certainty_map[y: y + winH, x: x + winW] = sub_certainty_map
                    sub_m = mask_prob[y: y + winH, x: x + winW]
                    hide_prob_scaled = target_class_score * self.scale_scores[step - 1]
                    sub_m_to_update = sub_m < hide_prob_scaled
                    # sub_m_to_update = np.logical_and(sub_m < hide_prob_scaled, sub_certainty_map_to_update)
                    sub_m[sub_m_to_update] = hide_prob_scaled
                    mask_prob[y: y + winH, x: x + winW] = sub_m
            #     # print('\n\n\n')
            # certainty_map /= certainty_map.max()
            # certainty_map *= self.global_hide_prob_scale
            # # return certainty_map
            return mask_prob
        elif self.loc_det_method == 'target_class_score_nms':
            mask_prob = np.zeros((224, 224))
            all_rects = []
            all_probs = []
            all_steps = []
            for step in list(window_history.keys()):
                for window_params, logits in window_history[step]:
                    target_class_score = logits[self.class_id]
                    if target_class_score > self.prob_thresh:
                        x, y, winW, winH = window_params
                        all_rects.append([x, y, winW, winH])
                        all_probs.append(target_class_score)
                        all_steps.append(step)

            if all_rects == []:
                return mask_prob
            all_rects = np.array(all_rects)
            pick_nms = torchvision.ops.nms(boxes=torch.from_numpy(all_rects.astype(np.float32)),
                                           scores=torch.from_numpy(np.array(all_probs)), iou_threshold=self.iou_threshold)
            steps_nms = [item for (idx, item) in enumerate(all_steps) if idx in pick_nms.numpy()]
            rects_nms = [item for (idx, item) in enumerate(all_rects) if idx in pick_nms.numpy()]
            probs_nms = [item for (idx, item) in enumerate(all_probs) if idx in pick_nms.numpy()]

            for step, rect, prob in zip(steps_nms, rects_nms, probs_nms):
                x, y, winW, winH = rect
                sub_m = mask_prob[y: y + winH, x: x + winW]
                hide_prob_scaled = prob * self.scale_scores[step - 1]
                sub_m[sub_m < hide_prob_scaled] = hide_prob_scaled
                mask_prob[y: y + winH, x: x + winW] = sub_m
            return mask_prob
        elif self.loc_det_method == 'el2n':
            mask_prob = np.zeros((224, 224))

            def get_el2n(logits):
                y = np.zeros(self.n_classes)
                y[self.class_id] = 1
                el2n = np.linalg.norm(y - logits)
                return el2n

            all_window_history = window_history[1] + window_history[2] + window_history[3]
            all_el2n = np.array([get_el2n(logits) for _, logits in all_window_history])
            el2n_max = all_el2n.max()

            for step in list(window_history.keys()):
                for window_params, logits in window_history[step]:
                    el2n = get_el2n(logits)
                    x, y, winW, winH = window_params
                    el2n_norm = el2n / el2n_max # normalization
                    hide_prob = - el2n_norm + 1 # just some linear relationship

                    sub_m = mask_prob[y: y + winH, x: x + winW]
                    hide_prob_scaled = hide_prob * self.scale_scores[step - 1]
                    sub_m[sub_m < hide_prob_scaled] = hide_prob_scaled
                    mask_prob[y: y + winH, x: x + winW] = sub_m
            return mask_prob
        else:
            raise NotImplemented

    def get_hide_probability_map(self, input_img):
        """

        :param input_img:
        :return: hide_probability_map (:class:`torch.Tensor`) 2D tensor [num_cells, num_cells] with probabilities if
        hiding the cell. By default, p is assigned globally (to all cells)
        """
        hide_probability_map = torch.zeros(self.num_cells, self.num_cells) + self.p

        if self.loc_det_method is not None:
            window_history = self._go_with_sliding_window(input_img)

            object_occurrence_probability_map = self._get_object_occurrence_probability_map(window_history)
            if self.loc_det_method != 'entropy_abs':
                object_occurrence_probability_map *= self.global_hide_prob_scale

            # Downscale the object_occurrence_probability_map
            object_occurrence_probability_map = cv2.resize(object_occurrence_probability_map, self.grid_size,
                                                           interpolation=cv2.INTER_LINEAR)

            hide_probability_map = self._update_default_hide_probability_map(hide_probability_map,
                                                                             object_occurrence_probability_map)
            return hide_probability_map.to(self.dev)

        return hide_probability_map.to(self.dev)


def _generate_saliency(num_masks, batch_size, filter_masks, num_cells, nsd, dev, hide_probability_map, input,
                       up_size, input_shape, cell_size, height, width, model, num_classes, saliency, class_id):
    num_chunks = (num_masks + batch_size - 1) // batch_size
    for chunk in range(num_chunks):
        # Generate RISE random masks on the fly.
        mask_bs = min(num_masks - batch_size * chunk, batch_size)

        if filter_masks is None:
            # Generate low-res, random binary masks.. Sampling from uniform distribution
            rand_nums = torch.rand(mask_bs, 1, *((num_cells,) * nsd), device=dev)
            grid = (rand_nums < hide_probability_map).float()
            # # Generate low-res, random binary masks. Sampling from Gaussian distribution (mean and sdt can be updated in time)
            # mean_init = 0.0
            # std_init = 0.4
            # mean_cellwise = torch.full((num_cells, num_cells), mean_init)
            # std_cellwise = torch.full((num_cells, num_cells), std_init)
            # mean_cellwise_batch = mean_cellwise.repeat(mask_bs, 1, 1, 1)
            # std_cellwise_batch = std_cellwise.repeat(mask_bs, 1, 1, 1)
            # rand_nums = torch.normal(mean=mean_cellwise_batch, std=std_cellwise_batch)
            # grid = (rand_nums < 0).float()

            # Upsample low-res masks to input shape + buffer.
            masks_up = _upsample_reflect(grid, up_size)

            # Save final RISE masks with random shift.
            masks = torch.empty(mask_bs, 1, *input_shape[2:], device=dev)
            shift_x = torch.randint(0,
                                    cell_size[0],
                                    (mask_bs,),
                                    device='cpu')
            shift_y = torch.randint(0,
                                    cell_size[1],
                                    (mask_bs,),
                                    device='cpu')
            for i in range(mask_bs):
                masks[i] = masks_up[i,
                           :,
                           shift_x[i]:shift_x[i] + height,
                           shift_y[i]:shift_y[i] + width]
        else:
            masks = filter_masks[
                    chunk * batch_size:chunk * batch_size + mask_bs]

        # Accumulate saliency mask.
        for i, inp in enumerate(input):
            out = torch.sigmoid(model(inp.unsqueeze(0) * masks))
            if len(out.shape) == 4:
                # TODO: Consider handling FC outputs more flexibly.
                assert out.shape[2] == 1
                assert out.shape[3] == 1
                out = out[:, :, 0, 0]
            sal = torch.matmul(out.data.transpose(0, 1),
                               masks.view(mask_bs, height * width))
            sal = sal.view((num_classes, height, width))
            saliency[i] = saliency[i] + sal

            # explanation = saliency[0, class_id, :, :].cpu().numpy()
            # delition, insertion = deletion_insertion_single_run(model, input, explanation, class_id)
            # print(
            #     f'saliency_aggregated so far: del {delition}, ins {insertion}, overall {insertion - delition}, num_masks {(chunk + 1) * batch_size}')

    return saliency, out.data, masks.data


def deletion_insertion_single_run(model, x: torch.tensor, saliency: np.array, class_id: int) -> Tuple[float, float]:
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)

    # Function that blurs input image
    blur = lambda x: F.conv2d(x, kern.cuda(), padding=klen // 2)

    insertion = CausalMetric(model, 'ins', 224, substrate_fn=blur)
    deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)

    # h_del = deletion.single_run_single_iteration(x, saliency, verbose=0,
    #                             class_id=class_id)
    # h_ins = insertion.single_run_single_iteration(x, saliency, verbose=0,
    #                              class_id=class_id)
    # return h_del, h_ins


    h_del = deletion.single_run(x, saliency, verbose=0,
                                class_id=class_id)  # , save_to='/home/etsykuno/tmp/'
    h_ins = insertion.single_run(x, saliency, verbose=0,
                                 class_id=class_id)  # , save_to='/home/etsykuno/tmp/'
    h_del_auc = auc(h_del)
    h_ins_auc = auc(h_ins)
    return h_del_auc, h_ins_auc


def deletion_insertion_single_run_few_iterations(model, x: torch.tensor, saliency: np.array, class_id: int) -> Tuple[float, float]:
    from torchray.benchmark.insertion_deletion import gkern
    from torchray.benchmark.insertion_deletion import CausalMetric
    from torchray.benchmark.insertion_deletion import auc

    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)

    # Function that blurs input image
    blur = lambda x: F.conv2d(x, kern.cuda(), padding=klen // 2)

    insertion = CausalMetric(model, 'ins', 224, substrate_fn=blur)
    deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)

    h_del = deletion.single_run_few_iterations(x, saliency, verbose=0,
                                class_id=class_id)
    h_ins = insertion.single_run_few_iterations(x, saliency, verbose=0,
                                 class_id=class_id)
    h_del_auc = auc(h_del)
    h_ins_auc = auc(h_ins)
    return h_del_auc, h_ins_auc


def update_hide_probability_map(masks, predictions, class_id, height, width, batch_size,
                                hide_probability_map_prev=0.5, update_step_size=0.05, beta=0.5,
                                update_prev=0, num_cells=7, resize_mode='bilinear', method_params=None, sal_history=None, saliency_shape=None):
    # # Calculate the update value (aka gradient)

    # # Take all masks (sal) from the prev batch, to update hide prob dist
    # sal_running = torch.matmul(predictions.data,
    #                    masks.view(batch_size, height * width))
    # sal_running = sal_running.view((1, 1, height, width))
    # s = torch.squeeze(F.interpolate(sal_running, (num_cells, num_cells), mode=resize_mode, align_corners=False))
    # hide_probability_update = s / s.max()  # Normalize to 0..1
    # hide_probability_update = hide_probability_update - hide_probability_update.mean()  # Make it zero mean

    # # Topk approach - Get saliency map of the topk best masks
    # k = 5
    # if method_params is not None:
    #     k = method_params[1]
    #     beta = method_params[2]
    # topk_pred_values, topk_pred_indices = torch.topk(predictions, k)
    # sal_running = torch.matmul(predictions[topk_pred_indices].data,
    #                            masks[topk_pred_indices].view(k, height * width))
    # sal_running = sal_running.view((1, 1, height, width))
    # s = torch.squeeze(F.interpolate(sal_running, (num_cells, num_cells), mode=resize_mode, align_corners=False))
    # hide_probability_update = s / s.max()  # Normalize to 0..1
    # hide_probability_update = hide_probability_update - hide_probability_update.mean()  # Make it zero mean


    # # Take hottest values from sal map and add them, take coldest values from sal and subtract them
    # # Does not work that well. All values are required
    # # sal_running = torch.matmul(predictions.data,
    # #                    masks.view(batch_size, height * width))
    # # sal_running = sal_running.view((1, 1, height, width))
    # # s = torch.squeeze(F.interpolate(sal_running, (num_cells, num_cells), mode=resize_mode, align_corners=False))
    # k = 5
    # topk_pred_values, topk_pred_indices = torch.topk(predictions, k)
    # sal_running = torch.matmul(predictions[topk_pred_indices].data,
    #                            masks[topk_pred_indices].view(k, height * width))
    # sal_running = sal_running.view((1, 1, height, width))
    # s = torch.squeeze(F.interpolate(sal_running, (num_cells, num_cells), mode=resize_mode, align_corners=False))
    # s = s / s.max()  # Normalize to 0..1
    #
    # p = 0.3
    # q_low, q_high = 0 + p, 1 - p
    # hide_probability_update = torch.zeros(7, 7).cuda()
    # hide_probability_update[s > torch.quantile(s, q_high)] = s[s > torch.quantile(s, q_high)] - s.mean()
    # hide_probability_update[s < torch.quantile(s, q_low)] = s[s < torch.quantile(s, q_low)] - s.mean()
    # # hide_probability_update = s - s.mean()  # Make it zero mean




    # sal_running - hide_probability_map_prev
    # TODO: remove clamp
    # Based on diff between running sal and current hide_probability_map?
    # Based on EP-like loss?
    # sal_running - hide_probability_map_prev

    update_step_size_base = 0.05
    if method_params is not None:
        update_step_size_base = method_params[0]
    update_step_size = update_step_size_base

    if len(sal_history)> 5:
        update_step_size = update_step_size_base / 2 # 0.035
    if len(sal_history)> 10:
        update_step_size = update_step_size_base / 4 # 0.02

    # if len(sal_history)> 3:
    #     update_step_size = update_step_size_base / 2 # 0.035
    # if len(sal_history)> 9:
    #     update_step_size = update_step_size_base / 4 # 0.02
    # if len(sal_history)> 12:
    #     update_step_size = update_step_size_base / 6 # 0.02
    #
    # if len(sal_history)> 7:
    #     update_step_size = update_step_size_base / 2 # 0.035
    # if len(sal_history)> 12:
    #     update_step_size = update_step_size_base / 4 # 0.02

    # gamma = 0.15
    # if method_params is not None:
    #     gamma = method_params[3]
    # step = len(sal_history) - 1
    # update_step_size = update_step_size_base * np.exp(-gamma * step)

    beta = 0.3
    k = 15
    if method_params is not None:
        k = method_params[1]
        beta = method_params[2]
    # Calculate the update value (aka gradient)
    # Preprocess current saliency map
    topk_pred_values, topk_pred_indices = torch.topk(predictions, k)
    sal_current = torch.matmul(predictions[topk_pred_indices].data,
                               masks[topk_pred_indices].view(k, height * width)) # Use just topk masks
    sal_current = sal_current.view((1, 1, height, width))
    sal_current = torch.squeeze(F.interpolate(sal_current, (num_cells, num_cells), mode=resize_mode, align_corners=False))
    sal_current /= sal_current.max() # Normalize to 0..1  # Normalize to 0..1
    sal_current_0_mean = sal_current - sal_current.mean()  # Make it zero-mean
    # p = 0.2
    # q_low, q_high = 0 + p, 1 - p
    # sal_current_0_mean = torch.zeros(7, 7).cuda()
    # s = sal_current
    # sal_current_0_mean[s > torch.quantile(s, q_high)] = s[s > torch.quantile(s, q_high)] - torch.cat((s[s > torch.quantile(s, q_high)], s[s < torch.quantile(s, q_low)])).mean()
    # sal_current_0_mean[s < torch.quantile(s, q_low)] = s[s < torch.quantile(s, q_low)] - torch.cat((s[s > torch.quantile(s, q_high)], s[s < torch.quantile(s, q_low)])).mean()
    sal_current_05_mean = sal_current_0_mean + 0.5  # Make it 0.5-mean
    # saliency_aggregated_so_far = torch.zeros((224, 224), device=predictions.device)
    # for sal_w in sal_history:
    #     saliency_aggregated_so_far += sal_w[0, class_id, :, :]
    # saliency_aggregated_so_far = torch.squeeze(
    #     F.interpolate(saliency_aggregated_so_far.view((1, 1, height, width)), (num_cells, num_cells), mode=resize_mode, align_corners=False))
    # saliency_aggregated_so_far /= saliency_aggregated_so_far.max()
    # # # Make saliency_aggregated_so_far zero mean and unit variance. then scale and shift like BN
    # # saliency_aggregated_so_far_norm = (saliency_aggregated_so_far - saliency_aggregated_so_far.mean()) / torch.std(saliency_aggregated_so_far.flatten())
    # # g, b = 0.03, 0.03
    # # saliency_aggregated_so_far_05mean = g * saliency_aggregated_so_far_norm + (0.5 + b)
    # # saliency_aggregated_so_far_05mean = torch.clamp(saliency_aggregated_so_far_05mean, 0, 1)

    # saliency_aggregated_so_far_0_mean = 0.5 * (saliency_aggregated_so_far - saliency_aggregated_so_far.mean())
    # saliency_aggregated_so_far_05mean = saliency_aggregated_so_far_0_mean + 0.5
    # # print(saliency_aggregated_so_far_05mean.max(), saliency_aggregated_so_far_05mean.min())
    # saliency_aggregated_so_far_05mean = torch.clamp(saliency_aggregated_so_far_05mean, 0, 1)

    # beta = 0.3
    # update_step_size = 0.015
    # if len(sal_history)> 5:
    #     update_step_size = 0.03
    # if len(sal_history)> 10:
    #     update_step_size = 0.05


    # hide_probability_update = saliency_aggregated_so_far_05mean - hide_probability_map_prev
    hide_probability_update = sal_current_05_mean - hide_probability_map_prev      #+ 0.2*(saliency_aggregated_so_far_05_mean - hide_probability_map_prev)  # Make it zero-mean again!

    # # Vanilla update
    # update = update_step_size*hide_probability_update
    # hide_probability_map = hide_probability_map_prev + update

    # Momentum update
    update = beta*update_prev + update_step_size*hide_probability_update
    hide_probability_map = hide_probability_map_prev + update

    # # Clamp
    # max_update = 0.15
    # if method_params is not None:
    #     max_update = method_params[3]
    # hide_probability_map = torch.clamp(hide_probability_map, min=(0.5 - max_update), max=(0.5 + max_update))

    return hide_probability_map, update


def rise(model,
         input,
         target=None,
         seed=0,
         num_masks=8000,
         num_cells=7,
         filter_masks=None,
         batch_size=32,
         p=0.5,
         resize=False,
         resize_mode='bilinear',
         loc_det_method=None,
         class_id=None,
         dataset=None,
         method_params=None,
         sal8000_name=None):
    r"""RISE.

    Args:
        model (:class:`torch.nn.Module`): a model.
        input (:class:`torch.Tensor`): input tensor.
        seed (int, optional): manual seed used to generate random numbers.
            Default: ``0``.
        num_masks (int, optional): number of RISE random masks to use.
            Default: ``8000``.
        num_cells (int, optional): number of cells for one spatial dimension
            in low-res RISE random mask. Default: ``7``.
        filter_masks (:class:`torch.Tensor`, optional): If given, use the
            provided pre-computed filter masks. Default: ``None``.
        batch_size (int, optional): batch size to use. Default: ``128``.
        p (float, optional): with prob p, a low-res cell is set to 0;
            otherwise, it's 1. Default: ``0.5``.
        resize (bool or tuple of ints, optional): If True, resize saliency map
            to size of :attr:`input`. If False, don't resize. If (width,
            height) tuple, resize to (width, height). Default: ``False``.
        resize_mode (str, optional): If resize is not None, use this mode for
            the resize function. Default: ``'bilinear'``.

    Returns:
        :class:`torch.Tensor`: RISE saliency map.
    """
    with torch.no_grad():
        print('num_masks', num_masks, 'seed', seed)
        # Get device of input (i.e., GPU).
        dev = input.device

        # Initialize saliency mask and mask normalization term.
        input_shape = input.shape
        saliency_shape = list(input_shape)

        height = input_shape[2]
        width = input_shape[3]

        out = model(input)
        num_classes = out.shape[1]

        saliency_shape[1] = num_classes
        saliency = torch.zeros(saliency_shape, device=dev)

        # Number of spatial dimensions.
        nsd = len(input.shape) - 2
        assert nsd == 2

        # Spatial size of low-res grid cell.
        cell_size = tuple([int(np.ceil(s / num_cells))
                           for s in input_shape[2:]])

        # Spatial size of upsampled mask with buffer (input size + cell size).
        up_size = tuple([input_shape[2 + i] + cell_size[i]
                         for i in range(nsd)])

        # Save current random number generator state.
        state = torch.get_rng_state()

        # # Set seed.
        torch.manual_seed(seed)

        if filter_masks is not None:
            assert len(filter_masks) == num_masks

        if loc_det_method == 'rise':
            sal_history = []
            predictions = None
            update_prev = 0
            num_chunks = (num_masks + batch_size - 1) // batch_size
            for chunk in range(num_chunks):
                # if predictions is None:
                #     hide_probability_map = p
                # else:
                #     # For batch 32
                #     # 100 0.1 (not optimal?)
                #     # 200 0.05
                #     # 400 0.025
                #     update_step_size = 10 / num_masks
                #     if method_params is not None:
                #         update_step_size = method_params[0]
                #     # Update running hide_probability_map
                #     hide_probability_map, update_prev = update_hide_probability_map(masks, predictions[:, class_id], class_id, height, width, batch_size,
                #                                                                     hide_probability_map_prev=hide_probability_map,
                #                                                                     update_step_size=update_step_size,
                #                                                                     update_prev=update_prev,
                #                                                                     method_params=method_params,
                #                                                                     sal_history=sal_history,
                #                                                                     saliency_shape=saliency_shape)
                mask_bs = min(num_masks - batch_size * chunk, batch_size)
                saliency = torch.zeros(saliency_shape, device=dev)





                sal_8000 = np.load(sal8000_name)
                sal_8000 = torch.tensor(sal_8000).cuda()
                sal_8000 = torch.squeeze(
                    F.interpolate(sal_8000, (num_cells, num_cells), mode=resize_mode, align_corners=False))
                # sal_8000 0..1

                # hide_probability_map = torch.zeros(7, 7).cuda() + 0.5
                # sal_8000 *= 0.7
                # hide_probability_map[sal_8000 > torch.quantile(sal_8000, 0.9)] = sal_8000[sal_8000 > torch.quantile(sal_8000, 0.9)]

                sal_8000 = sal_8000.cpu().numpy()

                # Make sal_8000 zero mean and unit variance. then scale and shift
                sal_8000_norm = (sal_8000 - sal_8000.mean()) / np.std(sal_8000.flatten())
                g, b = 0.14, 0.0
                sal_8000_05mean = g * sal_8000_norm + (0.5 + b)

                # sal_8000_0mean = (sal_8000 - sal_8000.mean())
                # sal_8000_05mean = sal_8000_0mean + 0.5

                hide_probability_map = sal_8000_05mean
                hide_probability_map = np.clip(hide_probability_map, 0.1, 0.9)


                hide_probability_map = torch.tensor(hide_probability_map).cuda()






                sal, predictions, masks = _generate_saliency(mask_bs, batch_size, filter_masks, num_cells, nsd, dev,
                                         hide_probability_map, input, up_size, input_shape, cell_size,
                                         height, width, model, num_classes, saliency, class_id)
                sal_history.append(sal)

            # from scipy import stats
            # print('hide_probability_map',
            #       stats.describe(hide_probability_map.cpu().numpy().flatten()))
            # # print(g, b)

            saliency_aggregated = torch.zeros(saliency_shape, device=dev)
            for sal_w in sal_history:
                saliency_aggregated += sal_w

            # Normalize saliency mask.
            saliency = saliency_aggregated / num_masks
        # if loc_det_method == 'rise':
        #     # batch_size = 50
        #     # hide_probability_map = p
        #     # saliency = _generate_saliency(num_masks, batch_size, filter_masks, num_cells, nsd, dev, hide_probability_map,
        #     #                               input, up_size, input_shape, cell_size, height, width, model, num_classes, saliency)
        #     # # Normalize saliency mask.
        #     # saliency /= num_masks
        #
        #
        #
        #     # hide_probability_map = 0.5
        #     # num_masks_ = num_masks // 2
        #     # saliency_accumulated = _generate_saliency(num_masks_, batch_size, filter_masks, num_cells, nsd, dev,
        #     #                                           hide_probability_map,
        #     #                                           input, up_size, input_shape, cell_size, height, width, model,
        #     #                                           num_classes,
        #     #                                           saliency)
        #     #
        #     # saliency = F.interpolate(saliency, (num_cells, num_cells), mode=resize_mode, align_corners=False)
        #     # hide_probability_map = saliency[0, class_id, :, :]
        #     # hide_probability_map /= hide_probability_map.max()  # Normalize to 0..1
        #     # hide_probability_map *= 0.1  # Normalize to 0..scale
        #     # hide_probability_map = 0.5 + (hide_probability_map - hide_probability_map.mean())
        #     #
        #     # saliency = _generate_saliency(num_masks_, batch_size, filter_masks, num_cells, nsd, dev,
        #     #                               hide_probability_map,
        #     #                               input, up_size, input_shape, cell_size, height, width, model, num_classes,
        #     #                               saliency_accumulated)
        #     # # Normalize saliency mask.
        #     # saliency /= num_masks
        #
        #
        #
        #     # RISE-based localization
        #     # batch_size = 25
        #     # num_masks = 200
        #
        #     pred_hist = []
        #     mask_hist = []
        #
        #     sal_history = []
        #     score_history = []
        #
        #     score_history_saliency_aggregated = []
        #     score_history_saliency_aggregated_weighted = []
        #
        #     predictions = None
        #
        #     # saliency_aggregated = torch.zeros(saliency_shape, device=dev)
        #     num_steps = num_masks // batch_size
        #     masks_per_step = batch_size
        #
        #     def get_updated_hide_probability_map(saliency, update_step_size=0.1):
        #         # print(f'Process saliency from the prev iteration with scale {scales[i]} ...')
        #         s = F.interpolate(saliency, (num_cells, num_cells), mode=resize_mode, align_corners=False)
        #         hide_probability_map = s[0, class_id, :, :]
        #         hide_probability_map /= hide_probability_map.max()  # Normalize to 0..1
        #
        #         # weight hide_probability_map based on trust (the more random masks infered -> more trust -> more weight to hide_probability_map update )
        #         hide_probability_map *= update_step_size # Normalize to 0..0.1
        #         hide_probability_update = hide_probability_map - hide_probability_map.mean()
        #         hide_probability_map = 0.5 + hide_probability_update
        #         return hide_probability_map
        #
        #     # def aggregare_best_sal_masks(sal_history, score_history):
        #     #     descending_indices = [idx for idx, item in sorted(enumerate(score_history), key=lambda k: k[1], reverse=True)]
        #     #     half = len(score_history) // 2
        #     #     sal_idx_to_pick = descending_indices[:half]
        #     #
        #     #     saliency_aggregated_best = torch.zeros(saliency_shape, device=dev)
        #     #     for sal_idx in sal_idx_to_pick:
        #     #         saliency_aggregated_best += sal_history[sal_idx]
        #     #     return saliency_aggregated_best
        #
        #     def process_scores(scores):
        #         if len(scores) < 3:
        #             return [1] * len(scores)
        #
        #         min_val = min(scores)
        #         if min_val < 0:
        #             scores = [item + (-min_val) for item in scores]
        #         max_val = max(scores)
        #         return [item/max_val for item in scores] # 0 .. 1
        #
        #         # min_val = min(scores)
        #         # if min_val < 0:
        #         #     scores = [item + (-min_val) for item in scores]
        #         # max_val = max(scores)
        #         # return [(item/(max_val * 2) + 0.75) for item in scores] # 0.75 .. 1.25
        #
        #         # ordered_indices = [key for key, val in sorted(enumerate(scores), key=lambda k: k[1])]
        #         # margine = 0.25
        #         # k = np.linspace(1 - margine, 1 + margine, len(scores))
        #         # return [k[idx] for idx in ordered_indices]
        #
        #
        #     def aggregare_available_sal_masks(sal_history, score_history, weight=False):
        #         scores = process_scores(score_history)
        #
        #         if weight:
        #             saliencies_weighted = [sal * score for sal, score in zip(sal_history, scores)]
        #         else:
        #             saliencies_weighted = sal_history
        #         saliency_aggregated = torch.zeros(saliency_shape, device=dev)
        #         for sal_w in saliencies_weighted:
        #             saliency_aggregated += sal_w
        #         return saliency_aggregated
        #
        #
        #     for i in range(num_steps):
        #         # if len(sal_history) > 3:
        #         #     # update hide_probability_map based on multiple best sal
        #         #     # aggregate sal from multiple best sal
        #         #     saliency_aggregated = aggregare_available_sal_masks(sal_history, score_history, weight=True)
        #         #     hide_probability_map = get_updated_hide_probability_map(saliency_aggregated)
        #         # else:
        #         hide_probability_map = p
        #
        #
        #
        #         # pred_hist_class_id = torch.cat(pred_hist)[:, class_id]
        #         # topk_pred_values, topk_pred_indices = torch.topk(torch.cat(pred_hist_class_id)[:, class_id], 5)
        #         # mask_hist = torch.cat(mask_hist)
        #
        #         # if predictions is not None:
        #         #     # Take all masks (sal) from the prev batch, to update hide prob dist
        #         #     sal = torch.matmul(predictions.data.transpose(0, 1),
        #         #                        masks.view(batch_size, height * width))
        #         #     sal = sal.view((1, num_classes, height, width))
        #         #     hide_probability_map = get_updated_hide_probability_map(sal)
        #         #
        #         #     # # Topk approach
        #         #     # k = 5
        #         #     # topk_pred_values, topk_pred_indices = torch.topk(predictions[:, class_id], k)
        #         #     # # Get saliency map of the topk best masks
        #         #     # # masks[topk_pred_indices] *
        #         #     # s = torch.matmul(masks[topk_pred_indices].view(224, 224, k), topk_pred_values)
        #
        #         saliency = torch.zeros(saliency_shape, device=dev)
        #         sal, predictions, masks = _generate_saliency(masks_per_step, batch_size, filter_masks, num_cells, nsd, dev,
        #                                  hide_probability_map, input, up_size, input_shape, cell_size,
        #                                  height, width, model, num_classes, saliency, class_id)
        #
        #         pred_hist.append(predictions)
        #         mask_hist.append(masks)
        #
        #         sal_history.append(sal)
        #
        #         # explanation = sal[0, class_id, :, :].cpu().numpy()
        #         # del_auc, ins_auc = deletion_insertion_single_run(model, input, explanation, class_id)
        #         # del_ins_overall = ins_auc - del_auc
        #         # score_history.append(del_ins_overall)
        #         #
        #         # # Collect history of aggregated saliency map
        #         # current_saliency_aggregated = aggregare_available_sal_masks(sal_history, score_history, weight=False)
        #         # explanation = current_saliency_aggregated[0, class_id, :, :].cpu().numpy()
        #         # del_auc, ins_auc = deletion_insertion_single_run(model, input, explanation, class_id)
        #         # del_ins_overall = ins_auc - del_auc
        #         # score_history_saliency_aggregated.append(del_ins_overall)
        #         #
        #         # # Collect history of aggregated saliency map (weighted)
        #         # current_saliency_aggregated = aggregare_available_sal_masks(sal_history, score_history, weight=True)
        #         # explanation = current_saliency_aggregated[0, class_id, :, :].cpu().numpy()
        #         # del_auc, ins_auc = deletion_insertion_single_run(model, input, explanation, class_id)
        #         # del_ins_overall = ins_auc - del_auc
        #         # score_history_saliency_aggregated_weighted.append(del_ins_overall)
        #
        #     # # Temporal aggregation metric
        #     # ta_metric = auc(np.array(score_history_saliency_aggregated_weighted)) - auc(np.array(score_history_saliency_aggregated))
        #     # print('ta_metric', ta_metric)
        #
        #     # pred_hist_class_id = torch.cat(pred_hist)[:, class_id]
        #     # topk_pred_values, topk_pred_indices = torch.topk(torch.cat(pred_hist_class_id)[:, class_id], 5)
        #     # mask_hist = torch.cat(mask_hist)
        #
        #     saliency_aggregated = aggregare_available_sal_masks(sal_history, score_history, weight=False)
        #     # explanation_aggregated = saliency_aggregated[0, class_id, :, :].cpu().numpy()
        #     # e = explanation_aggregated / explanation_aggregated.max()
        #     # delition, insertion = deletion_insertion_single_run(model, input, explanation_aggregated, class_id)
        #     # print(f'saliency_aggregated del {delition}, ins {insertion}, overall {insertion-delition}\n')
        #
        #     # Normalize saliency mask.
        #     saliency = saliency_aggregated / num_masks
        else:
            if loc_det_method is None:
                hide_probability_map = p
            else:
                a_rise = ARISE(model, loc_det_method, p, dev, num_cells, class_id, dataset=dataset)
                hide_probability_map = a_rise.get_hide_probability_map(input)

            saliency, _, _ = _generate_saliency(num_masks, batch_size, filter_masks, num_cells, nsd, dev, hide_probability_map,
                                          input, up_size, input_shape, cell_size, height, width, model, num_classes, saliency, class_id)
            # Normalize saliency mask.
            saliency /= num_masks

        # Restore original random number generator state.
        torch.set_rng_state(state)

        saliency224 = saliency

        # Resize saliency mask if needed.
        saliency_resized = resize_saliency(input,
                                   saliency224,
                                   resize,
                                   mode=resize_mode)
        return saliency_resized, saliency224


def _generate_saliency_smbo(num_masks, batch_size, filter_masks, num_cells, nsd, dev, hide_probability_map, input,
                       up_size, input_shape, cell_size, height, width, model, num_classes, saliency, class_id):
    num_chunks = (num_masks + batch_size - 1) // batch_size
    for chunk in range(num_chunks):
        # Generate RISE random masks on the fly.
        mask_bs = min(num_masks - batch_size * chunk, batch_size)

        # Generate low-res, random binary masks.. Sampling from uniform distribution
        rand_nums = torch.rand(mask_bs, 1, *((num_cells,) * nsd), device=dev)
        grid = (rand_nums < hide_probability_map).float()

        # Upsample low-res masks to input shape + buffer.
        masks_up = _upsample_reflect(grid, up_size)

        # Save final RISE masks with random shift.
        masks = torch.empty(mask_bs, 1, *input_shape[2:], device=dev)
        shift_x = torch.randint(0,
                                cell_size[0],
                                (mask_bs,),
                                device='cpu')
        shift_y = torch.randint(0,
                                cell_size[1],
                                (mask_bs,),
                                device='cpu')
        for i in range(mask_bs):
            masks[i] = masks_up[i,
                       :,
                       shift_x[i]:shift_x[i] + height,
                       shift_y[i]:shift_y[i] + width]

        # Accumulate saliency mask.
        for i, inp in enumerate(input):
            out = torch.sigmoid(model(inp.unsqueeze(0) * masks))
            if len(out.shape) == 4:
                # TODO: Consider handling FC outputs more flexibly.
                assert out.shape[2] == 1
                assert out.shape[3] == 1
                out = out[:, :, 0, 0]
            sal = torch.matmul(out.data.transpose(0, 1),
                               masks.view(mask_bs, height * width))
            sal = sal.view((num_classes, height, width))
            saliency[i] = saliency[i] + sal

    return saliency, out.data, masks.data


def smbo(model,
         input,
         target=None,
         seed=0,
         num_cells=7,
         resize=False,
         resize_mode='bilinear'):
    r"""RISE.

    Args:
        model (:class:`torch.nn.Module`): a model.
        input (:class:`torch.Tensor`): input tensor.
        seed (int, optional): manual seed used to generate random numbers.
            Default: ``0``.
        num_masks (int, optional): number of RISE random masks to use.
            Default: ``8000``.
        num_cells (int, optional): number of cells for one spatial dimension
            in low-res RISE random mask. Default: ``7``.
        filter_masks (:class:`torch.Tensor`, optional): If given, use the
            provided pre-computed filter masks. Default: ``None``.
        batch_size (int, optional): batch size to use. Default: ``128``.
        p (float, optional): with prob p, a low-res cell is set to 0;
            otherwise, it's 1. Default: ``0.5``.
        resize (bool or tuple of ints, optional): If True, resize saliency map
            to size of :attr:`input`. If False, don't resize. If (width,
            height) tuple, resize to (width, height). Default: ``False``.
        resize_mode (str, optional): If resize is not None, use this mode for
            the resize function. Default: ``'bilinear'``.

    Returns:
        :class:`torch.Tensor`: RISE saliency map.
    """
    with torch.no_grad():
        # Get device of input (i.e., GPU).
        dev = input.device

        # Initialize saliency mask and mask normalization term.
        input_shape = input.shape
        saliency_shape = list(input_shape)

        out = model(input)
        num_classes = out.shape[1]

        saliency_shape[1] = num_classes
        saliency = torch.zeros(saliency_shape, device=dev)

        # Number of spatial dimensions.
        nsd = len(input.shape) - 2
        assert nsd == 2

        # Save current random number generator state.
        state = torch.get_rng_state()

        # # Set seed.
        torch.manual_seed(seed)












        # # AISE LIPO
        # from dlib import find_min_global
        # import math
        # import dlib
        # import time
        # import matplotlib.pyplot as plt
        # gauss_params_hist = {}
        # pred_score_hist = {}
        #
        # inference_times = []
        #
        # class Perturbation:
        #     def __init__(self, input):
        #         self.input = input
        #
        #     def apply(self, mask):
        #         return self.input * mask
        #
        # class GaussPMask:
        #     def __init__(self, H, W):
        #         self.H = H
        #         self.W = W
        #         h = np.linspace(0, 1, self.H)
        #         w = np.linspace(0, 1, self.W)
        #         self.h, self.w = np.meshgrid(w, h)
        #
        #     def _gaussian_2d(self, gauss_params):
        #         mh, mw, sh, sw = gauss_params
        #         A = 1 / (2 * math.pi * sh * sw)
        #         B = (self.h - mh) ** 2 / (2 * sh ** 2)
        #         C = (self.w - mw) ** 2 / (2 * sw ** 2)
        #         return A * np.exp(-(B + C))
        #
        #     def generate_pmask(self, gauss_params, g_scale=1.0):
        #         pmask = np.zeros([self.H, self.W])
        #         for gauss_param in gauss_params:
        #             z = self._gaussian_2d(gauss_param)
        #             pmask += (z / z.max()) * g_scale
        #         return pmask
        #
        # def objective_function(*args):
        #     # tic = time.time()
        #     gauss_params = args
        #     params = []
        #     for mh, mw in zip(gauss_params[::2], gauss_params[1::2]):
        #         noise_level = 0.025
        #         # # print("add_noise", noise_level)
        #         # mh, mw = add_noise(mh, mw, noise_level=noise_level)
        #
        #         params.append((mh, mw, sigma, sigma))
        #     gauss_params = params
        #     if sigma not in gauss_params_hist:
        #         gauss_params_hist[sigma] = []
        #     if sigma not in pred_score_hist:
        #         pred_score_hist[sigma] = []
        #     gauss_params_hist[sigma].append(gauss_params)
        #     # print('\ngauss_params', gauss_params)
        #     pmask = GaussPMask(224, 224).generate_pmask(gauss_params, g_scale=1.0)
        #     pmask = np.clip(pmask, 0, 1)
        #     pmask = torch.tensor(pmask, dtype=torch.float32).to(dev)
        #
        #     # # PRESERVE_VARIANT
        #     # x = perturbation.apply(pmask)
        #     # y = model(x)
        #     # y = torch.sigmoid(y)
        #     # # Get reward.
        #     # pred_score = y.squeeze()[target]
        #     # loss = - pred_score
        #     # pred_score_hist[sigma].append(pred_score.data.cpu().numpy())
        #
        #     # # DELETE_VARIANT
        #     # x_del = perturbation.apply(1 - pmask)
        #     # y_del = model(x_del)
        #     # y_del = torch.sigmoid(y_del)
        #     # pred_score_del = y_del.squeeze()[target]
        #     # loss = pred_score
        #
        #     # DUAL_VARIANT
        #     x = perturbation.apply(pmask)
        #     y = model(x)
        #     y = torch.sigmoid(y)
        #     pred_score = y.squeeze()[target]
        #     x_del = perturbation.apply(1 - pmask)
        #     y_del = model(x_del)
        #     y_del = torch.sigmoid(y_del)
        #     pred_score_del = y_del.squeeze()[target]
        #     # print('pred_score', pred_score, 'pred_score_del', pred_score_del)
        #     loss = - (pred_score - pred_score_del)
        #     pred_score_hist[sigma].append(pred_score.data.cpu().numpy())
        #
        #     # toc = time.time()
        #     # inference_times.append(toc - tic)
        #
        #     return loss
        #
        # num_gaussians = 1
        # n_sub_calls = 1
        # # http://dlib.net/dlib/global_optimization/global_function_search_abstract.h.html#global_function_search
        # # http://dlib.net/dlib/global_optimization/upper_bound_function_abstract.h.html
        # spec = dlib.function_spec([0.01] * num_gaussians * 2, [0.99] * num_gaussians * 2)
        # perturbation = Perturbation(input)
        #
        # solver_epsilon = 0.1
        # n_calls = 50
        # sigma_list = np.linspace(0.1, 0.25, 3)
        # # sigma_list = np.linspace(0.075, 0.25, 4)
        #
        # # pure_random_search_probability = 0.02  # default is 0.02
        # seed_incr = 0
        # print('LIPO: eps', solver_epsilon, 'n_calls', n_calls, 'sigmas', sigma_list,
        #       # 'rand_search_prob',
        #       # pure_random_search_probability,
        #       'seed_incr', seed_incr)
        #
        # # import time
        # # tic = time.time()
        # for sigma_idx, sigma in enumerate(sigma_list):
        #     # Simple init from scratch
        #     opt = dlib.global_function_search(spec)
        #     opt.set_solver_epsilon(solver_epsilon)
        #     # opt.set_pure_random_search_probability(pure_random_search_probability)
        #     opt.set_seed(sigma_idx + seed_incr)
        #     for i in range(n_calls):
        #         next = opt.get_next_x()
        #         next.set(-objective_function(*next.x))
        #     a = 1
        # # toc = time.time()
        # # print(f"perturbation + inference total time took {sum(inference_times)}")
        # # print(f"Whole time took                          {toc - tic}")
        # # print(f"optimizer time took                      {toc - tic - sum(inference_times)}")
        #
        #
        # # Aggregate masks
        # actual_n_calls = n_calls - (n_calls % n_sub_calls)
        # saliency_map = np.zeros((len(sigma_list), 224, 224))
        # for sigma_idx, sigma in enumerate(sigma_list):
        #     saliency_map_tmp = np.zeros((224, 224))
        #     for j in range(actual_n_calls):
        #         gp = gauss_params_hist[sigma][j]
        #         pmask = GaussPMask(224, 224).generate_pmask(gp, g_scale=1.0)
        #         score = pred_score_hist[sigma][j]
        #         saliency_map_tmp += pmask * score
        #     saliency_map_tmp = saliency_map_tmp / saliency_map_tmp.max()
        #     saliency_map[sigma_idx] = saliency_map_tmp
        # saliency_map = saliency_map.sum(axis=0)
        # saliency_map /= saliency_map.max()
        # a = 1









        # AISE DIRECT
        from scipy.optimize import direct, Bounds
        import math
        import dlib
        import matplotlib.pyplot as plt
        import time
        from scipy import signal

        # gauss_params_hist = []
        # pred_score_hist = []
        gauss_params_hist = {}
        pred_score_hist = {}

        inference_times = []

        class Perturbation:
            def __init__(self, input):
                self.input = input

            def apply(self, mask):
                return self.input * mask

        def shift_image(X, dx, dy):
            X = np.roll(X, dy, axis=0)
            X = np.roll(X, dx, axis=1)
            if dy > 0:
                X[:dy, :] = 0
            elif dy < 0:
                X[dy:, :] = 0
            if dx > 0:
                X[:, :dx] = 0
            elif dx < 0:
                X[:, dx:] = 0
            return X

        class GaussPMask:
            def __init__(self, H, W):
                self.H = H
                self.W = W
                h = np.linspace(0, 1, self.H)
                w = np.linspace(0, 1, self.W)
                self.h, self.w = np.meshgrid(w, h)

            def _gaussian_2d(self, gauss_params):
                mh, mw, sh, sw = gauss_params
                A = 1 / (2 * math.pi * sh * sw)
                B = (self.h - mh) ** 2 / (2 * sh ** 2)
                C = (self.w - mw) ** 2 / (2 * sw ** 2)
                return A * np.exp(-(B + C))

            def _signal_2d(self, params, kern1d):
                mh, mw, sh, sw = params

                # Create a larger 2D image (you would replace this with your own image data)
                larger_image_size = (224, 224)
                larger_image = np.zeros(larger_image_size)

                # Create a smaller 2D image (you would replace this with your own image data)
                s, _ = kern1d.shape
                smaller_image_size = (s, s)
                smaller_image = np.outer(kern1d, kern1d)

                # Calculate the position to place the smaller image at the center of the larger image
                start_x = larger_image_size[1] // 2 - smaller_image_size[1] // 2
                start_y = larger_image_size[0] // 2 - smaller_image_size[0] // 2

                # Create a copy of the larger image and place the smaller image at the center
                result_image = np.copy(larger_image)
                result_image[start_y:start_y + smaller_image_size[0], start_x:start_x + smaller_image_size[1]] = smaller_image

                dh, dw = int((mh - 0.5) * 224), int((mw - 0.5) * 224)
                result_image = shift_image(result_image, dh, dw)
                return result_image

            def generate_pmask(self, gauss_params, g_scale=1.0):
                pmask = np.zeros([self.H, self.W])
                for gauss_param in gauss_params:
                    z = self._gaussian_2d(gauss_param)
                    pmask += (z / z.max()) * g_scale

                    # # # Different kernels
                    # # https://docs.scipy.org/doc/scipy-0.14.0/reference/signal.html#module-scipy.signal
                    # # kern1d = signal.gaussian(224, 35).reshape(224, 1)
                    # # kern1d = signal.cosine(150).reshape(150, 1)
                    # # kern1d = signal.triang(150).reshape(150, 1)
                    # size = {
                    #     0.1: 110,
                    #     0.175: 180,
                    #     0.25: 224,
                    # }
                    # kern1d = signal.hann(size[gauss_param[-1]]).reshape(size[gauss_param[-1]], 1)
                    # signal_kern = self._signal_2d(gauss_param, kern1d)
                    # signal_kern /= signal_kern.max()
                    # pmask += signal_kern

                return pmask

        # def add_noise(mh, mw, noise_level=0.03):
        #     clean = np.array([mh, mw])
        #     noise = (np.random.rand(2) - 0.5) * noise_level
        #     res = clean + noise
        #     res = np.clip(res, 0.01, 0.99)
        #     return res[0], res[1]

        def objective_function(args):
            # tic = time.time()
            mh, mw = args

            # noise_level = 0.025
            # # print("add_noise", noise_level)
            # mh, mw = add_noise(mh, mw, noise_level=noise_level)

            params = []
            params.append((mh, mw, sigma, sigma))
            gauss_params = params

            # gauss_params_hist.append(gauss_params)
            if sigma not in gauss_params_hist:
                gauss_params_hist[sigma] = []
            if sigma not in pred_score_hist:
                pred_score_hist[sigma] = []
            gauss_params_hist[sigma].append(gauss_params)

            # print('\ngauss_params', gauss_params)

            pmask = GaussPMask(224, 224).generate_pmask(gauss_params, g_scale=1.0)
            pmask = np.clip(pmask, 0, 1)
            pmask = torch.tensor(pmask, dtype=torch.float32).to(dev)


            # # PRESERVE_VARIANT
            # x = perturbation.apply(pmask)
            # y = model(x)
            # y = torch.sigmoid(y)
            # # Get reward.
            # pred_score = y.squeeze()[target]
            # loss = - pred_score

            # DUAL_VARIANT
            x = perturbation.apply(pmask)
            y = model(x)
            y = torch.sigmoid(y)
            pred_score = y.squeeze()[target]
            x_del = perturbation.apply(1 - pmask)
            y_del = model(x_del)
            y_del = torch.sigmoid(y_del)
            pred_score_del = y_del.squeeze()[target]
            loss = - (pred_score - pred_score_del)

            pred_score_hist[sigma].append(pred_score.data.cpu().numpy())

            # toc = time.time()
            # inference_times.append(toc - tic)

            return loss.data.cpu().numpy()

        perturbation = Perturbation(input)

        solver_epsilon = 0.1
        n_calls = 50
        sigma_list = np.linspace(0.1, 0.25, 3)
        # sigma_list = np.linspace(0.075, 0.25, 4)
        # sigma_list = [0.175]
        locally_biased = False

        # tic = time.time()
        for sigma_idx, sigma in enumerate(sigma_list):
            bounds = Bounds([0.01, 0.01], [0.99, 0.99])
            _ = direct(objective_function, bounds,
                            eps=solver_epsilon,
                            maxfun=n_calls,
                            locally_biased=locally_biased)
        # toc = time.time()
        # print(f"perturbation + inference total time took {sum(inference_times)}")
        # print(f"Whole time took                          {toc - tic}")
        # print(f"optimizer time took                      {toc - tic - sum(inference_times)}")

        print('DIRECT: solver_epsilon', solver_epsilon, 'locally_biased', locally_biased, 'n_calls', n_calls, 'sigma_list', sigma_list)

        # Aggregate masks
        saliency_map = np.zeros((len(sigma_list), 224, 224))
        for sigma_idx, sigma in enumerate(sigma_list):
            saliency_map_tmp = np.zeros((224, 224))
            for j in range(n_calls):
                gp = gauss_params_hist[sigma][j]
                pmask = GaussPMask(224, 224).generate_pmask(gp, g_scale=1.0)
                score = pred_score_hist[sigma][j]
                saliency_map_tmp += pmask * score
            saliency_map_tmp = saliency_map_tmp / saliency_map_tmp.max()
            saliency_map[sigma_idx] = saliency_map_tmp

        saliency_map = saliency_map.sum(axis=0)
        saliency_map /= saliency_map.max()

        a = 1





        # # Visualize masks
        # from PIL import Image
        # import cv2
        # import math
        # data_path = "/home/etsykuno/TorchRay/data/datasets/voc/VOCdevkit/VOCdevkit_2007/VOC2007/JPEGImages/"
        # init_image_name = '000333.jpg'
        # init_img = Image.open(os.path.join(data_path, init_image_name)).convert("RGB")
        # init_img = np.array(init_img)[:, :, ::-1]
        # init_img = cv2.resize(init_img, (224, 224))
        #
        # # # RISE
        # # s = (np.random.rand(7, 7) < 0.5).astype(np.float)
        # # s = cv2.resize(s, (init_img.shape[1], init_img.shape[0]))
        # # pert_img = np.transpose((np.transpose(init_img, (2, 0, 1)) * s), (1, 2, 0)).astype(np.uint8)
        # # pmask = GaussPMask(224, 224).generate_pmask([(0.7, 0.7, 0.2, 0.2)], g_scale=1.0)
        # # s = cv2.resize(pmask, (init_img.shape[1], init_img.shape[0]))
        # # pert_img = np.transpose((np.transpose(init_img, (2, 0, 1)) * s), (1, 2, 0)).astype(np.uint8)
        #
        # plt.imshow(init_img[:, :, ::-1])
        # plt.show()
        # plt.imshow(init_img[:, :, ::-1])
        # plt.imshow(saliency_map, cmap='jet', alpha=0.5)
        # plt.show()
        #
        # init_img_ori = np.copy(init_img)
        # for gp in gauss_params_hist:
        #     for idx, p in enumerate(gauss_params_hist[gp]):  # gauss_params_hist[gp]:
        #         x, y, _, _ = p[0]  # p[0]
        #         w = 1
        #         init_img[int(y * init_img.shape[0]) - w:int(y * init_img.shape[0]) + w,
        #         int(x * init_img.shape[0]) - w:int(x * init_img.shape[0]) + w, :] = np.array(
        #             [0, 0, 255])
        #         score = pred_score_hist[gp][idx]
        #         _ = cv2.putText(init_img, str(round(float(score) * 100)),
        #                         (int(x * init_img.shape[0]), int(y * init_img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX,
        #                         0.35, (0, 0, 255))
        # plt.imshow(init_img[:, :, ::-1])
        # plt.show()
        #
        # pert_img = np.transpose((np.transpose(init_img_ori[:, :, ::-1], (2, 0, 1)) * saliency_map), (1, 2, 0)).astype(np.uint8)
        # plt.imshow(pert_img)
        # plt.show()




        # # Accuracy-aware sigma iteration
        # y = model(input)
        # y = torch.sigmoid(y)
        # ori_pred_score = y.squeeze()
        #
        # n_calls = 15
        # # sigma_list = np.linspace(0.25, 0.075, 7)
        # sigma_list = np.linspace(0.3, 0.05, 7)
        #
        # actual_n_calls = n_calls - (n_calls % n_sub_calls)
        #
        # spec = dlib.function_spec([0.01] * num_gaussians * 2, [0.99] * num_gaussians * 2)
        # opt = dlib.global_function_search(spec)
        # opt.set_solver_epsilon(solver_epsilon)
        # opt.set_pure_random_search_probability(0.1) # 0.02
        # early_stopped = False
        # pred_score_thresh = 0.1 # ori_pred_score[target] / 6
        # print('sigma_list', sigma_list)
        # for sigma_idx, sigma in enumerate(sigma_list):
        #     print('sigma', sigma)
        #     for i in range(n_calls // n_sub_calls):
        #         next_pool = []
        #         for j in range(n_sub_calls):
        #             next = opt.get_next_x()
        #             next_pool.append(next)
        #         for next in next_pool:
        #             next.set(-objective_function(*next.x))
        #     max_pred_score = np.array(pred_score_hist[sigma_idx * actual_n_calls: (sigma_idx + 1) * actual_n_calls]).max()
        #     print('max_pred_score', max_pred_score)
        #     if max_pred_score < pred_score_thresh:
        #         print('*** early_stopped ***')
        #         early_stopped = True
        #         break
        #     a = 1
        # a = 1
        #
        # # Add new sigmas to the list
        # if early_stopped:
        #     left_iter_budget = len(sigma_list) - (sigma_idx + 1)
        #     # sigma_list_after_early_stop = np.linspace(sigma_list[0], sigma_list[sigma_idx], left_iter_budget + 1, False)[1:] # Wide range
        #     sigma_list_after_early_stop = np.linspace(sigma_list[sigma_idx - 1], sigma_list[sigma_idx], left_iter_budget + 1, False)[1:] # Narrow range
        #     for sigma in sigma_list_after_early_stop:
        #         print('sigma', sigma)
        #         for i in range(n_calls // n_sub_calls):
        #             next_pool = []
        #             for j in range(n_sub_calls):
        #                 next = opt.get_next_x()
        #                 next_pool.append(next)
        #             for next in next_pool:
        #                 next.set(-objective_function(*next.x))
        #
        # # Add more observations to the existed sigmas
        # # TODO
        #
        # saliency_map = np.zeros((len(sigma_list), 224, 224))
        # for i in range(len(sigma_list)):
        #     max_pred_score = np.array(pred_score_hist[i * actual_n_calls : (i + 1) * actual_n_calls]).max()
        #     saliency_map_tmp = np.zeros((224, 224))
        #
        #     # if max_pred_score > pred_score_thresh:
        #     for j in range(actual_n_calls):
        #         gp = gauss_params_hist[i * actual_n_calls + j]
        #         pmask = GaussPMask(224, 224).generate_pmask(gp, g_scale=1.0)
        #         score = pred_score_hist[i * actual_n_calls + j]
        #         saliency_map_tmp += pmask * score
        #
        #     saliency_map_tmp = saliency_map_tmp / saliency_map_tmp.max() # ???
        #     saliency_map[i] = saliency_map_tmp











        # # 3D plot of saliency
        # import numpy
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        #
        # # Set up grid and test data
        # nx, ny = 224, 224
        # x = range(nx)
        # y = range(ny)
        #
        # # data = numpy.random.random((nx, ny))
        #
        # hf = plt.figure()
        # ha = hf.add_subplot(111, projection='3d')
        #
        # X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
        # ha.plot_surface(X, Y, saliency_map)
        #
        # plt.show()







        # # KDE
        # from KDEpy import NaiveKDE
        # import matplotlib.pyplot as plt
        #
        # data = np.array([[item[0][1], item[0][0]] for item in gauss_params_hist])
        # weights = np.array(pred_score_hist)
        # x, y = NaiveKDE(kernel='gaussian', bw=sigma).fit(data, weights).evaluate(224)
        #
        # y_ = y.reshape(32, 32)
        # # y_ /= y_.max()
        #
        # plt.plot(x, y)
        # plt.show()













        # from KDEpy import FFTKDE
        #
        # # Create 2D data of shape (obs, dims)
        # data = np.random.randn(2 ** 4, 2)
        #
        # grid_points = 2 ** 7  # Grid points in each dimension
        # N = 16  # Number of contours
        #
        # fig = plt.figure(figsize=(30, 10))
        #
        # for plt_num, norm in enumerate([1, 2, np.inf], 1):
        #     ax = fig.add_subplot(1, 3, plt_num)
        #     ax.set_title(f'Norm $p={norm}$')
        #
        #     # Compute the kernel density estimate
        #     kde = FFTKDE(kernel='box', norm=norm)
        #     grid, points = kde.fit(data).evaluate(grid_points)
        #
        #     # The grid is of shape (obs, dims), points are of shape (obs, 1)
        #     x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
        #     z = points.reshape(grid_points, grid_points).T
        #
        #     # Plot the kernel density estimate
        #     ax.contour(x, y, z, N, linewidths=0.8, colors='k')
        #     ax.contourf(x, y, z, N, cmap="RdBu_r")
        #     ax.plot(data[:, 0], data[:, 1], 'ok', ms=3)
        #
        # plt.tight_layout()
        # plt.show()













        # # True score map
        # from dlib import find_min_global
        # import math
        # import dlib
        # import matplotlib.pyplot as plt
        #
        # gauss_params_hist = []
        # pred_score_hist = []
        #
        # class Perturbation:
        #     def __init__(self, input):
        #         self.input = input
        #
        #     def apply(self, mask):
        #         return self.input * mask
        #
        # class GaussPMask:
        #     def __init__(self, H, W):
        #         self.H = H
        #         self.W = W
        #         h = np.linspace(0, 1, self.H)
        #         w = np.linspace(0, 1, self.W)
        #         self.h, self.w = np.meshgrid(w, h)
        #
        #     def _gaussian_2d(self, gauss_params):
        #         mh, mw, sh, sw = gauss_params
        #         A = 1 / (2 * math.pi * sh * sw)
        #         B = (self.h - mh) ** 2 / (2 * sh ** 2)
        #         C = (self.w - mw) ** 2 / (2 * sw ** 2)
        #         return A * np.exp(-(B + C))
        #
        #     def generate_pmask(self, gauss_params, g_scale=1.2):
        #         pmask = np.zeros([self.H, self.W])
        #         for gauss_param in gauss_params:
        #             z = self._gaussian_2d(gauss_param)
        #             pmask += (z / z.max()) * g_scale
        #         return pmask
        #
        # def objective_function(mh, mw, sigma):
        #     params = []
        #     params.append((mh, mw, sigma, sigma))
        #     gauss_params = params
        #
        #     gauss_params_hist.append(gauss_params)
        #
        #     print('\ngauss_params', gauss_params)
        #
        #     pmask = GaussPMask(224, 224).generate_pmask(gauss_params, g_scale=1.0)
        #     pmask = np.clip(pmask, 0, 1)
        #
        #     pmask = torch.tensor(pmask, dtype=torch.float32).to(dev)
        #
        #     x = perturbation.apply(pmask)
        #     y = model(x)
        #
        #     y = torch.sigmoid(y)
        #
        #     # Get reward.
        #     pred_score = y.squeeze()[target]
        #     loss = pred_score
        #
        #     # print('pred_score', pred_score)
        #     pred_score_hist.append(pred_score.data.cpu().numpy())
        #
        #     return loss
        #
        # perturbation = Perturbation(input)
        #
        # sigma_list = [0.1, 0.15, 0.2, 0.25]
        # sigma_list = [0.1, 0.15, 0.2, 0.25]
        # print('sigma_list', sigma_list)
        # saliency_map = np.zeros((len(sigma_list), 224, 224))
        # for sigma_idx, sigma in enumerate(sigma_list):
        #     print('sigma', sigma)
        #     saliency_map_tmp = np.zeros((224, 224))
        #     for i in range(0, 224):
        #         tmp = []
        #         for j in range(0, 224):
        #             gauss_params = [(((i + 1) / 224), (j + 1) / 224, sigma, sigma)]
        #             pmask = GaussPMask(224, 224).generate_pmask(gauss_params, g_scale=1.0)
        #             pmask = torch.tensor(pmask, dtype=torch.float32).to(dev)
        #             x = perturbation.apply(pmask)
        #             tmp.append(x)
        #         x = torch.cat(tmp)
        #         y = model(x)
        #         y = torch.sigmoid(y)
        #         pred_score = y.squeeze()[:, target]
        #         saliency_map_tmp[:, i] = pred_score.cpu().numpy()
        #
        #         # score_map[j, i] = objective_function((i + 1) / 224, (j + 1) / 224, sigma)
        #         # print(i * 224)
        #
        #     saliency_map[sigma_idx] = saliency_map_tmp
        #
        # saliency_map = saliency_map.sum(axis=0)
        #
        # saliency_map /= saliency_map.max()
        #
        # a = 1

        # from PIL import Image
        #
        # data_path = "/home/etsykuno/TorchRay/data/datasets/voc/VOCdevkit/VOCdevkit_2007/VOC2007/JPEGImages/"
        # init_image_name = "000025.jpg"
        # init_img = Image.open(os.path.join(data_path, init_image_name)).convert("RGB")
        # init_img = np.array(init_img)
        # init_img_ = cv2.resize(init_img, (224, 224), interpolation=cv2.INTER_LINEAR)
        #
        # fig = plt.figure()
        #
        # plt.axis('off')
        # plt.imshow(init_img_)
        # plt.imshow(saliency_map[0], cmap='jet', alpha=0.5)
        # # plt.show()
        # fig.savefig('/home/etsykuno/tmp/AISE_paper/005.png', bbox_inches='tight', pad_inches=0)





        # # Run experiment for a single image
        # import itertools
        #
        # # n_calls_list = [25] #np.random.randint(15, high=30, size=2)  # n_calls = 25
        #
        # n_calls_list = [25, 30]  # np.random.randint(15, high=30, size=2)  # n_calls = 25
        #
        # sigma_lists = [[0.1, 0.15, 0.2, 0.25], [0.1, 0.175, 0.25]]
        # # for i in range(10):
        # #     sigma_lists.append(list(sorted(np.random.uniform(low=0.03, high=0.3, size=(4,)))))
        #
        # num_gaussians_list = [1, 2]
        #
        # n_sub_calls_list = [1]
        #
        # solver_epsilon_list = np.random.uniform(low=0.01, high=0.5, size=(10,))
        #
        # scale_with_max_list = [0]
        #
        # g_scale_list = [1.0, 1.1]
        #
        # random_search_probability_list = np.random.uniform(low=0.01, high=0.5, size=(10,))
        #
        # list_of_options = list(itertools.product(
        #     n_calls_list,
        #     sigma_lists,
        #     num_gaussians_list,
        #     n_sub_calls_list,
        #     solver_epsilon_list,
        #     scale_with_max_list,
        #     g_scale_list,
        #     random_search_probability_list,
        # ))
        # le = len(list_of_options)
        #
        # for i, (n_calls, sigma_list, num_gaussians, n_sub_calls, solver_epsilon, scale_with_max, g_scale,
        #         random_search_probability) in enumerate(list_of_options):
        #     print(f"\nrun {i} out of {le}")
        #
        #     gauss_params_hist = []
        #     pred_score_hist = []
        #
        #     def objective_function(*args):
        #         gauss_params = args
        #
        #         params = []
        #         for mh, mw in zip(gauss_params[::2], gauss_params[1::2]):
        #             params.append((mh, mw, sigma, sigma))
        #         gauss_params = params
        #
        #         gauss_params_hist.append(gauss_params)
        #
        #         # print('\ngauss_params', gauss_params)
        #
        #         pmask = GaussPMask(224, 224).generate_pmask(gauss_params, g_scale=g_scale)
        #         pmask = np.clip(pmask, 0, 1)
        #
        #         pmask = torch.tensor(pmask, dtype=torch.float32).to(dev)
        #
        #         x = perturbation.apply(pmask)
        #         y = model(x)
        #
        #         y = torch.sigmoid(y)
        #
        #         # Get reward.
        #         pred_score = y.squeeze()[target]
        #         loss = - pred_score
        #
        #         # print('pred_score', pred_score)
        #         pred_score_hist.append(pred_score.data.cpu().numpy())
        #
        #         return loss
        #
        #     # num_gaussians = 1
        #     # n_calls = 25
        #     # n_sub_calls = 1
        #     # solver_epsilon = 0.1
        #     # sigma_list = [0.1, 0.15, 0.2, 0.25]
        #
        #     spec = dlib.function_spec([0.01] * num_gaussians * 2, [0.99] * num_gaussians * 2)
        #     opt = dlib.global_function_search(spec)
        #     opt.set_solver_epsilon(solver_epsilon)
        #     # opt.set_seed()
        #     # opt.get_monte_carlo_upper_bound_sample_num() 5000
        #     # opt.set_relative_noise_magnitude(0.005) # 0.001
        #     opt.set_pure_random_search_probability(random_search_probability)  # 0.02
        #     # for sigma, n_calls in zip(sigma_list, n_calls_list):
        #     for sigma in sigma_list:
        #         for i in range(n_calls // n_sub_calls):
        #             next_pool = []
        #             for j in range(n_sub_calls):
        #                 next = opt.get_next_x()
        #                 next_pool.append(next)
        #             for next in next_pool:
        #                 mu_h = next.x[0]
        #                 mu_w = next.x[1]
        #                 next.set(-objective_function(mu_h, mu_w))
        #
        #     # Aggregate masks separately per sigma level
        #     actual_n_calls = n_calls - (n_calls % n_sub_calls)
        #     saliency_map = np.zeros((224, 224))
        #     for i in range(len(sigma_list)):
        #         saliency_map_tmp = np.zeros((224, 224))
        #         for j in range(actual_n_calls):
        #             gp = gauss_params_hist[i * actual_n_calls + j]
        #             pmask = GaussPMask(224, 224).generate_pmask(gp, g_scale=1.0)
        #             score = pred_score_hist[i * actual_n_calls + j]
        #             saliency_map_tmp += pmask * score
        #         saliency_map_tmp = saliency_map_tmp / saliency_map_tmp.max()
        #         if scale_with_max:
        #             saliency_map_tmp *= np.array(pred_score_hist[i * actual_n_calls: (i + 1) * actual_n_calls]).max()
        #         saliency_map += saliency_map_tmp
        #
        #     del_auc, ins_auc = deletion_insertion_single_run(model, input, saliency_map, target)
        #     print('n_calls', n_calls)
        #     print('n_sub_calls', n_sub_calls)
        #     print('sigma_list', sigma_list)
        #     print('num_gaussians', num_gaussians)
        #     print('solver_epsilon', solver_epsilon)
        #     print('scale_with_max', scale_with_max)
        #     print('g_scale', g_scale)
        #     print('random_search_probability', random_search_probability)
        #     print(f'Del: {del_auc:.3f}, Ins: {ins_auc:.3f}')
        #     if ins_auc > 0.8:
        #         print('*********************************************************')
        # print('It is over')




        # Restore original random number generator state.

        torch.set_rng_state(state)
        saliency_map = torch.tensor(saliency_map)
        saliency224 = saliency_map

        # Resize saliency mask if needed.
        saliency_map = saliency_map.unsqueeze(0)
        saliency_map = saliency_map.repeat(1, 3, 1, 1)
        saliency_resized = resize_saliency(input,
                                   saliency_map,
                                   resize,
                                   mode=resize_mode)
        return saliency_resized[0, 0, :, :], saliency224
