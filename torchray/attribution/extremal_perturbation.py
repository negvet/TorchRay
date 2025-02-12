# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
This module provides an implementation of the *Extremal Perturbations* (EP)
method of [EP]_ for saliency visualization. The interface is given by
the :func:`extremal_perturbation` function:

.. literalinclude:: ../examples/extremal_perturbation.py
   :language: python
   :linenos:

Extremal perturbations seek to find a region of the input image that maximally
excites a certain output or intermediate activation of a neural network.

.. _ep_perturbations:

Perturbation types
~~~~~~~~~~~~~~~~~~

The :class:`Perturbation` class supports the following perturbation types:

* :attr:`BLUR_PERTURBATION`: Gaussian blur.
* :attr:`FADE_PERTURBATION`: Fade to black.

.. _ep_variants:

Extremal perturbation variants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`extremal_perturbation` function supports the following variants:

* :attr:`PRESERVE_VARIANT`: Find a mask that makes the activations large.
* :attr:`DELETE_VARIANT`: Find a mask that makes the activations small.
* :attr:`DUAL_VARIANT`: Find a mask that makes the activations large and whose
  complement makes the activations small, rewarding the difference between
  these two.

References:

    .. [EP] Ruth C. Fong, Mandela Patrick and Andrea Vedaldi,
            *Understanding Deep Networks via Extremal Perturbations and Smooth Masks,*
            ICCV 2019,
            `<http://arxiv.org/>`__.

"""

from __future__ import division
from __future__ import print_function


__all__ = [
    "extremal_perturbation",
    "Perturbation",
    "simple_reward",
    "contrastive_reward",
    "BLUR_PERTURBATION",
    "FADE_PERTURBATION",
    "PRESERVE_VARIANT",
    "DELETE_VARIANT",
    "DUAL_VARIANT",
]

import time
import math
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from torchray.utils import imsmooth, imsc
from .common import resize_saliency

BLUR_PERTURBATION = "blur"
"""Blur-type perturbation for :class:`Perturbation`."""

FADE_PERTURBATION = "fade"
"""Fade-type perturbation for :class:`Perturbation`."""

PRESERVE_VARIANT = "preserve"
"""Preservation game for :func:`extremal_perturbation`."""

DELETE_VARIANT = "delete"
"""Deletion game for :func:`extremal_perturbation`."""

DUAL_VARIANT = "dual"
"""Combined game for :func:`extremal_perturbation`."""


class Perturbation:
    r"""Perturbation pyramid.

    The class takes as input a tensor :attr:`input` and applies to it
    perturbation of increasing strenght, storing the resulting pyramid as
    the class state. The method :func:`apply` can then be used to generate an
    inhomogeneously perturbed image based on a certain perturbation mask.

    The pyramid :math:`y` is the :math:`L\times C\times H\times W` tensor

    .. math::
        y_{lcvu} = [\operatorname{perturb}(x, \sigma_l)]_{cvu}

    where :math:`x` is the input tensor, :math:`c` a channel, :math:`vu`,
    the spatial location, :math:`l` a perturbation level,  and
    :math:`\operatorname{perturb}` is a perturbation operator.

    For the *blur perturbation* (:attr:`BLUR_PERTURBATION`), the perturbation
    operator amounts to convolution with a Gaussian whose kernel has
    standard deviation :math:`\sigma_l = \sigma_{\mathrm{max}} (1 -  l/ (L-1))`:

    .. math::
        \operatorname{perturb}(x, \sigma_l) = g_{\sigma_l} \ast x

    For the *fade perturbation* (:attr:`FADE_PERTURBATION`),

    .. math::
        \operatorname{perturb}(x, \sigma_l) = \sigma_l \cdot x

    where  :math:`\sigma_l =  l / (L-1)`.

    Note that in all cases the last pyramid level :math:`l=L-1` corresponds
    to the unperturbed input and the first :math:`l=0` to the maximally
    perturbed input.

    Args:
        input (:class:`torch.Tensor`): A :math:`1\times C\times H\times W`
            input tensor (usually an image).
        num_levels (int, optional): Number of pyramid leves. Defaults to 8.
        type (str, optional): Perturbation type (:ref:`ep_perturbations`).
        max_blur (float, optional): :math:`\sigma_{\mathrm{max}}` for the
            Gaussian blur perturbation. Defaults to 20.

    Attributes:
        pyramid (:class:`torch.Tensor`): A :math:`L\times C\times H\times W`
            tensor with :math:`L` ():attr:`num_levels`) increasingly
            perturbed versions of the input tensor.
    """

    def __init__(self, input, num_levels=8, max_blur=20, type=BLUR_PERTURBATION):
        self.type = type
        self.num_levels = num_levels
        self.pyramid = []
        assert num_levels >= 2
        assert max_blur > 0
        with torch.no_grad():
            for sigma in torch.linspace(0, 1, self.num_levels):
                if type == BLUR_PERTURBATION:
                    y = imsmooth(input, sigma=(1 - sigma) * max_blur)
                elif type == FADE_PERTURBATION:
                    y = input * sigma
                else:
                    assert False
                self.pyramid.append(y)
            self.pyramid = torch.cat(self.pyramid, dim=0)

    def apply(self, mask):
        r"""Generate a perturbetd tensor from a perturbation mask.

        The :attr:`mask` is a tensor :math:`K\times 1\times H\times W`
        with spatial dimensions :math:`H\times W` matching the input
        tensor passed upon instantiation of the class. The output
        is a :math:`K\times C\times H\times W` with :math:`K` perturbed
        versions of the input tensor, one for each mask.

        Masks values are in the range 0 to 1, where 1 means that the input
        tensor is copied as is, and 0 that it is maximally perturbed.

        Formally, the output is then given by:

        .. math::
            z_{kcvu} = y_{m_{k1cu}, c, v, u}

        where :math:`k` index the mask, :math:`c` the feature channel,
        :math:`vu` the spatial location, :math:`y` is the pyramid tensor,
        and :math:`m` the mask tensor :attr:`mask`.

        The mask must be in the range :math:`[0, 1]`. Linear interpolation
        is used to index the perturbation level dimension of :math:`y`.

        Args:
            mask (:class:`torch.Tensor`): A :math:`K\times 1\times H\times W`
                input tensor representing :math:`K` masks.

        Returns:
            :class:`torch.Tensor`: A :math:`K\times C\times H\times W` tensor
            with :math:`K` perturbed versions of the input tensor.
        """
        n = mask.shape[0]
        w = mask.reshape(n, 1, *mask.shape[1:])
        w = w * (self.num_levels - 1)
        k = w.floor()
        w = w - k
        k = k.long()

        y = self.pyramid[None, :]
        y = y.expand(n, *y.shape[1:])
        k = k.expand(n, 1, *y.shape[2:])
        y0 = torch.gather(y, 1, k)
        y1 = torch.gather(y, 1, torch.clamp(k + 1, max=self.num_levels - 1))

        return ((1 - w) * y0 + w * y1).squeeze(dim=1)

    def to(self, dev):
        """Switch to another device.

        Args:
            dev: PyTorch device.

        Returns:
            Perturbation: self.
        """
        self.pyramid.to(dev)
        return self

    def __str__(self):
        return (
            f"Perturbation:\n"
            f"- type: {self.type}\n"
            f"- num_levels: {self.num_levels}\n"
            f"- pyramid shape: {list(self.pyramid.shape)}"
        )


def simple_reward(activation, target, variant):
    r"""Simple reward.

    For the :attr:`PRESERVE_VARIANT`, the simple reward is given by:

    .. math::
        z_{k1vu} = y_{n, c, v, u}

    where :math:`y` is the :math:`K\times C\times H\times W` :attr:`activation`
    tensor, :math:`c` the :attr:`target` channel, :math:`k` the mask index
    and :math:`vu` the spatial indices. :math:`c` must be in the range
    :math:`[0, C-1]`.

    For the :attr:`DELETE_VARIANT`, the reward is the opposite.

    For the :attr:`DUAL_VARIANT`, it is given by:

    .. math::
        z_{n1vu} = y_{n, c, v, u} - y_{n + N/2, c, v, u}.

    Args:
        activation (:class:`torch.Tensor`): activation tensor.
        target (int): target channel.
        variant (str): A :ref:`ep_variants`.

    Returns:
        :class:`torch.Tensor`: reward tensor with the same shape as
        :attr:`activation` but a single channel.
    """
    assert isinstance(activation, torch.Tensor)
    assert len(activation.shape) >= 2 and len(activation.shape) <= 4
    assert isinstance(target, int)
    if variant == DELETE_VARIANT:
        reward = - activation[:, target]
    elif variant == PRESERVE_VARIANT:
        reward = activation[:, target]
    elif variant == DUAL_VARIANT:
        bs = activation.shape[0]
        assert bs % 2 == 0
        num_areas = int(bs / 2)
        reward = activation[:num_areas, target] - \
            activation[num_areas:, target]
    else:
        assert False
    return reward


def contrastive_reward(activation, target, variant):
    r"""Contrastive reward.

    For the :attr:`PRESERVE_VARIANT`, the contrastive reward is given by:

    .. math::
        z_{k1vu} = y_{n, c, v, u} - \max_{c'\not= c} y_{n, c', v, u}

    The other variants are derived in the same manner as for
    :func:`simple_reward`.

    Args:
        activation (:class:`torch.Tensor`): activation tensor.
        target (int): target channel.
        variant (str): A :ref:`ep_variants`.

    Returns:
        :class:`torch.Tensor`: reward tensor with the same shape as
        :attr:`activation` but a single channel.
    """
    assert isinstance(activation, torch.Tensor)
    assert len(activation.shape) >= 2 and len(activation.shape) <= 4
    assert isinstance(target, int)

    def get(pred_y, y):
        temp_y = pred_y.clone()
        temp_y[:, y] = -100
        return pred_y[:, y] - temp_y.max(dim=1, keepdim=True)[0]

    if variant == DELETE_VARIANT:
        reward = - get(activation, target)
    elif variant == PRESERVE_VARIANT:
        reward = get(activation, target)
    elif variant == DUAL_VARIANT:
        bs = activation.shape[0]
        assert bs % 2 == 0
        num_areas = int(bs / 2)
        reward = (
            get(activation[:num_areas], target) -
            get(activation[num_areas:], target)
        )
    else:
        assert False

    return reward


class MaskGenerator:
    r"""Mask generator.

    The class takes as input the mask parameters and returns
    as output a mask.

    Args:
        shape (tuple of int): output shape.
        step (int): parameterization step in pixels.
        sigma (float): kernel size.
        clamp (bool, optional): whether to clamp the mask to [0,1]. Defaults to True.
        pooling_mehtod (str, optional): `'softmax'` (default),  `'sum'`, '`sigmoid`'.

    Attributes:
        shape (tuple): the same as the specified :attr:`shape` parameter.
        shape_in (tuple): spatial size of the parameter tensor.
        shape_out (tuple): spatial size of the output mask including margin.
    """

    def __init__(self, shape, step, sigma, clamp=True, pooling_method='softmax'):
        self.shape = shape
        self.step = step
        self.sigma = sigma
        self.coldness = 20
        self.clamp = clamp
        self.pooling_method = pooling_method

        assert int(step) == step

        # self.kernel = lambda z: (z < 1).float()
        self.kernel = lambda z: torch.exp(-2 * ((z - .5).clamp(min=0)**2))

        self.margin = self.sigma
        # self.margin = 0
        self.padding = 1 + math.ceil((self.margin + sigma) / step)
        self.radius = 1 + math.ceil(sigma / step)
        self.shape_in = [math.ceil(z / step) for z in self.shape]
        self.shape_mid = [
            z + 2 * self.padding - (2 * self.radius + 1) + 1
            for z in self.shape_in
        ]
        self.shape_up = [self.step * z for z in self.shape_mid]
        self.shape_out = [z - step + 1 for z in self.shape_up]

        self.weight = torch.zeros((
            1,
            (2 * self.radius + 1)**2,
            self.shape_out[0],
            self.shape_out[1]
        ))

        step_inv = [
            torch.tensor(zm, dtype=torch.float32) /
            torch.tensor(zo, dtype=torch.float32)
            for zm, zo in zip(self.shape_mid, self.shape_up)
        ]

        for ky in range(2 * self.radius + 1):
            for kx in range(2 * self.radius + 1):
                uy, ux = torch.meshgrid(
                    torch.arange(self.shape_out[0], dtype=torch.float32),
                    torch.arange(self.shape_out[1], dtype=torch.float32)
                )
                iy = torch.floor(step_inv[0] * uy) + ky - self.padding
                ix = torch.floor(step_inv[1] * ux) + kx - self.padding

                delta = torch.sqrt(
                    (uy - (self.margin + self.step * iy))**2 +
                    (ux - (self.margin + self.step * ix))**2
                )

                k = ky * (2 * self.radius + 1) + kx

                self.weight[0, k] = self.kernel(delta / sigma)

    def generate(self, mask_in):
        r"""Generate a mask.

        The function takes as input a parameter tensor :math:`\bar m` for
        :math:`K` masks, which is a :math:`K\times 1\times H_i\times W_i`
        tensor where `H_i\times W_i` are given by :attr:`shape_in`.

        Args:
            mask_in (:class:`torch.Tensor`): mask parameters.

        Returns:
            tuple: a pair of mask, cropped and full. The cropped mask is a
            :class:`torch.Tensor` with the same spatial shape :attr:`shape`
            as specfied upon creating this object. The second mask is the same,
            but with an additional margin and shape :attr:`shape_out`.
        """
        mask = F.unfold(mask_in,
                        (2 * self.radius + 1,) * 2,
                        padding=(self.padding,) * 2)
        mask = mask.reshape(
            len(mask_in), -1, self.shape_mid[0], self.shape_mid[1])
        mask = F.interpolate(mask, size=self.shape_up, mode='nearest')
        mask = F.pad(mask, (0, -self.step + 1, 0, -self.step + 1))
        mask = self.weight * mask

        if self.pooling_method == 'sigmoid':
            if self.coldness == float('+Inf'):
                mask = (mask.sum(dim=1, keepdim=True) - 5 > 0).float()
            else:
                mask = torch.sigmoid(
                    self.coldness * mask.sum(dim=1, keepdim=True) - 3
                )
        elif self.pooling_method == 'softmax':
            if self.coldness == float('+Inf'):
                mask = mask.max(dim=1, keepdim=True)[0]
            else:
                mask = (
                    mask * F.softmax(self.coldness * mask, dim=1)
                ).sum(dim=1, keepdim=True)

        elif self.pooling_method == 'sum':
            mask = mask.sum(dim=1, keepdim=True)
        else:
            assert False, f"Unknown pooling method {self.pooling_method}"
        m = round(self.margin)
        if self.clamp:
            mask = mask.clamp(min=0, max=1)
        cropped = mask[:, :, m:m + self.shape[0], m:m + self.shape[1]]
        return cropped, mask

    def to(self, dev):
        """Switch to another device.

        Args:
            dev: PyTorch device.

        Returns:
            MaskGenerator: self.
        """
        self.weight = self.weight.to(dev)
        return self


def extremal_perturbation(model,
                          input,
                          target,
                          areas=[0.1],
                          perturbation=BLUR_PERTURBATION,
                          max_iter=800,
                          num_levels=8,
                          step=7,
                          sigma=21,
                          jitter=True,
                          variant=PRESERVE_VARIANT,
                          print_iter=None,
                          debug=False,
                          reward_func=simple_reward,
                          resize=False,
                          resize_mode='bilinear',
                          smooth=0):
    r"""Compute a set of extremal perturbations.

    The function takes a :attr:`model`, an :attr:`input` tensor :math:`x`
    of size :math:`1\times C\times H\times W`, and a :attr:`target`
    activation channel. It produces as output a
    :math:`K\times C\times H\times W` tensor where :math:`K` is the number
    of specified :attr:`areas`.

    Each mask, which has approximately the specified area, is searched
    in order to maximise the (spatial average of the) activations
    in channel :attr:`target`. Alternative objectives can be specified
    via :attr:`reward_func`.

    Args:
        model (:class:`torch.nn.Module`): model.
        input (:class:`torch.Tensor`): input tensor.
        target (int): target channel.
        areas (float or list of floats, optional): list of target areas for saliency
            masks. Defaults to `[0.1]`.
        perturbation (str, optional): :ref:`ep_perturbations`.
        max_iter (int, optional): number of iterations for optimizing the masks.
        num_levels (int, optional): number of buckets with which to discretize
            and linearly interpolate the perturbation
            (see :class:`Perturbation`). Defaults to 8.
        step (int, optional): mask step (see :class:`MaskGenerator`).
            Defaults to 7.
        sigma (float, optional): mask smoothing (see :class:`MaskGenerator`).
            Defaults to 21.
        jitter (bool, optional): randomly flip the image horizontally at each iteration.
            Defaults to True.
        variant (str, optional): :ref:`ep_variants`. Defaults to
            :attr:`PRESERVE_VARIANT`.
        print_iter (int, optional): frequency with which to print losses.
            Defaults to None.
        debug (bool, optional): If True, generate debug plots.
        reward_func (function, optional): function that generates reward tensor
            to backpropagate.
        resize (bool, optional): If True, upsamples the masks the same size
            as :attr:`input`. It is also possible to specify a pair
            (width, height) for a different size. Defaults to False.
        resize_mode (str, optional): Upsampling method to use. Defaults to
            ``'bilinear'``.
        smooth (float, optional): Apply Gaussian smoothing to the masks after
            computing them. Defaults to 0.

    Returns:
        A tuple containing the masks and the energies.
        The masks are stored as a :class:`torch.Tensor`
        of dimension
    """
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300
    device = input.device

    regul_weight_last = max(regul_weight / 2, 1)

    if debug:
        print(
            f"extremal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(input.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    # TODO(av): undo on leaving the function.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    perturbation = Perturbation(
        input,
        num_levels=num_levels,
        type=perturbation
    ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in

    use_gauss_mask = 0
    use_smbo = 0
    single_gauss = 0
    reuse_ep_mask_generation = 0
    mix_of_gauss = 1

    if not use_gauss_mask:
        pmask = torch.ones(len(areas), 1, h, w).to(device)
        if debug:
            print(f"- mask resolution:\n  {pmask.shape}")

        # Prepare reference area vector.
        max_area = np.prod(mask_generator.shape_out)
        reference = torch.ones(len(areas), max_area).to(device)
        for i, a in enumerate(areas):
            reference[i, :int(max_area * (1 - a))] = 0

        # Initialize optimizer.
        optimizer = optim.SGD([pmask],
                              lr=learning_rate,
                              momentum=momentum,
                              dampening=momentum)
        hist = torch.zeros((len(areas), 2, 0))

        for t in range(max_iter):
            pmask.requires_grad_(True)

            # print(pmask.min().data, pmask.max().data, pmask.mean().data)

            # Generate the mask.
            mask_, mask = mask_generator.generate(pmask)

            # Apply the mask.
            if variant == DELETE_VARIANT:
                x = perturbation.apply(1 - mask_)
            elif variant == PRESERVE_VARIANT:
                x = perturbation.apply(mask_)
            elif variant == DUAL_VARIANT:
                x = torch.cat((
                    perturbation.apply(mask_),
                    perturbation.apply(1 - mask_),
                ), dim=0)
            else:
                assert False

            # Apply jitter to the masked data.
            if jitter and t % 2 == 0:
                x = torch.flip(x, dims=(3,))




            # Evaluate the model on the masked data.
            # with torch.no_grad():
            y = model(x)




            # Get reward.
            reward = reward_func(y, target, variant=variant)

            # Reshape reward and average over spatial dimensions.
            reward = reward.reshape(len(areas), -1).mean(dim=1)

            # Area regularization.
            mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
            regul = - ((mask_sorted - reference)**2).mean(dim=1) * regul_weight
            energy = (reward + regul).sum()

            # Gradient step.
            optimizer.zero_grad()
            (- energy).backward()
            optimizer.step()

            pmask.data = pmask.data.clamp(0, 1)

            # Record energy.
            hist = torch.cat(
                (hist,
                 torch.cat((
                     reward.detach().cpu().view(-1, 1, 1),
                     regul.detach().cpu().view(-1, 1, 1)
                 ), dim=1)), dim=2)

            # Adjust the regulariser/area constraint weight.
            regul_weight *= 1.0035

            # Diagnostics.
            debug_this_iter = debug and (t in (0, max_iter - 1)
                                         or regul_weight / regul_weight_last >= 2)

            if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
                print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
                for i, area in enumerate(areas):
                    print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                        area,
                        hist[i, 0, -1],
                        hist[i, 1, -1]), end="")
                print()

            if debug_this_iter:
                regul_weight_last = regul_weight
                for i, a in enumerate(areas):
                    plt.figure(i, figsize=(20, 6))
                    plt.clf()
                    ncols = 4 if variant == DUAL_VARIANT else 3
                    plt.subplot(1, ncols, 1)
                    plt.plot(hist[i, 0].numpy())
                    plt.plot(hist[i, 1].numpy())
                    plt.plot(hist[i].sum(dim=0).numpy())
                    plt.legend(('energy', 'regul', 'both'))
                    plt.title(f'target area:{a:.2f}')
                    plt.subplot(1, ncols, 2)
                    imsc(mask[i], lim=[0, 1])
                    plt.title(
                        f"min:{mask[i].min().item():.2f}"
                        f" max:{mask[i].max().item():.2f}"
                        f" area:{mask[i].sum() / mask[i].numel():.2f}")
                    plt.subplot(1, ncols, 3)
                    imsc(x[i])
                    if variant == DUAL_VARIANT:
                        plt.subplot(1, ncols, 4)
                        imsc(x[i + len(areas)])
                    plt.pause(0.001)

        mask_ = mask_.detach()

        mask_of_input_shape = mask_

        # Resize saliency map.
        mask_ = resize_saliency(input,
                                mask_,
                                resize,
                                mode=resize_mode)

        # Smooth saliency map.
        if smooth > 0:
            mask_ = imsmooth(
                mask_,
                sigma=smooth * min(mask_.shape[2:]),
                padding_mode='constant'
            )
            mask_of_input_shape = imsmooth(
                mask_of_input_shape,
                sigma=smooth * min(mask_of_input_shape.shape[2:]),
                padding_mode='constant'
            )

        return mask_, hist, mask_of_input_shape
    else:
        if use_smbo and single_gauss:
            from skopt import gp_minimize
            from skopt import forest_minimize
            from skopt.space import Real
            from skopt.plots import plot_convergence

            areas = [0.2]

            def generate_plot_mask(gauss_params, debug=False):
                pmask = GaussPMask(h, w, areas).generate_pmask(gauss_params)
                pmask = np.clip(pmask, 0, 1)
                pmask = torch.tensor(pmask, dtype=torch.float32).to(device)
                mask_, mask = mask_generator.generate(pmask)
                # Apply the mask.
                if variant == DELETE_VARIANT:
                    x = perturbation.apply(1 - mask_)
                elif variant == PRESERVE_VARIANT:
                    x = perturbation.apply(mask_)
                elif variant == DUAL_VARIANT:
                    x = torch.cat((
                        perturbation.apply(mask_),
                        perturbation.apply(1 - mask_),
                    ), dim=0)
                else:
                    assert False

                # Evaluate the model on the masked data.
                y = model(x)
                y = F.softmax(y, dim=1)
                target_class_prediction_score = y[:, target, :, :]
                print('target class prediction', target_class_prediction_score)

                if debug:
                    for i, a in enumerate(areas):
                        plt.figure(i, figsize=(20, 12))
                        plt.clf()
                        ncols = 2
                        plt.subplot(1, ncols, 1)
                        imsc(mask[i], lim=[0, 1])
                        plt.title(
                            f"min:{mask[i].min().item():.2f}"
                            f" max:{mask[i].max().item():.2f}"
                            f" area:{mask[i].sum() / mask[i].numel():.2f}")
                        plt.subplot(1, ncols, 2)
                        imsc(x[i])
                        plt.pause(0.001)

                return mask_, mask, target_class_prediction_score
            class GaussPMask():
                def __init__(self, H, W, areas):
                    self.num_masks = len(areas)
                    self.H = H
                    self.W = W

                    h = np.linspace(0, 1, self.H)
                    w = np.linspace(0, 1, self.W)
                    self.h, self.w = np.meshgrid(w, h)

                    # self.mh = (torch.tensor([0.5] * self.num_masks))
                    # self.mw = (torch.tensor([0.5] * self.num_masks))

                    # # sigma_scale = 0.02 # 0.05 optimal for area regul
                    # self.sh = np.array([0.9] * self.num_masks)  # sigma_scale * H
                    # self.sw = np.array([0.9] * self.num_masks)  # !!!!!

                    # self.gauss_scale = (torch.tensor([9.] * self.num_masks))

                def _gaussian_2d(self, i):
                    # # https://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
                    # # https://peterroelants.github.io/posts/multivariate-normal-primer/
                    # covariance = torch.tensor([
                    #     [self.sh[i], 0],
                    #     [0, self.sw[i]]
                    # ])
                    # A = 1. / (torch.sqrt((2 * math.pi) ** 2 * torch.det(covariance)))
                    # B = (-1/2) * ((x-mu).T.dot(torch.inverse(cov))).dot((x-mu))
                    A = 1 / (2 * math.pi * self.sh[i] * self.sw[i])
                    B = (self.h - self.mh[i]) ** 2 / (2 * self.sh[i] ** 2)
                    C = (self.w - self.mw[i]) ** 2 / (2 * self.sw[i] ** 2)
                    return A * np.exp(-(B + C))

                def generate_pmask(self, gauss_params, g_scale=1.2):
                    # mh, mw = gauss_params
                    # self.mh = np.array([mh] * self.num_masks)
                    # self.mw = np.array([mw] * self.num_masks)

                    # sh, sw = gauss_params
                    # self.sh = np.array([sh] * self.num_masks)
                    # self.sw = np.array([sw] * self.num_masks)
                    # self.mh = np.array([0.5] * self.num_masks)
                    # self.mw = np.array([0.5] * self.num_masks)

                    mh, mw, sh, sw = gauss_params
                    self.mh = np.array([mh] * self.num_masks)
                    self.mw = np.array([mw] * self.num_masks)
                    self.sh = np.array([sh] * self.num_masks)
                    self.sw = np.array([sw] * self.num_masks)

                    pmask = np.zeros([len(areas), self.H, self.W])
                    for i in range(self.num_masks):
                        z = self._gaussian_2d(i)
                        pmask[i] = (z / z.max()) * g_scale
                        # pmask[i] = z * self.gauss_scale[i]
                        # pmask[i] = z
                        # z_ = pmask[i].detach().cpu().numpy()
                    pmask = np.expand_dims(pmask, 1)
                    return pmask

            t = 0
            sigma_upper_bound = 0.5
            regul_area_weight = 300


            # reference_covariance_det_vals = np.linspace(sigma_upper_bound**2, reference_covariance_det_target, num_score_area_iter)
            reference_covariance_det_vals = np.array([0.25, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.001]) # , 0.005, 0.0025, 0.001
            num_score_area_iter = len(reference_covariance_det_vals)

            smbo_time_hist = []
            smbo_time = time.time()

            def ep_func(gauss_params, area_penalty=True, optimize_sigma=True):
                nonlocal t
                nonlocal regul_area_weight
                nonlocal smbo_time
                # print('\ngauss_params', gauss_params)
                # print(t)

                t = time.time() - smbo_time
                print('smbo_time', t)
                smbo_time_hist.append(t)
                smbo_time = time.time()

                if not area_penalty and not optimize_sigma:
                    mh, mw = gauss_params
                    # use init large sigmas (not optimized)
                    gauss_params = mh, mw, sigma_upper_bound, sigma_upper_bound
                pmask = GaussPMask(h, w, areas).generate_pmask(gauss_params)
                pmask = np.clip(pmask, 0, 1)

                pmask = torch.tensor(pmask, dtype=torch.float32).to(device)

                # print('pmask stats:', pmask.min(), pmask.max(), pmask.mean())

                # Generate the mask.
                mask_, mask = mask_generator.generate(pmask)

                # Apply the mask.
                if variant == DELETE_VARIANT:
                    x = perturbation.apply(1 - mask_)
                elif variant == PRESERVE_VARIANT:
                    x = perturbation.apply(mask_)
                elif variant == DUAL_VARIANT:
                    x = torch.cat((
                        perturbation.apply(mask_),
                        perturbation.apply(1 - mask_),
                    ), dim=0)
                else:
                    assert False

                # # Apply jitter to the masked data.
                # if jitter and t % 2 == 0:
                #     x = torch.flip(x, dims=(3,))

                # Evaluate the model on the masked data.
                # with torch.no_grad():
                y = model(x)
                # TODO: will softmax work better?
                # y = F.softmax(y, dim=1)

                # Get reward.
                reward = reward_func(y, target, variant=variant)
                # Reshape reward and average over spatial dimensions.
                reward = reward.reshape(len(areas), -1).mean(dim=1) #* reward_weight
                # print('reward_weight', reward_weight)
                # reward_weight = 0.98 * reward_weight

                # Area regularization.
                if area_penalty:
                    # mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
                    # regul_area = - ((mask_sorted - reference) ** 2).mean(dim=1) * regul_weight
                    # sh, sw = gauss_params
                    _, _, sh, sw = gauss_params
                    covariance_det = sh * sw
                    regul_area = - ((covariance_det - reference_covariance_det) ** 2) * regul_area_weight
                    # print('reference_covariance_det', reference_covariance_det)
                    # print('regul_area_weight', regul_area_weight)
                    # regul_area_weight *= 1.2
                    # if t in [25, 50, 75]:
                    #     regul_area_weight *= 5
                    # # Warm up. Give time for localization
                    # if t < 10:
                    #     regul_area = regul_area * 0.0001
                else:
                    regul_area = 0

                # regul_gauss = - ((gauss_pmask_generator.sh - gauss_pmask_generator.sw) ** 2) * 1000
                # print('reward Energy', reward.detach().data.cpu().numpy())
                # print('regul_area Energy', regul_area)
                energy = (reward + regul_area).sum() #  + regul_area
                # energy = (regul_area)  # + regul_area

                score = - energy
                t += 1

                return score.item()
                # return score


            y = model(input)
            y = F.softmax(y, dim=1)
            print('[Non perturbed image] target class prediction', y[:, target, :, :])
            mask_hist = []


            # Optimize score w/o area penalty
            bounds = [Real(low=0.0, high=1.0), Real(low=0.0, high=1.0)]  # means
            res = gp_minimize(partial(ep_func, area_penalty=False, optimize_sigma=False), bounds, acq_func="PI", n_calls=12, n_initial_points=5, random_state=1234, noise=1e-10)
            if debug:
                print('res.x', res.x)
                print('res.fun', res.fun)
                # plot_convergence(res)
                # plt.show()
            mask_, mask, target_class_prediction_score = generate_plot_mask([res.x[0], res.x[1], sigma_upper_bound, sigma_upper_bound], debug)
            mask_hist.append((mask_, target_class_prediction_score))

            # Optimize score + area
            for i in range(num_score_area_iter - 1):
                print('\niter', i)
                t = 0
                reference_covariance_det = reference_covariance_det_vals[i + 1]
                sigma = math.sqrt(reference_covariance_det)
                # if i == 0:
                #     x0_sigma_h = x0_sigma_w = sigma_upper_bound
                # else:
                #     ratio = res.x[2]/res.x[3]
                #     x0_sigma_h = math.sqrt(reference_covariance_det * ratio)
                #     x0_sigma_w = x0_sigma_h / ratio
                #     # x0_sigma_h = res.x[2]
                #     # x0_sigma_w = res.x[3]
                mean_range = sigma
                sigma_range = sigma/5
                bounds = [
                          Real(low=max(0, res.x[0] - mean_range), high=min(1, res.x[0] + mean_range)),
                          Real(low=max(0, res.x[1] - mean_range), high=min(1, res.x[1] + mean_range)),
                          Real(low=sigma - sigma_range, high=sigma + sigma_range),
                          Real(low=sigma - sigma_range, high=sigma + sigma_range),
                          # Real(low=max(0, res.x[0] - x0_sigma_h/2), high=min(1, res.x[0] + x0_sigma_h/2)),
                          # Real(low=max(0, res.x[1] - x0_sigma_w/2), high=min(1, res.x[1] + x0_sigma_w/2)),
                          # Real(low=x0_sigma_h - sigma_range, high=x0_sigma_h + sigma_range),
                          # Real(low=x0_sigma_w - sigma_range, high=x0_sigma_w + sigma_range)
                          ]
                # print('bounds', bounds)
                x0 = [res.x[0], res.x[1], sigma, sigma]
                # x0 = [res.x[0], res.x[1], x0_sigma_h, x0_sigma_w]
                # x0 = [res.x[0], res.x[1]]
                # x0 = None
                res = gp_minimize(partial(ep_func, area_penalty=False), bounds, x0=x0, acq_func="PI", n_calls=12, n_initial_points=5, random_state=1234, noise=1e-10)
                if debug:
                    print('res.x', res.x)
                    print('res.fun', res.fun)
                    # plot_convergence(res)
                    # plt.show()
                mask_, mask, target_class_prediction_score = generate_plot_mask(res.x, debug)

                # if target_class_prediction_score < 0.2:
                #     print('target_class_prediction_score go below 0.2 -> Break!')
                #     break

                # TODO: which g_scale to use?
                pmask = GaussPMask(h, w, areas).generate_pmask(res.x, g_scale=1.2)
                pmask = torch.tensor(pmask, dtype=torch.float32).to(device)
                mask_, mask = mask_generator.generate(pmask)
                mask_hist.append((mask_, target_class_prediction_score))
                # mask_, mask, target_class_prediction_score = generate_plot_mask([res.x[0], res.x[1], sigma, sigma])

            mask_hist = [mask for (mask, score) in mask_hist if score > 0.1]

            # Combine 3 last masks (each mask obtained with a separate area contrain)
            mask_ = torch.cat(mask_hist[-3:])



            mask_of_input_shape = mask_

            # Resize saliency map.
            mask_ = resize_saliency(input,
                                    mask_,
                                    resize,
                                    mode=resize_mode)

            # Smooth saliency map.
            if smooth > 0:
                mask_ = imsmooth(
                    mask_,
                    sigma=smooth * min(mask_.shape[2:]),
                    padding_mode='constant'
                )
                mask_of_input_shape = imsmooth(
                    mask_of_input_shape,
                    sigma=smooth * min(mask_of_input_shape.shape[2:]),
                    padding_mode='constant'
                )

            return mask_, None, mask_of_input_shape
        elif use_smbo and reuse_ep_mask_generation:
            from skopt import gp_minimize
            from skopt.space import Real
            from skopt.plots import plot_convergence

            # Prepare reference area vector.
            max_area = np.prod(mask_generator.shape_out)
            reference = torch.ones(len(areas), max_area).to(device)
            for i, a in enumerate(areas):
                reference[i, :int(max_area * (1 - a))] = 0

            def generate_plot_mask(pmask_flatten, debug=False):
                pmask = np.array(pmask_flatten).reshape(1, 1, h, w)

                pmask = torch.tensor(pmask, dtype=torch.float32).to(device)
                mask_, mask = mask_generator.generate(pmask)
                # Apply the mask.
                if variant == DELETE_VARIANT:
                    x = perturbation.apply(1 - mask_)
                elif variant == PRESERVE_VARIANT:
                    x = perturbation.apply(mask_)
                elif variant == DUAL_VARIANT:
                    x = torch.cat((
                        perturbation.apply(mask_),
                        perturbation.apply(1 - mask_),
                    ), dim=0)
                else:
                    assert False

                # Evaluate the model on the masked data.
                y = model(x)
                y = F.softmax(y, dim=1)
                target_class_prediction_score = y[:, target, :, :]
                print('target class prediction', target_class_prediction_score)

                if debug:
                    for i, a in enumerate(areas):
                        plt.figure(i, figsize=(20, 12))
                        plt.clf()
                        ncols = 2
                        plt.subplot(1, ncols, 1)
                        imsc(mask[i], lim=[0, 1])
                        plt.title(
                            f"min:{mask[i].min().item():.2f}"
                            f" max:{mask[i].max().item():.2f}"
                            f" area:{mask[i].sum() / mask[i].numel():.2f}")
                        plt.subplot(1, ncols, 2)
                        imsc(x[i])
                        plt.pause(0.001)

                return mask_, mask, target_class_prediction_score
            class GaussPMask():
                def __init__(self, H, W, areas):
                    self.num_masks = len(areas)
                    self.H = H
                    self.W = W

                    h = np.linspace(0, 1, self.H)
                    w = np.linspace(0, 1, self.W)
                    self.h, self.w = np.meshgrid(w, h)

                    # self.mh = (torch.tensor([0.5] * self.num_masks))
                    # self.mw = (torch.tensor([0.5] * self.num_masks))

                    # # sigma_scale = 0.02 # 0.05 optimal for area regul
                    # self.sh = np.array([0.9] * self.num_masks)  # sigma_scale * H
                    # self.sw = np.array([0.9] * self.num_masks)  # !!!!!

                    # self.gauss_scale = (torch.tensor([9.] * self.num_masks))

                def _gaussian_2d(self, i):
                    # # https://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
                    # # https://peterroelants.github.io/posts/multivariate-normal-primer/
                    # covariance = torch.tensor([
                    #     [self.sh[i], 0],
                    #     [0, self.sw[i]]
                    # ])
                    # A = 1. / (torch.sqrt((2 * math.pi) ** 2 * torch.det(covariance)))
                    # B = (-1/2) * ((x-mu).T.dot(torch.inverse(cov))).dot((x-mu))
                    A = 1 / (2 * math.pi * self.sh[i] * self.sw[i])
                    B = (self.h - self.mh[i]) ** 2 / (2 * self.sh[i] ** 2)
                    C = (self.w - self.mw[i]) ** 2 / (2 * self.sw[i] ** 2)
                    return A * np.exp(-(B + C))

                def generate_pmask(self, gauss_params, g_scale=1.2):
                    # mh, mw = gauss_params
                    # self.mh = np.array([mh] * self.num_masks)
                    # self.mw = np.array([mw] * self.num_masks)

                    # sh, sw = gauss_params
                    # self.sh = np.array([sh] * self.num_masks)
                    # self.sw = np.array([sw] * self.num_masks)
                    # self.mh = np.array([0.5] * self.num_masks)
                    # self.mw = np.array([0.5] * self.num_masks)

                    mh, mw, sh, sw = gauss_params
                    self.mh = np.array([mh] * self.num_masks)
                    self.mw = np.array([mw] * self.num_masks)
                    self.sh = np.array([sh] * self.num_masks)
                    self.sw = np.array([sw] * self.num_masks)

                    pmask = np.zeros([len(areas), self.H, self.W])
                    for i in range(self.num_masks):
                        z = self._gaussian_2d(i)
                        pmask[i] = (z / z.max()) * g_scale
                        # pmask[i] = z * self.gauss_scale[i]
                        # pmask[i] = z
                        # z_ = pmask[i].detach().cpu().numpy()
                    pmask = np.expand_dims(pmask, 1)
                    return pmask

            t = 0
            sigma_upper_bound = 0.5
            regul_area_weight = 300


            # reference_covariance_det_vals = np.linspace(sigma_upper_bound**2, reference_covariance_det_target, num_score_area_iter)
            reference_covariance_det_vals = np.array([0.25, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.001]) # , 0.005, 0.0025, 0.001
            num_score_area_iter = len(reference_covariance_det_vals)


            def ep_func(pmask_flatten, area_penalty=True):
                nonlocal t
                nonlocal regul_area_weight
                pmask = np.array(pmask_flatten).reshape(1, 1, h, w)

                pmask = torch.tensor(pmask, dtype=torch.float32).to(device)

                # print('pmask stats:', pmask.min(), pmask.max(), pmask.mean())

                # Generate the mask.
                mask_, mask = mask_generator.generate(pmask)

                # Apply the mask.
                if variant == DELETE_VARIANT:
                    x = perturbation.apply(1 - mask_)
                elif variant == PRESERVE_VARIANT:
                    x = perturbation.apply(mask_)
                elif variant == DUAL_VARIANT:
                    x = torch.cat((
                        perturbation.apply(mask_),
                        perturbation.apply(1 - mask_),
                    ), dim=0)
                else:
                    assert False

                # # Apply jitter to the masked data.
                # if jitter and t % 2 == 0:
                #     x = torch.flip(x, dims=(3,))

                # Evaluate the model on the masked data.
                # with torch.no_grad():
                y = model(x)
                # TODO: will softmax work better?
                # y = F.softmax(y, dim=1)

                # Get reward.
                reward = reward_func(y, target, variant=variant)
                # Reshape reward and average over spatial dimensions.
                reward = reward.reshape(len(areas), -1).mean(dim=1) #* reward_weight
                # print('reward_weight', reward_weight)
                # reward_weight = 0.98 * reward_weight

                # Area regularization.
                if area_penalty:
                    mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
                    regul_area = - ((mask_sorted - reference) ** 2).mean(dim=1) * regul_area_weight
                    # print('reference_covariance_det', reference_covariance_det)
                    # print('regul_area_weight', regul_area_weight)
                    # regul_area_weight *= 1.2
                    # if t in [25, 50, 75]:
                    #     regul_area_weight *= 5
                    # # Warm up. Give time for localization
                    # if t < 10:
                    #     regul_area = regul_area * 0.0001
                else:
                    regul_area = 0

                # regul_gauss = - ((gauss_pmask_generator.sh - gauss_pmask_generator.sw) ** 2) * 1000
                # print('reward Energy', reward.detach().data.cpu().numpy())
                # print('regul_area Energy', regul_area)
                energy = (reward + regul_area).sum() #  + regul_area
                # energy = (regul_area)  # + regul_area

                score = - energy
                t += 1

                return score.item()
                # return score


            y = model(input)
            y = F.softmax(y, dim=1)
            print('[Non perturbed image] target class prediction', y[:, target, :, :])
            mask_hist = []


            # Optimize score w/o area penalty
            bounds = [Real(low=0.0, high=1.0) for _ in range(h * w)]
            x0 = list(np.ones(h * w))
            res = gp_minimize(ep_func, bounds, x0=x0, acq_func="PI", n_calls=20, n_initial_points=10, random_state=1234, noise=1e-10)
            if debug:
                print('res.x', res.x)
                print('res.fun', res.fun)
                # plot_convergence(res)
                # plt.show()
            mask_, mask, target_class_prediction_score = generate_plot_mask(res.x, debug)

            for i in range(5):
                x0 = res.x
                res = gp_minimize(ep_func, bounds, x0=x0, acq_func="PI", n_calls=20, n_initial_points=10, random_state=1234,
                                  noise=1e-10)
                if debug:
                    print('res.x', res.x)
                    print('res.fun', res.fun)
                    # plot_convergence(res)
                    # plt.show()
                mask_, mask, target_class_prediction_score = generate_plot_mask(res.x, debug)
                regul_area_weight *= 1.25

            mask_of_input_shape = mask_

            # Resize saliency map.
            mask_ = resize_saliency(input,
                                    mask_,
                                    resize,
                                    mode=resize_mode)

            # Smooth saliency map.
            if smooth > 0:
                mask_ = imsmooth(
                    mask_,
                    sigma=smooth * min(mask_.shape[2:]),
                    padding_mode='constant'
                )
                mask_of_input_shape = imsmooth(
                    mask_of_input_shape,
                    sigma=smooth * min(mask_of_input_shape.shape[2:]),
                    padding_mode='constant'
                )

            return mask_, None, mask_of_input_shape
        if use_smbo and mix_of_gauss:
            from skopt import gp_minimize, forest_minimize
            from skopt.space import Real
            from skopt.plots import plot_convergence
            from dlib import find_min_global

            areas = [0.2]
            random_state = 1234

            def generate_plot_mask(gauss_params, debug=False):
                pmask = GaussPMask(h, w, areas).generate_pmask(gauss_params)
                pmask = np.clip(pmask, 0, 1)
                pmask = torch.tensor(pmask, dtype=torch.float32).to(device)
                mask_, mask = mask_generator.generate(pmask)
                # Apply the mask.
                if variant == DELETE_VARIANT:
                    x = perturbation.apply(1 - mask_)
                elif variant == PRESERVE_VARIANT:
                    x = perturbation.apply(mask_)
                elif variant == DUAL_VARIANT:
                    x = torch.cat((
                        perturbation.apply(mask_),
                        perturbation.apply(1 - mask_),
                    ), dim=0)
                else:
                    assert False

                # Evaluate the model on the masked data.
                y = model(x)
                y = F.softmax(y, dim=1)
                target_class_prediction_score = y[:, target, :, :]
                print('target class prediction', target_class_prediction_score)

                if debug:
                    for i, a in enumerate(areas):
                        plt.figure(i, figsize=(20, 12))
                        plt.clf()
                        ncols = 2
                        plt.subplot(1, ncols, 1)
                        imsc(mask[i], lim=[0, 1])
                        plt.title(
                            f"min:{mask[i].min().item():.2f}"
                            f" max:{mask[i].max().item():.2f}"
                            f" area:{mask[i].sum() / mask[i].numel():.2f}")
                        plt.subplot(1, ncols, 2)
                        imsc(x[i])
                        plt.pause(0.001)

                return mask_, mask, target_class_prediction_score

            class GaussPMask():
                def __init__(self, H, W, areas):
                    self.num_masks = len(areas)
                    self.H = H
                    self.W = W

                    h = np.linspace(0, 1, self.H)
                    w = np.linspace(0, 1, self.W)
                    self.h, self.w = np.meshgrid(w, h)

                    # self.mh = (torch.tensor([0.5] * self.num_masks))
                    # self.mw = (torch.tensor([0.5] * self.num_masks))

                    # # sigma_scale = 0.02 # 0.05 optimal for area regul
                    # self.sh = np.array([0.9] * self.num_masks)  # sigma_scale * H
                    # self.sw = np.array([0.9] * self.num_masks)  # !!!!!

                    # self.gauss_scale = (torch.tensor([9.] * self.num_masks))

                def _gaussian_2d(self, i):
                    # # https://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
                    # # https://peterroelants.github.io/posts/multivariate-normal-primer/
                    # covariance = torch.tensor([
                    #     [self.sh[i], 0],
                    #     [0, self.sw[i]]
                    # ])
                    # A = 1. / (torch.sqrt((2 * math.pi) ** 2 * torch.det(covariance)))
                    # B = (-1/2) * ((x-mu).T.dot(torch.inverse(cov))).dot((x-mu))
                    A = 1 / (2 * math.pi * self.sh[i] * self.sw[i])
                    B = (self.h - self.mh[i]) ** 2 / (2 * self.sh[i] ** 2)
                    C = (self.w - self.mw[i]) ** 2 / (2 * self.sw[i] ** 2)
                    return A * np.exp(-(B + C))

                def generate_pmask(self, gausses_params, g_scale=1.2):
                    # mh, mw = gauss_params
                    # self.mh = np.array([mh] * self.num_masks)
                    # self.mw = np.array([mw] * self.num_masks)

                    # sh, sw = gauss_params
                    # self.sh = np.array([sh] * self.num_masks)
                    # self.sw = np.array([sw] * self.num_masks)
                    # self.mh = np.array([0.5] * self.num_masks)
                    # self.mw = np.array([0.5] * self.num_masks)

                    pmask = np.zeros([len(areas), self.H, self.W])
                    for gauss_params in gausses_params:
                        mh, mw, sh, sw = gauss_params
                        self.mh = np.array([mh] * self.num_masks)
                        self.mw = np.array([mw] * self.num_masks)
                        self.sh = np.array([sh] * self.num_masks)
                        self.sw = np.array([sw] * self.num_masks)
                        for i in range(self.num_masks):
                            z = self._gaussian_2d(i)
                            pmask[i] += (z / z.max()) * g_scale
                            # pmask[i] = z * self.gauss_scale[i]
                            # pmask[i] = z
                            # z_ = pmask[i].detach().cpu().numpy()
                    pmask = np.expand_dims(pmask, 1)
                    return pmask

            t = 0

            # reference_covariance_det_vals = np.linspace(sigma_upper_bound**2, reference_covariance_det_target, num_score_area_iter)
            # reference_covariance_det_vals = np.array([0.25, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.001]) # , 0.005, 0.0025, 0.001
            # num_score_area_iter = len(reference_covariance_det_vals)

            gauss_num = 3

            acq_func = 'LCB' # LCB
            sigma_upper_bound = 0.2

            smbo_time_hist = []
            smbo_time = time.time()

            # def ep_func(gauss_params, area_penalty=True, optimize_sigma=True):
            def ep_func(mh, mw, mh1, mw1, mh2, mw2):
                gauss_params = [mh, mw, mh1, mw1, mh2, mw2]


                area_penalty = False
                optimize_sigma = False
                nonlocal t
                nonlocal regul_area_weight
                nonlocal smbo_time
                print('\ngauss_params', gauss_params)
                print(t)

                t = time.time() - smbo_time
                print('smbo_time', t)
                smbo_time_hist.append(t)
                smbo_time = time.time()

                tic_ = time.time()
                print('start ep_func', tic_)
                if not area_penalty and not optimize_sigma:
                    params = []
                    for mh, mw in zip(gauss_params[::2], gauss_params[1::2]):
                        params.append((mh, mw, sigma_upper_bound, sigma_upper_bound))
                    gauss_params = params
                else:
                    params = []
                    for mh, mw, sigma in zip(gauss_params[::3], gauss_params[1::3], gauss_params[2::3]):
                        params.append((mh, mw, sigma, sigma))
                    gauss_params = params
                pmask = GaussPMask(h, w, areas).generate_pmask(gauss_params)
                pmask = np.clip(pmask, 0, 1)

                pmask = torch.tensor(pmask, dtype=torch.float32).to(device)

                # print('pmask stats:', pmask.min(), pmask.max(), pmask.mean())

                # Generate the mask.
                mask_, mask = mask_generator.generate(pmask)

                # Apply the mask.
                if variant == DELETE_VARIANT:
                    x = perturbation.apply(1 - mask_)
                elif variant == PRESERVE_VARIANT:
                    x = perturbation.apply(mask_)
                elif variant == DUAL_VARIANT:
                    x = torch.cat((
                        perturbation.apply(mask_),
                        perturbation.apply(1 - mask_),
                    ), dim=0)
                else:
                    assert False

                # # Apply jitter to the masked data.
                # if jitter and t % 2 == 0:
                #     x = torch.flip(x, dims=(3,))

                # Evaluate the model on the masked data.
                # with torch.no_grad():
                y = model(x)
                # TODO: will softmax work better?
                # y = F.softmax(y, dim=1)

                # Get reward.
                reward = reward_func(y, target, variant=variant)
                # Reshape reward and average over spatial dimensions.
                reward = reward.reshape(len(areas), -1).mean(dim=1) #* reward_weight
                # print('reward_weight', reward_weight)
                # reward_weight = 0.98 * reward_weight

                # Area regularization.
                if area_penalty:
                    covariance_det = 0
                    for _, _, sh, sw in gauss_params:
                         covariance_det += sh * sw
                    regul_area = - ((covariance_det - reference_covariance_det) ** 2) * regul_area_weight
                else:
                    regul_area = 0

                # regul_gauss = - ((gauss_pmask_generator.sh - gauss_pmask_generator.sw) ** 2) * 1000
                print('reward Energy', reward.detach().data.cpu().numpy())
                print('regul_area Energy', regul_area)
                energy = (reward + regul_area).sum() #  + regul_area
                # energy = (regul_area)  # + regul_area

                score = - energy
                t += 1

                print('finish ep_func', time.time() - tic_)

                return score.item()
                # return score

            def ep_func_area(mh, mw, sigma, mh1, mw1, sigma1, mh2, mw2, sigma2):
                gauss_params = [mh, mw, sigma, mh1, mw1, sigma1, mh2, mw2, sigma2]
                area_penalty = True
                optimize_sigma = True
                nonlocal t
                nonlocal regul_area_weight
                nonlocal smbo_time
                print('\ngauss_params', gauss_params)
                print(t)

                t = time.time() - smbo_time
                print('smbo_time', t)
                smbo_time_hist.append(t)
                smbo_time = time.time()

                tic_ = time.time()
                print('start ep_func', tic_)
                params = []
                for mh, mw, sigma in zip(gauss_params[::3], gauss_params[1::3], gauss_params[2::3]):
                    params.append((mh, mw, sigma, sigma))
                gauss_params = params
                pmask = GaussPMask(h, w, areas).generate_pmask(gauss_params)
                pmask = np.clip(pmask, 0, 1)

                pmask = torch.tensor(pmask, dtype=torch.float32).to(device)

                # print('pmask stats:', pmask.min(), pmask.max(), pmask.mean())

                # Generate the mask.
                mask_, mask = mask_generator.generate(pmask)

                # Apply the mask.
                if variant == DELETE_VARIANT:
                    x = perturbation.apply(1 - mask_)
                elif variant == PRESERVE_VARIANT:
                    x = perturbation.apply(mask_)
                elif variant == DUAL_VARIANT:
                    x = torch.cat((
                        perturbation.apply(mask_),
                        perturbation.apply(1 - mask_),
                    ), dim=0)
                else:
                    assert False

                # # Apply jitter to the masked data.
                # if jitter and t % 2 == 0:
                #     x = torch.flip(x, dims=(3,))

                # Evaluate the model on the masked data.
                # with torch.no_grad():
                y = model(x)
                # TODO: will softmax work better?
                # y = F.softmax(y, dim=1)

                # Get reward.
                reward = reward_func(y, target, variant=variant)
                # Reshape reward and average over spatial dimensions.
                reward = reward.reshape(len(areas), -1).mean(dim=1) #* reward_weight
                # print('reward_weight', reward_weight)
                # reward_weight = 0.98 * reward_weight

                # Area regularization.
                if area_penalty:
                    covariance_det = 0
                    for _, _, sh, sw in gauss_params:
                         covariance_det += sh * sw
                    regul_area = - ((covariance_det - reference_covariance_det) ** 2) * regul_area_weight
                else:
                    regul_area = 0

                # regul_gauss = - ((gauss_pmask_generator.sh - gauss_pmask_generator.sw) ** 2) * 1000
                print('reward Energy', reward.detach().data.cpu().numpy())
                print('regul_area Energy', regul_area)
                energy = (reward + regul_area).sum() #  + regul_area
                # energy = (regul_area)  # + regul_area

                score = - energy
                t += 1

                print('finish ep_func', time.time() - tic_)

                return score.item()
                # return score


            y = model(input)
            y = F.softmax(y, dim=1)
            print('[Non perturbed image] target class prediction', y[:, target, :, :])
            mask_hist = []





            # Optimize score w/o area penalty
            print('Start Optimize score w/o area penalty')
            tic = time.time()
            n_calls = 50
            n_initial_points = 10
            bounds = []
            for _ in range(gauss_num):
                bounds.extend([Real(low=0.0, high=1.0), Real(low=0.0, high=1.0)])
            # res = gp_minimize(partial(ep_func, area_penalty=False, optimize_sigma=False), bounds, acq_func=acq_func, n_calls=n_calls, n_initial_points=n_initial_points, random_state=random_state, noise=1e-10)
            # res = forest_minimize(partial(ep_func, area_penalty=False, optimize_sigma=False), bounds, acq_func=acq_func, n_calls=n_calls, n_initial_points=n_initial_points, random_state=random_state)

            # res_ = find_min_global(ep_func, [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], num_function_calls=n_calls)
            reference_covariance_det = 0.2 * 0.2 * gauss_num
            regul_area_weight = 100
            res_ = find_min_global(ep_func_area, [0, 0, 0.199, 0, 0, 0.199, 0, 0, 0.199], [1, 1, 0.2, 1, 1, 0.2, 1, 1, 0.2], num_function_calls=n_calls)
            res = lambda: None
            res.x = res_[0]
            res.fun = res_[1]

            if debug:
                print('res.x', res.x)
                print('res.fun', res.fun)
                plot_convergence(res)
                plt.show()
            res_gausses_params = []
            for mh, mw in zip(res.x[::2], res.x[1::2]):
                res_gausses_params.append((mh, mw, sigma_upper_bound, sigma_upper_bound))
            mask_, mask, target_class_prediction_score = generate_plot_mask(res_gausses_params, debug)
            mask_hist.append((mask_, target_class_prediction_score))
            print('Stop Optimize score w/o area penalty', time.time() - tic)





            # Optimize score w/ area penalty 1
            n_calls = 20
            n_initial_points = 1
            t = 0
            regul_area_weight = 300
            sigma_target = 0.12
            sigma_min = 0.07
            reference_covariance_det = sigma_target * sigma_target * gauss_num
            bounds = []
            for _ in range(gauss_num):
                bounds.extend([Real(low=0.0, high=1.0), Real(low=0.0, high=1.0), Real(low=sigma_min, high=sigma_upper_bound)])

            x0 = []
            for iter in res.x_iters:
                iter_res = []
                for mh, mw in zip(iter[::2], iter[1::2]):
                    iter_res.extend([mh, mw, sigma_upper_bound])
                x0.append(iter_res)

            regul_area = ((sigma_upper_bound * sigma_upper_bound * gauss_num - reference_covariance_det) ** 2) * regul_area_weight
            y0 = [(func_val + regul_area) for func_val in res.func_vals]

            # x0 = []
            # for mh, mw in zip(res.x[::2], res.x[1::2]):
            #     x0.extend([mh, mw, sigma_upper_bound])
            # regul_area = ((sigma_upper_bound * sigma_upper_bound * gauss_num - reference_covariance_det) ** 2) * regul_area_weight
            # y0 = res.fun + regul_area

            res = gp_minimize(ep_func, bounds, x0=x0, y0=y0, acq_func=acq_func, n_calls=n_calls, n_initial_points=n_initial_points, random_state=random_state, noise=1e-10)
            # res = forest_minimize(ep_func, bounds, x0=x0, y0=y0, acq_func=acq_func, n_calls=n_calls, n_initial_points=n_initial_points, random_state=random_state)
            if debug:
                print('res.x', res.x)
                print('res.fun', res.fun)
                plot_convergence(res)
                plt.show()
            res_gausses_params = []
            for mh, mw, sigma in zip(res.x[::3], res.x[1::3], res.x[2::3]):
                res_gausses_params.append((mh, mw, sigma, sigma))
            mask_, mask, target_class_prediction_score = generate_plot_mask(res_gausses_params, debug)
            mask_hist.append((mask_, target_class_prediction_score))





            # Optimize score w/ area penalty 2
            t = 0

            regul_area_weight_prev = regul_area_weight
            reference_covariance_det_prev = reference_covariance_det
            regul_area_weight = 400
            sigma_target = 0.07
            sigma_min = 0.035
            reference_covariance_det = sigma_target * sigma_target * gauss_num

            x0 = []
            regul_area_hist_prev = []
            regul_area_hist = []
            for iter in res.x_iters:
                iter_res = []
                regul = 0
                for mh, mw, sigma in zip(iter[::3], iter[1::3], iter[2::3]):
                    iter_res.extend([mh, mw, sigma])
                    regul += sigma * sigma
                x0.append(iter_res)
                regul_area_hist_prev.append((((regul - reference_covariance_det_prev) ** 2) * regul_area_weight_prev))
                regul_area_hist.append((((regul - reference_covariance_det) ** 2) * regul_area_weight))

            y0 = [(func_val - regul_area_prev + regul_area) for func_val, regul_area_prev, regul_area in zip(res.func_vals, regul_area_hist_prev, regul_area_hist)]

            bounds = []
            for _ in range(gauss_num):
                bounds.extend([Real(low=0.0, high=1.0), Real(low=0.0, high=1.0), Real(low=sigma_min, high=sigma_upper_bound)])
            res = gp_minimize(ep_func, bounds, x0=x0, y0=y0, acq_func=acq_func, n_calls=n_calls, n_initial_points=n_initial_points, random_state=random_state, noise=1e-10)
            # res = forest_minimize(ep_func, bounds, x0=x0, y0=y0, acq_func=acq_func, n_calls=n_calls, n_initial_points=n_initial_points, random_state=random_state)
            if debug:
                print('res.x', res.x)
                print('res.fun', res.fun)
                plot_convergence(res)
                plt.show()
            res_gausses_params = []
            for mh, mw, sigma in zip(res.x[::3], res.x[1::3], res.x[2::3]):
                res_gausses_params.append((mh, mw, sigma, sigma))
            mask_, mask, target_class_prediction_score = generate_plot_mask(res_gausses_params, debug)
            mask_hist.append((mask_, target_class_prediction_score))








            # Optimize score w/ area penalty 3
            t = 0

            regul_area_weight_prev = regul_area_weight
            reference_covariance_det_prev = reference_covariance_det
            regul_area_weight = 500
            sigma_target = 0.035
            sigma_min = 0.02
            reference_covariance_det = sigma_target * sigma_target * gauss_num

            x0 = []
            regul_area_hist_prev = []
            regul_area_hist = []
            for iter in res.x_iters:
                iter_res = []
                regul = 0
                for mh, mw, sigma in zip(iter[::3], iter[1::3], iter[2::3]):
                    iter_res.extend([mh, mw, sigma])
                    regul += sigma * sigma
                x0.append(iter_res)
                regul_area_hist_prev.append((((regul - reference_covariance_det_prev) ** 2) * regul_area_weight_prev))
                regul_area_hist.append((((regul - reference_covariance_det) ** 2) * regul_area_weight))

            y0 = [(func_val - regul_area_prev + regul_area) for func_val, regul_area_prev, regul_area in zip(res.func_vals, regul_area_hist_prev, regul_area_hist)]

            bounds = []
            for _ in range(gauss_num):
                bounds.extend([Real(low=0.0, high=1.0), Real(low=0.0, high=1.0), Real(low=sigma_min, high=sigma_upper_bound)])
            res = gp_minimize(ep_func, bounds, x0=x0, y0=y0, acq_func=acq_func, n_calls=n_calls, n_initial_points=n_initial_points, random_state=random_state, noise=1e-10)
            # res = forest_minimize(ep_func, bounds, x0=x0, y0=y0, acq_func=acq_func, n_calls=n_calls, n_initial_points=n_initial_points, random_state=random_state)
            if debug:
                print('res.x', res.x)
                print('res.fun', res.fun)
                plot_convergence(res)
                plt.show()
            res_gausses_params = []
            for mh, mw, sigma in zip(res.x[::3], res.x[1::3], res.x[2::3]):
                res_gausses_params.append((mh, mw, sigma, sigma))
            mask_, mask, target_class_prediction_score = generate_plot_mask(res_gausses_params, debug)
            mask_hist.append((mask_, target_class_prediction_score))





            # # Optimize score w/ area penalty 4
            # t = 0
            #
            # regul_area_weight_prev = regul_area_weight
            # reference_covariance_det_prev = reference_covariance_det
            # regul_area_weight = 2000
            # sigma_target = 0.01
            # reference_covariance_det = sigma_target * sigma_target * gauss_num
            #
            # x0 = []
            # regul_area_hist_prev = []
            # regul_area_hist = []
            # for iter in res.x_iters:
            #     iter_res = []
            #     regul = 0
            #     for mh, mw, sh, sw in zip(iter[::4], iter[1::4], iter[2::4], iter[3::4]):
            #         iter_res.extend([mh, mw, sh, sw])
            #         regul += sh * sw
            #     x0.append(iter_res)
            #     regul_area_hist_prev.append((((regul - reference_covariance_det_prev) ** 2) * regul_area_weight_prev))
            #     regul_area_hist.append((((regul - reference_covariance_det) ** 2) * regul_area_weight))
            #
            # y0 = [(func_val - regul_area_prev + regul_area) for func_val, regul_area_prev, regul_area in zip(res.func_vals, regul_area_hist_prev, regul_area_hist)]
            #
            # bounds = []
            # for _ in range(gauss_num):
            #     bounds.extend([Real(low=0.0, high=1.0), Real(low=0.0, high=1.0), Real(low=0.001, high=0.2), Real(low=0.001, high=0.2)])
            # res = gp_minimize(ep_func, bounds, x0=x0, y0=y0, acq_func=acq_func, n_calls=n_calls, n_initial_points=n_initial_points, random_state=random_state, noise=1e-10)
            # if debug:
            #     print('res.x', res.x)
            #     print('res.fun', res.fun)
            #     plot_convergence(res)
            #     plt.show()
            # res_gausses_params = []
            # for mh, mw, sh, sw in zip(res.x[::4], res.x[1::4], res.x[2::4], res.x[3::4]):
            #     res_gausses_params.append((mh, mw, sh, sw))
            # mask_, mask, target_class_prediction_score = generate_plot_mask(res_gausses_params, debug)
            # mask_hist.append((mask_, target_class_prediction_score))







            # # Optimize score + area
            # for i in range(num_score_area_iter - 1):
            #     print('\niter', i)
            #     t = 0
            #     reference_covariance_det = reference_covariance_det_vals[i + 1]
            #     sigma = math.sqrt(reference_covariance_det)
            #     # if i == 0:
            #     #     x0_sigma_h = x0_sigma_w = sigma_upper_bound
            #     # else:
            #     #     ratio = res.x[2]/res.x[3]
            #     #     x0_sigma_h = math.sqrt(reference_covariance_det * ratio)
            #     #     x0_sigma_w = x0_sigma_h / ratio
            #     #     # x0_sigma_h = res.x[2]
            #     #     # x0_sigma_w = res.x[3]
            #     mean_range = sigma
            #     sigma_range = sigma/5
            #     bounds = [
            #               Real(low=max(0, res.x[0] - mean_range), high=min(1, res.x[0] + mean_range)),
            #               Real(low=max(0, res.x[1] - mean_range), high=min(1, res.x[1] + mean_range)),
            #               Real(low=sigma - sigma_range, high=sigma + sigma_range),
            #               Real(low=sigma - sigma_range, high=sigma + sigma_range),
            #               # Real(low=max(0, res.x[0] - x0_sigma_h/2), high=min(1, res.x[0] + x0_sigma_h/2)),
            #               # Real(low=max(0, res.x[1] - x0_sigma_w/2), high=min(1, res.x[1] + x0_sigma_w/2)),
            #               # Real(low=x0_sigma_h - sigma_range, high=x0_sigma_h + sigma_range),
            #               # Real(low=x0_sigma_w - sigma_range, high=x0_sigma_w + sigma_range)
            #               ]
            #     # print('bounds', bounds)
            #     x0 = [res.x[0], res.x[1], sigma, sigma]
            #     # x0 = [res.x[0], res.x[1], x0_sigma_h, x0_sigma_w]
            #     # x0 = [res.x[0], res.x[1]]
            #     # x0 = None
            #     res = gp_minimize(partial(ep_func, area_penalty=False), bounds, x0=x0, acq_func="PI", n_calls=12, n_initial_points=5, random_state=random_state, noise=1e-10)
            #     if debug:
            #         print('res.x', res.x)
            #         print('res.fun', res.fun)
            #         # plot_convergence(res)
            #         # plt.show()
            #     mask_, mask, target_class_prediction_score = generate_plot_mask(res.x, debug)
            #
            #     # if target_class_prediction_score < 0.2:
            #     #     print('target_class_prediction_score go below 0.2 -> Break!')
            #     #     break
            #
            #     # TODO: which g_scale to use?
            #     pmask = GaussPMask(h, w, areas).generate_pmask(res.x, g_scale=1.2)
            #     pmask = torch.tensor(pmask, dtype=torch.float32).to(device)
            #     mask_, mask = mask_generator.generate(pmask)
            #     mask_hist.append((mask_, target_class_prediction_score))
            #     # mask_, mask, target_class_prediction_score = generate_plot_mask([res.x[0], res.x[1], sigma, sigma])
            #
            # mask_hist = [mask for (mask, score) in mask_hist if score > 0.1]
            #
            # # Combine 3 last masks (each mask obtained with a separate area contrain)
            # mask_ = torch.cat(mask_hist[-3:])



            mask_of_input_shape = mask_

            # Resize saliency map.
            mask_ = resize_saliency(input,
                                    mask_,
                                    resize,
                                    mode=resize_mode)

            # Smooth saliency map.
            if smooth > 0:
                mask_ = imsmooth(
                    mask_,
                    sigma=smooth * min(mask_.shape[2:]),
                    padding_mode='constant'
                )
                mask_of_input_shape = imsmooth(
                    mask_of_input_shape,
                    sigma=smooth * min(mask_of_input_shape.shape[2:]),
                    padding_mode='constant'
                )

            return mask_, None, mask_of_input_shape
        else:
            class GaussPMask(torch.nn.Module):
                def __init__(self, H, W, areas):
                    super(GaussPMask, self).__init__()
                    self.num_masks = len(areas)
                    self.H = H
                    self.W = W

                    h = torch.linspace(0, 1, self.H).to(device)
                    w = torch.linspace(0, 1, self.W).to(device)
                    self.h, self.w = torch.meshgrid(h, w)

                    self.mh = torch.nn.Parameter(torch.tensor([0.5] * self.num_masks))
                    self.mw = torch.nn.Parameter(torch.tensor([0.5] * self.num_masks))

                    # sigma_scale = 0.02 # 0.05 optimal for area regul
                    self.sh = torch.nn.Parameter(torch.tensor([0.9] * self.num_masks)) # sigma_scale * H
                    self.sw = torch.nn.Parameter(torch.tensor([0.9] * self.num_masks))  # !!!!!
                    # self.sh = torch.nn.Parameter(torch.tensor([0.5] * self.num_masks))
                    # self.sw = torch.nn.Parameter(torch.tensor([0.5] * self.num_masks))

                    # self.gauss_scale = (torch.tensor([9.] * self.num_masks))

                def get_covariance(self):
                    return torch.tensor([
                        [self.sh, 0],
                        [0, self.sw]
                    ])

                def _gaussian_2d(self, i):
                    # # https://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
                    # # https://peterroelants.github.io/posts/multivariate-normal-primer/
                    # covariance = torch.tensor([
                    #     [self.sh[i], 0],
                    #     [0, self.sw[i]]
                    # ])
                    # A = 1. / (torch.sqrt((2 * math.pi) ** 2 * torch.det(covariance)))
                    # B = (-1/2) * ((x-mu).T.dot(torch.inverse(cov))).dot((x-mu))
                    A = 1 / (2 * math.pi * self.sh[i] * self.sw[i])
                    B = (self.h - self.mh[i]) ** 2 / (2 * self.sh[i] ** 2)
                    C = (self.w - self.mw[i]) ** 2 / (2 * self.sw[i] ** 2)
                    return A * torch.exp(-(B + C))

                def forward(self):
                    pmask = torch.zeros(len(areas), self.H, self.W).to(device)
                    for i in range(self.num_masks):
                        z = self._gaussian_2d(i)
                        pmask[i] = (z / z.max()) * 1.2
                        # pmask[i] = z * self.gauss_scale[i]
                        # pmask[i] = z
                        # z_ = pmask[i].detach().cpu().numpy()
                    return pmask.unsqueeze(1)

            gauss_pmask_generator = GaussPMask(h, w, areas)
            gauss_pmask_generator.to(device)
            pmask = gauss_pmask_generator()

            for p in gauss_pmask_generator.named_parameters():
                print(p)

            if debug:
                print(f"- mask resolution:\n  {pmask.shape}")

            # Prepare reference area vector.
            max_area = np.prod(mask_generator.shape_out)
            reference = torch.ones(len(areas), max_area).to(device)
            for i, a in enumerate(areas):
                reference[i, :int(max_area * (1 - a))] = 0

            # Initialize optimizer.
            learning_rate = 0.03
            # optimizer = optim.SGD(gauss_pmask_generator.parameters(),
            #                       lr=learning_rate,
            #                       momentum=momentum,
            #                       dampening=momentum
            #                       )
            # optimizer = optim.SGD(gauss_pmask_generator.parameters(),
            #                       lr=learning_rate)
            optimizer = optim.Adam(gauss_pmask_generator.parameters(),
                                  lr=learning_rate)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.25)

            hist = torch.zeros((len(areas), 2, 0))
            reward_weight = 1
            regul_area_weight = 50 # 10 is good
            for t in range(max_iter):
                print()
                print(t)
                for p in gauss_pmask_generator.named_parameters():
                    print(p)
                pmask = gauss_pmask_generator()

                pmask.data = pmask.data.clamp(0, 1)

                # print('pmask stats:', pmask.min().data, pmask.max().data, pmask.mean().data)

                # Generate the mask.
                mask_, mask = mask_generator.generate(pmask)

                # Apply the mask.
                if variant == DELETE_VARIANT:
                    x = perturbation.apply(1 - mask_)
                elif variant == PRESERVE_VARIANT:
                    x = perturbation.apply(mask_)
                elif variant == DUAL_VARIANT:
                    x = torch.cat((
                        perturbation.apply(mask_),
                        perturbation.apply(1 - mask_),
                    ), dim=0)
                else:
                    assert False

                # # Apply jitter to the masked data.
                # if jitter and t % 2 == 0:
                #     x = torch.flip(x, dims=(3,))

                # Evaluate the model on the masked data.
                # with torch.no_grad():
                y = model(x)
                print('target class prediction', y[:, target, :, :])

                # Get reward.
                reward = reward_func(y, target, variant=variant)
                # Reshape reward and average over spatial dimensions.
                reward = reward.reshape(len(areas), -1).mean(dim=1) * reward_weight
                print('reward_weight', reward_weight)
                # reward_weight = 0.98 * reward_weight

                # Area regularization.
                # mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
                # regul_area = - ((mask_sorted - reference) ** 2).mean(dim=1) * regul_weight
                covariance_det = gauss_pmask_generator.sh * gauss_pmask_generator.sw
                reference_covariance_det = 0.05
                regul_area = - ((covariance_det - reference_covariance_det) ** 2) * regul_area_weight
                print('regul_area_weight', regul_area_weight)
                regul_area_weight *= 1.05
                # if t in [25, 50, 75]:
                #     regul_area_weight *= 5
                # # Warm up. Give time for localization
                if t < 10:
                    regul_area = regul_area * 0.0001

                # regul_gauss = - ((gauss_pmask_generator.sh - gauss_pmask_generator.sw) ** 2) * 1000

                print('reward Energy', reward.detach().data.cpu().numpy())
                print('regul_area Energy', regul_area.detach().data.cpu().numpy())
                energy = (reward + regul_area).sum()

                # last_lr = scheduler.get_last_lr()
                # print('last_lr', last_lr)

                # Gradient step.
                optimizer.zero_grad()
                (- energy).backward()
                optimizer.step()
                # scheduler.step()

                for name, p in gauss_pmask_generator.named_parameters():
                    if name == 'mh':
                        p.data = p.data.clamp(0, 1)
                    if name == 'mw':
                        p.data = p.data.clamp(0, 1)
                    if name == 'sh':
                        p.data = p.data.clamp(0.1, 1.5)
                    if name == 'sw':
                        p.data = p.data.clamp(0.1, 1.5)
                    # if name == 'gauss_scale':
                    #     p.data = p.data.clamp(0., 3000.)

                print('sh * sw', gauss_pmask_generator.sh * gauss_pmask_generator.sw)
                print('sh.grad, sw.grad', gauss_pmask_generator.sh.grad, gauss_pmask_generator.sw.grad)
                print('mh.grad, mw.grad', gauss_pmask_generator.mh.grad, gauss_pmask_generator.mw.grad)

                # Record energy.
                hist = torch.cat(
                    (hist,
                     torch.cat((
                         reward.detach().cpu().view(-1, 1, 1),
                         regul_area.detach().cpu().view(-1, 1, 1)
                     ), dim=1)), dim=2)

                # # Adjust the regulariser/area constraint weight.
                # regul_weight *= 1.0035

                # Diagnostics.
                debug_this_iter = debug and (t in (0, max_iter - 1)
                                             or regul_weight / regul_weight_last >= 2)

                # if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
                print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
                for i, area in enumerate(areas):
                    print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                        area,
                        hist[i, 0, -1],
                        hist[i, 1, -1]), end="")
                print()

                debug_this_iter = (t % 10 == 0)
                if debug_this_iter:
                    regul_weight_last = regul_weight
                    for i, a in enumerate(areas):
                        plt.figure(i, figsize=(20, 6))
                        plt.clf()
                        ncols = 4 if variant == DUAL_VARIANT else 3
                        plt.subplot(1, ncols, 1)
                        plt.plot(hist[i, 0].numpy())     # target class score
                        plt.plot(hist[i, 1].numpy())   # regul_area
                        # plt.plot(hist[i].sum(dim=0).numpy())
                        plt.legend(('taget_class_score', 'regul_area', 'both'))
                        plt.title(f'target area:{a:.2f}')
                        plt.subplot(1, ncols, 2)
                        imsc(mask[i], lim=[0, 1])
                        plt.title(
                            f"min:{mask[i].min().item():.2f}"
                            f" max:{mask[i].max().item():.2f}"
                            f" area:{mask[i].sum() / mask[i].numel():.2f}")
                        plt.subplot(1, ncols, 3)
                        imsc(x[i])
                        if variant == DUAL_VARIANT:
                            plt.subplot(1, ncols, 4)
                            imsc(x[i + len(areas)])
                        plt.pause(0.001)





            mask_ = mask_.detach()

            mask_of_input_shape = mask_

            # Resize saliency map.
            mask_ = resize_saliency(input,
                                    mask_,
                                    resize,
                                    mode=resize_mode)

            # Smooth saliency map.
            if smooth > 0:
                mask_ = imsmooth(
                    mask_,
                    sigma=smooth * min(mask_.shape[2:]),
                    padding_mode='constant'
                )
                mask_of_input_shape = imsmooth(
                    mask_of_input_shape,
                    sigma=smooth * min(mask_of_input_shape.shape[2:]),
                    padding_mode='constant'
                )

            return mask_, hist, mask_of_input_shape























        # class GaussPMask(torch.nn.Module):
        #     def __init__(self, H, W, areas):
        #         super(GaussPMask, self).__init__()
        #         self.num_masks = len(areas)
        #         self.H = H
        #         self.W = W
        #
        #         h = torch.linspace(0, 1, self.H).to(device)
        #         w = torch.linspace(0, 1, self.W).to(device)
        #         self.h, self.w = torch.meshgrid(h, w)
        #
        #         self.mh = (torch.tensor([0.5] * self.num_masks))
        #         self.mw = (torch.tensor([0.5] * self.num_masks))
        #
        #         self.sh = torch.nn.Parameter(torch.tensor([2.0] * self.num_masks))
        #         self.sw = torch.nn.Parameter(torch.tensor([2.0] * self.num_masks))  # !!!!!
        #
        #     def _gaussian_2d(self, i):
        #         A = 1 / (2 * math.pi * self.sh[i] * self.sw[i])
        #         B = (self.h - self.mh[i]) ** 2 / (2 * self.sh[i] ** 2)
        #         C = (self.w - self.mw[i]) ** 2 / (2 * self.sw[i] ** 2)
        #         return A * torch.exp(-(B + C))
        #
        #     def forward(self):
        #         pmask = torch.zeros(len(areas), self.H, self.W).to(device)
        #         for i in range(self.num_masks):
        #             z = self._gaussian_2d(i)
        #             # print(z.max())
        #             pmask[i] = z / z.max()
        #         return pmask.unsqueeze(1)
        #
        # gauss_pmask_generator = GaussPMask(h, w, areas)
        # gauss_pmask_generator.to(device)
        # pmask = gauss_pmask_generator()
        #
        # for p in gauss_pmask_generator.named_parameters():
        #     print(p)
        #
        # # Initialize optimizer.
        # learning_rate = 0.005
        # # optimizer = optim.SGD(gauss_pmask_generator.parameters(),
        # #                       lr=learning_rate,
        # #                       momentum=momentum,
        # #                       dampening=momentum)
        # optimizer = optim.SGD(gauss_pmask_generator.parameters(),
        #                       lr=learning_rate)
        # # optimizer = optim.Adam(gauss_pmask_generator.parameters(),
        # #                       lr=learning_rate)
        #
        # grad_hist = []
        # regul_area_weight = 10
        # for t in range(max_iter):
        #     print()
        #     print(t)
        #     pmask = gauss_pmask_generator()
        #
        #     # pmask.data = pmask.data.clamp(0, 1)
        #
        #     print('pmask stats:', pmask.min().data, pmask.max().data, pmask.mean().data)
        #
        #     # Generate the mask.
        #     mask_, mask = mask_generator.generate(pmask)
        #
        #     # Area regularization.
        #     # mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        #     # regul_area = - ((mask_sorted - reference) ** 2).mean(dim=1) * regul_weight
        #
        #     covariance_det = gauss_pmask_generator.sh * gauss_pmask_generator.sw
        #     reference_covariance_det = 0.005
        #     regul_area = - ((covariance_det - reference_covariance_det) ** 2) * regul_area_weight
        #     print(regul_area_weight)
        #     regul_area_weight *= 1.1
        #
        #     energy = (regul_area)
        #
        #     # Gradient step.
        #     optimizer.zero_grad()
        #     (- energy).backward()
        #     optimizer.step()
        #
        #     for name, p in gauss_pmask_generator.named_parameters():
        #         if name == 'mh':
        #             p.data = p.data.clamp(0, 1)
        #         if name == 'mw':
        #             p.data = p.data.clamp(0, 1)
        #         if name == 'sh':
        #             p.data = p.data.clamp(0.025, 0.1 * h)
        #         if name == 'sw':
        #             p.data = p.data.clamp(0.025, 0.1 * w)
        #         # if name == 'gauss_scale':
        #         #     p.data = p.data.clamp(0., 3000.)
        #
        #     for p in gauss_pmask_generator.named_parameters():
        #         print(p)
        #     print('sh * sw', gauss_pmask_generator.sh * gauss_pmask_generator.sw)
        #     print('sh.grad, sw.grad', gauss_pmask_generator.sh.grad, gauss_pmask_generator.sw.grad)
        #     grad_hist.append((gauss_pmask_generator.sh.grad.cpu(), gauss_pmask_generator.sw.grad.cpu()))
        #
        # for i, g in enumerate(grad_hist):
        #     print(i, g)
        #
        # return