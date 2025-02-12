# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

__all__ = ['_shap']

import numpy as np
import torch

import shap

from .common import resize_saliency


def _shap(model,
          input,
          target=None,
          num_evals=8000,
          batch_size=32,
          resize=224,
          resize_mode='bilinear',
          ):
    r"""SHAP.

    Args:
        model (:class:`torch.nn.Module`): a model.
        input (:class:`torch.Tensor`): input tensor.
        num_evals (int, optional): number of SHAP trials.
            Default: ``10000``.
        batch_size (int, optional): batch size to use. Default: ``128``.

    Returns:
        :class:`torch.Tensor`: SHAP saliency map.
    """
    input_shap = input.permute(0, 2, 3, 1)
    device = input.device

    def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
        return x

    def predict(img: np.ndarray) -> torch.Tensor:
        img = nhwc_to_nchw(torch.Tensor(img))
        img = img.to(device)
        output = model(img)
        return output[:, :, 0, 0]
        # return output.squeeze(2, 3)

    # out = predict(input)
    # classes = torch.argmax(out, axis=1).cpu().numpy()

    masker_blur = shap.maskers.Image("blur(128,128)", input_shap[0].shape)
    explainer = shap.Explainer(predict, masker_blur)

    shap_values = explainer(
        input_shap,
        max_evals=num_evals,
        batch_size=batch_size,
        outputs=[target],
    )

    saliency = torch.Tensor(shap_values.values[..., 0])
    saliency = saliency.permute(0, 3, 1, 2)[:, 1:2, ...]  # extract only single channel between duplicated RGB-channels

    # Resize saliency map.
    saliency = resize_saliency(input,
                               saliency,
                               resize,
                               mode=resize_mode)

    return saliency
