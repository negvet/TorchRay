import os
from typing import List, Tuple, Union
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import sys

from openvino.inference_engine import ExecutableNetwork
import torch

# from algorithms.extreme_perturb.extremal_perturbation import ExtremePerturbAnalysis
from torchray.attribution.adaptive_rise_torch_ov import AdaptiveRISEAnalysis
from torchray.benchmark.models import get_model, get_transform


######################################################## SET SEED ###################################################
seed = 0
np.random.seed(seed)


ID = 11
INPUT_SIZE = (224, 224)


def get_argiment_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fw', type=str, help='Framework to use', choices=['ov', 'torch'], default='ov')
    return parser


def visualize(img_file, init_img, saliency_maps):
    if isinstance(init_img, Image.Image):
        init_img = np.array(init_img)

    saliency_maps = cv2.resize(saliency_maps, (init_img.shape[1], init_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    fig = plt.figure(figsize=(7, 7))
    plt.imshow(init_img[:, :, ::-1])
    plt.imshow(saliency_maps, cmap='jet', alpha=0.5)#, vmin=0, vmax=1)
    plt.axis('off')

    result_path = os.path.join(os.path.expanduser('~'), "TorchRay/examples/results/adaptive_rise/")
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    fig.savefig(result_path + img_file[:-5] + '_id' + str(ID) + '.png')


def get_model_ov() -> ExecutableNetwork:
    from openvino.inference_engine import IECore

    model_path = os.path.join(os.path.expanduser('~'), "TorchRay/examples/model/public/resnet-50-pytorch/FP16/resnet-50-pytorch.xml")
    print(model_path)
    ie_core = IECore()
    net = ie_core.read_network(model=model_path)
    device = "CPU"
    model = ie_core.load_network(network=net, device_name=device)
    return model


def get_model_torch() -> torch.nn.Module:
    model = get_model(
                arch='resnet50',
                dataset='voc_2007',
                convert_to_fully_convolutional=True,
            )
    return model.eval().cuda()


def preprocess_img_ov(init_img: np.ndarray):
    from examples.utils.engine import COCOInference
    infer = COCOInference
    preprocessed_img = infer.process_input(frame=init_img, dsize=INPUT_SIZE)
    return preprocessed_img


def preprocess_img_torch(init_img: Image.Image):
    transform = get_transform(size=INPUT_SIZE,
                                  dataset='voc_2007')
    preprocessed_img = transform(init_img).unsqueeze(0).numpy()
    return preprocessed_img


def get_input_img(framework) -> Tuple[str, Union[np.ndarray, Image.Image], int]:
    data = "imagenet" # pothole id 5는 못찾는 경우
    data_path = os.path.join(os.path.expanduser('~'), 'TorchRay/examples/data/imagenet/')
    json_path = os.path.join(os.path.expanduser('~'), 'TorchRay/examples/data/imagenet.json')

    with open(json_path) as json_file:
        data = json.load(json_file)
        img_file = data[str(ID)]["image"]
        target_class = data[str(ID)]["category_id"]
    print(img_file, ID)

    if framework == 'ov':
        frame = cv2.imread(os.path.join(data_path, img_file))
    elif framework == 'torch':
        frame = Image.open(os.path.join(data_path, img_file)).convert("RGB")

    return img_file, frame, target_class


def main(argv):
    parser = get_argiment_parser()
    args = parser.parse_args(args=argv)

    framework = args.fw

    img_file, init_img, target_class = get_input_img(framework)

    if framework == 'ov':
        model = get_model_ov()
        input_img = preprocess_img_ov(init_img)
        n_classes = 1000 # for imagenet
    elif framework == 'torch':
        model = get_model_torch()
        input_img = preprocess_img_torch(init_img)
        target_class = 11 # dog in VOC
        n_classes = 20 # for voc_2007 dataset

    rise = AdaptiveRISEAnalysis(model=model, n_classes=n_classes, framework=framework, n_masks=100, grid_size=(16, 16))
    saliency_map = rise.generate_sailancy_map(input_img, target_class)

    visualize(img_file, init_img, saliency_map)


if __name__ == '__main__':
    main(sys.argv[1:])
