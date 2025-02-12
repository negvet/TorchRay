import os
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import time
from openvino.inference_engine import IECore

# from algorithms.extreme_perturb.extremal_perturbation import ExtremePerturbAnalysis
from torchray.attribution.adaptive_rise import AdaptiveRISEAnalysis
from examples.utils.engine import COCOInference, OTEInference

data = "imagenet" # pothole id 5는 못찾는 경우
data_path = './data/imagenet'
json_path = './data/imagenet.json'

# model path
model_name = "resnet-50-pytorch"
precision = "FP16"
model_path = f"./model/public/{model_name}/{precision}/{model_name}.xml"

# model_path = f"./model/public/resnet-50-pytorch/batch4/resnet-50-pytorch.xml"



result_path = './results/adaptive_rise/'
if not os.path.exists(result_path):
    os.mkdir(result_path)


print(model_path)
ie_core = IECore()
net = ie_core.read_network(model=model_path)
exec_net = ie_core.load_network(network=net, device_name="CPU")

input_key = list(exec_net.input_info)[0]
output_key = list(exec_net.outputs.keys())[0]
height, width = exec_net.input_info[input_key].tensor_desc.dims[2:]




for id in range(0, 16):
    rise = AdaptiveRISEAnalysis(model_path=model_path, n_masks=100,
                        grid_size=(16, 16), device="CPU")

    start = time.time()
    with open(json_path) as json_file:
        data = json.load(json_file)
        img_file = data[str(id)]["image"]
        target_class = data[str(id)]["category_id"]
    print('\n', img_file, id)

    frame = cv2.imread(os.path.join(data_path, img_file))
    print("data loading time (sec): ", time.time()-start)

    infer = COCOInference
    input_img = infer.process_input(frame=frame, dsize=(width, height))
    saliency_maps = rise.generate_sailancy_map(frame, input_img, target_class)


    saliency_maps = cv2.resize(saliency_maps, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    fig = plt.figure(figsize=(7, 7))
    plt.imshow(frame[:, :, ::-1])
    plt.imshow(saliency_maps, cmap='jet', alpha=0.5)#, vmin=0, vmax=1)
    plt.axis('off')
    fig.savefig(result_path+img_file[:-5]+'_id'+str(id)+'.png')
    print("visualization time (sec): ", time.time()-start)
