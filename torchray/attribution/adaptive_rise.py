from typing import List, Tuple
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from openvino.inference_engine import IECore
import itertools
import copy

class AdaptiveRISEAnalysis:
    def __init__(
        self, 
        model_path: str,
        n_masks: int = 1000,
        grid_size: Tuple = [16, 16],
        device = "CPU",
    ):
        ie_core = IECore()
        net = ie_core.read_network(model=model_path)
        self.exec_net = ie_core.load_network(network=net, device_name=device)

        self.n_masks = n_masks
        self.grid_size = grid_size
        self.vertices = {}
        self.threshold = [0.2, 0.3, 0.3, 0.4, 0.4]
        self.prob = [0.1, 0.1, 0.2, 0.4, 0.8]

    def _generate_bf_mask(self, image_size: Tuple[int,int], vertices = List[List[int]]) -> np.ndarray:
        image_w, image_h = image_size
        grid_w, grid_h = self.grid_size

        indices = [[0, 1]] * len(vertices)
        cand = list(itertools.product(*indices))
        print(cand)
        
        masks = np.zeros((len(cand), 3, image_w, image_h), dtype=np.float32)
        for i in range(len(cand)):
            mask = np.zeros((grid_w, grid_h), dtype=np.float32)
            for j in range(len(vertices)):
                if cand[i][j] == 1:
                    mask[vertices[j][0]:vertices[j][0]+vertices[j][2], \
                        vertices[j][1]:vertices[j][1]+vertices[j][2]] = 1.0
            mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
            masks[i,:,:,:] = np.transpose(np.dstack([mask] * 3), (2, 0, 1))

        return masks

    def _mask_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask = mask[np.newaxis, ...]
        masked = ((image.astype(np.float32) / 255 * mask) * 255).astype(np.uint8)
        return masked

    def _generate_mask(self, image_size: Tuple[int,int], filtered_mask = np.ndarray) -> np.ndarray:
        image_w, image_h = image_size
        grid_w, grid_h = self.grid_size
        cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
        up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h
        
        # mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) < 0.5).astype(np.float32)
        mask = np.ones(self.grid_size, dtype=np.float32)
        for h in range(grid_h):
            for w in range(grid_w):
                for j in self.vertices.keys():
                    if w >= self.vertices[j][0] and w < self.vertices[j][2] and \
                        h >= self.vertices[j][1] and h < self.vertices[j][2] and np.random.uniform(0,1) < 0.5:
                        mask[h][w] = 0.0
        
        # mask = mask * filtered_mask
        # mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_LINEAR)

        mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
        offset_w = np.random.randint(0, cell_w)
        offset_h = np.random.randint(0, cell_h)
        mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
        mask = np.transpose(np.dstack([mask] * 3), (2, 0, 1))

        return mask

    def _generate_mask_prob(self, image_size: Tuple[int,int]) -> np.ndarray:
        image_w, image_h = image_size
        grid_w, grid_h = self.grid_size
        cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
        up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h
        
        norm_prop = []
        for v in self.vertices.values():
            norm_prop.append(v)

        # mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) < 0.5).astype(np.float32)
        mask = np.zeros(self.grid_size, dtype=np.float32)
        for v, score in self.vertices.items():
            temp = (np.random.uniform(0, 1, size=(v[2], v[2])) < 0.5 / max(norm_prop) * score).astype(np.float32)
            # temp = (np.random.uniform(0, 1, size=(v[2], v[2])) < score).astype(np.float32)
            for r in range(v[2]): 
                for c in range(v[2]):
                    mask[v[0]+r][v[1]+c] = temp[r][c]

        mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
        offset_w = np.random.randint(0, cell_w)
        offset_h = np.random.randint(0, cell_h)
        mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
        mask = np.transpose(np.dstack([mask] * 3), (2, 0, 1))

        return mask

    def generate_sailancy_map(
        self, 
        vis: np.ndarray,
        frame: np.ndarray, 
        target_class: int, 
    ) -> np.ndarray:

        input_key = list(self.exec_net.input_info)[0]
        image_h, image_w = frame.shape[2:] # 1 x 3 x 224 x 224

        results = self.exec_net.infer(inputs={input_key: frame})
        y = results['prob'][0]
        logits = np.exp(y)/sum(np.exp(y))
        base_score = logits[target_class]   
        entropy = - np.sum(logits * np.log(logits)) / np.log(1000)
        # base_score = results['prob'][0][target_class]
        print("base:", base_score, entropy, np.argmax(logits))

        trial = 0
        self.vertices[(0, 0, self.grid_size[0])] = self.prob[trial]
        if entropy < self.threshold[trial]:
            dropped_v = {}
            for _ in range(3):
                trial += 1
                # print("######################## {} #######################".format(trial))
                next_v = {}
                for idx, (test_v, test_score) in enumerate(self.vertices.items()):
                    v_size = test_v[2] // 2
                    split_test_v = [[test_v[0], test_v[1], v_size],\
                                    [test_v[0]+v_size, test_v[1], v_size],\
                                    [test_v[0], test_v[1]+v_size, v_size],\
                                    [test_v[0]+v_size, test_v[1]+v_size, v_size]]
                    
                    temp_dropped_v = []
                    temp_next_v = []
                    for idxx, v in enumerate(split_test_v):
                        cropped_frame = frame[:,:,v[0]*14:(v[0]+v[2])*14, v[1]*14:(v[1]+v[2])*14]
                        cropped_frame = np.stack([cv2.resize(frm, dsize=(image_h,image_w)) for frm in cropped_frame[0]],axis=0)
                        cropped_frame = cropped_frame[np.newaxis,...]

                        results = self.exec_net.infer(inputs={input_key: cropped_frame})
                        y = results['prob'][0]
                        logits = np.exp(y)/sum(np.exp(y))
                        base_score = logits[target_class]
                        entropy = - np.sum(logits * np.log(logits)) / np.log(1000)
                        # print('base_score', trial, idx, idxx, v, base_score, entropy, np.argmax(logits))

                        if entropy < self.threshold[trial]:
                            temp_next_v.append((v, self.prob[trial+1]))
                        else:
                            temp_dropped_v.append((v, self.prob[trial]))
                    if not temp_next_v:
                        dropped_v[test_v] = test_score
                    else:
                        for v in temp_next_v:
                            next_v[tuple(v[0])] = v[1]
                        for v in temp_dropped_v:
                            dropped_v[tuple(v[0])] = v[1]
                    # print('next', next_v, dropped_v)
                if next_v is None: 
                    break
                else: 
                    self.vertices = next_v
            self.vertices.update(dropped_v)
            print(self.vertices)
        
        saliency_map  = np.zeros((image_h, image_w), dtype=np.float32)
        for idx in range(self.n_masks):
            mask = self._generate_mask_prob(image_size=(image_w, image_h)) # 224 x 224
            masked = self._mask_image(frame, mask)

            results = self.exec_net.infer(inputs={input_key: masked})
            y = results['prob'][0]
            logits = np.exp(y)/sum(np.exp(y))
            score = logits[target_class]
            saliency_map += mask[0,:,:] * score

        saliency_map /= self.n_masks

        return saliency_map
