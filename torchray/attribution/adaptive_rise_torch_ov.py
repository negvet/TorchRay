# # from cgi import print_arguments
# import time
# from typing import Tuple, Union
# import math
# import numpy as np
# import cv2
#
#
# from openvino.inference_engine import ExecutableNetwork
# import torch
# import torch.nn.functional as F
# import torchvision
# from torchray.attribution.rise import _upsample_reflect
#
# # Some utils
# def sliding_window(image, stepSize, windowSize):
#     # slide a window across the image
#     for y in range(0, image.shape[2], stepSize):
#         for x in range(0, image.shape[3], stepSize):
#             # yield the current window
#             yield (x, y, image[:, :, y:y + windowSize[1], x:x + windowSize[0]])
#
#
# class AdaptiveRISEAnalysis:
#     def __init__(
#         self,
#         model : Union[ExecutableNetwork, torch.nn.Module],
#         n_classes: int,
#         framework: str = 'ov',
#         input_size: tuple = (224, 224),
#         n_masks: int = 1000,
#         grid_size: Tuple = [16, 16],
#         batch_size: int = 100,
#         loc_det_method: str = 'entropy',
#         method_params = None,
#     ):
#         self.model = model
#         self.batch_size = batch_size
#         self.framework = framework
#         self.input_size = input_size
#         self.n_masks = n_masks
#         self.grid_size = grid_size
#         self.vertices = {}
#         self.n_classes = n_classes
#
#         # Select from: 'entropy' 'cross_entropy' 'sw' 'sw_entropy'
#         self.loc_det_method = loc_det_method
#         self.test_hide_prob_mask_only = False
#
#         if method_params is not None:
#             # print('method_params:', method_params)
#             self.method_params = method_params
#
#         self.window_history = {}
#         self.offline_method = False
#
#         torch.manual_seed(0)
#
#         if self.loc_det_method == 'entropy':
#             # print('Use Entropy')
#
#             # window_sizes, step_sizes, init_prob, hide_probabilities, n_windows, grid_size, detect_prob_threshs, e_threshold = self.method_params
#             # self.e_threshold = list(e_threshold)
#             # self.hide_prob_vs_level = list(hide_probabilities)
#
#
#             self.window_sizes = [(112, 112), (56, 56), (28, 28)]
#             self.step_sizes = [112, 56, 28]
#
#             # # Entropy, Wonju params
#             # pointing 49.0, pointing_difficult 22.0
#             # self.e_threshold = [0.2, 0.3, 0.3, 0.4, 0.4]
#             # self.hide_prob_vs_level = [0.1, 0.1, 0.2, 0.4, 0.8]
#
#             # Entropy, HP search (VOC)
#             # pointing 67.7, pointing_difficult 39.7 with local rand_mask
#             # pointing 46.0, pointing_difficult 28.3 with global rand_mask (more efficient, accidentally perform worse)
#             # self.e_threshold = [0.3, 0.35, 0.375, 0.4]
#             # self.hide_prob_vs_level = [0.1, 0.2, 0.4, 0.8]
#
#             # # ### fixed mask rescale 75.6/55.3 ### w/o else target_class_score * hide_prob_vs_level_list
#             # self.e_threshold = [0.06007340188403569, 0.43843258458685014, 0.5386484722834396, 0.5870263431427508]
#             # self.hide_prob_vs_level = [0.22689047130725148, 0.3842258310595026, 0.5375861935093168, 0.8488834769344369]
#
#             # ### with fixed mask rescale 77.4/53.4
#             self.e_threshold = [0.23251454410368397, 0.23614372720474272, 0.24337482948630954, 0.2443019107288976]
#             self.hide_prob_vs_level = [0.6918573994196947, 0.8500709653048456, 0.8816043065385348, 0.9021287437462904]
#         elif self.loc_det_method == 'cross_entropy':
#             print('Use Cross-Entropy')
#             # # Cross-entropy, HP search (VOC)
#             # # pointing 0.568, pointing_difficult 0.234
#             self.predict_prob = [0.6, 0.55, 0.5, 0.45]
#             self.hide_prob_vs_level = [0.1, 0.1, 0.2, 0.4, 0.8]
#
#             self.e_threshold = [- np.log(item) for item in self.predict_prob]
#         elif self.loc_det_method == 'sw':
#             # window_sizes, step_sizes, init_prob, hide_probabilities, n_windows, grid_size, detect_prob_threshs, e_threshold = self.method_params
#             # self.hide_prob_vs_level = hide_probabilities
#             # self.detect_prob_thresh = detect_prob_threshs
#
#             self.window_sizes = [(224 // 2, 224 // 2), (224 // 4, 224 // 4), (26, 26)]
#             self.step_sizes = [56, 42, 22]
#             # self.hide_prob_vs_level = [0.2, 0.3, 0.5, 0.8]
#
#             # # 72.0/44.1 set target_class_score to 1
#             # # sub_m[sub_m < (target_class_score * self.hide_prob_vs_level[step])] = self.hide_prob_vs_level[step]
#             # self.hide_prob_vs_level = [0.2717125293986743, 0.7583562806562215, 0.8287564401881778, 0.9696532883865291]
#             # self.detect_prob_thresh = 0.2
#
#             # # fixed scale mask 77.9/54.9
#             # self.hide_prob_vs_level = [0.11735906910661309, 0.3246868485929505, 0.5344194647380829, 0.96622523487]
#             # self.detect_prob_thresh = 0.4 # for bbox to keep
#
#             # fixed scale mask 80.6/55.6
#             self.hide_prob_vs_level = [0.125362904220766, 0.21426969464793433, 0.4723551533281406, 0.8062740744111907]
#             self.detect_prob_thresh = 0.4
#         elif self.loc_det_method == 'sw_entropy':
#             # window_sizes, step_sizes, init_prob, hide_probabilities, n_windows, grid_size, detect_prob_threshs, e_threshold = self.method_params
#             # self.e_threshold = list(e_threshold)
#             # self.hide_prob_vs_level = list(hide_probabilities)
#
#             self.window_sizes = [(224 // 2, 224 // 2), (224 // 4, 224 // 4), (26, 26)]
#             self.step_sizes = [56, 42, 22]
#
#             # self.e_threshold = [0.3, 0.35, 0.375, 0.4]
#             # self.hide_prob_vs_level = [0.1, 0.2, 0.4, 0.8]
#
#
#             # # GPU 3, 82.7/68.5 fixed mask rescaling
#             # self.e_threshold = [0.26196496858557666, 0.2757221829172699, 0.2945323687173077, 0.4845579917462672]
#             # self.hide_prob_vs_level = [0.046005727605316404, 0.11088501683743379, 0.412654903504557, 0.8103466847332615]
#
#             # # GPU 3, 82.3/67.0 fixed mask rescaling
#             # self.e_threshold = [0.19825521542284316, 0.2831804210777451, 0.298054410538121, 0.49946195690572714]
#             # self.hide_prob_vs_level = [0.007116259038177919, 0.12344201074247663, 0.25279658371866665, 0.4334308277741359]
#
#             # # GPU 3, 83.9/67.6 fixed mask rescaling
#             # self.e_threshold = [0.12973385410518623, 0.6322141495374082, 0.8889306968033414, 0.8905167570753474]
#             # self.hide_prob_vs_level = [0.15696128091043826, 0.5225205496541462, 0.5638283725289605, 0.9710179304684956]
#
#             # GPU 3, 84.7/70.1
#             self.e_threshold = [0.1944647444319103, 0.4181195741885608, 0.7643505172424835, 0.9120056599401741]
#             self.hide_prob_vs_level = [0.03159610438591043, 0.12718202211911767, 0.5777885681578225, 0.8469101435208534]
#         elif self.loc_det_method == 'entropy_abs':
#             self.grid_size = (7, 7)
#
#             # window_sizes, step_sizes, init_prob, hide_probabilities, n_windows, grid_size, detect_prob_threshs, e_threshold, mask_scale = self.method_params
#             # self.e_threshold = list(e_threshold)
#             # self.hide_prob_vs_level = list(hide_probabilities)
#             # self.grid_size = grid_size
#             # self.mask_scale = mask_scale
#
#             self.window_sizes = [(224 // 2, 224 // 2), (224 // 4, 224 // 4), (26, 26)] # 3*3, 5*5, 10*10 (134 inferences at max)
#             self.step_sizes = [56, 42, 22]
#
#             # # 84.6/69.4 . Note: requires mask downscaling with self.mask_scale
#             # self.e_threshold = [0.1944647444319103, 0.4181195741885608, 0.7643505172424835, 0.9120056599401741]
#             # self.hide_prob_vs_level = [0.03159610438591043, 0.12718202211911767, 0.5777885681578225, 0.8469101435208534]
#             # self.mask_scale = 0.2
#
#             # # 85.9/68.3
#             # self.e_threshold = [0.10811919161146769, 0.30397746805780057, 0.3964587526024516, 0.8676537859601777]
#             # self.hide_prob_vs_level = [0.015763249849159243, 0.04054041587280869, 0.09979496445560347, 0.24787281033454847]
#
#             # 86.2/68.4
#             self.e_threshold = [0.08391669513844599, 0.3926695687128092, 0.4485679560147107, 0.837019970367532]
#             self.hide_prob_vs_level = [0.02404863575342404, 0.05835518311472475, 0.1503570413649637, 0.2762894266319287]
#         elif self.loc_det_method == 'entropy_rel':
#             self.offline_method = True
#             # window_sizes, step_sizes, init_prob, hide_probabilities, n_windows, grid_size, detect_prob_threshs, e_threshold, mask_scale, scale_scores = self.method_params
#             # self.global_hide_prob_scale = mask_scale
#             # self.scale_scores = scale_scores
#
#             self.grid_size = (7, 7)
#             self.window_sizes = [(224 // 2, 224 // 2), (224 // 4, 224 // 4), (26, 26)] # 3*3, 5*5, 10*10 (134 inferences at max)
#             self.step_sizes = [56, 42, 22]
#             # tmp
#             self.hide_prob_vs_level = [0]
#
#             # 85.1/70.8
#             self.global_hide_prob_scale = 0.3
#             self.scale_scores = (0.6, 0.75, 1)
#         elif self.loc_det_method == 'target_class_score':
#             self.offline_method = True
#             # window_sizes, step_sizes, init_prob, hide_probabilities, n_windows, grid_size, detect_prob_threshs, e_threshold, mask_scale, scale_scores = self.method_params
#             # self.global_hide_prob_scale = mask_scale
#             # self.scale_scores = scale_scores
#
#             self.grid_size = (7, 7)
#             self.window_sizes = [(224 // 2, 224 // 2), (224 // 4, 224 // 4), (26, 26)] # 3*3, 5*5, 10*10 (134 inferences at max)
#             self.step_sizes = [56, 42, 22]
#             # tmp
#             self.hide_prob_vs_level = [0]
#
#             # 84.6
#             self.global_hide_prob_scale = 0.3
#             self.scale_scores = (0.5, 0.7, 1)
#         elif self.loc_det_method == 'target_class_score_nms':
#             self.offline_method = True
#             # window_sizes, step_sizes, init_prob, hide_probabilities, n_windows, grid_size, detect_prob_threshs, e_threshold, mask_scale, scale_scores = self.method_params
#             # self.global_hide_prob_scale = mask_scale
#             # self.scale_scores = scale_scores
#
#             self.grid_size = (7, 7)
#             self.window_sizes = [(224 // 2, 224 // 2), (224 // 4, 224 // 4), (26, 26)] # 3*3, 5*5, 10*10 (134 inferences at max)
#             self.step_sizes = [56, 42, 22]
#             # tmp
#             self.hide_prob_vs_level = [0]
#
#             # based on target_class_score, pointing 0.868, pointing_difficult 0.639
#             # hide_prob_scaled = prob * self.scale_scores[step - 1] * self.global_hide_prob_scale
#             self.prob_thresh = 0.1
#             self.iou_threshold = 0.7
#
#             self.global_hide_prob_scale = 0.6
#             self.scale_scores = (0.6, 0.75, 1)
#         elif self.loc_det_method == 'sw_el2n':
#             self.offline_method = True
#             # window_sizes, step_sizes, init_prob, hide_probabilities, n_windows, grid_size, detect_prob_threshs, e_threshold, mask_scale, scale_scores = self.method_params
#             # self.global_hide_prob_scale = mask_scale
#             # self.scale_scores = scale_scores
#
#             self.grid_size = (7, 7)
#             self.window_sizes = [(224 // 2, 224 // 2), (224 // 4, 224 // 4), (26, 26)] # 3*3, 5*5, 10*10 (134 inferences at max)
#             self.step_sizes = [56, 42, 22]
#             # tmp
#             self.hide_prob_vs_level = [0]
#
#             # 83.8/69.2
#             self.global_hide_prob_scale = 0.3
#             self.scale_scores = (0.6, 0.8, 1)
#         elif self.loc_det_method == 'rise':
#             pass
#         else:
#             raise RuntimeError('Not implemented')
#
#
#     def _mask_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
#         mask = mask[np.newaxis, ...]
#         if self.framework == 'ov':
#             # masked = ((image.astype(np.float32) / 255 * mask) * 255).astype(np.uint8)
#             masked = (image.astype(np.float32) * mask).astype(np.uint8)
#         elif self.framework == 'torch':
#             masked = image * mask
#         else:
#             raise RuntimeError(f'Not implemented framework {self.framework}')
#         return masked
#
#     def _infer(self, frame: np.ndarray):
#         if self.framework == 'ov':
#             input_key = list(self.model.input_info)[0]
#             results = self.model.infer(inputs={input_key: frame})
#             y = results['prob'][0]
#             logits = np.exp(y)/sum(np.exp(y))
#         elif self.framework == 'torch':
#             with torch.no_grad():
#                 frame = torch.from_numpy(frame).cuda()
#                 y = self.model(frame)
#
#                 # sm = torch.nn.Softmax(dim=1)
#                 # logits = sm(y)
#
#                 logits = torch.sigmoid(y)
#
#                 logits = logits.cpu().numpy().squeeze()
#         return logits
#
#     def _calculate_entropy(self, logits, target_class):
#         if self.loc_det_method == 'entropy':
#             return - np.sum(logits * np.log(logits)) / np.log(self.n_classes)
#         elif self.loc_det_method == 'cross_entropy':
#             y = np.zeros(self.n_classes)
#             y[target_class] = 1
#             return - np.sum(y * np.log(logits))
#         elif self.loc_det_method in ['sw_entropy', 'entropy_abs', 'entropy_rel']:
#             return - np.sum(logits * np.log(logits)) / np.log(self.n_classes)
#         else:
#             raise RuntimeError('Not implemented')
#
#     # def _generate_vertices_with_probs(self, frame, target_class):
#     #     det_infer_counter = 1
#     #     logits = self._infer(frame)
#     #     target_class_score = logits[target_class]
#     #
#     #     # if np.argmax(logits) != target_class:
#     #     #     self.e_threshold = [- np.log(item/2.5) for item in self.predict_prob]
#     #
#     #     entropy = self._calculate_entropy(logits, target_class)
#     #     # print("\nInit image infer: \ntarget_class_score:", target_class_score, "\nentropy", entropy, "\nnp.argmax(logits)", np.argmax(logits), "\n")
#     #
#     #     trial = 0
#     #     self.vertices[(0, 0, self.grid_size[0])] = self.hide_prob_vs_level[trial]
#     #     if entropy < self.e_threshold[trial]:
#     #         dropped_v = {}
#     #         for _ in range(3):
#     #             trial += 1
#     #             # print("######################## {} #######################".format(trial))
#     #             next_v = {}
#     #             for idx, (test_v, test_score) in enumerate(self.vertices.items()):
#     #                 v_size = test_v[2] // 2
#     #                 split_test_v = [[test_v[0], test_v[1], v_size],\
#     #                                 [test_v[0]+v_size, test_v[1], v_size],\
#     #                                 [test_v[0], test_v[1]+v_size, v_size],\
#     #                                 [test_v[0]+v_size, test_v[1]+v_size, v_size]]
#     #
#     #                 temp_dropped_v = []
#     #                 temp_next_v = []
#     #                 for idxx, v in enumerate(split_test_v):
#     #                     cropped_frame = frame[:,:,v[0]*14:(v[0]+v[2])*14, v[1]*14:(v[1]+v[2])*14]
#     #                     cropped_frame = np.stack([cv2.resize(frm, dsize=self.input_size) for frm in cropped_frame[0]],axis=0)
#     #                     cropped_frame = cropped_frame[np.newaxis,...]
#     #
#     #                     logits = self._infer(cropped_frame)
#     #                     det_infer_counter += 1
#     #                     target_class_score = logits[target_class]
#     #                     entropy = self._calculate_entropy(logits, target_class)
#     #                     # print('target_class_score', trial, idx, idxx, v, target_class_score, entropy, np.argmax(logits))
#     #
#     #                     if entropy < self.e_threshold[trial]:
#     #                         temp_next_v.append((v, self.hide_prob_vs_level[trial+1]))
#     #                     else:
#     #                         temp_dropped_v.append((v, self.hide_prob_vs_level[trial]))
#     #                 if not temp_next_v:
#     #                     dropped_v[test_v] = test_score
#     #                 else:
#     #                     for v in temp_next_v:
#     #                         next_v[tuple(v[0])] = v[1]
#     #                     for v in temp_dropped_v:
#     #                         dropped_v[tuple(v[0])] = v[1]
#     #                 # print('next', next_v, "dropped", dropped_v)
#     #             if next_v is None:
#     #                 break
#     #             else:
#     #                 self.vertices = next_v
#     #             a = 1
#     #         self.vertices.update(dropped_v)
#     #         # print('vertices:\n', self.vertices)
#     #     else:
#     #         print('Whole image not certain enough for class', target_class)
#     #     print('localization infer_counter', det_infer_counter)
#     #
#     # def _generate_mask_prob_from_vertices(self, image_size: Tuple[int,int]) -> np.ndarray:
#     #     image_w, image_h = image_size
#     #     grid_w, grid_h = self.grid_size
#     #     cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
#     #     up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h
#     #
#     #     norm_prop = []
#     #     for v in self.vertices.values():
#     #         norm_prop.append(v)
#     #
#     #     mask = np.zeros(self.grid_size, dtype=np.float32)
#     #     # rand_mask = np.random.uniform(0, 1, size=self.grid_size)
#     #     for v, score in self.vertices.items():
#     #         rand_sub_mask = np.random.uniform(0, 1, size=(v[2], v[2]))
#     #         # rand_sub_mask = rand_mask[v[0]: v[0] + v[2], v[1]: v[1] + v[2]]
#     #         temp = (rand_sub_mask < 0.5 / max(norm_prop) * score).astype(np.float32)
#     #         mask[v[0] : v[0]+v[2], v[1] : v[1]+v[2]] = temp
#     #
#     #     mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
#     #     offset_w = np.random.randint(0, cell_w)
#     #     offset_h = np.random.randint(0, cell_h)
#     #     mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
#     #     mask = np.transpose(np.dstack([mask] * 3), (2, 0, 1))
#     #
#     #     return mask
#
#     def _should_do_loc_det_image(self, input_img, target_class):
#         # if self.loc_det_method == 'entropy':
#         #     # logits = self._infer(input_img)
#         #     # entropy = self._calculate_entropy(logits, target_class)
#         #     # return entropy < self.e_threshold[0]
#         #     return True
#         # elif self.loc_det_method == 'sw':
#         #     return True
#         # elif self.loc_det_method == 'sw_entropy':
#             return True
#
#     def _update_mask_prob(self, mask_prob, updated_mask_prob_current, step, window_params, logits, target_class):
#         x, y, winW, winH = window_params
#         if self.loc_det_method == 'entropy':
#             entropy = self._calculate_entropy(logits, target_class)
#             # TODO: consider to remove else, experiment with masking needed
#             hide_prob_vs_level_list = [self.hide_prob_vs_level[0]] + [self.hide_prob_vs_level[1]] + self.hide_prob_vs_level[1:]
#             target_class_score = logits[target_class]
#             if entropy < self.e_threshold[step]:
#                 mask_prob[y: y + winH, x: x + winW] = target_class_score * hide_prob_vs_level_list[step + 1]
#                 updated_mask_prob_current[y: y + winH, x: x + winW] = True
#             else:
#                 mask_prob[y: y + winH, x: x + winW] = target_class_score * hide_prob_vs_level_list[step]
#         elif self.loc_det_method == 'sw':
#             target_class_score = logits[target_class]
#             if target_class_score > self.detect_prob_thresh:
#                 sub_m = mask_prob[y : y+winH, x : x+winW]
#                 # TODO: set target_class_score to 1
#                 sub_m[sub_m < (target_class_score * self.hide_prob_vs_level[step])] = target_class_score * self.hide_prob_vs_level[step]
#                 mask_prob[y : y+winH, x : x+winW] = sub_m
#                 updated_mask_prob_current[y: y + winH, x: x + winW] = True
#
#             # if target_class_score > self.detect_prob_thresh:
#             #     sub_m = mask_prob[y : y+winH, x : x+winW]
#             #     # TODO: set target_class_score to 1
#             #     sub_m[sub_m < (target_class_score * self.hide_prob_vs_level[step])] = self.hide_prob_vs_level[step]
#             #     mask_prob[y : y+winH, x : x+winW] = sub_m
#             #     updated_mask_prob_current[y: y + winH, x: x + winW] = True
#         elif self.loc_det_method == 'sw_entropy':
#             entropy = self._calculate_entropy(logits, target_class)
#             # # TODO: consider to remove else, experiment with masking needed
#             # hide_prob_vs_level_list = [self.hide_prob_vs_level[0]] + [self.hide_prob_vs_level[1]] + self.hide_prob_vs_level[1:]
#             # target_class_score = logits[target_class]
#             # if entropy < self.e_threshold[step]:
#             #     mask_prob[y: y + winH, x: x + winW] = target_class_score * hide_prob_vs_level_list[step + 1]
#             #     updated_mask_prob_current[y: y + winH, x: x + winW] = True
#             # else:
#             #     mask_prob[y: y + winH, x: x + winW] = target_class_score * hide_prob_vs_level_list[step]
#
#             # GPU 3
#             # TODO: remove bias towards small scale object detection?
#             # TODO: update hide probability at small scale only if more confident then at big scale
#             target_class_score = logits[target_class]
#             if entropy < self.e_threshold[step]:
#                 sub_m = mask_prob[y: y + winH, x: x + winW]
#                 sub_m[sub_m < (target_class_score * self.hide_prob_vs_level[step])] = target_class_score * self.hide_prob_vs_level[step]
#                 mask_prob[y: y + winH, x: x + winW] = sub_m
#                 updated_mask_prob_current[y: y + winH, x: x + winW] = True
#
#             # # GPU 2
#             # target_class_score = logits[target_class]
#             # if entropy < self.e_threshold[step]:
#             #     sub_m = mask_prob[y: y + winH, x: x + winW]
#             #     sub_m[sub_m < (target_class_score * self.hide_prob_vs_level[step])] = self.hide_prob_vs_level[step]
#             #     mask_prob[y: y + winH, x: x + winW] = sub_m
#             #     updated_mask_prob_current[y: y + winH, x: x + winW] = True
#
#                                         # # GPU 1 71...
#                                         # target_class_score = logits[target_class]
#                                         # if entropy < self.e_threshold[step]:
#                                         #     mask_prob[y: y + winH, x: x + winW] = target_class_score * self.hide_prob_vs_level[step]
#                                         #     updated_mask_prob_current[y: y + winH, x: x + winW] = True
#
#                                         # # GPU 0, 70.4/35.7
#                                         # target_class_score = logits[target_class]
#                                         # if entropy < self.e_threshold[step]:
#                                         #     sub_m = mask_prob[y: y + winH, x: x + winW]
#                                         #     sub_m[sub_m < (self.hide_prob_vs_level[step])] = self.hide_prob_vs_level[step]
#                                         #     mask_prob[y: y + winH, x: x + winW] = sub_m
#                                         #     updated_mask_prob_current[y: y + winH, x: x + winW] = True
#         elif self.loc_det_method == 'entropy_abs':
#             entropy = self._calculate_entropy(logits, target_class)
#
#             if step not in self.window_history:
#                 self.window_history[step] = []
#             self.window_history[step].append(entropy)
#
#             # GPU 3
#             # TODO: remove bias towards small scale object detection?
#             # TODO: update hide probability at small scale only if more confident then at big scale
#             target_class_score = logits[target_class]
#             if entropy < self.e_threshold[step]:
#                 sub_m = mask_prob[y: y + winH, x: x + winW]
#                 sub_m[sub_m < (target_class_score * self.hide_prob_vs_level[step])] = target_class_score * self.hide_prob_vs_level[step]
#                 mask_prob[y: y + winH, x: x + winW] = sub_m
#                 updated_mask_prob_current[y: y + winH, x: x + winW] = True
#         elif self.loc_det_method == 'entropy_rel':
#             entropy = self._calculate_entropy(logits, target_class)
#             target_class_score = logits[target_class]
#             if step not in self.window_history:
#                 self.window_history[step] = []
#             self.window_history[step].append((window_params, entropy, target_class_score))
#         elif self.loc_det_method in ['target_class_score', 'target_class_score_nms']:
#             if step not in self.window_history:
#                 self.window_history[step] = []
#             self.window_history[step].append((window_params, logits, target_class))
#         elif self.loc_det_method == 'sw_el2n':
#             y = np.zeros(self.n_classes)
#             y[target_class] = 1
#             el2n = np.linalg.norm(y - logits)
#
#             if step not in self.window_history:
#                 self.window_history[step] = []
#             self.window_history[step].append((window_params, logits, el2n))
#         else:
#             raise NotImplemented
#
#     def _update_mask_prob_offline(self):
#         if self.loc_det_method == 'entropy_rel':
#             # TODO: scale with target_class_score
#             # TODO: add more weight to the later steps (small scale), yes yes
#             mask_prob = np.zeros((224, 224))
#
#             all_window_history = self.window_history[1] + self.window_history[2] + self.window_history[3]
#             all_entropies = np.array([entropy for _, entropy, _ in all_window_history])
#             entropy_max = all_entropies.max()
#
#             for step in list(self.window_history.keys()):
#                 for window_params, entropy, target_class_score in self.window_history[step]:
#                     x, y, winW, winH = window_params
#                     entropy_norm = entropy / entropy_max
#                     hide_prob = - entropy_norm + 1
#
#                     sub_m = mask_prob[y: y + winH, x: x + winW]
#                     hide_prob_scaled = hide_prob * target_class_score * self.scale_scores[step - 1] * self.global_hide_prob_scale
#                     sub_m[sub_m < hide_prob_scaled] = hide_prob_scaled
#                     mask_prob[y: y + winH, x: x + winW] = sub_m
#             return mask_prob
#         if self.loc_det_method == 'target_class_score_nms':
#             mask_prob = np.zeros((224, 224))
#
#             all_rects = []
#             all_probs = []
#             all_steps = []
#             for step in list(self.window_history.keys()):
#                 for window_params, logits, target_class in self.window_history[step]:
#                     target_class_score = logits[target_class]
#                     if target_class_score > self.prob_thresh:
#                         x, y, winW, winH = window_params
#                         all_rects.append([x, y, winW, winH])
#                         all_probs.append(target_class_score)
#                         all_steps.append(step)
#
#             if all_rects == []:
#                 return mask_prob
#             all_rects = np.array(all_rects)
#             pick_nms = torchvision.ops.nms(boxes=torch.from_numpy(all_rects.astype(np.float32)),
#                                            scores=torch.from_numpy(np.array(all_probs)), iou_threshold=self.iou_threshold)
#             steps_nms = [item for (idx, item) in enumerate(all_steps) if idx in pick_nms.numpy()]
#             rects_nms = [item for (idx, item) in enumerate(all_rects) if idx in pick_nms.numpy()]
#             probs_nms = [item for (idx, item) in enumerate(all_probs) if idx in pick_nms.numpy()]
#
#             for step, rect, prob in zip(steps_nms, rects_nms, probs_nms):
#             # for step, rect, prob in zip(all_steps, all_rects, all_probs):
#                 x, y, winW, winH = rect
#                 sub_m = mask_prob[y: y + winH, x: x + winW]
#                 hide_prob_scaled = prob * self.scale_scores[step - 1] * self.global_hide_prob_scale
#                 # hide_prob_scaled = prob * self.global_hide_prob_scale
#                 sub_m[sub_m < hide_prob_scaled] = hide_prob_scaled
#                 mask_prob[y: y + winH, x: x + winW] = sub_m
#
#             # for (xA, yA, xB, yB) in rects_nms:
#             #     cv2.rectangle(mask_prob, (xA, yA), (xB, yB), (0, 255, 0), 2)
#
#             return mask_prob
#         if self.loc_det_method == 'target_class_score':
#             # TODO: scale with target_class_score
#             # TODO: add more weight to the later steps (small scale), yes yes
#             mask_prob = np.zeros((224, 224))
#             certainty_map = np.zeros((224, 224))
#             # all_window_history = self.window_history[1] + self.window_history[2] + self.window_history[3]
#             # all_target_class_score = [logits[target_class] for window_params, logits, target_class in all_window_history]
#             # target_class_score_max = max(all_target_class_score)
#             # print('\ntarget_class_score_max global', target_class_score_max)
#
#             # el2n_all = []
#             # for window_params, logits, target_class in all_window_history:
#             #     y = np.zeros(self.n_classes)
#             #     y[target_class] = 1
#             #     el2n = np.linalg.norm(y - logits)
#             #     el2n_all.append(el2n)
#             # el2n_max = max(el2n_all)
#
#             # e_to_use_all = []
#             # for window_params, logits, target_class in all_window_history:
#             #     y_eps = np.zeros(self.n_classes) + 1e-7
#             #     y_eps[target_class] = 1
#             #     e_to_use = np.sum(logits * np.log(logits/y_eps))
#             #     e_to_use_all.append(e_to_use)
#             # e_to_use_max = max(e_to_use_all)
#
#             # e_ce_all = []
#             # ce_all = []
#             # for window_params, logits, target_class in all_window_history:
#             #     e = - np.sum(logits * np.log(logits))
#             #     y = np.zeros(self.n_classes)
#             #     y[target_class] = 1
#             #     ce = - np.sum(y * np.log(logits))
#             #     e_ce_all.append(e / ce)
#             #     ce_all.append(ce)
#             # e_ce_max = max(e_ce_all)
#             for step in list(self.window_history.keys()):
#                 # target_class_scores = [logits[target_class] for window_params, logits, target_class in self.window_history[step]]
#                 # target_class_score_max_within_step = max(target_class_scores)
#                 # print('step', step, 'target_class_score_max_within_step', target_class_score_max_within_step)
#
#                 # el2n_all = []
#                 # for window_params, logits, target_class in self.window_history[step]:
#                 #     y = np.zeros(self.n_classes)
#                 #     y[target_class] = 1
#                 #     el2n = np.linalg.norm(y - logits)
#                 #     el2n_all.append(el2n)
#                 # el2n_max = max(el2n_all)
#
#                 # e_to_use_all = []
#                 # for window_params, logits, target_class in self.window_history[step]:
#                 #     y_eps = np.zeros(self.n_classes) + 1e-7
#                 #     y_eps[target_class] = 1
#                 #     e_to_use = np.sum(logits * np.log(logits/y_eps))
#                 #     e_to_use_all.append(e_to_use)
#                 # e_to_use_max = max(e_to_use_all)
#
#                 # e_ce_all = []
#                 # for window_params, logits, target_class in self.window_history[step]:
#                 #     e = - np.sum(logits * np.log(logits))
#                 #     y = np.zeros(self.n_classes)
#                 #     y[target_class] = 1
#                 #     ce = - np.sum(y * np.log(logits))
#                 #     e_ce_all.append(e/ce)
#                 # e_ce_max = max(e_ce_all)
#                 for window_params, logits, target_class in self.window_history[step]:
#                     target_class_score = logits[target_class]
#                     # e = - np.sum(logits * np.log(logits))
#                     # y = np.zeros(self.n_classes)
#                     # y[target_class] = 1
#                     # ce = - np.sum(y * np.log(logits))
#                     # print(f'\nstep {step}, window_params {window_params}, target_class_score {target_class_score:.3f}, target_class_is_max {np.argmax(logits) == target_class}')
#                     # print(f'e {e}, ce {ce}')
#                     # print(f'e/ce {e / ce}, ce/e {ce / e}')
#                     #
#                     # y_eps = np.zeros(self.n_classes) + 1e-7
#                     # y_eps[target_class] = 1
#                     # print(f'Dkl(logits||y_eps) {np.sum(logits * np.log(logits/y_eps))}, H(logits, y_eps) {- np.sum(logits * np.log(y_eps))}, H(logits) {- np.sum(logits * np.log(logits))}')
#                     # print(f'Dkl(y_eps||logits) {np.sum(y_eps * np.log(y_eps/logits))}, H(y_eps, logits) {- np.sum(y_eps * np.log(logits))}, H(y_eps) {- np.sum(y_eps * np.log(y_eps))}')
#                     # # el2n = np.linalg.norm(y - logits)
#                     # # print(f'el2n {el2n:.3f}')
#
#
#
#                     # el2n = el2n / el2n_max
#                     # hide_prob = - el2n + 1
#                     # e_to_use = np.sum(logits * np.log(logits/y_eps))
#                     # e_to_use = e_to_use / e_to_use_max
#                     # hide_prob = - e_to_use + 1
#                     # hide_prob = (e/ce) / e_ce_max
#                     # x, y, winW, winH = window_params
#                     # sub_m = mask_prob[y: y + winH, x: x + winW]
#                     # hide_prob_scaled = hide_prob * self.scale_scores[step - 1] * self.global_hide_prob_scale
#                     # sub_m[sub_m < hide_prob_scaled] = hide_prob_scaled
#                     # mask_prob[y: y + winH, x: x + winW] = sub_m
#
#
#                     # if target_class_score > 0.5:
#
#                     x, y, winW, winH = window_params
#
#                     # sub_certainty_map = certainty_map[y: y + winH, x: x + winW]
#                     # # sub_certainty_map_to_update = np.logical_and(((1/e) * target_class_score) > sub_certainty_map, sub_m_to_update)
#                     # sub_certainty_map_to_update = ((1 / e) * target_class_score) > sub_certainty_map
#                     # sub_certainty_map[sub_certainty_map_to_update] = (1/e) * target_class_score
#                     # certainty_map[y: y + winH, x: x + winW] = sub_certainty_map
#
#                     sub_m = mask_prob[y: y + winH, x: x + winW]
#                     hide_prob_scaled = target_class_score * self.scale_scores[step - 1]
#                     sub_m_to_update = sub_m < hide_prob_scaled
#                     # sub_m_to_update = np.logical_and(sub_m < hide_prob_scaled, sub_certainty_map_to_update)
#                     sub_m[sub_m_to_update] = hide_prob_scaled
#                     mask_prob[y: y + winH, x: x + winW] = sub_m
#
#
#             #     # print('\n\n\n')
#             # certainty_map /= certainty_map.max()
#             # certainty_map *= self.global_hide_prob_scale
#             # # return certainty_map
#
#             return mask_prob * self.global_hide_prob_scale
#
#         # if self.loc_det_method == 'entropy_rel':
#         #     # TODO: scale with target_class_score
#         #     # TODO: add more weight to the later steps (small scale), yes yes
#         #     mask_prob = np.zeros((224, 224))
#         #
#         #     all_window_history = self.window_history[1] + self.window_history[2] + self.window_history[3]
#         #     all_entropies = np.array([entropy for _, entropy, _ in all_window_history])
#         #     entropy_max = all_entropies.max()
#         #
#         #     e_x = np.exp(all_entropies - np.max(all_entropies))
#         #     all_entropies_softmax =e_x / e_x.sum()
#         #     all_entropies_softmax_hist = {}
#         #     for step in list(self.window_history.keys()):
#         #         all_entropies_softmax_hist[step] = []
#         #
#         #         for i in range(len(self.window_history[step])):
#         #             all_entropies_softmax_hist[step].append(all_entropies_softmax)
#         #
#         #     for step in list(self.window_history.keys()):
#         #         for window_params, entropy, target_class_score in enumerate(self.window_history[step]):
#         #             x, y, winW, winH = window_params
#         #             entropy_norm = entropy / entropy_max
#         #             hide_prob = - entropy_norm + 1
#         #
#         #             sub_m = mask_prob[y: y + winH, x: x + winW]
#         #             hide_prob_scaled = hide_prob * target_class_score * self.scale_scores[step - 1] * self.global_hide_prob_scale
#         #             sub_m[sub_m < hide_prob_scaled] = hide_prob_scaled
#         #             mask_prob[y: y + winH, x: x + winW] = sub_m
#         #     return mask_prob
#
#         elif self.loc_det_method == 'sw_el2n':
#             mask_prob = np.zeros((224, 224))
#
#             all_window_history = self.window_history[1] + self.window_history[2] + self.window_history[3]
#             all_el2n = np.array([el2n for _, _, el2n in all_window_history])
#             el2n_max = all_el2n.max()
#
#             for step in list(self.window_history.keys()):
#                 for window_params, logits, el2n in self.window_history[step]:
#                     x, y, winW, winH = window_params
#                     el2n_norm = el2n / el2n_max
#                     hide_prob = - el2n_norm + 1
#
#                     sub_m = mask_prob[y: y + winH, x: x + winW]
#                     hide_prob_scaled = hide_prob * self.scale_scores[step - 1] * self.global_hide_prob_scale
#                     sub_m[sub_m < hide_prob_scaled] = hide_prob_scaled
#                     mask_prob[y: y + winH, x: x + winW] = sub_m
#             return mask_prob
#
#     def _get_mask_prob(self, input_img, target_class):
#         mask_prob = np.zeros((224, 224)) + self.hide_prob_vs_level[0]
#         updated_mask_prob = np.ones((224, 224), dtype=bool) # all mask is updated with initial mask_prob
#
#         should_do_loc_det_image = self._should_do_loc_det_image(input_img, target_class)
#
#         # logits = self._infer(input_img)
#         # e = - np.sum(logits * np.log(logits))
#         # y = np.zeros(self.n_classes)
#         # y[target_class] = 1
#         # ce = - np.sum(y * np.log(logits))
#         # # print(f'All image. target_class_score {logits[target_class]}, e {e}, ce {ce}')
#         # # if logits[target_class] < 0.2:
#         # #     should_do_loc_det_image = False
#
#         if should_do_loc_det_image:
#             step = 1
#             for (winW, winH), STEP_SIZE in zip(self.window_sizes, self.step_sizes):
#                 updated_mask_prob_current = np.zeros((224, 224), dtype=bool)
#
#                 windows = []
#                 window_params_history = []
#                 for (x, y, window) in sliding_window(input_img, stepSize=STEP_SIZE, windowSize=(winW, winH)):
#                     if window.shape[2] != winH or window.shape[3] != winW:
#                         continue
#                     if (not np.any(updated_mask_prob[y : y+winH, x : x+winW])) and (not self.offline_method):
#                         # if mask was not updated last iteration - skip further analysis of this region of the image
#                         # print('SKIIIIIP')
#                         continue
#
#                     window = np.stack([cv2.resize(frm, dsize=self.input_size) for frm in window[0]], axis=0)
#                     window = window[np.newaxis,...]
#                     windows.append(window)
#
#                     window_params = x, y, winW, winH
#                     window_params_history.append(window_params)
#
#                 if len(windows) > 0:
#                     window_vs_logit = self._infer(np.concatenate(windows)) # n_windows x 20
#                     for i in range(window_vs_logit.shape[0]):
#                         logits = window_vs_logit[i]
#                         window_params = window_params_history[i]
#                         self._update_mask_prob(mask_prob, updated_mask_prob_current, step, window_params, logits,
#                                                target_class)
#
#                 step += 1
#                 updated_mask_prob = updated_mask_prob_current
#
#         # offline hide_prob_mask update methods
#         if self.loc_det_method in ['entropy_rel', 'sw_el2n', 'target_class_score', 'target_class_score_nms']:
#             if not should_do_loc_det_image:
#                 return mask_prob
#             mask_prob = self._update_mask_prob_offline()
#
#         return mask_prob
#
#     # def _get_mask_prob(self, input_img, target_class):
#     #     # SW
#     #
#     #     # window_sizes, step_sizes, init_prob, hide_probabilities, n_windows, grid_size, detect_prob_threshs = self.method_params
#     #     # self.grid_size = grid_size
#     #
#     #     window_sizes = [(224 // 2, 224 // 2), (224 // 4, 224 // 4), (26, 26)]
#     #     step_sizes = [56, 42, 22]
#     #     hide_probabilities = [0.3, 0.5, 0.8]
#     #     init_prob = 0.2
#     #     detect_prob_thresh = 0.4 # for bbox to keep
#     #
#     #     mask_prob = np.zeros((224, 224)) + self.hide_prob_vs_level[0]
#     #     updated_mask_prob = np.ones((224, 224), dtype=bool)  # all mask is updated with initial mask_prob
#     #     prev_prob_range = [0, init_prob]
#     #     step = 1
#     #     for (winW, winH), STEP_SIZE, hide_prob in zip(window_sizes, step_sizes, hide_probabilities):
#     #         # print('Update mask prob', i)
#     #         updated_mask_prob_current = np.zeros((224, 224), dtype=bool)
#     #         for (x, y, window) in sliding_window(input_img, stepSize=STEP_SIZE, windowSize=(winW, winH)):
#     #             if window.shape[2] != winH or window.shape[3] != winW:
#     #                 continue
#     #
#     #             # sub_m = mask_prob[y : y+winH, x : x+winW]
#     #             # if not np.any(np.logical_and(sub_m > prev_prob_range[0], sub_m <= prev_prob_range[1])):
#     #             #     continue
#     #             if not np.any(updated_mask_prob[y : y+winH, x : x+winW]):
#     #                 # if mask was not updated last iteration - skip further analysis of this region of the image
#     #                 continue
#     #
#     #
#     #             window = np.stack([cv2.resize(frm, dsize=self.input_size) for frm in window[0]], axis=0)
#     #             window = window[np.newaxis,...]
#     #             logits = self._infer(window)
#     #             target_class_score = logits[target_class]
#     #
#     #             if target_class_score > detect_prob_thresh:
#     #                 sub_m = mask_prob[y : y+winH, x : x+winW]
#     #                 sub_m[sub_m < (target_class_score * hide_prob)] = target_class_score * hide_prob
#     #                 mask_prob[y : y+winH, x : x+winW] = sub_m
#     #                 updated_mask_prob_current[y: y + winH, x: x + winW] = True
#     #
#     #         step += 1
#     #         updated_mask_prob = updated_mask_prob_current
#     #
#     #     return mask_prob
#
#     def _get_hide_prob_mask_to_compare_with(self, mask):
#         if self.loc_det_method == 'entropy':
#             # return 0.5 / mask.max() * mask
#             return mask
#         elif self.loc_det_method == 'cross_entropy':
#             return 0.5 / mask.max() * mask
#         elif self.loc_det_method == 'sw':
#             return mask
#         elif self.loc_det_method == 'sw_entropy':
#             # return 0.5 / mask.max() * mask
#             return mask
#         elif self.loc_det_method in ['entropy_abs', 'entropy_rel', 'sw_el2n', 'target_class_score', 'target_class_score_nms', 'rise']:
#             mask_prob_base = np.zeros(self.grid_size) + 0.5  # RISE
#             mask_to_update_with = mask - mask.mean() # zero mean
#             return mask_prob_base + mask_to_update_with
#         else:
#             raise NotImplemented
#
#     def _generate_mask(self, mask, image_size) -> np.ndarray:
#         # image_w, image_h = image_size
#         # grid_w, grid_h = self.grid_size
#         # cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
#         # up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h
#         #
#         # mask = mask.astype(np.float32)
#         # mask = cv2.resize(mask, self.grid_size, interpolation=cv2.INTER_LINEAR)
#         # mask = self._get_hide_prob_mask_to_compare_with(mask)
#         # mask = (np.random.uniform(0, 1, size=self.grid_size) < mask).astype(np.float32)
#         #
#         # # TODO: add more granularity to the confident regions, and less to the less confident layers
#         #
#         # mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
#         # offset_w = np.random.randint(0, cell_w)
#         # offset_h = np.random.randint(0, cell_h)
#         # mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
#         # mask = np.transpose(np.dstack([mask] * 3), (2, 0, 1))
#         # return mask
#
#
#         nsd = 2
#         mask_bs = 1
#         dev = 'cpu'
#         grid_w, grid_h = self.grid_size
#
#         num_cells = grid_w
#
#         # Spatial size of low-res grid cell.
#         cell_size = tuple([int(np.ceil(s / num_cells))
#                            for s in self.input_size])
#
#         up_size = tuple([self.input_size[i] + cell_size[i]
#                          for i in range(nsd)])
#
#         grid = (torch.rand(mask_bs, 1, grid_h, grid_w, device=dev) < 0.5).float()
#
#         masks_up = _upsample_reflect(grid, up_size)
#
#
#
#         # Save final RISE masks with random shift.
#         masks = torch.empty(mask_bs, 1, *self.input_size, device=dev)
#         shift_x = torch.randint(0,
#                                 cell_size[0],
#                                 (mask_bs,),
#                                 device='cpu')
#         shift_y = torch.randint(0,
#                                 cell_size[1],
#                                 (mask_bs,),
#                                 device='cpu')
#         for i in range(mask_bs):
#             masks[i] = masks_up[i,
#                        :,
#                        shift_x[i]:shift_x[i] + self.input_size[0],
#                        shift_y[i]:shift_y[i] + self.input_size[1]]
#
#         masks = masks.squeeze().numpy()
#         return masks
#
#
#
#     def _get_saliency_map(self, mask_prob, frame, target_class):
#         saliency = np.zeros(self.input_size)
#         num_chunks = (self.n_masks + self.batch_size - 1) // self.batch_size
#         for chunk in range(num_chunks):
#             n_masks_in_chank = min(self.n_masks - self.batch_size * chunk, self.batch_size)
#
#             mask_batch = []
#             masked_batch = []
#             for _ in range(n_masks_in_chank):
#                 # mask = self._generate_mask_prob_from_vertices(image_size=self.input_size) # 224 x 224
#                 mask = self._generate_mask(mask_prob, image_size=self.input_size)  # 224 x 224
#
#                 # mask_ = np.zeros(self.grid_size, dtype=np.float32)
#                 # for v, score in self.vertices.items():
#                 #     for r in range(v[2]):
#                 #         for c in range(v[2]):
#                 #             mask_[v[0] + r][v[1] + c] = score
#                 # mask_ = cv2.resize(mask_, self.input_size, interpolation=cv2.INTER_LINEAR)
#                 #
#                 # mask__ = cv2.resize(mask_prob, self.grid_size, interpolation=cv2.INTER_LINEAR)
#                 # mask__ = cv2.resize(mask__, (224, 224), interpolation=cv2.INTER_LINEAR)
#
#                 masked = self._mask_image(frame, mask)
#                 mask_batch.append(mask)
#                 masked_batch.append(masked)
#
#             # mask_batch = np.array(mask_batch)[:, 0, :, :] # 100 x 224 x 224
#             mask_batch = np.array(mask_batch)  # 100 x 224 x 224
#             masked_batch = np.concatenate(masked_batch) # 100 x 3 x 224 x 224
#             logits = self._infer(masked_batch) # 100 x 20
#
#             if logits.ndim == 1:
#                 logits = logits[np.newaxis, ...]
#
#             scores = logits[:, target_class][..., np.newaxis, np.newaxis] # 100 x 1 x 1
#             saliency_map = np.sum((mask_batch * scores), axis=0)
#             saliency += saliency_map
#
#         saliency /= self.n_masks
#
#         return saliency
#
#     def generate_sailancy_map(self,
#                               frame: np.ndarray,
#                               target_class: int) -> np.ndarray:
#         # # self._generate_vertices_with_probs(frame, target_class)
#         # mask_prob = self._get_mask_prob(frame, target_class)
#         #
#         # # # test
#         # # mask_prob = np.zeros((224, 224))
#         #
#         # if self.test_hide_prob_mask_only:
#         #     return mask_prob
#         #
#         # saliency = self._get_saliency_map(mask_prob, frame, target_class)
#
#
#
#
#
#         # RISE-based localization
#         num_steps = 4
#         masks_per_step = 50
#         self.n_masks = masks_per_step
#
#         global_hide_prob_scale = 0.3
#         scales = np.linspace(0, global_hide_prob_scale, num_steps) # linearly spaced scales of hide prob
#
#         saliency_to_return = np.zeros(self.input_size)
#         for i in range(num_steps):
#             print(f'\nIteration {i}')
#             if i == 0:
#                 print('Start with 0-prob mask')
#                 mask_prob = np.zeros((224, 224))# + 0.5
#             else:
#                 print(f'Process saliency from the prev iteration with scale {scales[i]} ...')
#                 saliency /= saliency.max()  # Normalize to 0..1
#                 saliency *=  scales[i] # Normalize to 0..scale
#                 mask_prob = saliency
#
#             saliency = self._get_saliency_map(mask_prob, frame, target_class)
#             saliency_to_return += saliency
#             print('Saliency map generated')
#             if i < (num_steps - 1):
#                 print('Use saliency map (normalized 0..1) as a hide prob map for the next iteration')
#             elif i == num_steps - 1:
#                 print('Return final saliency map')
#                 return saliency_to_return / num_steps
#
#
#
#
#
#         return saliency
#
#     # def _generate_vertices_with_probs(self, frame, target_class):
#     #     det_infer_counter = 1
#     #     logits = self._infer(frame)
#     #     target_class_score = logits[target_class]
#     #
#     #     # if np.argmax(logits) != target_class:
#     #     #     self.e_threshold = [- np.log(item/2.5) for item in self.predict_prob]
#     #
#     #     entropy = self._calculate_entropy(logits, target_class)
#     #     # print("\nInit image infer: \ntarget_class_score:", target_class_score, "\nentropy", entropy, "\nnp.argmax(logits)", np.argmax(logits), "\n")
#     #
#     #     trial = 0
#     #     self.vertices[(0, 0, self.grid_size[0])] = self.hide_prob_vs_level[trial]
#     #     if entropy < self.e_threshold[trial]:
#     #         dropped_v = {}
#     #         for _ in range(3):
#     #             trial += 1
#     #             # print("######################## {} #######################".format(trial))
#     #             next_v = {}
#     #             for idx, (test_v, test_score) in enumerate(self.vertices.items()):
#     #                 v_size = test_v[2] // 2
#     #                 split_test_v = [[test_v[0], test_v[1], v_size], \
#     #                                 [test_v[0] + v_size, test_v[1], v_size], \
#     #                                 [test_v[0], test_v[1] + v_size, v_size], \
#     #                                 [test_v[0] + v_size, test_v[1] + v_size, v_size]]
#     #
#     #                 temp_dropped_v = []
#     #                 temp_next_v = []
#     #                 for idxx, v in enumerate(split_test_v):
#     #                     cropped_frame = frame[:, :, v[0] * 14:(v[0] + v[2]) * 14, v[1] * 14:(v[1] + v[2]) * 14]
#     #                     cropped_frame = np.stack([cv2.resize(frm, dsize=self.input_size) for frm in cropped_frame[0]],
#     #                                              axis=0)
#     #                     cropped_frame = cropped_frame[np.newaxis, ...]
#     #
#     #                     logits = self._infer(cropped_frame)
#     #                     det_infer_counter += 1
#     #                     target_class_score = logits[target_class]
#     #                     entropy = self._calculate_entropy(logits, target_class)
#     #                     # print('target_class_score', trial, idx, idxx, v, target_class_score, entropy, np.argmax(logits))
#     #
#     #                     if entropy < self.e_threshold[trial]:
#     #                         temp_next_v.append((v, target_class_score * self.hide_prob_vs_level[trial + 1]))
#     #                     else:
#     #                         temp_dropped_v.append((v, target_class_score * self.hide_prob_vs_level[trial]))
#     #                 if not temp_next_v:
#     #                     dropped_v[test_v] = test_score
#     #                 else:
#     #                     for v in temp_next_v:
#     #                         next_v[tuple(v[0])] = v[1]
#     #                     for v in temp_dropped_v:
#     #                         dropped_v[tuple(v[0])] = v[1]
#     #                 # print('next', next_v, "dropped", dropped_v)
#     #             if next_v is None:
#     #                 break
#     #             else:
#     #                 self.vertices = next_v
#     #         self.vertices.update(dropped_v)
#     #         print('vertices:\n', self.vertices)
#     #     else:
#     #         print('Whole image not certain enough for class', target_class)
#     #     print('det_infer_counter', det_infer_counter)
#     #
#     # def _generate_mask_prob(self, image_size: Tuple[int,int]) -> np.ndarray:
#     #     image_w, image_h = image_size
#     #     grid_w, grid_h = self.grid_size
#     #     cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
#     #     up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h
#     #
#     #     norm_prop = []
#     #     for v in self.vertices.values():
#     #         norm_prop.append(v)
#     #
#     #     # mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) < 0.5).astype(np.float32)
#     #     mask = np.zeros(self.grid_size, dtype=np.float32)
#     #     for v, score in self.vertices.items():
#     #         # temp = (np.random.uniform(0, 1, size=(v[2], v[2])) < 0.5 / max(norm_prop) * score).astype(np.float32)
#     #         # temp = (np.random.uniform(0, 1, size=(v[2], v[2])) < score).astype(np.float32)
#     #         for r in range(v[2]):
#     #             for c in range(v[2]):
#     #                 mask[v[0]+r][v[1]+c] = score
#     #
#     #     mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
#     #     return mask
#     #
#     # def generate_sailancy_map(
#     #     self,
#     #     frame: np.ndarray,
#     #     target_class: int,
#     # ) -> np.ndarray:
#     #     self._generate_vertices_with_probs(frame, target_class)
#     #
#     #     mask = self._generate_mask_prob(image_size=self.input_size) # 224 x 224
#     #
#     #     return mask
