# -*- coding: utf-8 -*-
import os
import time
import random
import math
import scipy.io
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils
from tqdm import tqdm
import json
import shutil
import argparse
from tqdm import tqdm



'''
    PREPROCESS COCOSEARCH18 DATASET FOR USE IN TRAINING
    TARGET IMAGES BASED ON
        https://github.com/NeuroLIAA/visions
    DATASET DOWNLOADED FROM
        https://sites.google.com/view/cocosearch/
    ORIGINAL README FILE OF THE DATASET
        https://drive.google.com/file/d/1WKnswfq8dVvyEpKQcQrf-d6dHljOKA--/view
'''



print('--> PREPARE COCOSEARCH18 DATASET')



# DATA SOURCE PATH
print('LOAD DATA')
root_folder = './env_data/'
path_fixation_data = '/data/COCOSearch18_dataset/fixations/coco_search18_fixations_TP_train_split1.json'
path_search_images = '/data/COCOSearch18_dataset/images'
path_target_images = '/data/COCOSearch18_dataset/target_image_visions'




# LOAD TRAJECTORY INFO ORIGINAL
TRAJECTORY_DATA_LIST = []
with open(path_fixation_data, 'r') as f:
    data_org = json.load(f)
trajectory_num = len(data_org)



# SHOW SOME INFO
subject_id_dict = {}
for traj_i in range(len(data_org)):
    temp_info = data_org[traj_i]
    temp_subject = temp_info['subject']
    if temp_subject in subject_id_dict.keys():
        subject_id_dict[temp_subject] += 1
    else:
        subject_id_dict[temp_subject] = 1
print('ALL SUBJECTS')
print(subject_id_dict)



# SAMPLE SCANPATHS
selected_scanpath_index_all = []
for subj_id in range(1, 11):
    candidate_index_list = []
    for traj_i in range(len(data_org)):
        if len(data_org[traj_i]['X']) < 2:
            continue
        if data_org[traj_i]['subject'] == subj_id:
            candidate_index_list.append(traj_i)
    random_instance = random.Random(123)
    random_instance.shuffle(candidate_index_list)
    index_list = candidate_index_list[:80]
    selected_scanpath_index_all += index_list



# =============================================
# SEARCH IMAGE
SEARCH_IMAGE_DICT_ALL = {}
for traj_i in tqdm(selected_scanpath_index_all, desc='SEARCH IMG', leave=False):
    traj_data_org = data_org[traj_i]
    search_image_name = traj_data_org['name']
    task_string = traj_data_org['task']
    assert traj_data_org['condition'] == 'present'
    if search_image_name in SEARCH_IMAGE_DICT_ALL.keys():   # DUPLICATE
        continue
    search_image_filepath = os.path.join(path_search_images, task_string, search_image_name)
    load_img = mpimg.imread(search_image_filepath)
    resize_img = cv2.resize(load_img, [512, 320], cv2.INTER_LINEAR)
    SEARCH_IMAGE_DICT_ALL[search_image_name] = resize_img
print('TOTAL SEARCH IMG NUM', len(SEARCH_IMAGE_DICT_ALL.keys()))



# TARGET IMAGE
TARGET_IMAGE_DICT_ALL = {}
path_target_images
for traj_i in tqdm(selected_scanpath_index_all, desc='TARGET IMG', leave=False):
    traj_data_org = data_org[traj_i]
    task_string = traj_data_org['task']
    if task_string in TARGET_IMAGE_DICT_ALL.keys():         # DUPLICATE
        continue
    target_image_filepath = os.path.join(path_target_images, task_string + '.jpg')
    load_img = mpimg.imread(target_image_filepath)
    TARGET_IMAGE_DICT_ALL[task_string] = load_img
print('TOTAL TARGET IMG NUM', len(TARGET_IMAGE_DICT_ALL.keys()))



# SCANPATH DATA
TASK_NAME_FULL_LIST = ['bottle', 'chair', 'fork', 'laptop', 'oven', 'stop sign', 'bowl', 'clock', 'keyboard', 'microwave', 'potted plant', 'toilet', 'car', 'cup', 'knife', 'mouse', 'sink', 'tv']
def get_target_index(task_name):
    assert task_name in TASK_NAME_FULL_LIST
    task_index = TASK_NAME_FULL_LIST.index(task_name)
    return task_index
SCANPATH_LIST_ALL = []
traj_universal_counter = 0
for traj_i in tqdm(selected_scanpath_index_all, desc='SCANPATH', leave=False):
    traj_data_org = data_org[traj_i]
    x_resize = [min(max(math.floor(temp_item / 1680 * 512), 0), 511) for temp_item in traj_data_org['X']]
    y_resize = [min(max(math.floor(temp_item / 1050 * 320), 0), 319) for temp_item in traj_data_org['Y']]
    if len(x_resize) < 2:       # DISCARD SCANPATHS THAT ARE TOO SHORT
        continue
    [bbox_x, bbox_y, bbox_w, bbox_h] = traj_data_org['bbox']
    bbox_x1 = bbox_x
    bbox_x2 = bbox_x + bbox_w
    bbox_y1 = bbox_y
    bbox_y2 = bbox_y + bbox_h
    bbox_x1_resize = math.floor(bbox_x1 / 1680 * 512)
    bbox_x2_resize = math.floor(bbox_x2 / 1680 * 512)
    bbox_y1_resize = math.floor(bbox_y1 / 1050 * 320)
    bbox_y2_resize = math.floor(bbox_y2 / 1050 * 320)
    task_index = get_target_index(traj_data_org['task'])
    # SORT RESULT
    traj_data_transformed = {}
    traj_data_transformed['search_image_name'] = traj_data_org['name']
    traj_data_transformed['initial_fixation_X'] = int(512 * 0.5)
    traj_data_transformed['initial_fixation_Y'] = int(320 * 0.5)
    traj_data_transformed['X'] = x_resize
    traj_data_transformed['Y'] = y_resize
    traj_data_transformed['target_bbox_X'] = [bbox_x1_resize, bbox_x2_resize]
    traj_data_transformed['target_bbox_Y'] = [bbox_y1_resize, bbox_y2_resize]
    traj_data_transformed['target_type_string'] = traj_data_org['task']
    traj_data_transformed['target_type_index'] = task_index
    traj_data_transformed['traj_universal_id'] = traj_universal_counter
    traj_data_transformed['subject_id'] = traj_data_org['subject']
    traj_universal_counter += 1
    SCANPATH_LIST_ALL.append(traj_data_transformed)



with open(os.path.join(root_folder, 'COCOSearch18.pkl'), 'wb') as outfile:
    pickle.dump([
            SEARCH_IMAGE_DICT_ALL,
            TARGET_IMAGE_DICT_ALL,
            SCANPATH_LIST_ALL,
        ], outfile)


print('FINISHED')

