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

'''
    PREPROCESS ASD DATASET FOR USE IN TRAINING
'''

print('--> PREPARE ASD DATASET')



# DATA SOURCE PATH
print('LOAD DATA')
root_folder = './env_data/'
if not os.path.exists(root_folder):
    utils.print_info('Error data not found')
    exit()
else:
    rootdata_path = root_folder + 'Documents/VS_Data_All.mat'
    image_path = root_folder + 'Tasks/Visual Search Array New/'
    imagelabel_path = root_folder + 'Tasks/ROI New/'
    roi_path = root_folder + 'Tasks/Visual Search ROI New/'
    # ROOT DATA
    root_data = scipy.io.loadmat(rootdata_path)



# IMAGE DATABASE
print('GET IMAGE DATA')
IMG_number = root_data['behAll'][0, 1]['nImg'].item().item()        # IMG_number == 20
IMG_ROI_LIST = []
for img_i in range(IMG_number):
    img_data = {}
    # IMG
    img_filename = image_path + 'vs%d.tif' % (img_i + 1)
    load_img = mpimg.imread(img_filename)
    # ROI
    roi_filename = imagelabel_path + 'vs%d.mat' % (img_i + 1)
    temp_roi = scipy.io.loadmat(roi_filename)
    temp_roi = temp_roi['ROI']
    roi_number = temp_roi.shape[1]
    load_roi = np.zeros([roi_number, 5], dtype=np.float32)
    for line_i in range(roi_number):
        load_roi[line_i, 0:2] = temp_roi['x'][0, line_i][:, 0]
        load_roi[line_i, 2:4] = temp_roi['y'][0, line_i][:, 0]
        load_roi[line_i, 4] = temp_roi['category'][0, line_i].item()
    # STORE DATA
    img_data['image'] = load_img
    img_data['img_filename'] = 'vs%d.jpg' % (img_i + 1)
    img_data['roi_filename'] = 'vs%d.mat' % (img_i + 1)
    img_data['roi'] = load_roi
    img_data['roi_center'] = [np.mean(load_roi[:, 0:2], axis=1), np.mean(load_roi[:, 2:4], axis=1)]
    IMG_ROI_LIST.append(img_data)
with open(os.path.join(root_folder, 'IMG_ROI.pkl'), 'wb') as outfile:
    pickle.dump(IMG_ROI_LIST, outfile)


# FUNCTION FOR TRAJECTORY PROCESSOR
def search_cue_preprocess(image_input):
    SIZE_1, SIZE_2 = 96, 128
    # GRAY BACKGROUND
    output_image = np.ones([SIZE_1, SIZE_2, 3], dtype=np.uint8) * 128
    ratio_0 = SIZE_1 / image_input.shape[0]
    ratio_1 = SIZE_2 / image_input.shape[1]
    # RATIO FOR RESIZE
    if ratio_0 < ratio_1:
        re_size_0 = SIZE_1
        re_size_1 = math.floor(image_input.shape[1] * ratio_0)
        input_image_resize = cv2.resize(image_input, (re_size_1, re_size_0), interpolation=cv2.INTER_AREA)
    else:
        re_size_0 = math.floor(image_input.shape[0] * ratio_1)
        re_size_1 = SIZE_2
        input_image_resize = cv2.resize(image_input, (re_size_1, re_size_0), interpolation=cv2.INTER_AREA)
    # POSITION OF THE IMAGE
    pos0_1 = math.floor(SIZE_1*0.5 - input_image_resize.shape[0] / 2)
    pos0_2 = pos0_1 + input_image_resize.shape[0]
    pos1_1 = math.floor(SIZE_2*0.5 - input_image_resize.shape[1] / 2)
    pos1_2 = pos1_1 + input_image_resize.shape[1]
    output_image[pos0_1:pos0_2, pos1_1:pos1_2, :] = input_image_resize
    return output_image

def get_subj_type(id_matlab):
    assert id_matlab >= 1 and id_matlab <= 50
    if id_matlab in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        id_type = 'Epilepsy'
    elif id_matlab in [14, 15, 16]:
        id_type = 'Amygdala'
    elif id_matlab in [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
        id_type = 'ASD'
    elif id_matlab in [30, 31, 32, 33, 34, 35, 36, 37]:
        id_type = 'ASD_Ctrl'
    elif id_matlab in [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]:
        id_type = 'NUS'
    elif id_matlab in [49, 50]:
        id_type = 'Stroke'
    return id_type


# TRAJECTORIES
print('PROCESS TRAJECTORIES')
TRAJECTORY_LIST_ALL = []
traj_counter = 0
for session_i in range(1, 51):
    # PLACEHOLDER
    subj_data = {}
    subj_data['id_matlab'] = session_i
    subj_data['id_python'] = session_i - 1
    subj_data['subj_type_str'] = get_subj_type(session_i)
    subj_data['traj_list'] = []
    # LOAD DATA
    em_data = root_data['EM'][0, session_i - 1]
    trial_num = int(np.max(em_data[:, 8]))
    IMG_index_list = root_data['beh'][0, session_i - 1]['iT']['imgInd'][0]
    ROI_index_list = root_data['beh'][0, session_i - 1]['iT']['ROIInd'][0]
    trial_num = min(trial_num, IMG_index_list.shape[0])
    image_name_session_list = []
    # TRAJECTORY DATA
    for trial_i in range(1, trial_num + 1):     # MATLAB value, starting from 1
        # SESSION INFO
        traj_data = {}
        traj_data['__subject_id_matlab'] = '%02d' % session_i
        traj_data['__trial_id_matlab'] = trial_i
        traj_data['traj_universal_id'] = traj_counter
        traj_counter += 1
        # TRAJECTORY PIXEL INFO
        traj_X = em_data[em_data[:, 8] == trial_i, 1]
        traj_Y = em_data[em_data[:, 8] == trial_i, 2]
        traj_X = np.clip(np.round(traj_X), 0, 1023).astype(np.int32).tolist()
        traj_Y = np.clip(np.round(traj_Y), 0, 767).astype(np.int32).tolist()
        if len(traj_X) < 2:             # EMPTY TRAJECTORY
            continue
        traj_data['X'] = traj_X
        traj_data['Y'] = traj_Y
        traj_data['length'] = len(traj_X)
        traj_data['initial_fixation_X'] = int(1024 * 0.5)
        traj_data['initial_fixation_Y'] = int(768 * 0.5)
        # IMAGE + ROI
        img_correspond_index = IMG_index_list[trial_i - 1].item()
        roi_correspond_index = ROI_index_list[trial_i - 1].item()
        traj_data['image_id_python'] = img_correspond_index - 1
        # SEARCH CUE TYPE
        target_type = int(root_data['beh'][0, session_i - 1]['target'][0, trial_i - 1])
        suffix_list = ['_S.mat', '_NS.mat']
        target_type_str = ['none', 'social', 'nonsocial']
        traj_data['target_type'] = target_type
        traj_data['target_type_str'] = target_type_str[target_type]
        traj_data['target_object'] = traj_data['target_type_str']
        traj_data['task'] = traj_data['target_object']
        target_exist = target_type in [1, 2]
        if not target_exist:            # Target-absent IS NOT USED
            continue
        # TARGET FOUND INFO
        target_found_list = em_data[em_data[:, 8] == trial_i, 7]
        target_found_flag = sum(target_found_list) > 0
        traj_data['target_found'] = bool(target_found_flag)
        # SEARCH CUE ROI
        target_roi_filename_suffix = suffix_list[target_type - 1]
        target_roi_filename = roi_path + 'vs%d_%d' % (img_correspond_index, roi_correspond_index) + target_roi_filename_suffix
        target_roi = scipy.io.loadmat(target_roi_filename)
        target_roi_x = target_roi['x'][:, 0]
        target_roi_y = target_roi['y'][:, 0]
        # SEARCH CUE ID WITHIN IMAGE
        image_roi_list_x_center = IMG_ROI_LIST[img_correspond_index - 1]['roi_center'][0]
        image_roi_list_y_center = IMG_ROI_LIST[img_correspond_index - 1]['roi_center'][1]
        target_roi_x_center = (target_roi_x[0] + target_roi_x[1]) * 0.5
        target_roi_y_center = (target_roi_y[0] + target_roi_y[1]) * 0.5
        image_roi_distance = np.sqrt(np.square(image_roi_list_x_center - target_roi_x_center) + 
                                     np.square(image_roi_list_y_center - target_roi_y_center))
        min_distance = np.min(image_roi_distance)
        traj_data['target_idINimg'] = int(np.argmin(image_roi_distance))
        # SEARCH CUE ROI CONVERSION
        x1, x2, y1, y2 = IMG_ROI_LIST[img_correspond_index - 1]['roi'][traj_data['target_idINimg'], 0:4]
        x1 = math.floor(x1)
        x2 = math.floor(x2)
        y1 = math.floor(y1)
        y2 = math.floor(y2)
        # SEARCH CUE INFO
        traj_data['target_roi_X'] = [min(x1, x2), max(x1, x2)]
        traj_data['target_roi_Y'] = [min(y1, y2), max(y1, y2)]
        subj_data['traj_list'].append(traj_data)
    subj_data['traj_num'] = len(subj_data['traj_list'])
    # RECORD SUBJECT-LEVEL DATA
    TRAJECTORY_LIST_ALL.append(subj_data)
with open(os.path.join(root_folder, 'TRAJECTORY_DATA.pkl'), 'wb') as outfile:
    pickle.dump(TRAJECTORY_LIST_ALL, outfile)



print('FINISHED')

