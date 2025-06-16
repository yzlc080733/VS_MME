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
import torch
import colorsys



'''
    ENV DATASET
    ASD
        Requested from Wang et al., see reference
        "Autism Spectrum Disorder, but Not Amygdala Lesions, Impairs Social Attention in Visual Search"
    COCOSearch18
        https://sites.google.com/view/cocosearch/
        /data/COCOSearch18_dataset/images
    IVSN
        https://github.com/kreimanlab/VisualSearchZeroShot
'''

print('--> DATASET ENVIRONMENT')


# FUNCTIONS
def search_cue_preprocess(image_input, s1=96, s2=128):
    SIZE_1, SIZE_2 = s1, s2
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



def get_wang_dataset(dataset_variant=['All', 'both']):
    print('--> GET wang DATASET')
    root_folder = './env_data/'
    with open(os.path.join(root_folder, 'IMG_ROI.pkl'), 'rb') as infile:
        IMG_ROI_LIST = pickle.load(infile)
    with open(os.path.join(root_folder, 'TRAJECTORY_DATA.pkl'), 'rb') as infile:
        TRAJECTORY_LIST_ALL = pickle.load(infile)
    # EXPAND TRAJECTORY DATA
    print('--> Expand trajectory data')    
    for subj_data in TRAJECTORY_LIST_ALL:
        for traj_data in subj_data['traj_list']:
            img_correspond_index = traj_data['image_id_python'] + 1
            x1, x2 = traj_data['target_roi_X']
            y1, y2 = traj_data['target_roi_Y']
            image_save = np.copy(IMG_ROI_LIST[img_correspond_index - 1]['image'])
            target_roi_image = image_save[y1:y2, x1:x2]
            target_roi_image = search_cue_preprocess(target_roi_image, s1=96, s2=128)
            traj_data['target_image'] = target_roi_image
    # DATASET PLACEHOLDER
    img_cue = []
    img_search = []
    traj = []
    universal_id = []
    target_bbox = []
    # DATASET VARIANT -- SUBJECT
    if dataset_variant[0] == 'All':
        subj_index_list = list(range(50))
    elif dataset_variant[0] in ['Epilepsy', 'Amygdala', 'ASD', 'ASD_Ctrl', 'NUS', 'Stroke']:
        subj_index_list = []
        for subj_i in range(50):
            if TRAJECTORY_LIST_ALL[subj_i]['subj_type_str'] == dataset_variant[0]:
                subj_index_list.append(subj_i)
    elif dataset_variant[0] in range(1, 51):
        subj_index_list = [dataset_variant[0] - 1]
    else:
        raise Exception('/// Error in dataaset_name[0]')
    # DATASET VARIANT -- TRAJECTORY (TARGET TYPE)
    for subj_i in subj_index_list:
        subj_data = TRAJECTORY_LIST_ALL[subj_i]
        for traj_data in subj_data['traj_list']:
            if dataset_variant[1] == 'both':
                pass
            elif dataset_variant[1] in ['social', 'nonsocial']:
                if traj_data['target_type_str'] != dataset_variant[1]:
                    continue
            else:
                raise Exception('/// Error in dataaset_name[1]')
            
            img_cue.append(traj_data['target_image'])
            img_search.append(IMG_ROI_LIST[traj_data['image_id_python']]['image'])
            traj.append(np.array([traj_data['X'], traj_data['Y']]))
            universal_id.append(traj_data['traj_universal_id'])
            x1, x2 = traj_data['target_roi_X'];  y1, y2 = traj_data['target_roi_Y']
            target_bbox.append([x1, x2, y1, y2])
    print(len(traj))
    return img_search, img_cue, traj, universal_id, target_bbox


def get_cocosearch18_dataset(dataset_variant=[1]):
    root_folder = './env_data/'
    with open(os.path.join(root_folder, 'COCOSearch18.pkl'), 'rb') as infile:
        [SEARCH_IMAGE_DICT_ALL, TARGET_IMAGE_DICT_ALL, SCANPATH_LIST_ALL] = pickle.load(infile)
    if dataset_variant[0] == 'all':
        temp_list = []
        for traj_i in range(len(SCANPATH_LIST_ALL)):
            temp_traj_data = SCANPATH_LIST_ALL[traj_i]
            temp_list.append(temp_traj_data)
        SCANPATH_LIST_ALL = temp_list
    else:           # SELECT ONE SUBJECT
        selected_subject_id = dataset_variant[0]
        assert selected_subject_id in range(1, 11)
        temp_list = []
        for traj_i in range(len(SCANPATH_LIST_ALL)):
            temp_traj_data = SCANPATH_LIST_ALL[traj_i]
            if temp_traj_data['subject_id'] == selected_subject_id:
                temp_list.append(temp_traj_data)
        SCANPATH_LIST_ALL = temp_list
    
    
    traj_number = len(SCANPATH_LIST_ALL)

    img_search = []
    img_cue = []
    traj = []
    universal_id = []
    target_bbox = []
    for traj_i in range(traj_number):
        temp_traj_data = SCANPATH_LIST_ALL[traj_i]
        # SEARCH IMAGE
        img_search.append(SEARCH_IMAGE_DICT_ALL[temp_traj_data['search_image_name']])
        # TARGET IMAGE -- NOTE CROPPED FROM SEARCH IMAGE -- NOT CATEGORY REPRESENTATIVE IMAGE FROM VISIONS PAPER
        x1, x2 = temp_traj_data['target_bbox_X']
        y1, y2 = temp_traj_data['target_bbox_Y']
        image_search_temp = np.copy(SEARCH_IMAGE_DICT_ALL[temp_traj_data['search_image_name']])
        target_roi_image = image_search_temp[y1:y2, x1:x2]
        target_roi_image = search_cue_preprocess(target_roi_image, s1=80, s2=128)
        img_cue.append(target_roi_image)
        # SCANPATH DATA
        traj.append(np.array([temp_traj_data['X'], temp_traj_data['Y']]))
        # UNIVERSAL ID
        universal_id.append(temp_traj_data['traj_universal_id'])
        # TARGET BBOX INFO
        x1, x2 = temp_traj_data['target_bbox_X'];  y1, y2 = temp_traj_data['target_bbox_Y']
        target_bbox.append([x1, x2, y1, y2])
    return img_search, img_cue, traj, universal_id, target_bbox


def get_ivsnarray_dataset(dataset_variant=[0]):
    root_folder = './env_data/'
    with open(os.path.join(root_folder, 'IVSNarray.pkl'), 'rb') as infile:
        [SEARCH_IMAGE_LIST, TARGET_IMAGE_LIST, TARGET_BBOX_LIST, TRAJECTORIES_ALL] = pickle.load(infile)
    if dataset_variant[0] == 'all':
        selected_subj_id_list = list(range(16))
    else:           # SELECT ONE SUBJECT
        assert dataset_variant[0] in range(0, 16)
        selected_subj_id_list = [dataset_variant[0]]
    
    img_search = []
    img_cue = []
    traj = []
    universal_id = []
    target_bbox = []
    for subj_i in range(0, 16):
        if subj_i in selected_subj_id_list:
            subject_data = TRAJECTORIES_ALL[subj_i]
            traj_number = len(subject_data['traj_list'])
            for traj_i in range(traj_number):
                traj_dict = subject_data['traj_list'][traj_i]
                temp_id = traj_dict['traj_id']
                img_search.append(np.copy(SEARCH_IMAGE_LIST[temp_id]))
                img_cue.append(np.copy(TARGET_IMAGE_LIST[temp_id]))
                traj.append(np.int32(np.array([traj_dict['traj_X'], traj_dict['traj_Y']])))
                universal_id.append(traj_dict['universal_id'])
                target_bbox.append(TARGET_BBOX_LIST[temp_id])
    
    return img_search, img_cue, traj, universal_id, target_bbox



class Dataset:
    def __init__(self, dataset_name='asd', dataset_variant=[None], device=None, split_valid=True, split_seed=0) -> None:
        self.dataset_name = dataset_name
        self.dataset_variant = dataset_variant
        # INITIALIZE DATA
        if self.dataset_name == 'asd':
            self.img_search, self.img_cue, self.traj, self.universal_id, self.target_bbox = get_wang_dataset(self.dataset_variant)
            self.img_size_X, self.img_size_Y, self.img_channel_num = 1024, 768, 3
            self.observation_radius = 105
            print('DATA_NUM=', len(self.img_search), 'IMG_SHAPE=', self.img_search[0].shape)
        elif self.dataset_name == 'cocosearch18':
            self.img_search, self.img_cue, self.traj, self.universal_id, self.target_bbox = get_cocosearch18_dataset(self.dataset_variant)
            self.img_size_X, self.img_size_Y, self.img_channel_num = 512, 320, 3
            self.observation_radius = 50
            print('DATA_NUM=', len(self.img_search), 'IMG_SHAPE=', self.img_search[0].shape)
        elif self.dataset_name == 'ivsnarray':
            self.img_search, self.img_cue, self.traj, self.universal_id, self.target_bbox = get_ivsnarray_dataset(self.dataset_variant)
            self.img_size_X, self.img_size_Y, self.img_channel_num = 676, 756, 3
            self.observation_radius = 85
            print('DATA_NUM=', len(self.img_search), 'IMG_SHAPE=', self.img_search[0].shape)
        else:
            raise Exception('/// Error in dataset_name')
        # TRAIN -- VALID INDEX
        self.data_num_total = len(self.img_search)
        index_list_all = list(range(self.data_num_total))
        if split_valid == True:
            # RANDOM SHUFFLING
            if self.data_num_total <= 10:
                raise Exception('/// WARNING: TOO FEW DATA FOR VALID SPLIT')
            random_instance = random.Random(0)
            random_instance.shuffle(index_list_all)
            index_list_train = index_list_all[:int(0.9 * self.data_num_total)]
            index_list_valid = index_list_all[int(0.9 * self.data_num_total):]
        else:
            index_list_train = index_list_all
            index_list_valid = index_list_all
        # TRAIN -- VALID DATA
        # TRAIN DATA
        self.train_img_search, self.train_img_cue, self.train_traj, self.train_universal_id, self.train_target_bbox = [], [], [], [], []
        for d_i in range(len(index_list_train)):
            self.train_img_search.append(self.img_search[index_list_train[d_i]])
            self.train_img_cue.append(self.img_cue[index_list_train[d_i]])
            self.train_traj.append(self.traj[index_list_train[d_i]])
            self.train_universal_id.append(self.universal_id[index_list_train[d_i]])
            self.train_target_bbox.append(self.target_bbox[index_list_train[d_i]])
        # VALIDATION DATA
        if split_valid == True:
            self.valid_img_search, self.valid_img_cue, self.valid_traj, self.valid_universal_id, self.valid_target_bbox = [], [], [], [], []
            for d_i in range(len(index_list_valid)):
                self.valid_img_search.append(self.img_search[index_list_valid[d_i]])
                self.valid_img_cue.append(self.img_cue[index_list_valid[d_i]])
                self.valid_traj.append(self.traj[index_list_valid[d_i]])
                self.valid_universal_id.append(self.universal_id[index_list_valid[d_i]])
                self.valid_target_bbox.append(self.target_bbox[index_list_valid[d_i]])
        else:
            self.valid_img_search = self.train_img_search
            self.valid_img_cue = self.train_img_cue
            self.valid_traj = self.train_traj
            self.valid_universal_id = self.train_universal_id
            self.valid_target_bbox = self.train_target_bbox
        del self.img_search, self.img_cue, self.traj, self.universal_id, self.target_bbox
        self.train_num = len(self.train_img_search)
        self.valid_num = len(self.valid_img_search)


