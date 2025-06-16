import scipy
import cv2
import numpy as np
import math
import os
import shutil
import pickle
import colorsys
import random
from tqdm import tqdm


'''
PREPROCESS THE DATASET FROM
    https://github.com/kreimanlab/VisualSearchZeroShot
WHICH IS ASSOCIATED WITH PAPER
    "Finding Any Waldo with Zero-Shot Invariant and Efficient Visual Search" 
'''



# ================== DEFINITIONS ================
boxctrx = np.array([790, 490, 340, 490, 790, 940, ])            # VERTICAL
boxctry = 1000 - np.array([772, 772, 512, 252, 252, 512, ])     # HORIZONTAL
boxsize = 156
bbox_x1 = np.int32(boxctrx - boxsize * 0.5)
bbox_x2 = np.int32(boxctrx + boxsize * 0.5)
bbox_y1 = np.int32(boxctry - boxsize * 0.5)
bbox_y2 = np.int32(boxctry + boxsize * 0.5)
x_min = np.min(bbox_x1)
x_max = np.max(bbox_x2)
y_min = np.min(bbox_y1)
y_max = np.max(bbox_y2)
dataset_rootpath = '../reference_IVSN_org/draft_space/array'
random.seed(123)



# ================== LOAD DATA ==================
root_data = scipy.io.loadmat(dataset_rootpath + '/psy/array.mat')
root_data = root_data['MyData']

SEARCH_IMAGE_LIST = []
TARGET_IMAGE_LIST = []
TARGET_BBOX_LIST = []
for d_i in tqdm(range(root_data.shape[0]), desc='IMAGE'):
    base_i = d_i

    # TARGET BBOX
    base_data = root_data[base_i]
    base_target_cate = base_data[0][1][0][0]
    base_arraycate = base_data[0][3]
    target_position_id = base_arraycate[:, 0].tolist().index(base_target_cate)
    b_x1 = bbox_x1[target_position_id] - x_min
    b_x2 = bbox_x2[target_position_id] - x_min
    b_y1 = bbox_y1[target_position_id] - y_min
    b_y2 = bbox_y2[target_position_id] - y_min
    TARGET_BBOX_LIST.append([b_y1, b_y2, b_x1, b_x2])

    # SEARCH IMAGE
    img_search = cv2.imread(os.path.join(dataset_rootpath + '/stimuli/array_%d.jpg' % (d_i%300 + 1)))
    SEARCH_IMAGE_LIST.append(img_search)
    # TARGET IMAGE
    img_target = cv2.imread(os.path.join(dataset_rootpath + '/target/target_%d.jpg' % (d_i + 1)))
    TARGET_IMAGE_LIST.append(img_target)



# ================== FUNCIONS ===================
def clean_nan_values(raw_x, raw_y):
    dr_x = []
    dr_y = []
    for p_i in range(raw_x.shape[0]):
        if np.isnan(raw_x[p_i]) or np.isnan(raw_y[p_i]):
            # print('CLEAN nan: ', p_i, raw_x.shape[0])
            pass
        else:
            dr_x.append(raw_x[p_i])
            dr_y.append(raw_y[p_i])
    dr_x = np.array(dr_x)
    dr_y = np.array(dr_y)
    return dr_x, dr_y


def get_distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)



# ================== LOAD SCANPATH ==============
# THIS FILE CONTAINS seq -- SEQUENCE OF DATA USED IN HumanExp.m
seq_data = scipy.io.loadmat(os.path.join(dataset_rootpath + '/psy/array_seq.mat'))
seq_data = seq_data['seq']
data_sequence = (seq_data[0, :] - 1).tolist()
seq_sort = np.argsort(data_sequence)

TRAJECTORIES = []
universal_id = 0
subjectname_list = [
    'subj01-mm', 'subj02-el', 'subj03-yu', 'subj05-je',
    'subj07-pr', 'subj08-bo', 'subj09-az', 'subj10-oc',
    'subj11-lu', 'subj12-al', 'subj13-ni', 'subj14-ji',
    'subj15-ma', 'subj17-ga', 'subj18-an', 'subj19-ni',
]
for subj_i in range(len(subjectname_list)):
    subject_foldername = subjectname_list[subj_i]
    subject_data = {
        'subject_name': subject_foldername,
        'subject_id': subj_i,
        'traj_list': [],
    }
    # THIS FILE CONTAINS trialRecord
    trialRecord = scipy.io.loadmat(os.path.join(dataset_rootpath + '/psy/subjects_array/', subject_foldername, 'fix.mat'))
    trialRecord = trialRecord['trialRecord']
    data_scanpath = trialRecord[seq_sort, 0]
    data_number = data_scanpath.shape[0]
    print('SUBJ', subj_i, subject_foldername, data_number)
    

    # ================== CLUSTER POINTS =============
    DIST_THRESHOLD = 5

    for trial_i in range(data_number):
        raw_x = data_scanpath[trial_i][1][:, 0]
        raw_y = data_scanpath[trial_i][2][:, 0]
        raw_x, raw_y = clean_nan_values(raw_x, raw_y)
        point_number = raw_x.shape[0]
        
        cluster_x = []
        cluster_y = []
        rec_flag = False
        for p_i in range(1, point_number):
            dist_value = get_distance(raw_x[p_i], raw_y[p_i], raw_x[p_i-1], raw_y[p_i-1])
            if rec_flag == False:
                if dist_value < DIST_THRESHOLD:
                    rec_list_x = [raw_x[p_i-1], raw_x[p_i]]
                    rec_list_y = [raw_y[p_i-1], raw_y[p_i]]
                    rec_flag = True
                else:
                    continue
            else:
                if dist_value < DIST_THRESHOLD:
                    rec_list_x.append(raw_x[p_i])
                    rec_list_y.append(raw_y[p_i])
                else:
                    cluster_x.append(np.mean(np.array(rec_list_x)) - x_min)
                    cluster_y.append(np.mean(np.array(rec_list_y)) - y_min)
                    rec_flag = False
        if rec_flag == True:        # FINAL POINTS
            cluster_x.append(np.mean(np.array(rec_list_x)) - x_min)
            cluster_y.append(np.mean(np.array(rec_list_y)) - y_min)
        # REMOVE FIRST POINT FOR SAVING
        cluster_x_nostart = cluster_x[1:]
        cluster_y_nostart = cluster_y[1:]
        if len(cluster_x_nostart) < 2:
            continue
        else:
            for tp_i in range(len(cluster_x_nostart)):
                cluster_x_nostart[tp_i] = max(min(cluster_x_nostart[tp_i], 755), 0)
                cluster_y_nostart[tp_i] = max(min(cluster_y_nostart[tp_i], 675), 0)
            subject_data['traj_list'].append({
                'traj_id': trial_i,
                'traj_X': cluster_y_nostart,        # save X -> HORIZONTAL
                'traj_Y': cluster_x_nostart,
                'universal_id': universal_id,
            })
            universal_id += 1

    # SAMPLE 80 SCANPATHS FOR EACH SUBJECT
    subject_data['traj_list'] = random.sample(subject_data['traj_list'], k=80)
    TRAJECTORIES.append(subject_data)



# SAVE TO FILE
with open(os.path.join('./env_data/', 'IVSNarray.pkl'), 'wb') as outfile:
    pickle.dump([
            SEARCH_IMAGE_LIST,
            TARGET_IMAGE_LIST,
            TARGET_BBOX_LIST,
            TRAJECTORIES,
        ], outfile)


print('FINISHED')
