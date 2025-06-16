import random
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
import os
import pickle
import cv2
import time
import datetime
import utils
from utils import look
import multimatch_gaze
import models.model_rwta as model_rwta



'''
    METHOD IRL WITH PPO SAMPLING + STATE STACKING
    SVPG AS CANDIDATE
'''



# FOR DEBUG ONLY
S1 = 0
S2 = 2

NEGLECT_TRAIN_VAL_SPLIT = True
REWARD_GAMMA = 0.90
AGENT_GAMMA = 0.90
HUMAN_SCANPATH_THRESHOLD = +0.30
GREEDY_VALIDATION = True
GREEDY_DEBUG = True
GREEDY_EVALUATION = True
TRAIN_LENGTH = 30000

DEBUG = False
np.set_printoptions(precision=3, suppress=True)

GX = 8
GY = 6
HISTORY_LEN = 2





# ================ FUNCTIONS ================
def action_pixel_to_grid(x, y):
    x_new = math.floor(x / 1024 * GX)
    y_new = math.floor(y /  768 * GY)
    return x_new, y_new

def action_grid_to_pixel(x, y):
    x_pixel = math.floor((x + 0.5) / GX * 1023)
    y_pixel = math.floor((y + 0.5) / GY *  767)
    return x_pixel, y_pixel

def action_grid_to_index(x, y):
    action_index = x * GY + y
    return action_index
    
def action_index_to_grid(action_index):
    x_grid = action_index // GY
    y_grid = action_index % GY
    return x_grid, y_grid

def clip_value(x, min_x, max_x):
    return min(max(x, min_x), max_x)

def calc_rewards(r_map_output, state_x, state_y, old_grid_x, old_grid_y, revisitation_flag, PARAM_AMP, PARAM_PEL):
    # DYNAMIC REWARD
    # >-- AMPLITUDE PENALTY
    new_grid_x, new_grid_y = action_pixel_to_grid(x=state_x, y=state_y)
    grid_distance = math.sqrt((new_grid_x - old_grid_x)**2 + (new_grid_y - old_grid_y)**2) * (1024 / GX)
    if grid_distance > PARAM_AMP * 1280:
        reward_amplitude = -1
    else:
        reward_amplitude = 0
    # >-- REVISITATION PENALTY
    if revisitation_flag == True:
        reward_revisitation = PARAM_PEL * (-1)
    else:
        reward_revisitation = 0
    # >-- PREDICTED REWRAD
    reward_predict = r_map_output[new_grid_x, new_grid_y]
    return reward_amplitude + reward_revisitation + reward_predict

def grid_distance(i_ax, i_ay, i_hx, i_hy):
    grid_h_x = np.floor(i_hx / 1024 * GX)
    grid_h_y = np.floor(i_hy /  768 * GY)
    grid_a_x = np.floor(i_ax / 1024 * GX)
    grid_a_y = np.floor(i_ay /  768 * GY)
    grid_distance_value = np.mean(np.sqrt(np.square(grid_a_x-grid_h_x) + np.square(grid_a_y-grid_h_y)))
    return grid_distance_value

def multimatch_distance(i_ax, i_ay, i_hx, i_hy):
    screensize_X = 1024
    screensize_Y = 768
    # PREPROCESS TRAJECTORIES
    agent_X_list = i_ax.tolist()
    agent_Y_list = i_ay.tolist()
    human_X_list = i_hx.tolist()
    human_Y_list = i_hy.tolist()
    # ADD START POINT
    agent_X_list.insert(0, int(screensize_X * 0.5))
    agent_Y_list.insert(0, int(screensize_Y * 0.5))
    human_X_list.insert(0, int(screensize_X * 0.5))
    human_Y_list.insert(0, int(screensize_Y * 0.5))
    # TIME -- NO TIME INFO USED FOR HUMAN SCANPATHS
    agent_duration_list = [1 for _ in range(len(agent_X_list))]
    human_duration_list = [1 for _ in range(len(human_X_list))]
    # CONVERT FORMAT
    agent_scanpath = np.array(list(zip(agent_X_list, agent_Y_list, agent_duration_list)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])
    human_scanpath = np.array(list(zip(human_X_list, human_Y_list, human_duration_list)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])
    [v1, v2, v3, v4, v5] = multimatch_gaze.docomparison(fixation_vectors1=human_scanpath, fixation_vectors2=agent_scanpath, screensize=(screensize_X, screensize_Y))
    ''' 0->3: shape, direction, length, position. 4: duration'''
    return v1, v2, v3, v4, v5




# ================ MODELS ================
class RewardNet(nn.Module):
    def __init__(self, basis_flag, att_map_basis, dev):
        super(RewardNet, self).__init__()
        self.basis_flag = basis_flag
        self.reward_basis = torch.clone(att_map_basis).to(dev)
        # LAYERS
        self.full_conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5, stride=2), nn.ReLU(), nn.MaxPool2d(kernel_size=4))
        self.full_conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=4))
        self.full_conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.full_conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, stride=1), nn.ReLU(),)
        self.full_flatten = nn.Flatten(start_dim=1)
        self.cue_conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=4) )
        self.cue_conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2) )
        self.cue_conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=5, stride=1), nn.ReLU(), )
        self.cue_flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Sequential(nn.Linear(7200, 512), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(512, self.reward_basis.shape[0]))
        self.dev = dev
        self.to(self.dev)

    def forward(self, i_full, i_cue):
        process_full = i_full - 0.5
        process_cue = i_cue - 0.5
        # NETWORK
        x_full = self.full_conv4(self.full_conv3(self.full_conv2(self.full_conv1(process_full))))
        x_full = self.full_flatten(x_full)
        x_cue = self.cue_conv3(self.cue_conv2(self.cue_conv1(process_cue)))
        x_cue = self.cue_flatten(x_cue)
        x = torch.cat([x_full, x_cue], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        reward_weight = self.fc3(x)
        # REWARD MAP
        target_shape = [reward_weight.shape[0], self.reward_basis.shape[0], self.reward_basis.shape[1], self.reward_basis.shape[2]]
        weight_mm = reward_weight.unsqueeze(-1).unsqueeze(-1).expand(target_shape)
        basis_mm = self.reward_basis.unsqueeze(0).expand(target_shape)
        reward_map = torch.sum(weight_mm * basis_mm, dim=1)
        return reward_map[0]
    
    def save_model(self, name_fix=''):
        torch.save(self.state_dict(), './log_model/' + name_fix + '_rewardnet.pt')

    def load_model(self, name_fix=''):
        self.load_state_dict(torch.load('./log_model/' + name_fix + '_rewardnet.pt'))




class AgentNet(nn.Module):
    def __init__(self, dev, critic_flag=False):
        super(AgentNet, self).__init__()
        self.critic_flag = critic_flag
        # PARTIAL OBS
        self.obs_conv1 = nn.Sequential(nn.Conv2d(3*HISTORY_LEN, 16, kernel_size=5, stride=3), nn.ReLU(), nn.MaxPool2d(kernel_size=4))
        self.obs_conv2 = nn.Sequential(nn.Conv2d(16, 48, kernel_size=5, stride=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.obs_flatten = nn.Flatten(start_dim=1)
        # CUE
        self.cue_conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5, stride=3), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.cue_conv2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=5, stride=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.cue_flatten = nn.Flatten(start_dim=1)
        # POSITION EMBEDDING
        # self.pos_fc = nn.Sequential(nn.Linear(240, 48), nn.ReLU())
        # AFTER CONCATENATE
        self.fc1 = nn.Sequential(nn.Linear(768, 1536), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1536, 512), nn.ReLU())        # NO POS USED
        if self.critic_flag == False:
            self.fc3 = nn.Sequential(nn.Linear(512, GX*GY), nn.Softmax(dim=1))
        else:   # Q-VALUES FOR CRITIC
            self.fc3 = nn.Sequential(nn.Linear(512, GX*GY))
        self.dev = dev
        self.to(self.dev)

    def forward(self, i_obs, i_cue, i_pos):
        pre_obs = i_obs - 0.5
        pre_cue = i_cue - 0.5
        x_obs = self.obs_flatten(self.obs_conv2(self.obs_conv1(pre_obs)))
        x_cue = self.cue_flatten(self.cue_conv2(self.cue_conv1(pre_cue)))
        x = torch.cat([x_obs, x_cue], dim=1)
        x = self.fc2(self.fc1(x))
        action_probability = self.fc3(x)
        return action_probability

    def save_model(self, name_fix=''):
        if self.critic_flag == False:
            torch.save(self.state_dict(), './log_model/' + name_fix + '_agentnet.pt')
        else:
            torch.save(self.state_dict(), './log_model/' + name_fix + '_criticnet.pt')

    def load_model(self, name_fix=''):
        if self.critic_flag == False:
            self.load_state_dict(torch.load('./log_model/' + name_fix + '_agentnet.pt'))
        else:
            self.load_state_dict(torch.load('./log_model/' + name_fix + '_criticnet.pt'))




class Memory_AGENT:     # MEMORY BUFFER
    def __init__(self, dev=torch.device('cpu')):
        self.dev = dev
        self.reset()

    def reset(self):
        self.s_obs_list = []
        self.s_cue_list = []
        self.s_pos_list = []
        self.model_output_list = []
        self.a_index_list = []
        self.a_logprob_list = []
        self.r_list = []
        self.SVPG_other_list = []

    def add_transition(self, s_obs, s_cue, s_pos, model_output, a_index, a_logprob, r_value):
        self.s_obs_list.append(s_obs)
        self.s_cue_list.append(s_cue)
        self.s_pos_list.append(s_pos)
        self.model_output_list.append(model_output)
        self.a_index_list.append(a_index)
        self.a_logprob_list.append(a_logprob)
        self.r_list.append(r_value)
    
    def add_transition_SVPG(self, s_obs, s_cue, s_pos, model_output, a_index, a_logprob, r_value, other):
        self.add_transition(s_obs, s_cue, s_pos, model_output, a_index, a_logprob, r_value)
        self.SVPG_other_list.append(other)

    def tune_reward(self):
        for step_i in reversed(range(len(self.r_list) - 1)):
            self.r_list[step_i] += self.r_list[step_i+1] * AGENT_GAMMA

    def get_batch(self):
        self.tune_reward()
        batch_obs = torch.cat(self.s_obs_list)
        batch_cue = torch.cat(self.s_cue_list)
        batch_pos = torch.cat(self.s_pos_list)
        batch_model_output = torch.cat(self.model_output_list)
        batch_a_index = torch.stack(self.a_index_list)
        batch_a_logprob = torch.stack(self.a_logprob_list)
        batch_r_value = torch.stack(self.r_list)
        return batch_obs, batch_cue, batch_pos, batch_model_output, batch_a_index, batch_a_logprob, batch_r_value




# DATASET TO RL ENVIRONMENT
class VSDataloader:
    def __init__(self, img_search_list, img_target_list, traj_list, target_bbox_list, img_size, param_rand):
        '''
            x: 0-7, 0-1023
            y: 0-5, 0-767
        '''
        self.img_search_list = []
        self.img_target_list = []
        self.trajectory_list = []
        self.target_bbox_list = []
        self.org_size_X = img_size[0]
        self.org_size_Y = img_size[1]
        # PARAMS
        self.parameter_random_action = param_rand
        # PREPROCESS IMAGES
        for t_i in range(len(img_search_list)):
            load_image = img_search_list[t_i]
            img_resize = cv2.resize(load_image, (1024, 768), interpolation=cv2.INTER_LINEAR)
            self.img_search_list.append(img_resize)
        for t_i in range(len(img_target_list)):
            load_image = img_target_list[t_i]
            img_target = cv2.resize(load_image, (128, 96), interpolation=cv2.INTER_LINEAR)
            self.img_target_list.append(img_target)
        # PREPROCESS SCANPATH
        for t_i in range(len(traj_list)):
            temp_array = np.copy(traj_list[t_i])
            temp_array[0, :] = temp_array[0, :] / self.org_size_X * 1024
            temp_array[1, :] = temp_array[1, :] / self.org_size_Y * 768
            temp_array = np.int64(temp_array)
            self.trajectory_list.append(temp_array)
        # PREPROCESS HUMAN VISITATION RECORD
        self.human_visitation_record = []
        for t_i in range(len(self.trajectory_list)):
            temp_vc = np.zeros([GX, GY])
            temp_human_traj = self.trajectory_list[t_i]
            for step_i in range(temp_human_traj.shape[1]):
                h_x_pixel = temp_human_traj[0, step_i]
                h_y_pixel = temp_human_traj[1, step_i]
                human_grid_x, human_grid_y = action_pixel_to_grid(h_x_pixel, h_y_pixel)
                temp_vc[human_grid_x, human_grid_y] += 1
            self.human_visitation_record.append(temp_vc)
        # PREPROCESS TARGET BBOX
        for t_i in range(len(target_bbox_list)):
            x1, x2, y1, y2 = target_bbox_list[t_i]
            x1 = x1 / self.org_size_X * 1024
            x2 = x2 / self.org_size_X * 1024
            y1 = y1 / self.org_size_Y * 768
            y2 = y2 / self.org_size_Y * 768
            self.target_bbox_list.append([x1, x2, y1, y2])
        # TRAIN -- VALIDATION SPLIT
        self.trajectory_num = len(self.trajectory_list)
        full_list = list(range(self.trajectory_num))
        
        random_instance = random.Random(123)
        random_instance.shuffle(full_list)
        self.train_list = full_list[:int(0.9 * self.trajectory_num)]
        self.val_list = full_list[int(0.9 * self.trajectory_num):]
        
        if NEGLECT_TRAIN_VAL_SPLIT == True:
            self.train_list = full_list
            self.val_list = full_list

        self.TRAIN_NUM, self.VAL_NUM = len(self.train_list), len(self.val_list)
        print('VAL:   %d   TRAIN:   %d' % (self.VAL_NUM, self.TRAIN_NUM))
    
    def get_partial_observation(self, in_pos_x, in_pos_y):
        # CLIP CURRENT POSITION
        pos_x = clip_value(in_pos_x, 0, 1023)
        pos_y = clip_value(in_pos_y, 0, 767)
        # CROP
        px1 = math.ceil(max(pos_x - 0.20 * 1024, 0))
        px2 = math.floor(min(pos_x + 0.20 * 1024, 1024))
        py1 = math.ceil(max(pos_y - 0.20 *  768, 0))
        py2 = math.floor(min(pos_y + 0.20 *  768, 768))
        # TARGET FILLING
        dx1 = math.floor(0.20 * 1024) - (pos_x - px1)
        dx2 = math.floor(0.20 * 1024) + (px2 - pos_x)
        dy1 = math.floor(0.20 *  768) - (pos_y - py1)
        dy2 = math.floor(0.20 *  768) + (py2 - pos_y)
        partial_obs_org = np.ones([math.floor(0.20*768)*2, math.floor(0.20*1024)*2, 3]) * 128
        partial_obs_org[dy1:dy2, dx1:dx2, :] = self.image_search[py1:py2, px1:px2, :]
        # RESIZE
        partial_obs = cv2.resize(partial_obs_org, (256, 192), interpolation=cv2.INTER_LINEAR)
        partial_obs = np.float32(partial_obs)
        return partial_obs

    def get_position_embedding(self, pos_X, pos_Y):     # NOT USED BY THE MODEL
        grid_x, grid_y = action_pixel_to_grid(pos_X, pos_Y)
        index = action_grid_to_index(grid_x, grid_y)
        index_embedding = torch.zeros([1, 240])
        index_embedding[0, (index*5):(index*5+5)] = 1
        return index_embedding

    def init_common(self, index):
        # BASIC
        self.index = index
        self.step_num = 0
        self.done_signal = False
        # IMAGE
        self.image_search = np.copy(self.img_search_list[index])
        self.image_target = np.copy(self.img_target_list[index])
        # VISITATION RECORD
        self.human_vc = np.copy(self.human_visitation_record[index])
        # INITIAL FIXATION AT CENTER
        self.state_position_x = math.floor(1024 * 0.5)
        self.state_position_y = math.floor( 768 * 0.5)
        self.a_initial_pos_grid_x, self.a_initial_pos_grid_y = action_pixel_to_grid(self.state_position_x, self.state_position_y)
        self.pos_embedding = self.get_position_embedding(self.state_position_x, self.state_position_y)
        # PARTIAL OBSERVATION
        self.partial_obs = self.get_partial_observation(self.state_position_x, self.state_position_y)
        # P OBSERVATION HISTORY
        candidate_search_image = cv2.resize(np.copy(self.img_search_list[index]), (256, 192), interpolation=cv2.INTER_LINEAR)
        self.pobs_history = [candidate_search_image.astype(np.float32) for _ in range(HISTORY_LEN-1)]
        self.pobs_history.append(self.partial_obs)
        self.posembed_history = [np.zeros_like(self.pos_embedding) for _ in range(HISTORY_LEN-1)]
        self.posembed_history.append(self.pos_embedding)
        # SCANPATH
        self.human_traj_length = self.trajectory_list[index].shape[1]
        self.traj_X = np.copy(self.trajectory_list[index][0, :])
        self.traj_Y = np.copy(self.trajectory_list[index][1, :])
        # AGENT BUFFER
        self.agent_trajectory_x = []
        self.agent_trajectory_y = []
        self.agent_visitation_count = np.zeros([8, 6])
        self.step_num = 0
        self.done_signal = False

    def init_train(self):
        self.index = self.train_list[random.randint(0, self.TRAIN_NUM - 1)]
        self.init_common(self.index)

    def init_val(self):
        self.index = self.val_list[random.randint(0, self.VAL_NUM - 1)]
        self.init_common(self.index)
        
    def make_action(self, action_index):
        # CONVERT ACTION FORMAT
        act_grid_x, act_grid_y = action_index_to_grid(action_index)
        act_pixel_x, act_pixel_y = action_grid_to_pixel(act_grid_x, act_grid_y)
        # ADD ACTION NOISE <- ENVIRONMENTAL, NOT EXPLORATION
        if self.parameter_random_action > 0:
            # GAUSSIAN NOISE
            random_noise = np.random.multivariate_normal([0, 0], np.array([[1, 0], [0, 1]]), 1) * self.parameter_random_action * 1024
            rand_x = act_pixel_x + random_noise[0, 0]
            rand_y = act_pixel_y + random_noise[0, 1]
            rand_x = clip_value(x=rand_x, min_x=0, max_x=1023)
            rand_y = clip_value(x=rand_y, min_x=0, max_x=767)
            act_grid_x, act_grid_y = action_pixel_to_grid(x=rand_x, y=rand_y)
        else:
            pass    # USE AGENT DECISION
        # REVISITATION CHECK
        if self.agent_visitation_count[act_grid_x, act_grid_y] >= 1:
            self.flag_action_is_revisit = True
        else:
            self.flag_action_is_revisit = False
        # TRAJECTORY RECORD
        self.agent_trajectory_x.append(act_pixel_x)
        self.agent_trajectory_y.append(act_pixel_y)
        self.agent_visitation_count[act_grid_x, act_grid_y] += 1
        # UPDATE STATE (POSITION)
        self.state_position_x = act_pixel_x
        self.state_position_y = act_pixel_y
        self.partial_obs = self.get_partial_observation(self.state_position_x, self.state_position_y)
        self.pos_embedding = self.get_position_embedding(self.state_position_x, self.state_position_y)
        self.pobs_history.pop(0)
        self.pobs_history.append(self.partial_obs)
        self.posembed_history.pop(0)
        self.posembed_history.append(self.pos_embedding)
        # DONE
        self.step_num += 1
        if self.step_num >= self.human_traj_length:
            self.done_signal = True
        else:
            self.done_signal = False




# MAIN METHOD
class MEPPO:
    def __init__(self, size_X, size_Y, size_obs_radius, exp_name, device=torch.device('cpu'),
                 v_basis=0, v_rand=0.0, v_amp=1.0, v_pel=1.0, entropy_ratio=1.0, svpg_enabled=True,
                 lr_agent=0.0001, lr_reward=0.0001):
        # VARIANT
        self.flag_SVPG = svpg_enabled
        self.entropy_ratio = entropy_ratio
        # PARAMS
        self.param_basis = v_basis
        self.param_rand = v_rand
        self.param_amp = v_amp
        self.param_pel = v_pel
        self.learning_rate_reward = lr_reward
        self.learning_rate_agent = lr_agent
        # SETTINGS
        self.size_X = size_X
        self.size_Y = size_Y
        self.device = device
        self.exp_name = exp_name
        self.init_reward_basis()
        self.init_models()
        # HYPER-PARAMETERS
        self.train_length = TRAIN_LENGTH
        self.agent_train_num = 20
        self.reward_train_num = 20
        self.reward_agent_sample_traj_num = 5
        if DEBUG == True:
            self.train_length = 1000
            self.agent_train_num = 4
        # LOG TEXT FILE
        self.init_log_file()
    
    def init_log_file(self):
        self.File = open('./log_text/log_' + self.exp_name + '.txt', 'w')
        self.log_text_flush_time = time.time()
        self.log_text(self.File, 'init', str(datetime.datetime.now()))

    def log_text(self, file_handle, type_str, record_text, onscreen=False):
        if onscreen:
            print('\033[92m%s\033[0m' % (type_str).ljust(10), record_text)
        file_handle.write((type_str+',').ljust(10) + record_text + '\n')
        if time.time() - self.log_text_flush_time > 10:
            self.log_text_flush_time = time.time()
            file_handle.flush()
            os.fsync(file_handle.fileno())

    def init_reward_basis(self):
        if self.param_basis == 1:       # USE BASIS
            reward_basis_path = './env_data/reward_basis_0909.pkl'
            with open(reward_basis_path, 'rb') as temp_file:
                attention_map_restore = pickle.load(temp_file)
            att_map_basis = torch.from_numpy(attention_map_restore.astype(np.float32))
            att_map_basis_unsqueeze = torch.unsqueeze(att_map_basis, dim=1)
            attention_map_resized = torch.nn.functional.interpolate(input=att_map_basis_unsqueeze, size=[12, 16], mode='nearest')
            attention_map_resized_squeezed = torch.squeeze(attention_map_resized, dim=1)
            self.att_map_basis = torch.transpose(att_map_basis, dim0=1, dim1=2)
        else:                           # NO REWARD FUNCTION BASIS    
            att_map_basis = torch.zeros([GX*GY, GX, GY])
            for map_i in range(GX*GY):
                index = map_i % (GX*GY)
                att_map_basis[map_i, :, :] = 0
                x_grid, y_grid = action_index_to_grid(index)
                att_map_basis[map_i, x_grid, y_grid] = 1
            self.att_map_basis = att_map_basis

    def init_models(self):
        self.model_reward = RewardNet(basis_flag=self.param_basis, att_map_basis=self.att_map_basis, dev=self.device)
        self.opt_reward = torch.optim.Adam(self.model_reward.parameters(), lr=self.learning_rate_reward) # weight_decay=0.1
        self.loss_function_reward = nn.MSELoss()

        self.model_critic = AgentNet(dev=self.device, critic_flag=True)
        self.opt_critic = torch.optim.Adam(self.model_critic.parameters(), lr=0.0001)
        
        if self.flag_SVPG == True:
            self.model_SVPG = model_rwta.RWTA_Prob(
                input_size=32400,
                output_size=GX*GY,
                hid_num=100, hid_size=10,
                remove_connection_pattern='none',
                optimizer_name='sgd', optimizer_learning_rate=self.learning_rate_agent,
                entropy_ratio=self.entropy_ratio,
                device=self.device)
        else:
            self.model_agent = AgentNet(dev=self.device)
            self.opt_agent = torch.optim.Adam(self.model_agent.parameters(), lr=self.learning_rate_agent)
        self.memory_agent = Memory_AGENT()

    def SVPG_state_process(self, s_obs, s_cue):
        batch_size = s_obs.shape[0]
        s_obs_interpolate = torch.nn.functional.interpolate(s_obs, size=[60, 80])
        s_obs_reshape = torch.reshape(s_obs_interpolate, [batch_size, -1])
        s_cue_interpolate = torch.nn.functional.interpolate(s_cue, size=[30, 40])
        s_cue_reshape = torch.reshape(s_cue_interpolate, [batch_size, -1])
        s_concat = torch.cat([s_obs_reshape, s_cue_reshape], dim=1)
        return s_concat

    def list_image_to_tensor(self, in_img_list, gpu=True):
        image_bundle = np.concatenate(in_img_list, axis=2)
        image = np.moveaxis(np.float32(image_bundle), 2, 0)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        if gpu == True:
            image = image.to(self.device)
        image = image / 255
        return image

    def image_to_tensor(self, in_img, gpu=True):
        image = np.moveaxis(np.float32(in_img), 2, 0)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        if gpu == True:
            image = image.to(self.device)
        image = image / 255
        return image

    def train(self, img_search_list, img_target_list, traj_list, target_bbox_list, **kwargs):
        if DEBUG == True:
            DATA = VSDataloader(img_search_list=img_search_list[S1:S2], img_target_list=img_target_list[S1:S2], traj_list=traj_list[S1:S2], target_bbox_list=target_bbox_list[S1:S2], img_size=[self.size_X, self.size_Y], param_rand=self.param_rand)
        else:
            DATA = VSDataloader(img_search_list=img_search_list, img_target_list=img_target_list, traj_list=traj_list, target_bbox_list=target_bbox_list, img_size=[self.size_X, self.size_Y], param_rand=self.param_rand)
        train_record_value_max = -1000
        # >>>>>>>>>> REWARD LOOP
        agent_sample_step = 0
        for train_i in tqdm(range(self.train_length), desc='TRAIN', leave=False):
            # >>>>>>>>>> AGENT TRAIN LOOP
            for epi_i in range(self.agent_train_num):
                DATA.init_train()
                self.memory_agent.reset()
                data_index = DATA.index
                s_full = self.image_to_tensor(DATA.image_search)
                s_cue = self.image_to_tensor(DATA.image_target)
                # REWARD MODEL FORWARD
                with torch.no_grad():
                    r_map_output = self.model_reward.forward(i_full=s_full, i_cue=s_cue)
                # >>>>>>>>>> AGENT EPISODE
                s_obs = self.list_image_to_tensor(DATA.pobs_history)
                s_posembed = DATA.pos_embedding.to(self.device)
                if random.random() < HUMAN_SCANPATH_THRESHOLD:      # USE GUIDE
                    use_guide = 2       # GREEDY GUIDE
                else:
                    use_guide = 0
                if use_guide == 2:
                    # GET READY FOR OPTION 2 -- REWARD-BASED GUIDANCE
                    r_map_scratch = r_map_output.cpu().numpy()
                for step_i in range(DATA.human_traj_length):
                    old_grid_x, old_grid_y = action_pixel_to_grid(x=DATA.state_position_x, y=DATA.state_position_y)
                    # ACTION
                    if self.flag_SVPG:
                        model_output, model_other_output = self.model_SVPG(self.SVPG_state_process(s_obs, s_cue))
                        action_distribution = torch.distributions.Categorical(model_output)
                        action_chosen_index = torch.unsqueeze(torch.argmax(model_other_output[0]), dim=0)
                        # print(action_distribution.entropy().item())
                    else:
                        with torch.no_grad():
                            model_output = self.model_agent.forward(i_obs=s_obs, i_cue=s_cue, i_pos=s_posembed)
                        action_distribution = torch.distributions.Categorical(model_output)
                        action_chosen_index = action_distribution.sample()
                    # ADD HUMAN SCANPATH FOR FASTER EXPLORATION -- ONCE EVERY BATCH
                    if use_guide == 1:
                        # [[[[ OPTION 1 ]]]] -- USE HUMAN SCANPATH AS GUIDE
                        hg_x, hg_y = action_pixel_to_grid(DATA.traj_X[step_i], DATA.traj_Y[step_i])
                        action_index_human = action_grid_to_index(hg_x, hg_y)
                        action_chosen_index[0] = action_index_human
                    elif use_guide == 2:
                        # [[[[ OPTION 2 ]]]] -- USE GREEDY ACTION (REWARD ONLY, NO ENV PENALTY)
                        x_g, y_g = np.unravel_index(r_map_scratch.argmax(), r_map_scratch.shape)
                        action_index = action_grid_to_index(x=x_g, y=y_g)
                        r_map_scratch[x_g, y_g] = -1 * self.param_pel
                        action_chosen_index[0] = action_index
                    action_logprob = action_distribution.log_prob(action_chosen_index)
                    DATA.make_action(action_index=action_chosen_index.item())
                    # CALC REWARD
                    r_combined = calc_rewards(r_map_output, DATA.state_position_x, DATA.state_position_y, old_grid_x, old_grid_y, DATA.flag_action_is_revisit, self.param_amp, self.param_pel)
                    # TRANSITION DATA
                    done_signal = (step_i == DATA.human_traj_length - 1)
                    new_s_obs = self.list_image_to_tensor(DATA.pobs_history)
                    new_s_posembed = DATA.pos_embedding.to(self.device)
                    if self.flag_SVPG:
                        # UPDATE SVPG OTHER_OUTPUT WITH POSSIBLE HUMAN PREFERENCE
                        model_other_output[0] = model_other_output[0] * 0
                        model_other_output[0][0, action_chosen_index.item()] = 1
                        model_other_output[1] = action_logprob
                        model_other_output[3][:, self.model_SVPG.dim_h:self.model_SVPG.dim_ha] = model_other_output[0]
                        self.memory_agent.add_transition_SVPG(s_obs=s_obs, s_cue=s_cue, s_pos=s_posembed, model_output=model_output, a_index=action_chosen_index, a_logprob=action_logprob, r_value=r_combined, other=model_other_output)
                    else:
                        self.memory_agent.add_transition(s_obs=s_obs, s_cue=s_cue, s_pos=s_posembed, model_output=model_output, a_index=action_chosen_index, a_logprob=action_logprob, r_value=r_combined)
                    # TIME STEP TRANSITION
                    s_obs = torch.clone(new_s_obs)
                    s_posembed = torch.clone(new_s_posembed)
                # UPDATE
                # TRANSITION DATA
                b_obs, b_cue, b_pos, b_model_output, b_a_index, b_a_logprob, b_r_value = self.memory_agent.get_batch()
                batch_size = b_obs.shape[0]
                a1_index = b_a_index[:, 0].int()
                a1_onehot = torch.zeros([batch_size, GX*GY], device=self.device)
                a1_onehot[torch.arange(batch_size), a1_index] = 1
                # <4>
                # CRITIC VALUES
                s1_value = self.model_critic.forward(i_obs=b_obs, i_cue=b_cue, i_pos=b_pos)
                s1_value_ave = torch.sum(b_model_output * s1_value, dim=1).detach()
                # UPDATE CRITIC
                state_value_target = s1_value.clone().detach()
                state_value_target[torch.arange(batch_size), a1_index] = b_r_value
                loss_critic = torch.mean(torch.sum(torch.square(s1_value - state_value_target), dim=1))
                self.opt_critic.zero_grad()
                loss_critic.backward()
                self.opt_critic.step()
                # UPDATE AGENT
                advantage = (b_r_value - s1_value_ave).detach()
                old_logprob = torch.clone(b_a_logprob)
                for ppo_i in range(3):
                    epsilon_clip = 0.2
                    if self.flag_SVPG:
                        model_output, model_other_output = self.model_SVPG(self.SVPG_state_process(b_obs, b_cue))
                        model_output.requires_grad_()
                        action_distribution = torch.distributions.Categorical(model_output)
                        action_logprob = action_distribution.log_prob(b_a_index[:, 0])
                        action_entropy = action_distribution.entropy()
                        old_other = self.memory_agent.SVPG_other_list
                        loss_agent = self.model_SVPG.learn_episode_ppo(action_logprob, old_logprob[:, 0], advantage,
                            epsilon_clip, action_entropy,
                            other=[old_other, model_other_output], model_output=model_output, a_onehot=a1_onehot)
                    else:
                        model_output_ppo = self.model_agent.forward(i_obs=b_obs, i_cue=b_cue, i_pos=b_pos)
                        action_distribution = torch.distributions.Categorical(model_output_ppo)
                        action_logprob_ppo = action_distribution.log_prob(b_a_index[:, 0])
                        action_entropy = action_distribution.entropy()
                        ratio = torch.exp(action_logprob_ppo - old_logprob[:, 0].detach())
                        target_1 = ratio * advantage
                        target_2 = torch.clamp(ratio, 1-epsilon_clip, 1+epsilon_clip) * advantage
                        loss_agent = -torch.min(target_1, target_2).mean() - self.entropy_ratio * action_entropy.mean()
                        self.opt_agent.zero_grad()
                        loss_agent.backward()
                        self.opt_agent.step()
                    if ppo_i == 0:
                        record_text = '%10d, %10d, %5d, %10.8f' % (agent_sample_step, train_i, epi_i, loss_agent.detach().cpu().numpy())
                        self.log_text(self.File, 'train_agent', record_text=record_text)
            # >>>>>>>>>> GATHER DATA FOR REWARD MODEL
            reward_loss = 0
            for epi_i in range(self.reward_train_num):
                DATA.init_train()
                data_index = DATA.index
                s_full = self.image_to_tensor(DATA.image_search)
                s_cue = self.image_to_tensor(DATA.image_target)
                r_map_output = self.model_reward.forward(i_full=s_full, i_cue=s_cue)
                # (1) EVALUATE THIS HUMAN SCANPATH
                human_reward_record = []
                for step_i in range(DATA.human_traj_length):
                    old_grid_x, old_grid_y = action_pixel_to_grid(x=DATA.state_position_x, y=DATA.state_position_y)
                    hg_x, hg_y = action_pixel_to_grid(DATA.traj_X[step_i], DATA.traj_Y[step_i])
                    action_index = action_grid_to_index(hg_x, hg_y)
                    DATA.make_action(action_index=action_index)
                    r_combined = calc_rewards(r_map_output, DATA.state_position_x, DATA.state_position_y, old_grid_x, old_grid_y, DATA.flag_action_is_revisit, self.param_amp, self.param_pel)
                    human_reward_record.append(r_combined)
                for step_i in reversed(range(DATA.human_traj_length - 1)):
                    human_reward_record[step_i] += human_reward_record[step_i+1] * REWARD_GAMMA
                # (2) GET AGENT SCANPATH
                agent_reward_collect = []
                agent_prob_collect = []
                for agent_collect_i in range(self.reward_agent_sample_traj_num):
                    DATA.init_common(data_index)        # IMPORTANT: RE INIT ENV
                    agent_reward_record = []
                    agent_traj_logprob = 0
                    for step_i in range(DATA.human_traj_length):
                        s_obs = self.list_image_to_tensor(DATA.pobs_history)
                        s_posembed = DATA.pos_embedding.to(self.device)
                        old_grid_x, old_grid_y = action_pixel_to_grid(x=DATA.state_position_x, y=DATA.state_position_y)
                        
                        if self.flag_SVPG:
                            model_output, model_other_output = self.model_SVPG(self.SVPG_state_process(s_obs, s_cue))
                        else:
                            with torch.no_grad():
                                model_output = self.model_agent.forward(i_obs=s_obs, i_cue=s_cue, i_pos=s_posembed)
                        # 0422 REVISION: SMOOTHER DISTRIBUTION
                        model_output = model_output+0.10

                        action_distribution = torch.distributions.Categorical(model_output)
                        action_chosen_index = action_distribution.sample()
                        action_logprob = action_distribution.log_prob(action_chosen_index)
                        action_index = action_chosen_index.item()
                        DATA.make_action(action_index=action_index)
                        r_combined = calc_rewards(r_map_output, DATA.state_position_x, DATA.state_position_y, old_grid_x, old_grid_y, DATA.flag_action_is_revisit, self.param_amp, self.param_pel)
                        agent_reward_record.append(r_combined)
                        agent_traj_logprob += action_logprob
                    for step_i in reversed(range(DATA.human_traj_length - 1)):
                        agent_reward_record[step_i] += agent_reward_record[step_i+1] * REWARD_GAMMA
                    agent_reward_collect.append(agent_reward_record[0])
                    agent_prob_collect.append(agent_traj_logprob)
                # # (3) ADD HUMAN PATH PROB ESTIMATION      # PART OF ABLATION
                # DATA.init_common(data_index)        # IMPORTANT: RE INIT ENV
                # agent_reward_record = []
                # agent_traj_logprob = 0
                # for step_i in range(DATA.human_traj_length):
                #     s_obs = self.list_image_to_tensor(DATA.pobs_history)
                #     s_posembed = DATA.pos_embedding.to(self.device)
                #     old_grid_x, old_grid_y = action_pixel_to_grid(x=DATA.state_position_x, y=DATA.state_position_y)
                    
                #     if self.flag_SVPG:
                #         model_output, model_other_output = self.model_SVPG(self.SVPG_state_process(s_obs, s_cue))
                #     else:    
                #         with torch.no_grad():
                #             model_output = self.model_agent.forward(i_obs=s_obs, i_cue=s_cue, i_pos=s_posembed)
                    
                #     action_distribution = torch.distributions.Categorical(model_output)
                #     hg_x, hg_y = action_pixel_to_grid(DATA.traj_X[step_i], DATA.traj_Y[step_i])
                #     action_chosen_index = action_distribution.sample()
                #     human_action_index = action_grid_to_index(hg_x, hg_y)
                #     action_chosen_index[0] = human_action_index
                #     action_logprob = action_distribution.log_prob(action_chosen_index)
                #     action_index = action_chosen_index.item()
                #     DATA.make_action(action_index=action_index)
                #     r_combined = calc_rewards(r_map_output, DATA.state_position_x, DATA.state_position_y, old_grid_x, old_grid_y, DATA.flag_action_is_revisit, self.param_amp, self.param_pel)
                #     agent_reward_record.append(r_combined)
                #     agent_traj_logprob += action_logprob
                # for step_i in reversed(range(DATA.human_traj_length - 1)):
                #     agent_reward_record[step_i] += agent_reward_record[step_i+1] * REWARD_GAMMA
                # agent_reward_collect.append(agent_reward_record[0])
                # agent_prob_collect.append(agent_traj_logprob)
                # (4) CALC REWARD GRADIENT AND TARGET
                loss_reward_1 = human_reward_record[0]
                loss_reward_2 = torch.log(torch.mean(torch.exp(torch.stack(agent_reward_collect)) / (torch.exp(torch.cat(agent_prob_collect)) + 1e-15)))
                
                loss_reward_to_maximize = loss_reward_1 - loss_reward_2
                reward_loss = reward_loss + (-1) * loss_reward_to_maximize
                if torch.isinf(reward_loss) or torch.isnan(reward_loss):
                    print('INF CHECK')
                    record_text = 'INF CHECK'
                    self.log_text(self.File, 'error', record_text=record_text)
                    if DEBUG:
                        breakpoint()
                    else:
                        exit()
                # (5) REWARD MAP CLAMP LOSS
                reward_map_sum = torch.sum(r_map_output)
                ratio_limit_max = torch.nn.functional.relu(torch.max(r_map_output).detach() - 10)
                ratio_limit_sum = torch.nn.functional.relu(torch.sum(r_map_output).detach() - 40)
                reward_loss = reward_loss + 1 * 0.5 * torch.sum(torch.square(r_map_output)) * torch.max(ratio_limit_max, ratio_limit_sum) + 5 * torch.sum(torch.nn.functional.relu(r_map_output * (-1)))
            # UPDATE REWARD MODEL
            loss_reward = reward_loss / self.agent_train_num
            self.opt_reward.zero_grad()
            loss_reward.backward()
            self.opt_reward.step()
            
            record_text = '%10d, %10.8f' % (train_i, loss_reward.detach().cpu().numpy())
            self.log_text(self.File, 'train_reward', record_text=record_text)
            
            # DEBUG
            if DEBUG == True:
                DATA.init_train()
                data_index = DATA.index
                s_full = self.image_to_tensor(DATA.image_search)
                s_cue = self.image_to_tensor(DATA.image_target)
                with torch.no_grad():
                    r_map_output = self.model_reward.forward(i_full=s_full, i_cue=s_cue)
                for step_i in range(DATA.human_traj_length):
                    s_obs = self.list_image_to_tensor(DATA.pobs_history)
                    s_posembed = DATA.pos_embedding.to(self.device)
                    old_grid_x, old_grid_y = action_pixel_to_grid(x=DATA.state_position_x, y=DATA.state_position_y)
                    
                    if self.flag_SVPG:
                        model_output, model_other_output = self.model_SVPG(self.SVPG_state_process(s_obs, s_cue))
                    else:
                        with torch.no_grad():
                            model_output = self.model_agent.forward(i_obs=s_obs, i_cue=s_cue, i_pos=s_posembed)
                    action_distribution = torch.distributions.Categorical(model_output)
                    action_chosen_index = action_distribution.sample()
                    if GREEDY_DEBUG:
                        action_chosen_index[0] = torch.argmax(model_output)
                    action_index = action_chosen_index.item()
                    DATA.make_action(action_index=action_index)
                DEBUG_DISPLAY = [np.copy(DATA.human_vc), np.copy(DATA.agent_visitation_count),
                                    np.copy(DATA.traj_X), np.copy(DATA.traj_Y),
                                    DATA.agent_trajectory_x.copy(), DATA.agent_trajectory_y.copy(), ]
                print(loss_reward.detach().item())
                print(r_map_output.detach().cpu().numpy())
                print(DEBUG_DISPLAY[0])
                print(DEBUG_DISPLAY[1])
                for s_i in range(len(DEBUG_DISPLAY[4])):
                    x, y = action_pixel_to_grid(DEBUG_DISPLAY[2][s_i], DEBUG_DISPLAY[3][s_i])
                    print('(%d,%d)'%(x, y), ' ', end='')
                print()
                for s_i in range(len(DEBUG_DISPLAY[4])):
                    x, y = action_pixel_to_grid(DEBUG_DISPLAY[4][s_i], DEBUG_DISPLAY[5][s_i])
                    print('(%d,%d)'%(x, y), ' ', end='')
                print()

            # ========== VALIDATION AND SAVE MODEL ==========
            if train_i % 1000 == 999:
                validation_record = np.zeros([len(DATA.val_list), 1])
                for epi_i in range(len(DATA.val_list)):
                    # GET SCANPATH INDEX
                    val_index = DATA.val_list[epi_i]
                    DATA.init_common(val_index)
                    data_index = DATA.index
                    # INIT EPISODE
                    s_full = self.image_to_tensor(DATA.image_search)
                    s_cue = self.image_to_tensor(DATA.image_target)
                    # EPISODE BEGIN
                    for step_i in range(DATA.human_traj_length):
                        s_obs = self.list_image_to_tensor(DATA.pobs_history)
                        s_posembed = DATA.pos_embedding.to(self.device)
                        old_grid_x, old_grid_y = action_pixel_to_grid(x=DATA.state_position_x, y=DATA.state_position_y)
                        
                        if self.flag_SVPG:
                            model_output, model_other_output = self.model_SVPG(self.SVPG_state_process(s_obs, s_cue))
                        else:
                            with torch.no_grad():
                                model_output = self.model_agent.forward(i_obs=s_obs, i_cue=s_cue, i_pos=s_posembed)
                        
                        action_distribution = torch.distributions.Categorical(model_output)
                        action_chosen_index = action_distribution.sample()
                        if GREEDY_VALIDATION:
                            action_chosen_index[0] = torch.argmax(model_output)
                        action_logprob = action_distribution.log_prob(action_chosen_index)
                        action_index = action_chosen_index.item()
                        DATA.make_action(action_index=action_index)
                    # MEASUREMENT
                    vc_human = np.copy(DATA.human_vc)
                    vc_agent = np.copy(DATA.agent_visitation_count)
                    scanpath_x_human = np.copy(DATA.traj_X)
                    scanpath_y_human = np.copy(DATA.traj_Y)
                    scanpath_x_agent = np.array(DATA.agent_trajectory_x)
                    scanpath_y_agent = np.array(DATA.agent_trajectory_y)
                    valid_measurement_value = grid_distance(i_ax=scanpath_x_agent, i_ay=scanpath_y_agent,
                                                            i_hx=scanpath_x_human, i_hy=scanpath_y_human)
                    validation_record[epi_i, 0] = (-1) * valid_measurement_value
                criteria_value = np.mean(validation_record[:, 0])       # MULTIMATCH SIMILARITY SCORE
                record_text = '%10d, %10.8f' % (train_i, criteria_value)
                self.log_text(self.File, 'valid', record_text=record_text, onscreen=True)
                # SAVE MODEL
                if criteria_value >= train_record_value_max:
                    train_record_value_max = criteria_value
                    if self.flag_SVPG:
                        self.model_SVPG.save_model(name=self.exp_name)
                    else:
                        self.model_agent.save_model(name_fix=self.exp_name)
                    self.model_critic.save_model(name_fix=self.exp_name)
                    self.model_reward.save_model(name_fix=self.exp_name)
                    record_text = '%10d' % (train_i)
                    self.log_text(self.File, 'save', record_text=record_text, onscreen=True)
            if train_i % 100 == 99:
                if self.flag_SVPG:
                    self.model_SVPG.save_model(name=self.exp_name+'_latest')
                else:
                    self.model_agent.save_model(name_fix=self.exp_name+'_latest')
                self.model_critic.save_model(name_fix=self.exp_name+'_latest')
                self.model_reward.save_model(name_fix=self.exp_name+'_latest')
            if train_i % 2000 == 1999:
                exp_name_postfix = '_%d' % (train_i)
                if self.flag_SVPG:
                    self.model_SVPG.save_model(name=self.exp_name+exp_name_postfix)
                else:
                    self.model_agent.save_model(name_fix=self.exp_name+exp_name_postfix)
                self.model_critic.save_model(name_fix=self.exp_name+exp_name_postfix)
                self.model_reward.save_model(name_fix=self.exp_name+exp_name_postfix)
    
    def evaluate(self, img_search_list, img_target_list, traj_list, target_bbox_list, **kwargs):
        print('EVALUATION')
        if DEBUG == True:
            DATA = VSDataloader(img_search_list=img_search_list[S1:S2], img_target_list=img_target_list[S1:S2], traj_list=traj_list[S1:S2], target_bbox_list=target_bbox_list[S1:S2], img_size=[self.size_X, self.size_Y], param_rand=self.param_rand)
        else:
            DATA = VSDataloader(img_search_list=img_search_list, img_target_list=img_target_list, traj_list=traj_list, target_bbox_list=target_bbox_list, img_size=[self.size_X, self.size_Y], param_rand=self.param_rand)
        
        if self.flag_SVPG:
            self.model_SVPG.load_model(name=self.exp_name)
        else:
            self.model_agent.load_model(name_fix=self.exp_name)
        
        agent_traj_list = []

        for epi_i in tqdm(range(len(DATA.trajectory_list)), desc='EVA'):
            # GET SCANPATH INDEX
            eva_index = epi_i
            DATA.init_common(eva_index)
            data_index = DATA.index
            # INIT EPISODE
            s_full = self.image_to_tensor(DATA.image_search)
            s_cue = self.image_to_tensor(DATA.image_target)
            
            # EPISODE BEGIN
            for step_i in range(DATA.human_traj_length):
                s_obs = self.list_image_to_tensor(DATA.pobs_history)
                s_posembed = DATA.pos_embedding.to(self.device)
                old_grid_x, old_grid_y = action_pixel_to_grid(x=DATA.state_position_x, y=DATA.state_position_y)
                
                if self.flag_SVPG:
                    model_output, model_other_output = self.model_SVPG(self.SVPG_state_process(s_obs, s_cue))
                else:
                    with torch.no_grad():
                        model_output = self.model_agent.forward(i_obs=s_obs, i_cue=s_cue, i_pos=s_posembed)
                
                action_distribution = torch.distributions.Categorical(model_output)
                action_chosen_index = action_distribution.sample()
                if GREEDY_EVALUATION:
                    action_chosen_index[0] = torch.argmax(model_output)
                action_logprob = action_distribution.log_prob(action_chosen_index)
                action_index = action_chosen_index.item()
                DATA.make_action(action_index=action_index)

            # SCANPATH CONVERSION
            store_traj_x = [value / 1024 * DATA.org_size_X for value in DATA.agent_trajectory_x]
            store_traj_y = [value /  768 * DATA.org_size_Y for value in DATA.agent_trajectory_y]
            agent_trajectory = np.zeros_like(DATA.trajectory_list[epi_i])
            agent_trajectory[0, :] = store_traj_x
            agent_trajectory[1, :] = store_traj_y
            agent_traj_list.append(agent_trajectory)
        
        return agent_traj_list

