import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import skimage
from tqdm import tqdm
import datetime
import time
import cv2
import math
import utils
import os
import pickle



'''
    TRANSFORMER-BASED SINGLE-STEP PREDICTION
    NETWORK BASED ON
        https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
'''



class AttentionBlock(nn.Module):
    def __init__(self, dim_embedding, dim_hidden, num_heads, dropout=0):
        super().__init__()
        self.layer_normalization_1 = nn.LayerNorm(dim_embedding)
        self.attention = nn.MultiheadAttention(dim_embedding, num_heads)
        ''' NOTE: BATCH_FIRST NOT ENABLED'''
        self.layer_normalization_2 = nn.LayerNorm(dim_embedding)
        self.linear = nn.Sequential(
            nn.Linear(dim_embedding, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_embedding),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x_norm = self.layer_normalization_1(x)
        attention_output, att_weights = self.attention(x_norm, x_norm, x_norm)
        x = x + attention_output
        x = x + self.linear(self.layer_normalization_2(x))
        # ATTENTION OUTPUT WEIGHTS -- RESIDUAL AND RENORMALIZATION
        residual_attention = torch.eye(att_weights.shape[1], device=x.device)
        att_weights = att_weights + residual_attention
        att_weights = att_weights / torch.sum(att_weights, dim=-1).unsqueeze(dim=-1)
        return x, att_weights


class Classifier_Net(nn.Module):
    def __init__(self, action_dim=48, history_len=10):
        super(Classifier_Net, self).__init__()
        self.action_dim = action_dim
        self.history_len = history_len
        
        # SEARCH IMAGE CNN
        self.conv2_1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=6, stride=4), nn.ReLU(), nn.MaxPool2d(kernel_size=2) )
        self.conv2_2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2) )
        self.flatten2 = nn.Flatten(start_dim=1)
        # TARGET IMAGE CNN
        self.conv3_1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=6, stride=4), nn.ReLU(), nn.MaxPool2d(kernel_size=2) )
        self.conv3_2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2) )
        self.flatten3 = nn.Flatten(start_dim=1)
        
        # ATTENTION
        self.dim_embedding = 384
        self.dim_hidden = 384
        self.n_head = 8
        self.dropout = 0.2
        self.attention_block_1 = AttentionBlock(dim_embedding=self.dim_embedding, dim_hidden=self.dim_hidden, num_heads=self.n_head, dropout=self.dropout)
        self.attention_block_2 = AttentionBlock(dim_embedding=self.dim_embedding, dim_hidden=self.dim_hidden, num_heads=self.n_head, dropout=self.dropout)
        self.attention_block_3 = AttentionBlock(dim_embedding=self.dim_embedding, dim_hidden=self.dim_hidden, num_heads=self.n_head, dropout=self.dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.dim_embedding),
            nn.Linear(self.dim_embedding, self.dim_embedding),
            nn.ReLU(),
            nn.Linear(self.dim_embedding, self.action_dim),
        )
        self.dropout = nn.Dropout(self.dropout)
        self.token = nn.Parameter(torch.randn(1, 1, self.dim_embedding))

    def forward(self, i_obs_hist, i_cue):
        # INFO
        i_obs_hist = i_obs_hist.permute(0, 1, 4, 2, 3)
        i_cue = torch.squeeze(i_cue.permute(0, 1, 4, 2, 3), dim=1)
        batch_size = i_obs_hist.shape[0]
        history_len = i_obs_hist.shape[1]
        i_obs_hist_reshape = torch.reshape(i_obs_hist, [batch_size*history_len, 3, 96, 128])
        # IMAGE TO FEATURE
        x2_comb = self.conv2_2(self.conv2_1(i_obs_hist_reshape))
        x2_comb = self.flatten2(x2_comb)
        x2 = torch.reshape(x2_comb, [batch_size, history_len, 384])
        # ---
        x3 = self.conv3_2(self.conv3_1(i_cue))
        x3 = self.flatten3(x3)
        x3 = torch.unsqueeze(x3, dim=1)
        # ATTENTION LAYERS
        token_batch = self.token.repeat(batch_size, 1, 1)
        x = torch.cat([token_batch, x2, x3], dim=1)
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x, att_weight_1 = self.attention_block_1(x)
        x, att_weight_2 = self.attention_block_2(x)
        x, att_weight_3 = self.attention_block_3(x)
        total_attention = torch.matmul(torch.matmul(att_weight_3, att_weight_2), att_weight_1)
        total_attention = total_attention[:, 0, :]
        # CLASSIFIER
        output = self.classifier(x[0])
        # OUTPUT
        return output, total_attention
        
    def save_model(self, name_fix=''):
        torch.save(self.state_dict(), './log_model/' + name_fix + '_best.pt')

    def load_model(self, name_fix=''):
        self.load_state_dict(torch.load('./log_model/' + name_fix + '_best.pt'))



class BCtransformer:
    def __init__(self, size_X, size_Y, device=torch.device('cpu'), exp_name=''):
        # SEARCH IMAGE SIZE
        self.size_X = size_X;  self.size_Y = size_Y
        # AGENT ACTION GRID
        self.grid_size_X = 8
        self.grid_size_Y = 6
        self.action_num = self.grid_size_X * self.grid_size_Y
        # PARTIAL OBSEARVATION SIZE
        self.partial_obs_size_X = round(self.size_X * 0.25)
        self.partial_obs_size_Y = round(self.size_Y * 0.25)
        # ATTENTION SETTING
        self.history_len = 10
        # TRAINING SETTING
        self.batch_size = 128
        self.device = device
        self.exp_name = exp_name
        # INITIALIZATION
        self.init_model()
        self.init_log_file()

    def init_model(self):               # MODEL AND OPTIMIZER
        self.classifier_model = Classifier_Net(action_dim=self.action_num, history_len=self.history_len).to(device=self.device)
        self.optimizer = torch.optim.Adam(self.classifier_model.parameters(), lr=0.0001, weight_decay=0.01)
        # self.loss_function = torch.nn.MSELoss()
        self.loss_function = torch.nn.CrossEntropyLoss()
    
    def init_log_file(self):
        self.File = open('./log_text/log_' + self.exp_name + '.txt', 'w')
        self.log_text_flush_time = time.time()
        self.log_text(self.File, 'init', str(datetime.datetime.now()))

    def log_text(self, file_handle, type_str, record_text, onscreen=True):
        if onscreen:
            print('\033[92m%s\033[0m' % (type_str).ljust(10), record_text)
        file_handle.write((type_str+',').ljust(10) + record_text + '\n')
        if time.time() - self.log_text_flush_time > 10:
            self.log_text_flush_time = time.time()
            file_handle.flush()
            os.fsync(file_handle.fileno())

    def action_pixel_convert(self, pixel_x, pixel_y):
        grid_x = math.floor(pixel_x / self.size_X * self.grid_size_X)
        grid_y = math.floor(pixel_y / self.size_Y * self.grid_size_Y)
        action_index = grid_x + grid_y * self.grid_size_X
        action_onehot = np.zeros([self.grid_size_X * self.grid_size_Y])
        action_onehot[action_index] = 1
        return grid_x, grid_y, action_index, action_onehot

    def action_index_convert(self, index):
        grid_x = index % self.grid_size_X
        grid_y = index // self.grid_size_X
        pixel_x = math.floor((grid_x + 0.5) / self.grid_size_X * self.size_X)
        pixel_y = math.floor((grid_y + 0.5) / self.grid_size_Y * self.size_Y)
        return grid_x, grid_y, pixel_x, pixel_y

    def train(self, img_search_list, img_target_list, traj_list, **kwargs):
        # EXTRACT DATA ==========================
        data_partial_obs_list = []
        data_target_image_list = []
        data_human_fixation_list = []
        for traj_i in tqdm(range(len(traj_list)), leave=False, desc='EXTRACT DATA 1'):
            traj_data = traj_list[traj_i]
            traj_length = traj_data.shape[1]
            # IMAGE SEARCH EXPAND
            image_search = np.copy(img_search_list[traj_i])
            image_target = np.copy(img_target_list[traj_i])
            s1, s2, s3 = image_search.shape
            image_search_expand = np.ones([s1*3, s2*3, s3]) * 128
            image_search_expand[s1:(s1*2), s2:(s2*2), :] = image_search
            # TRAJECTORY DATA
            temp_partial_obs = []       # PARTIAL OBSERVATIONS IN A SCANPATH
            temp_human_fixation = []    # HUMAN FIXATIONS IN A SCANPATH
            for step_i in range(traj_length):
                # AGENT FIXATION FOR PARTIAL OBS GENERATION. START AT CENTER.
                if step_i == 0:
                    agent_fixation_x = round(self.size_X * 0.5)
                    agent_fixation_y = round(self.size_Y * 0.5)
                else:
                    agent_fixation_x = traj_data[0, step_i - 1]
                    agent_fixation_y = traj_data[1, step_i - 1]
                # PARTIAL OBSERVATION
                px_1 = agent_fixation_x - self.partial_obs_size_X + s2
                px_2 = agent_fixation_x + self.partial_obs_size_X + s2
                py_1 = agent_fixation_y - self.partial_obs_size_Y + s1
                py_2 = agent_fixation_y + self.partial_obs_size_Y + s1
                image_partial = np.copy(image_search_expand[py_1:py_2, px_1:px_2, :])
                image_partial_resize = cv2.resize(image_partial, (128, 96), interpolation=cv2.INTER_LINEAR)
                image_partial_resize = image_partial_resize / 255
                # HUMAN FIXATION -- TARGET
                human_fixation_x = traj_data[0, step_i]
                human_fixation_y = traj_data[1, step_i]
                _1, _2, _3, human_fixation_onehot = self.action_pixel_convert(pixel_x=human_fixation_x, pixel_y=human_fixation_y)
                # DATASET
                temp_human_fixation.append(human_fixation_onehot)
                temp_partial_obs.append(image_partial_resize)
            # DATASET ALL
            image_target_resize = cv2.resize(image_target, (128, 96), interpolation=cv2.INTER_LINEAR)
            image_target_resize = image_target_resize / 255
            data_target_image_list.append(np.float32(np.expand_dims(image_target_resize, axis=0)))
            data_partial_obs_list.append(np.float32(np.stack(temp_partial_obs)))
            data_human_fixation_list.append(np.float32(np.stack(temp_human_fixation)))
        # =======================================
        # CONVERT TO POINT PREDICTION TASK
        data_pobshist_list = []
        data_timg_list = []
        data_hfix_list = []
        data_validlength_list = []
        for traj_i in tqdm(range(len(traj_list)), leave=False, desc='EXTRACT DATA 2'):
            step_num = data_human_fixation_list[traj_i].shape[0]
            for step_i in range(step_num):
                i_start = max(step_i + 1 - self.history_len, 0)
                sto_number = step_i + 1 - i_start
                pobs_hist = np.ones([self.history_len, 96, 128, 3]) * 0.5
                pobs_hist[(self.history_len-sto_number):self.history_len] = np.copy(data_partial_obs_list[traj_i][i_start:(step_i+1)])
                data_pobshist_list.append(pobs_hist)
                data_timg_list.append(np.copy(data_target_image_list[traj_i]))
                data_hfix_list.append(np.copy(data_human_fixation_list[traj_i][step_i]))
                data_validlength_list.append(sto_number)
        del data_human_fixation_list
        del data_target_image_list
        del data_partial_obs_list
        # UPDATE: USE TORCH TO ACCELARATE STACK OPERATION
        data_pobshist_list = torch.stack([torch.FloatTensor(data_pobshist_list[i]) for i in range(len(data_pobshist_list))])
        data_timg_list = torch.stack([torch.FloatTensor(data_timg_list[i]) for i in range(len(data_timg_list))])
        data_hfix_list = torch.stack([torch.FloatTensor(data_hfix_list[i]) for i in range(len(data_hfix_list))])
        data_validlength_list = np.float32(np.stack(data_validlength_list))
        # =======================================
        # TRAIN-VALIDATION SPLIT
        index_list_all = list(range(len(data_pobshist_list)))
        assert len(data_pobshist_list) > 10
        random_instance = random.Random(1)
        random_instance.shuffle(index_list_all)
        index_list_train = index_list_all
        index_list_valid = index_list_all
        loss_valid_record = 10000
        # TRAIN
        for train_i in range(10000):
            batch_list = np.random.choice(index_list_train, size=self.batch_size, replace=False)
            # DATA SAMPLE
            d1 = (data_pobshist_list[batch_list]).to(self.device)
            d2 = (data_timg_list[batch_list]).to(self.device)
            d_label = (data_hfix_list[batch_list]).to(self.device)
            d_obslen = data_validlength_list[batch_list]
            d_label_argmax = torch.argmax(d_label, dim=1)
            # MODEL PREDICTION
            predict, attention_out = self.classifier_model(d1, d2)
            # LOSS
            _, predict_argmax = torch.max(predict, dim=1)
            predict_onehot = F.one_hot(predict_argmax.detach(), num_classes=self.action_num)
            correct_num = torch.sum(torch.sum(torch.abs(predict_onehot - d_label), dim=1) == 0)
            loss_value = self.loss_function(predict, d_label_argmax)
            # OPTIMIZE
            train_accuracy = correct_num / self.batch_size
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()
            
            if train_i % 1000 == 999:
                torch.save(self.classifier_model.state_dict(), './log_model/' + self.exp_name + '_%06d' % train_i + '.pt')
                self.log_text(self.File, 'save', record_text='%10d' % (train_i))
            if train_i == 9999:
                self.classifier_model.save_model(name_fix=self.exp_name)

    def evaluate(self, img_search_list, img_target_list, traj_list, **kwargs):
        agent_traj_list = []
        attention_record = []
        self.classifier_model.load_model(name_fix=self.exp_name)
        for traj_i in tqdm(range(len(traj_list)), desc='AGENT', leave=False):
            # PLACEHOLDER
            agent_trajectory = np.zeros_like(traj_list[traj_i])
            # TRAJECTORY DATA
            traj_data = traj_list[traj_i]
            traj_length = traj_data.shape[1]
            # IMAGE SEARCH EXPAND
            image_search = np.copy(img_search_list[traj_i])
            image_target = np.copy(img_target_list[traj_i])
            s1, s2, s3 = image_search.shape
            image_search_expand = np.ones([s1*3, s2*3, s3]) * 128
            image_search_expand[s1:(s1*2), s2:(s2*2), :] = image_search
            # IMAGE TARGET
            image_target_resize = cv2.resize(image_target, (128, 96), interpolation=cv2.INTER_LINEAR)
            image_target_resize = image_target_resize / 255
            # INITIAL FIXATION POINT
            agent_fixation_x = round(self.size_X * 0.5)
            agent_fixation_y = round(self.size_Y * 0.5)
            # INITIALIZE OBSERVATION HISTORY
            pobs_hist = np.ones([self.history_len, 96, 128, 3]) * 0.5
            for step_i in range(traj_length):
                # PARTIAL OBSERVATION
                px_1 = agent_fixation_x - self.partial_obs_size_X + s2
                px_2 = agent_fixation_x + self.partial_obs_size_X + s2
                py_1 = agent_fixation_y - self.partial_obs_size_Y + s1
                py_2 = agent_fixation_y + self.partial_obs_size_Y + s1
                image_partial = np.copy(image_search_expand[py_1:py_2, px_1:px_2, :])
                image_partial_resize = cv2.resize(image_partial, (128, 96), interpolation=cv2.INTER_LINEAR)
                image_partial_resize = image_partial_resize / 255
                # CONVERT TO TENSOR -- HISTORY
                pobs_hist = np.delete(pobs_hist, 0, axis=0)
                pobs_hist = np.concatenate([pobs_hist, np.expand_dims(image_partial_resize, axis=0)])
                d1 = torch.from_numpy(np.expand_dims(np.float32(pobs_hist), axis=0)).to(self.device)
                d2 = torch.from_numpy(np.expand_dims(np.expand_dims(np.float32(image_target_resize), axis=0), axis=0)).to(self.device)
                valid_history_length = step_i + 1
                # AGENT PREDICTION
                with torch.no_grad():
                    predict, attention_out = self.classifier_model(d1, d2)
                _, predict_argmax = torch.max(predict, dim=1)
                predict_argmax = predict_argmax.item()
                # ACTION CONVERT
                grid_x, grid_y, pixel_x, pixel_y = self.action_index_convert(index=predict_argmax)
                # ACTION REPLACEMENT
                agent_fixation_x = pixel_x
                agent_fixation_y = pixel_y
                agent_trajectory[0, step_i] = pixel_x
                agent_trajectory[1, step_i] = pixel_y
                # STORE ATTENTION
                attention_record.append([attention_out.cpu().numpy(), valid_history_length])
            agent_traj_list.append(agent_trajectory)
        with open(os.path.join('log_text', 'mapAttention_%s.pkl' % self.exp_name), 'wb') as outfile:
            pickle.dump(attention_record, outfile)
        return agent_traj_list


