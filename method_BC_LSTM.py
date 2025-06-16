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



'''
    METHOD BEHAVIOR CLONING.
    LSTM ENABLED
'''



class Classifier_Net(nn.Module):
    def __init__(self, action_dim=48, lstm_size=80):
        super(Classifier_Net, self).__init__()
        self.action_dim = action_dim
        self.lstm_size = lstm_size
        
        self.conv2_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=6, stride=4), nn.ReLU(), nn.MaxPool2d(kernel_size=2) )
        self.conv2_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=4, stride=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2) )
        self.conv3_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=6, stride=4), nn.ReLU(), nn.MaxPool2d(kernel_size=2) )
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=4, stride=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2) )
        
        self.flatten2 = nn.Flatten(start_dim=1)
        self.flatten3 = nn.Flatten(start_dim=1)
        self.lstm_layer = nn.LSTM(input_size=384, hidden_size=self.lstm_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(384 + self.lstm_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, action_dim)

    def forward(self, i_obs, i_cue, history_feature):
        # CONV
        x2 = self.conv2_2(self.conv2_1(i_obs))
        x3 = self.conv3_2(self.conv3_1(i_cue))
        x2 = self.flatten2(x2)
        x3 = self.flatten3(x3)
        # LSTM
        x2 = torch.unsqueeze(x2, dim=1)
        x2, (h_new, c_new) = self.lstm_layer(x2, history_feature)
        x2 = torch.squeeze(x2, dim=1)
        # FC
        x = torch.cat([x2, x3], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, (h_new, c_new)

    def save_model(self, name_fix=''):
        torch.save(self.state_dict(), './log_model/' + name_fix + '_best.pt')

    def load_model(self, name_fix=''):
        self.load_state_dict(torch.load('./log_model/' + name_fix + '_best.pt'))



class BClstm:
    def __init__(self, size_X, size_Y, device=torch.device('cpu'), exp_name='', lstm_step_num=0, lstm_size=10):
        # SEARCH IMAGE SIZE
        self.size_X = size_X;  self.size_Y = size_Y
        # AGENT ACTION GRID
        self.grid_size_X = 8
        self.grid_size_Y = 6
        self.action_num = self.grid_size_X * self.grid_size_Y
        # PARTIAL OBSEARVATION SIZE
        self.partial_obs_size_X = round(self.size_X * 0.25)
        self.partial_obs_size_Y = round(self.size_Y * 0.25)
        # TRAINING SETTING
        self.batch_size = 16
        self.device = device
        self.exp_name = exp_name
        # LSTM SETTING
        self.lstm_size = lstm_size
        self.lstm_step_num = lstm_step_num		# NOT USED
        # INITIALIZATION
        self.init_model()
        self.init_log_file()

    def init_model(self):       # MODEL AND OPTIMIZER
        self.classifier_model = Classifier_Net(lstm_size=self.lstm_size).to(device=self.device)
        self.optimizer = torch.optim.Adam(self.classifier_model.parameters(), 0.001)
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
        # EXTRACT DATA
        data_partial_obs_list = []
        data_target_image_list = []
        data_human_fixation_list = []
        for traj_i in tqdm(range(len(traj_list)), leave=False, desc='EXTRACT DATA'):
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
        
        # TRAIN
        train_index_list = list(range(len(traj_list)))
        for train_i in range(500):
            point_num = 0
            correct_num = 0
            batch_list = train_index_list
            loss_value = 0
            for traj_i in batch_list:
                DATA_1 = torch.from_numpy(np.moveaxis(data_partial_obs_list[traj_i], 3, 1)).to(self.device)
                DATA_2 = torch.from_numpy(np.moveaxis(data_target_image_list[traj_i], 3, 1)).to(self.device)
                LABEL = torch.from_numpy(data_human_fixation_list[traj_i]).to(self.device)
                LABEL_argmax = torch.argmax(LABEL, dim=1)
                for step_i in range(DATA_1.shape[0]):
                    d1 = DATA_1[step_i:(step_i+1)]
                    d2 = DATA_2
                    if step_i == 0:
                        predict, (h, c) = self.classifier_model(d1, d2, history_feature=None)
                    else:
                        predict, (h, c) = self.classifier_model(d1, d2, history_feature=(h, c))
                    _, predict_argmax = torch.max(predict, dim=1)
                    predict_onehot = F.one_hot(predict_argmax.detach(), num_classes=self.action_num)
                    truth = LABEL[step_i:(step_i+1), :]
                    if torch.sum(torch.abs(predict_onehot - truth)) == 0:
                        correct_num = correct_num + 1
                    loss_value = loss_value + self.loss_function(predict, LABEL_argmax[step_i:(step_i+1)])
                    point_num = point_num + 1
            # OPTIMIZE
            train_accuracy = correct_num / point_num
            loss_value = loss_value / point_num
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()
            record_text = '%10d, %10.8f' % (train_i, loss_value.detach().cpu().numpy())
            self.log_text(self.File, 'train', record_text=record_text)
        self.classifier_model.save_model(name_fix=self.exp_name)


    def evaluate_human(self, img_search_list, img_target_list, traj_list, **kwargs):
        # EVALUATION BASED ON HUMAN FIXATIONS
        # EXTRACT DATA
        data_partial_obs_list = []
        data_target_image_list = []
        data_human_fixation_list = []
        for traj_i in tqdm(range(len(traj_list)), leave=False, desc='EXTRACT DATA VAL'):
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
        # VALIDATION
        train_index_list = list(range(len(traj_list)))
        agent_traj_list = []
        self.classifier_model.load_model(name_fix=self.exp_name)
        for traj_i in train_index_list:
            agent_trajectory = np.zeros_like(traj_list[traj_i])
            DATA_1 = torch.from_numpy(np.moveaxis(data_partial_obs_list[traj_i], 3, 1)).to(self.device)
            DATA_2 = torch.from_numpy(np.moveaxis(data_target_image_list[traj_i], 3, 1)).to(self.device)
            for step_i in range(DATA_1.shape[0]):
                d1 = DATA_1[step_i:(step_i+1)]
                d2 = DATA_2
                if step_i == 0:
                    predict, (h, c) = self.classifier_model(d1, d2, history_feature=None)
                else:
                    predict, (h, c) = self.classifier_model(d1, d2, history_feature=(h, c))
                _, predict_argmax = torch.max(predict, dim=1)
                predict_argmax = predict_argmax.item()
                grid_x, grid_y, pixel_x, pixel_y = self.action_index_convert(index=predict_argmax)
                agent_trajectory[0, step_i] = pixel_x
                agent_trajectory[1, step_i] = pixel_y
            agent_traj_list.append(agent_trajectory)
        return agent_traj_list


    def evaluate(self, img_search_list, img_target_list, traj_list, **kwargs):
        agent_traj_list = []
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
            for step_i in range(traj_length):
                # PARTIAL OBSERVATION
                px_1 = agent_fixation_x - self.partial_obs_size_X + s2
                px_2 = agent_fixation_x + self.partial_obs_size_X + s2
                py_1 = agent_fixation_y - self.partial_obs_size_Y + s1
                py_2 = agent_fixation_y + self.partial_obs_size_Y + s1
                image_partial = np.copy(image_search_expand[py_1:py_2, px_1:px_2, :])
                image_partial_resize = cv2.resize(image_partial, (128, 96), interpolation=cv2.INTER_LINEAR)
                image_partial_resize = image_partial_resize / 255
                # CONVERT TO TENSOR
                d1 = torch.from_numpy(np.expand_dims(np.moveaxis(np.float32(image_partial_resize), 2, 0), axis=0)).to(self.device)
                d2 = torch.from_numpy(np.expand_dims(np.moveaxis(np.float32(image_target_resize), 2, 0), axis=0)).to(self.device)
                # AGENT PREDICTION
                with torch.no_grad():
                    if step_i == 0:
                        predict, (h, c) = self.classifier_model(d1, d2, history_feature=None)
                    else:
                        predict, (h, c) = self.classifier_model(d1, d2, history_feature=(h, c))
                _, predict_argmax = torch.max(predict, dim=1)
                predict_argmax = predict_argmax.item()
                # ACTION CONVERT
                grid_x, grid_y, pixel_x, pixel_y = self.action_index_convert(index=predict_argmax)
                # ACTION REPLACEMENT
                agent_fixation_x = pixel_x
                agent_fixation_y = pixel_y
                agent_trajectory[0, step_i] = pixel_x
                agent_trajectory[1, step_i] = pixel_y
            agent_traj_list.append(agent_trajectory)
        return agent_traj_list


