import numpy as np
import random
import argparse
import time
import os
import torch
import pickle
# CUSTOM LIB
import env_dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DESCRIPTION')
    parser.add_argument('--dataset', type=str, default='asd-1-social')
    parser.add_argument('--valsplit', type=int, default=1)
    parser.add_argument('--method', type=str, default='MEPPO-0-0.0-1.0-10.0-snn-0.1-0.0001-0.0001')
    parser.add_argument('--rep', type=int, default=61)
    parser.add_argument('--thread', type=int, default=5)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    EXP_NAME = '%s_%d_%s_%02d' % (args.dataset, args.valsplit, args.method, args.rep)

    # DEVICE SETTINGS
    if args.cuda == -2:     # NUMPY ONLY
        torch_device = None
    elif args.cuda == -1:
        torch_device = torch.device('cpu')
    elif args.cuda < 0:
        raise Exception('/// ERROR -- args.cuda')
    else:
        torch_device = torch.device('cuda:%d' % args.cuda)
    if args.thread < 0:
        pass
    else:
        torch.set_num_threads(args.thread)


    # INITIALIZE DATASET, TRAIN-VALID SPLIT
    dataset_info = args.dataset.split('-')
    if dataset_info[0] == 'asd':
        if dataset_info[1] in ['All', 'Amygdala', 'ASD', 'ASD_Ctrl', 'NUS']:
            dataset_variant = [dataset_info[1]]
        else:
            dataset_variant = [int(dataset_info[1])]
        dataset_variant.append(dataset_info[2])
    if dataset_info[0] == 'cocosearch18':
        if dataset_info[1] == 'All':
            dataset_variant = ['all']
        else:
            dataset_variant = [int(dataset_info[1])]
    if dataset_info[0] == 'ivsnarray':
        if dataset_info[1] == 'All':
            dataset_variant = ['all']
        else:
            dataset_variant = [int(dataset_info[1])]
    env = env_dataset.Dataset(dataset_name=dataset_info[0], dataset_variant=dataset_variant, split_valid=args.valsplit, split_seed=args.rep, device=torch_device)
    print('--> DATANUM SPLIT', env.data_num_total, env.train_num, env.valid_num)
    
    
    # METHOD SETTING
    if args.method == 'randuniform':
        import method_rand_uniform
        model = method_rand_uniform.RandomUniform(size_X=env.img_size_X, size_Y=env.img_size_Y)
    elif args.method[0:6] == 'BClstm':
        import method_BC_LSTM
        method_variant_lstm_length = int(args.method.split('-')[1])
        method_variant_lstm_size = int(args.method.split('-')[2])
        model = method_BC_LSTM.BClstm(size_X=env.img_size_X, size_Y=env.img_size_Y,device=torch_device, exp_name=EXP_NAME, lstm_step_num=method_variant_lstm_length, lstm_size=method_variant_lstm_size)
    elif args.method[0:5] == 'MEPPO':
        import method_MEPPO
        method_variant_usebasis         = int(args.method.split('-')[1])
        method_variant_randomness       = float(args.method.split('-')[2])
        method_variant_maxamplitude     = float(args.method.split('-')[3])
        method_variant_revisitpenalty   = float(args.method.split('-')[4])
        method_variant_nettype          = str(args.method.split('-')[5])
        method_variant_entropy_ratio    = float(args.method.split('-')[6])
        method_variant_lr_reward        = float(args.method.split('-')[7])
        method_variant_lr_agent         = float(args.method.split('-')[8])
        if method_variant_nettype == 'ann':
            flag_SVPG_enabled = False
        elif method_variant_nettype == 'snn':
            flag_SVPG_enabled = True
        else:
            raise Exception('Err method_variant_nettype ', method_variant_nettype)
        model = method_MEPPO.MEPPO(size_X=env.img_size_X, size_Y=env.img_size_Y,
                                   size_obs_radius=env.observation_radius,
                                   device=torch_device, exp_name=EXP_NAME,
                                   v_basis=method_variant_usebasis, v_rand=method_variant_randomness, v_amp=method_variant_maxamplitude, v_pel=method_variant_revisitpenalty,
                                   entropy_ratio=method_variant_entropy_ratio, svpg_enabled=flag_SVPG_enabled,
                                   lr_agent=method_variant_lr_agent, lr_reward=method_variant_lr_reward)
    elif args.method == 'BCtransformer':
        import method_BC_TRAN
        model = method_BC_TRAN.BCtransformer(size_X=env.img_size_X, size_Y=env.img_size_Y, device=torch_device, exp_name=EXP_NAME)
    else:
        raise Exception('/// ERROR args.method')
    

    # TRAIN
    model.train(img_search_list=env.train_img_search, img_target_list=env.train_img_cue, traj_list=env.train_traj, target_bbox_list=env.train_target_bbox)
    

    # EVALUATE
    agent_traj = model.evaluate(img_search_list=env.valid_img_search, img_target_list=env.valid_img_cue, traj_list=env.valid_traj, target_bbox_list=env.valid_target_bbox)
    human_traj = env.valid_traj
    traj_index = env.valid_universal_id
    
    for t_i in range(len(agent_traj)):
        print('%9d -------------' % (t_i))
        print(agent_traj[t_i])
        print(human_traj[t_i])
    with open(os.path.join('./log_text/agenttraj_%s.pkl' % EXP_NAME), 'wb') as outfile:
        pickle.dump([agent_traj, human_traj, env.img_size_X, env.img_size_Y, traj_index], outfile)
    

    print('FINISHED')

