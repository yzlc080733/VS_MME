import random
import numpy as np
from tqdm import tqdm


'''
    METHOD RANDOM UNIFORM
'''

class RandomUniform:
    def __init__(self, size_X, size_Y):
        self.size_X = size_X;  self.size_Y = size_Y
    
    def train(self, **kwargs):      # RANDOM, TRAINING NOT NEEDED
        pass

    def evaluate(self, traj_list, **kwargs):        # OTHER INPUTS NOT NEEDED
        agent_traj_list = []
        for data_i in tqdm(range(len(traj_list)), desc='RAND', leave=False):
            agent_traj = np.zeros_like(traj_list[data_i])   # SHAPE==[2, N_STEP]
            for step_i in range(agent_traj.shape[1]):
                random_X = random.randint(0, self.size_X - 1)
                random_Y = random.randint(0, self.size_Y - 1)
                agent_traj[0, step_i] = random_X
                agent_traj[1, step_i] = random_Y
            agent_traj_list.append(agent_traj)
        return agent_traj_list


