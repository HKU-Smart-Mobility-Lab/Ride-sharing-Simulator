from array import array
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .PrioritizedMemory import Memory


# approximate Q function using Neural Network
# state is input and Q Value of each action is output of network
class MODEL_MLP(nn.Module):
    def __init__(self,
                cfg,
                total_grid_num,
                total_time_step):
        super(MODEL_MLP, self).__init__()
        
        self.cfg = cfg
        self.total_grid_num = total_grid_num
        self.total_time_step = total_time_step
        self.loc_embed_num = self.cfg.MODEL.LOCATION_EMBED_NUM
        self.time_embed_num = self.cfg.MODEL.TIME_EMBED_NUM
        self.max_capacity = self.cfg.VEHICLE.MAXCAPACITY


        self.path_input_dim = (2 * self.max_capacity + 1) * (self.loc_embed_num + 1)
        
        self.embedding1 = nn.Embedding(self.total_grid_num + 1, self.loc_embed_num)
        self.fc1 = nn.Sequential(
            nn.Linear(self.path_input_dim, 200),
            nn.Tanh()
        )

        self.embedding2 = nn.Embedding(total_time_step, self.time_embed_num)
        
        self.fc2 = nn.Sequential(
            nn.Linear(200+self.loc_embed_num+self.time_embed_num, 200),
            nn.Tanh(),
            nn.Linear(200,200),
            nn.Tanh()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(2*total_grid_num, 200),
            nn.Tanh()
        )
        
        self.fc4 = nn.Sequential(
            nn.Linear(200*2, 1)
        )

    # params
    # veh_grid_list: the current grid id list of the vehicle: batch_size * (2 * max_capacity + 1)
    # veh_t_delay: the delay time of each node in the current state
    # cur_loc: the current grid id of the vehicle
    # cur_t: the current timepoint of the simulation system
    # veh_dis: the distribution of vehicles in the previous 15 minutes
    # req_dis: the distribution of requests in the previous 15 minutes
    def forward(self, state):
        veh_grid_list, veh_t_delay, cur_loc, cur_t, veh_dis, req_dis = state
       
        batch_size = veh_grid_list.shape[0]
        '''matching'''
        path_emedb = self.embedding1(veh_grid_list) # batch_size * (2 * max_capacity + 1) * loc_embed_num
        path_ori_inp = torch.cat((path_emedb, veh_t_delay.unsqueeze(-1)), axis = -1)
        path_ori_inp = path_ori_inp.view(batch_size, -1)
        path_ori = self.fc1(path_ori_inp) # batch_size * 200
        

        # the current location's embbeding
        cur_loc_embed = self.embedding1(cur_loc).squeeze() # batch_size * loc_embed_num
        # the current time's embbeding
        cur_t_embed = self.embedding2(cur_t).squeeze() # batch_size * time_embed_num

        matching_input = torch.cat((path_ori, cur_loc_embed, cur_t_embed), axis = -1)
        m_inp = self.fc2(matching_input) # batch_size * 200

        '''repositioning'''
        veh_dis = veh_dis.view(batch_size, -1)
        req_dis = req_dis.view(batch_size, -1) # batch_size * total_grid_num
        repositioning_inp = torch.cat((veh_dis, req_dis), axis = 1) # batch_size * (2*total_grid_num)
        r_inp = self.fc3(repositioning_inp) # batch_size * 200
        
        '''combination'''
        inp = torch.cat((m_inp, r_inp), axis = 1).type(torch.float) # batch_size * 400

        value = self.fc4(inp) # batch_size * 1

        return m_inp


# CNN Model
class MODEL_CNN(nn.Module):
    def __init__(self,
                cfg):
        super(MODEL_CNN, self).__init__()

        self.cfg = cfg
        self.inp_layer_num = self.cfg.VEHICLE.MAXCAPACITY * 2 + 3
        self.t_inp = int((self.cfg.SIMULATION.END - self.cfg.SIMULATION.START) / self.cfg.MODEL.TIME_INTERVAL)

        self.conv1 = nn.Conv2d(self.inp_layer_num, 16, 3, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Sequential(
            nn.Linear(self.t_inp, 256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64*2*2 + 256, 1),
        )

        self._initialize_weights()
    
    # Initialize weights
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain = 0.01)
    
    
    def forward(self, states): # batch_size * 7 * 10 * 10
        state, t_onehot = states
        value = F.relu(self.bn1(self.conv1(state)), inplace = True)
        value = F.relu(self.bn2(self.conv2(value)), inplace = True)
        value = F.relu(self.bn3(self.conv3(value)), inplace = True)
        value = F.relu(self.bn4(self.conv4(value)), inplace = True)
        value = value.view(len(value), -1) # batch_size * 256
        t = self.fc1(t_onehot) # batch_size * 256
        value = torch.cat((value, t), axis = 1) # batch_size * 512
        value = self.fc2(value) # batch_size * 1

        return value


# Model consists of positions and time only
class MODEL(nn.Module):
    def __init__(self,
                cfg):
        super(MODEL, self).__init__()
        self.cfg = cfg
        self.layer_num = self.cfg.VEHICLE.MAXCAPACITY * 2 + 1
        self.p_num = self.cfg.ENVIRONMENT.CITY.X_GRID_NUM * self.cfg.ENVIRONMENT.CITY.Y_GRID_NUM
        self.t_num = int((self.cfg.SIMULATION.END - self.cfg.SIMULATION.START) / self.cfg.MODEL.TIME_INTERVAL)
        self.fc1_num = 100
        self.fc2_num = 200

        self.fc1 = nn.Sequential(
            nn.Linear(self.p_num + self.t_num, 100),
            nn.Tanh()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.layer_num * self.fc1_num, self.fc2_num),
            nn.Tanh()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.fc2_num, self.fc2_num),
            nn.Tanh()
        )
        self.fc4 = nn.Linear(self.fc2_num,1)

        self._initialize_weights()
    
    # Initialize weights
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain = 0.01)
    
    def forward(self, state): 
        pos, t = state
        batch_size = pos.shape[0]

        value = torch.cat((pos, t), axis = -1)  # batch_size * layer_num * (pos_num + t_num)
        value = self.fc1(value)                 # batch_size * layer_num * 100
        value = value.view(batch_size, -1)      # batch_size * (layer_num * 100)
        value = self.fc2(value)                 # batch_size * 200
        value = self.fc3(value)                 # batch_size * 200
        value = self.fc4(value)                 # batch_size * 1
        
        return value





# Agent for the Ride-pooling
# it uses Neural Network to approximate q function
# and prioritized experience replay memory & target q network
class Agent():
    def __init__(self, 
                cfg,
                total_grid_num,
                total_time_step):
        self.cfg = cfg
        self.total_grid_num = total_grid_num
        self.total_time_step = total_time_step

        # These are hyper parameters for the DQN
        self.discount_factor = self.cfg.MODEL.DISCOUNT_FACTOR
        self.learning_rate = self.cfg.MODEL.LEARNING_RATE
        self.memory_size = self.cfg.MODEL.MEMORY_SIZE
        self.batch_size = self.cfg.MODEL.BATCH_SIZE
        self.train_frequency = self.cfg.MODEL.TRAIN_FREQUENCY
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = 'cpu'
        self.MSELoss = nn.MSELoss(reduction = 'none')

        # create prioritized replay memory using SumTree
        self.memory = Memory(self.memory_size)

        # # create main model and target model
        # self.model = MODEL(cfg,
        #                 total_grid_num,
        #                 total_time_step)
        # self.model = self.model.to(self.device)
        # self.model.apply(self.weights_init)
        # self.target_model = MODEL(cfg,
        #                     total_grid_num,
        #                     total_time_step)
        # self.target_model = self.target_model.to(self.device)
        # # initialize target model
        # self.update_target_model()

        self.model = MODEL(cfg)
        self.target_model = MODEL(cfg)
        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)


    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Convert the components of the state to tensor and send them to the specific device
    def state2tensor(self, state, device = None):
        if device is None:
            device = self.device
        
        state_tensor = []
        for item in state:
            if not isinstance(item, array):
                item = np.array(item)
            if item.dtype == 'float64':
                item = item.astype(np.float32)
            item = torch.from_numpy(item)
            item = item.to(device)
            state_tensor.append(item)
        
        return state_tensor


    # Score the states during the simulation
    def get_value(self, state):
        state_tensor = self.state2tensor(state)
        #state_tensor = torch.from_numpy(state).to(self.device)
        value = self.model(state_tensor)
        value = value.detach().cpu().numpy()

        return value

    # Score the states using target model, which will be used as target when training the model
    def get_value_target(self, state):
        state_tensor = self.state2tensor(state)
        #state_tensor = torch.from_numpy(state).to(self.device)
        value = self.target_model(state_tensor)
        value = value.detach().cpu().numpy()

        return value


    # save sample (error,<s,a,r,s'>) to the replay memory
    def append_sample(self, states, scores_target, done):
        states_tensor = self.state2tensor(states, self.device)
        #states_tensor = torch.from_numpy(states).to(self.device)
        value = self.model(states_tensor).detach().cpu().numpy()
        
        # We use the mean TD Difference of all vehicles to calculate the priority of the experience
        error = np.mean(abs(value - scores_target) + 1e-6)

        self.memory.add(error, [states, scores_target, done])


    # Convert the samples to formats that can be handeled by the model
    def FormatSampleBatch(self, batch):
        # [[1,2],[3,4]] --> [[1,3], [2,4]]
        def TransposeList(batch):
            new_batch = [[] for _ in range(len(batch[0]))]
            for sample in batch:
                for i in range(len(sample)):
                    new_batch[i].append(sample[i])
            
            return new_batch
        
        # concatenate state batch
        def FormatState(state):
            for i in range(len(state)):
                item = np.array(state[i])
                state[i] = np.vstack(item)
            
            return state

        # states, scores_target, dones
        batch = TransposeList(batch)
        # veh_grid_list, veh_t_delay, cur_loc, cur_t, veh_dis, req_dis 
        batch[0] = FormatState(TransposeList(batch[0])) # states

        return batch


    # pick samples from prioritized replay memory (with batch_size)
    def train_model(self):
        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        #states, scores_target, dones = self.FormatSampleBatch(mini_batch)
        for batch_idx, (states, scores_target, dones) in enumerate(mini_batch):
            scores_target = np.array(scores_target, dtype = np.float32).reshape(-1, 1)
            # bool to binary
            dones = np.vstack(np.array(dones, dtype = np.float32))
            # for i in range(5):
            #     print(states[0][i])
            # Value of current states and next states
            states_tensor = self.state2tensor(states, self.device)
            #states_tensor = torch.from_numpy(states).to(self.device)
            pred = self.model(states_tensor)
            
            # Convert data format
            scores_target = torch.from_numpy(scores_target).to(self.device)
            dones = torch.from_numpy(dones).to(self.device)
            #is_weight = torch.from_numpy(is_weights[batch_idx] * self.cfg.VEHICLE.NUM).to(self.device)
            is_weight = is_weights[batch_idx] * self.cfg.VEHICLE.NUM
            
            errors = torch.abs(pred - scores_target).detach().cpu().numpy()
            #errors = errors.reshape(self.batch_size, -1)
            errors = np.mean(errors, axis = 0)
            # update priority
            idx = idxs[batch_idx]
            self.memory.update(idx, errors)

            # MSE Loss function
            loss = self.MSELoss(pred, scores_target)
            loss = is_weight * loss
            loss = loss.mean()
            #loss = loss.sum()
            #print('pred: {}, target: {}, is_weight: {}, loss: {}'.format(pred, scores_target, is_weight, loss))
            
            self.optimizer.zero_grad()
            #loss = (is_weights * F.mse_loss(pred, target, reduction = 'none').view(self.batch_size,-1).mean(axis = 0)).mean()
            loss.backward()
            # and train
            self.optimizer.step
        
        #print(loss)
        
        # Update target_model based on the learned model
        self.update_target_model()