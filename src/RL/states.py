import numpy as np
import math

'''Object of the data distribution including requests and vehicles'''
class Distribution():
    '''parameters:
    type: the type of distribution (requests or vehicles)
    step_time: The step time of the simulation system
    record_time: The previous time needed to be recorded
    x_grid_num: the horizontal grid number
    y_grid_num: the vertical grid number
    '''
    def __init__(self,
                type,
                step_time,
                record_time,
                x_grid_num,
                y_grid_num,
                node_coord_to_grid,
                area_box):
        self.type = type
        self.record_steps = int(record_time / step_time)
        self.current_step = 0
        self.y_grid_num = y_grid_num
        self.x_grid_num = x_grid_num
        self.node_coord_to_grid = node_coord_to_grid
        self.area_box = area_box

        self.distribution = []


    # Update the distribution
    def Update(self, data_list):
        dis_tmp = np.zeros((self.y_grid_num, self.x_grid_num))
        for data in data_list:
            # Update each data
            if self.type == 'requests':
                coord = data.pickup_position
            else:
                coord = data.current_position
            try:
                (x_num, y_num) = self.node_coord_to_grid[coord]
            except:
                x_num, y_num = Coord2Grid(coord, self.area_box, self.x_grid_num, self.y_grid_num)
            
            dis_tmp[y_num, x_num] += 1
        
        if self.current_step < self.record_steps:
            self.distribution.append(dis_tmp)
        else:
            self.distribution[self.current_step % self.record_steps] = dis_tmp
        
        self.current_step += 1
        
    
    # Get the mean distribution in the previous record time
    def GetDistribution(self):
        if len(self.distribution) == 0:
            return np.zeros((self.y_grid_num, self.x_grid_num), dtype = np.float32)
        
        dis = np.array(self.distribution, dtype = np.float32)
        dis = np.sum(dis, axis = 0) # 10 * 10
        
        #dis = (dis - np.mean(dis)) / (np.std(dis) + 1e-6)
        dis = dis / (np.max(dis) + 1e-6)
        return dis


'''objective of states'''
class States():
    def __init__(self,
                cfg,
                environment,
                requests_record_time = 1800,
                vehicle_record_time = None):
        self.cfg = cfg
        self.environment = environment
        self.node_coord_to_grid = self.environment.node_coord_to_grid
        self.area_box = self.environment.area_box
        
        #self.nodes_coordinate_grid = nodes_coordinate_grid
        self.x_grid_num = self.cfg.ENVIRONMENT.CITY.X_GRID_NUM
        self.y_grid_num = self.cfg.ENVIRONMENT.CITY.Y_GRID_NUM
        self.step_time = self.cfg.SIMULATION.STEP_TIME
        
        # We input the mean distribution of the requests in the previous 30 minutes
        self.requests_distribution = Distribution(type = 'requests',
                                                step_time = self.step_time,
                                                record_time = requests_record_time,
                                                x_grid_num = self.x_grid_num,
                                                y_grid_num = self.y_grid_num,
                                                node_coord_to_grid = self.node_coord_to_grid,
                                                area_box=self.area_box)
        # We input the current distribution of the vehicles
        self.vehicles_distribution = Distribution(type = 'vehicles',
                                                step_time = self.step_time,
                                                record_time = self.step_time,
                                                x_grid_num = self.x_grid_num,
                                                y_grid_num = self.y_grid_num,
                                                node_coord_to_grid = self.node_coord_to_grid,
                                                area_box=self.area_box)
    
    # Get the vehicles' states: veh_grid_list, veh_t_delay, cur_loc
    def Vehicles2States(self, vehicles):
        max_capacity = self.cfg.VEHICLE.MAXCAPACITY
        
        veh_grid_list = np.zeros((len(vehicles), 2*max_capacity+1), dtype = np.int64)
        veh_t_delay = np.zeros((len(vehicles), 2*max_capacity+1), dtype = float)
        cur_loc = np.ones((len(vehicles), 1), dtype = np.int64)

        for idx, vehicle in enumerate(vehicles):
            # current position
            cur_loc[idx] = self.node_coord_to_grid_id(vehicle.current_position)
            veh_grid_list[idx, 0] = self.node_coord_to_grid_id(vehicle.current_position)
            
            if vehicle.path is not None:
                # time delay
                time_delay = vehicle.path.time_delay_to_each_position
                veh_t_delay[idx, 1:len(time_delay)+1] = time_delay
                # grid list
                next_positions = vehicle.path.next_positions
                for ip, pos in enumerate(next_positions):
                    veh_grid_list[idx, ip+1] = self.node_coord_to_grid_id(pos)
        
        assert veh_grid_list.any() >=0 and veh_grid_list.any() <= self.x_grid_num * self.y_grid_num
        
        return [veh_grid_list, veh_t_delay, cur_loc]
    

    # Convert grid coordinate to grid id
    def node_coord_to_grid_id(self, coord):
        try:
            (x_num, y_num) = self.node_coord_to_grid[coord]
        except:
            x_num, y_num = Coord2Grid(coord, self.area_box, self.x_grid_num, self.y_grid_num)
        
        grid_id = int(y_num * self.x_grid_num + x_num + 1)
        
        return grid_id

    
    # Get states
    def GetStates_MLP(self, vehicles, step):
        states = []
        states_veh = self.Vehicles2States(vehicles)
        states.extend(states_veh)

        cur_t = math.floor(step * self.step_time / self.cfg.MODEL.TIME_INTERVAL)
        cur_t = np.ones((len(vehicles), 1), dtype = np.int64) * cur_t
        veh_dis = self.vehicles_distribution.GetDistribution()
        veh_dis = np.repeat(veh_dis.reshape(1, veh_dis.shape[0], veh_dis.shape[1]), len(vehicles), axis = 0)
        req_dis = self.requests_distribution.GetDistribution()
        req_dis = np.repeat(req_dis.reshape(1, req_dis.shape[0], req_dis.shape[1]), len(vehicles), axis = 0)
        
        states.extend([cur_t, veh_dis, req_dis])
        
        return states
    

    # Get states for CNN
    def GetStates_CNN(self, vehicles, step):
        # Get distribution of vehicles and requests
        veh_dis = np.zeros((len(vehicles), 1, self.y_grid_num, self.x_grid_num), dtype = np.float32)
        req_dis = np.zeros((len(vehicles), 1, self.y_grid_num, self.x_grid_num), dtype = np.float32)
        veh_dis[:, 0] = self.vehicles_distribution.GetDistribution()
        req_dis[:, 0] = self.requests_distribution.GetDistribution()
        
        # Get location lists
        veh_grids = np.zeros((len(vehicles), 2*self.cfg.VEHICLE.MAXCAPACITY+1, self.y_grid_num, self.x_grid_num), dtype = np.float32)
        for veh_idx, vehicle in enumerate(vehicles):
            # current position
            try:
                (x_num, y_num) = self.node_coord_to_grid[vehicle.current_position]
            except:
                # if the position is not located at a intersection, we calculate the grid location
                x_num, y_num = Coord2Grid(vehicle.current_position, self.area_box, self.x_grid_num, self.y_grid_num)
            
            veh_grids[veh_idx, 0, y_num, x_num] = 1.
            # next positions
            if vehicle.path is not None:
                next_positions = vehicle.path.next_positions
                for ip, pos in enumerate(next_positions):
                    try:
                        (x_num, y_num) = self.node_coord_to_grid[pos]
                    except:
                        x_num, y_num = Coord2Grid(pos, self.area_box, self.x_grid_num, self.y_grid_num)
                    veh_grids[veh_idx, ip+1, y_num, x_num] = 1.
        
        # Concatenate
        state = np.concatenate((veh_grids, req_dis, veh_dis), axis = 1) # len(vehicles) * 5 * 10 * 10

        # Get time
        cur_t = math.floor(step * self.step_time / self.cfg.MODEL.TIME_INTERVAL)
        t_num = int((self.cfg.SIMULATION.END - self.cfg.SIMULATION.START) / self.cfg.MODEL.TIME_INTERVAL)
        t_onehot = np.zeros((len(vehicles), t_num), dtype = np.float32)
        t_onehot[:, cur_t] = 1.0
        
        states = [state, t_onehot]
        
        return states


    def GetStates(self, vehicles, step):
        # parameters
        layer_num = self.cfg.VEHICLE.MAXCAPACITY * 2 + 1
        p_num = self.cfg.ENVIRONMENT.CITY.X_GRID_NUM * self.cfg.ENVIRONMENT.CITY.Y_GRID_NUM
        t_num = int((self.cfg.SIMULATION.END - self.cfg.SIMULATION.START) / self.cfg.MODEL.TIME_INTERVAL)
        # Initialize position and time
        position = np.zeros((len(vehicles), layer_num, p_num), dtype = np.float32)
        time = np.zeros((len(vehicles), layer_num, t_num), dtype = np.float32)
        
        # current time
        t_idx = math.floor(step * self.step_time / self.cfg.MODEL.TIME_INTERVAL)
        time[:, 0, t_idx] = 1.0
        
        # Get positions
        for idx, vehicle in enumerate(vehicles):
            # current position
            x_num, y_num = Coord2Grid(vehicle.current_position, self.area_box, self.x_grid_num, self.y_grid_num)
            p_idx = y_num * self.x_grid_num + x_num
            position[idx, 0, p_idx] = 1.0
            
            if vehicle.path is not None:
                node_idx1 = 0
                t = 0
                # next positionsand time points
                next_positions = vehicle.path.next_positions
                for ip, pos in enumerate(next_positions):
                    # positions
                    x_num, y_num = Coord2Grid(pos, self.area_box, self.x_grid_num, self.y_grid_num)
                    p_idx = y_num * self.x_grid_num + x_num
                    position[idx, ip+1, p_idx] = 1.
                    
                    # # time points
                    # try:
                    #     node_idx2 = vehicle.path.next_itinerary_nodes.index(pos, node_idx1)
                    #     t = sum(vehicle.path.time_needed_to_next_node[0:node_idx2+1])
                    # except:
                    if ip == 0:
                        _, t = self.environment.GetDistanceandTime(vehicle.current_position, pos, type = 'Manhattan')
                    else:
                        t += self.environment.GetDistanceandTime(next_positions[ip-1], pos, type = 'Manhattan')[1]
                    
                    t_idx = math.floor((step * self.step_time + t)/ self.cfg.MODEL.TIME_INTERVAL)
                    if t_idx > 47:
                        t_idx = 47
                    time[idx, ip+1, t_idx] = 1.0
                    #node_idx1 = node_idx2
                    
        
        return position, time


'''
If we assume drivers move from origin to destination along the straight line, then itinerary nodes are not located at intersections.
Therefore, we need to recalculate grid N.O. of each position
'''
def Coord2Grid(coord, area_box, x_grid_num, y_grid_num):
    lng_min, lng_max, lat_min, lat_max = area_box
    delta_x = (lng_max - lng_min) / x_grid_num
    delta_y = (lat_max - lat_min) / y_grid_num
    lng, lat = coord
    
    x_num = math.floor((lng - lng_min) / delta_x)
    y_num = math.floor((lat - lat_min) / delta_y)
    if x_num >= 10 :
        x_num = 9
    if y_num >= 10:
        y_num = 9
    
    return x_num, y_num
