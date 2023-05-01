import numpy as np
import math


'''Object of the data distribution including requests and vehicles'''
class HotDistribution():
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
                area_box):
        self.type = type
        self.record_steps = int(record_time / step_time)
        self.current_step = 0
        self.y_grid_num = y_grid_num
        self.x_grid_num = x_grid_num
        self.area_box = area_box

        self.distribution = []
        self.distribution_all = np.zeros((self.y_grid_num, self.x_grid_num), dtype = np.float32)


    # Update the distribution
    def Update(self, data_list):
        dis_tmp = np.zeros((self.y_grid_num, self.x_grid_num))
        for data in data_list:
            # Update each data
            if self.type == 'requests':
                coord = data.pickup_position
            elif self.type == 'vehicles':
                # We only count vehicles that are not full
                if data.current_capacity + sum(req.num_person for req in data.next_requests) >= data.max_capacity:
                    continue
                coord = data.current_position
            else:
                raise NotImplementedError
            
            x_num, y_num = self.Coord2Grid(coord, self.area_box, self.x_grid_num, self.y_grid_num)
            
            dis_tmp[y_num, x_num] += 1
            self.distribution_all[y_num, x_num] += 1
        
        if self.current_step < self.record_steps:
            self.distribution.append(dis_tmp)
        else:
            self.distribution[self.current_step % self.record_steps] = dis_tmp
        
        self.current_step += 1

    
    '''
    If we assume drivers move from origin to destination along the straight line, then itinerary nodes are not located at intersections.
    Therefore, we need to recalculate grid N.O. of each position
    '''
    # Calculate grid n.o. of node given the longtitude and latitude
    def Coord2Grid(self, coord, area_box, x_grid_num, y_grid_num):
        lng_min, lng_max, lat_min, lat_max = area_box
        delta_x = (lng_max - lng_min) / x_grid_num
        delta_y = (lat_max - lat_min) / y_grid_num
        lng, lat = coord
        
        x_num = math.floor((lng - lng_min) / delta_x)
        y_num = math.floor((lat - lat_min) / delta_y)
        if x_num >= x_grid_num :
            x_num = x_grid_num - 1
        if y_num >= y_grid_num:
            y_num = y_grid_num - 1
        
        return x_num, y_num


    # Get the mean distribution in the previous record time
    def GetDistribution(self):
        if len(self.distribution) == 0:
            return np.zeros((self.y_grid_num, self.x_grid_num), dtype = np.float32), self.distribution_all
        
        dis = np.array(self.distribution, dtype = np.float32)
        dis = np.sum(dis, axis = 0) # 10 * 10
        
        dis /= self.record_steps # mean value
        dis_all = self.distribution_all / self.current_step

        return dis, dis_all
