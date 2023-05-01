import random
import numpy as np
from ..component.Trip import  Path
from docplex.mp.model import Model 
from random import choice
import copy

'''
The subsystem of the control center that simulates actions and manages all vehicles
The parameters of all vehicles are changed here
'''
class Reposition:
    def __init__(self,
                environment,
                method):
       
        self.environment = environment
        self.method = method
        
        self.x_grid_num = self.environment.x_grid_num
        self.y_grid_num = self.environment.y_grid_num

        # If we assume vechiles move from the origin to destination along the straight line, we calculate the Euclidean distance
        if self.environment.consider_itinerary and self.environment.itinerary_method == 'straight':
            self.cal_dis_method = 'Linear'
        # Else we calculate the Manhattan distance
        else:
            self.cal_dis_method = 'Manhattan'
    
    
    # function: Repositioning idle vehicles according to the given method
    def Reposition(self, vehicles):
        if self.method == 'Random':
            self.RepositioningRandomly(vehicles)
        elif self.method == 'NearGrid':
            self.Repositioning2NearGrids(vehicles)
        elif self.method == 'HotGrid':
            self.Repositioning2HotGrid(vehicles)
        elif self.method == 'HotNode':
            self.Repositioning2HotNode(vehicles)
        else:
            raise NotImplementedError


    # function: Manage idle vehicles to another node randomly (within 10 kms)
    # params: The vehicle needed to be relocated
    # return: Actions of idle vehicles or updated vehicles
    def RepositioningRandomly(self, vehicles):
        for vehicle in vehicles:
            while True:
                node = self.environment.nodes_coordinate[int(random.random()*len(self.environment.nodes_coordinate))] # Coordinate
                if node != vehicle.current_position: # Repositioning to another position
                    break
            dis, time = self.environment.GetDistanceandTime(node, vehicle.current_position, type = self.cal_dis_method)
            
            # Initialize path
            path = Path(current_position=vehicle.current_position,
                        next_positions=[vehicle.current_position, node],
                        time_needed_to_next_position=np.array([0,time]),
                        dis_to_next_position=np.array([0,dis]),
                        time_delay_to_each_position=np.zeros((2)))           
            
            # Update the vehicle's path
            vehicle.path = path



    # function: Manage idle vehicles to a grid nearby
    # params: The vehicle needed to be relocated
    # return: Actions of idle vehicles or updated vehicles
    def Repositioning2NearGrids(self, vehicles):
        for vehicle in vehicles:
            reposition_locations = self.GetRepositionLocation(vehicle.current_position)
            loc_idx = int(random.random() * len(reposition_locations))
            lng, lat, pickup_grid_id, dropoff_grid_id, distance, time = reposition_locations[loc_idx]
            
            # Initialize path
            path = Path(current_position=vehicle.current_position,
                        next_positions=[vehicle.current_position, (lng, lat)],
                        time_needed_to_next_position=np.array([0,time]),
                        dis_to_next_position=np.array([0,distance]),
                        time_delay_to_each_position=np.zeros((2)))
            
            # Update the vehicle's path
            vehicle.path = path


    '''
    Here we record the distribtion of vehicles and requests in the previous 30 mins and all time,
    and combine the two distributions to guide repositioning and matching.
    Specifically, we regard the diifference between the average number of requests and vehicles of each grid as the future reward.
    '''
    # function: repositioning idle vehicles to hot grids nearby (not only near 8 grids but also further grids)
    # params: vehicles needed to be relocated
    # return: None
    # Note: max_grid_num = 2 means there are (2 + 1 + 2)^2 - 1 = 24 candidate grids
    def Repositioning2HotGrid(self, vehicles, alpha = 0.0, max_x_grid_num = 4, max_y_grid_num = 4):
        req_dis, req_dis_all = self.environment.requests_distribution.GetDistribution()
        veh_dis, veh_dis_all = self.environment.vehicles_distribution.GetDistribution()
        # the relevantly hot areas
        r_dis = alpha * (req_dis - veh_dis) + (1 - alpha) * (req_dis_all - veh_dis_all)
        for vehicle in vehicles:
            target_x, target_y = None, None
            # Get the grid n.o. of the vehicle's current position
            try:
                vx, vy = self.environment.node_coord_to_grid[vehicle.current_position]
            except:
                vx, vy = self.environment.Coord2Grid(vehicle.current_position)

            # the range of candidate grids
            xmin, xmax = max(0, vx - max_x_grid_num), min(self.x_grid_num-1, vx + max_x_grid_num)
            ymin, ymax = max(0, vy - max_y_grid_num), min(self.y_grid_num-1, vy + max_y_grid_num)
            
            # Get the reward matrix
            r_mat = r_dis[ymin : ymax+1, xmin : xmax+1]
            # If no reward is more than 0, the vehicle just stays put
            # if np.max(r_mat) <= 0:
            #     continue
            
            # Get the repositioning probability of each grid
            p_mat = np.exp(r_mat) / np.sum(np.exp(r_mat), axis = (0,1)) # softmax
            
            # Sample a grid according to the probability matrix
            ranp = random.random()
            gridp = 0
            for row in range(len(p_mat)):
                for col in range(len(p_mat[0])):
                    gridp += p_mat[row, col]
                    if ranp < gridp:
                        target_x = xmin + col
                        target_y = ymin + row
                        break
                if target_x:
                    break
            
            # Get the center node the target grid and travel distance and time
            info = self.GetCenterNode(target_x, target_y, vehicle.current_position)
            if info is not None:
                lng, lat, r_grid, distance, time = info
                # Initialize path
                path = Path(current_position=vehicle.current_position,
                            next_positions=[vehicle.current_position, (lng, lat)],
                            time_needed_to_next_position=np.array([0,time]),
                            dis_to_next_position=np.array([0,distance]),
                            time_delay_to_each_position=np.zeros((2)))
                
                # Update the vehicle's path
                vehicle.path = path



    # function: Manage idle vehicles to hot areas
    # params: requests at the current time step or predicted demand at next time step
    # return: Actions of idle vehicles or updated vehicles
    '''
    This repositioning approach is provided by the paper: Neural Approximate Dynamic Programming for On-Demand Ride-Pooling,
    which will be compared with our joint RL optimization approach
    '''
    def Repositioning2HotNode(self, vehicles):
        # If there haven't been requests, we don't reposition idle vehicles
        if len(self.environment.past_requests) == 0:
            return
        
        # Get a list of possible targets by sampling from recent_requests
        possible_targets = []
        num_targets = min(500, len(vehicles))
        for _ in range(num_targets):
            target = choice(self.environment.past_requests)
            possible_targets.append(target)

        # Solve an LP to assign each vehicle to closest possible target
        model = Model()

        # Define variables, a matrix defining the assignment of vehicles to targets
        assignments = model.continuous_var_matrix(range(len(vehicles)), range(len(possible_targets)), name='assignments')

        # Make sure one vehicle can only be assigned to one target
        for vehicle_id in range(len(vehicles)):
            model.add_constraint(model.sum(assignments[vehicle_id, target_id] for target_id in range(len(possible_targets))) == 1)

        # Make sure one target can only be assigned to *ratio* vehicles
        num_fractional_targets = len(vehicles) - (int(len(vehicles) / num_targets) * num_targets)
        for target_id in range(len(possible_targets)):
            num_vehicles_to_target = int(len(vehicles) / num_targets) + (1 if target_id < num_fractional_targets else 0)
            model.add_constraint(model.sum(assignments[vehicle_id, target_id] for vehicle_id in range(len(vehicles))) == num_vehicles_to_target)

        # Define the objective: Minimise distance travelled
        model.minimize(model.sum(assignments[vehicle_id, target_id] * self.environment.GetDistanceandTime(vehicles[vehicle_id].current_position, possible_targets[target_id].pickup_position, type = self.cal_dis_method)[1] for target_id in range(len(possible_targets)) for vehicle_id in range(len(vehicles))))

        # Solve
        solution = model.solve()
        assert solution  # making sure that the model doesn't fail

        # Get the assigned targets
        for vehicle_id, vehicle in enumerate(vehicles):
            for target_id, target in enumerate(possible_targets):
                if (solution.get_value(assignments[vehicle_id, target_id]) == 1):
                    # Initialize path
                    distance, time = self.environment.GetDistanceandTime(vehicle.current_position, target.pickup_position, type = self.cal_dis_method)
                    path = Path(current_position=vehicle.current_position,
                                next_positions=[vehicle.current_position, target.pickup_position],
                                time_needed_to_next_position=np.array([0,time]),
                                dis_to_next_position=np.array([0,distance]),
                                time_delay_to_each_position=np.zeros((2)))
                    # Update the vehicle's path
                    vehicle.path = path
                    
                    break


    # function: get 8 (or less 8) repositioning locations nearby according to the vehicle's location
    # params: the vehicle's location
    # return: repositioning coordinates, grid coordinates, repositioning distance and time
    def GetRepositionLocation(self, vehicle_location, method = 'NearGrid'):
        reposition = []
        
        try:
            vx, vy = self.environment.node_coord_to_grid[vehicle_location]
        except:
            vx, vy = self.environment.Coord2Grid(vehicle_location)
        
        v_grid = vy * self.x_grid_num + vx # current(vehicle) grid id

        # reposition idle vehicles to a near grid
        if method == 'NearGrid':
            grids = [(vy-1, vx), (vy-1, vx+1), (vy, vx+1), (vy+1, vx+1), (vy+1, vx), (vy+1, vx-1), (vy, vx-1), (vy-1, vx-1)]
            for (ry, rx) in grids:
                # Check the bound
                if ry >=0 and ry < self.y_grid_num and rx >= 0 and rx < self.x_grid_num:
                    info = self.GetCenterNode(rx, ry, vehicle_location)
                    if info is not None:
                        lng, lat, r_grid, distance, time = info
                        reposition.append((lng, lat, v_grid, r_grid, distance, time))
        
        # Reposition idle vehicles to a hot area
        # We choose 8 candidate hot areas according to the previous requests, and we also choose 2 random areas for exploration
        elif method == 'HotGrid':
            # If there is no requests, we don't reposition idle vehicles
            requests_distribution,_ = self.environment.requests_distribution.GetDistribution()
            if np.max(requests_distribution) == 0:
                return reposition
            
            requests_distribtion = copy.deepcopy(self.requests_distribution).reshape(-1)
            # # 8 hot areas
            # pos_idx = np.argpartition(requests_distribtion, -8)[-8:] # The index of 8 hot areas
            # pos_idx = list(pos_idx)
            
            pos_idx_hot = [33, 34, 35, 43, 44, 45, 53, 54, 67]
            pos_idx = random.sample(pos_idx_hot, 2)
            # 2 random areas
            num, cnt = 0, 0 
            while cnt < 80:
                idx = random.randint(0,99)
                cnt += 1
                if idx not in pos_idx_hot and requests_distribtion[idx] > 1:
                    ry, rx = int(idx / self.x_grid_num), idx % self.x_grid_num
                    if abs(rx - vx) < 3 and abs(ry - vy) < 3: # grids not too far from the current grid
                        num += 1
                        pos_idx.append(idx)
                if num >= 3:
                    break

            # Get info
            for idx in pos_idx:
                if requests_distribtion[idx] <= 1:
                    continue
                ry, rx = int(idx / self.x_grid_num), idx % self.x_grid_num
                info = self.GetCenterNode(rx, ry, vehicle_location)
                if info is not None:
                    lng, lat, r_grid, distance, time = info
                    reposition.append((lng, lat, v_grid, r_grid, distance, time))        
        
        else:
            raise NotImplementedError
        
        return reposition 
    
    
    # function: Get the center node given a grid
    # params: the grid coordinate and the vehicle's current position
    # return: center node, target grid n.o, and travel distance and time
    def GetCenterNode(self, rx, ry, vehicle_location):
        # Choose the center node in the grid for repositioning
        center_node = self.environment.grid_center_node[ry, rx]
        # Make sure there exists a node in the grid and the vehicle is not at the center node
        if center_node is not None and center_node != vehicle_location:
            distance, time = self.environment.GetDistanceandTime(vehicle_location, center_node, type = self.cal_dis_method)
            r_grid = ry * self.x_grid_num + rx # repositioning grid id
            return [center_node[0], center_node[1], r_grid, distance, time]
        else:
            return None