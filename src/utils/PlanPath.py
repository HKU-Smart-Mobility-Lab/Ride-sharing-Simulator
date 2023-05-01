from ..component.Trip import Trip, Path
import random
import time
import numpy as np
import itertools as it



'''
The subsystem of the control center that handles requests, vehicles and trips
'''
class PlanPath:
    def __init__(self,
                environment,
                check_itinerary,
                method = 'NearestFirst'):
        
        self.environment = environment
        self.check_itinerary = check_itinerary
        self.method = method
        self.itinerary_method = self.environment.itinerary_method
        self.consider_congestion = self.environment.consider_congestion
        
        # If we assume vechiles move from the origin to destination along the straight line, we calculate the Euclidean distance
        if self.itinerary_method == 'straight':
            self.cal_dis_method = 'Linear'
        # Else we calculate the Manhattan distance to approximate the distance for the real road network
        else:
            self.cal_dis_method = 'Manhattan'

    # function: Plan the best route for a given trip and a vehicle according to the method
    # params: the vehicle and the trip
    # return: None if the vehicle can not take the trip due to constraints, or the path
    def PlanPath(self, vehicle, trip):
        # Check if the vehicle is still online or open to requests
        new_passenger_num = sum(request.num_person for request in trip.requests)
        next_passenger_num = sum(request.num_person for request in vehicle.next_requests)
        if vehicle.current_capacity + new_passenger_num + next_passenger_num > vehicle.max_capacity or vehicle.online == False or vehicle.open2request == False:
            return None
        
        # If the vehicle is idle and there is only one request in the trip, no need to check
        if vehicle.current_capacity + next_passenger_num == 0 and len(trip.requests) == 1:
            '''
                If we consider itinerary nodes when checking constraints of maximal pickup and detour time, the itinerary nodes of each possible trip are calculated in advance.
                If not, we check the constraints using Manhattan distance and calculate itinerary nodes after the optimal trip is given by ILP.
            '''
            if self.check_itinerary:
                iti1_nodes, iti1_dis, iti1_t = self.environment.GetItinerary(vehicle.current_position, trip.requests[0].pickup_position, method = self.itinerary_method)
                iti2_nodes, iti2_dis, iti2_t = trip.requests[0].iti_nodes, trip.requests[0].iti_dis, trip.requests[0].iti_t # There is only one request, we use the shortest route of the request directly
                d1, t1, d2, t2 = sum(iti1_dis), sum(iti1_t), sum(iti2_dis), sum(iti2_t)
                iti_nodes = iti1_nodes + iti2_nodes[1:]
                iti_dis = iti1_dis + iti2_dis
                iti_t = iti1_t + iti2_t
            else:
                d1, t1 = self.environment.GetDistanceandTime(vehicle.current_position, trip.requests[0].pickup_position, type = self.cal_dis_method)
                d2, t2 = trip.requests[0].original_travel_distance, trip.requests[0].original_travel_time
                iti_nodes, iti_dis, iti_t = [], [], [] # we do not calculate itinerary nodes here
            t_delay = np.zeros((2))
            t_delay[0] = t1 / trip.requests[0].max_con_pickup_time # pickup time delay, no travel time delay
            
            # Initialize the route of individual requests
            path = Path(current_position = vehicle.current_position,
                    next_positions = [trip.requests[0].pickup_position, trip.requests[0].dropoff_position],
                    time_needed_to_next_position = [t1, t2],
                    dis_to_next_position = [d1, d2],
                    time_delay_to_each_position = t_delay,
                    next_itinerary_nodes = iti_nodes,
                    dis_to_next_node = iti_dis,
                    time_needed_to_next_node = iti_t)
            
            return path
        
        '''
        If the passenger is not willing to share a vehicle with others, no need to check
        '''
        if vehicle.current_capacity + next_passenger_num == 0 and len(trip.requests) > 1:
            for req in trip.requests:
                if req.max_tol_num_person == 1:
                    return None
        
        # Choose a method
        if self.method == 'CompleteSearth':
            path = self.PlanPath_CompleteSearch(vehicle, trip)
        elif self.method == 'NearestFirst':
            path = self.PlanPath_NearestFirst(vehicle, trip)
        else:
            raise NotImplementedError
        
        return path


    # function: Plan the best route for a given trip and a vehicle (nearest position first)
    # params: the vehicle and the trip
    # return: None if the vehicle can not take the trip due to constraints, or the path
    def PlanPath_NearestFirst(self, vehicle, trip):
        
        # function: calculate the distance between the vehicle and requests
        def CalDis(current_position, requests, dis, dis_to_pos, NEXT_REQ = True):
            for req in requests:
                # Dropoff position
                Dd, _ = self.environment.GetDistanceandTime(current_position, req.dropoff_position, type = self.cal_dis_method)
                dis.append(Dd)
                dis_to_pos[Dd] = req.dropoff_position
                # pickup position
                if NEXT_REQ:
                    Dp, _ = self.environment.GetDistanceandTime(current_position, req.pickup_position, type = self.cal_dis_method)
                    dis.append(Dp)
                    dis_to_pos[Dp] = req.pickup_position
                    # If the Distance(vehicle, pickup_position) > Distance(vehicle, dropoff_position), we don't assign the request to the vehicle
                    if Dp > Dd:
                        return None, None
            return dis, dis_to_pos

        current_position = vehicle.current_position
        dis = []              # used to sort the distance
        dis_to_position = {}  # used to find the position according to the distance
        next_requests = list(set(vehicle.next_requests) | set(trip.requests))
        
        # Update the dis and dis_to_position
        # the next requests
        dis, dis_to_position = CalDis(current_position, next_requests, dis, dis_to_position)
        if dis is None:
            return None
        # the current requests of the vehicle
        if vehicle.current_capacity < len(vehicle.current_requests): # virtual request should not be considered
            cur_reqs = []
        else:
            cur_reqs = vehicle.current_requests
        
        dis, dis_to_position = CalDis(current_position, cur_reqs, dis, dis_to_position, False)

        # Set the positions' order
        next_positions = []
        dis.sort() # sort the distance
        for d in dis:    
            next_positions.append(dis_to_position[d])
        
        # Check the constraints
        MEET_CONSTRAINTS, path = self.CheckConstraints(next_positions, current_position, cur_reqs, next_requests)
        if not MEET_CONSTRAINTS:
            return None
        else:
            return path



    # function: Plan the best route for a given trip and a vehicle (minimum delay)
    # params: the vehicle and the trip
    # return: None if the vehicle can not take the trip due to constraints, or the path
    def PlanPath_CompleteSearch(self, vehicle, trip):
        
        # Search all possible paths
        all_possible_paths = self.SearchAllPossiblePath(vehicle.current_position, vehicle.current_requests, vehicle.next_requests, trip)
        # No possible path
        if len(all_possible_paths) == 0:
            return None
        # Choose the best path (minimum time delay)
        best_path = self.ChooseBestPath(all_possible_paths)
             
        return best_path


    # function: Enumerate all possible paths (under all constraints, i.e., pickup time, travel time, and etc.)
    # params: The current position of the vehicle, the currents requests and next reuqests of the vehicle
    # return: list[[positions], [], ...]
    # Note: Each path is represented by a position list
    def SearchAllPossiblePath(self, current_position, current_requests, next_requests, trip):
        path_all = []
        positions = []
        next_requests = list(set(next_requests) | set(trip.requests))
        
        # Integrate all next positions
        for request in current_requests:
            positions.append(request.dropoff_position)
        for request in next_requests:
            positions.append(request.pickup_position)
            positions.append(request.dropoff_position)
        positions = list(set(positions) | set([])) # Merge the same positions
        
        # Permutate all next positions
        positions_lists = it.permutations(positions, len(positions))
        for positions_list in positions_lists:
            
            # pickup position is front of dropoff position
            ORDER_CORRECT = True
            MEET_CONSTRAINTS = True
            
            for request in next_requests:
                if positions_list.index(request.pickup_position) >= positions_list.index(request.dropoff_position):
                    ORDER_CORRECT = False
                    break
            if not ORDER_CORRECT:
                continue
           
            # Check if the position list meets all constraints of each request
            MEET_CONSTRAINTS, path = self.CheckConstraints(positions_list, current_position, current_requests, next_requests)
            if not MEET_CONSTRAINTS:
                continue
            path_all.append(path)
        
        return path_all

    
    # function: Check if the position list meets the pickup time, travel time and travel distance constraints
    # params: The position list, the current position of the vehicle, the currents requests and next reuqests of the vehicle
    # return: bool value and the new path
    def CheckConstraints(self, pon_list, cur_pos, cur_reqs, next_reqs):
        assert len(pon_list) > 0
        MEET_CON = True
        
        time_needed_to_next_position = np.zeros((len(pon_list)))
        dis_to_next_position = np.zeros((len(pon_list)))
        time_delay_to_each_position = np.zeros((len(pon_list))) # Used to initialize the path of the trip
        
        # The first position
        distance, time = self.environment.GetDistanceandTime(cur_pos, pon_list[0], type = self.cal_dis_method)
        dis_to_next_position[0] = distance
        time_needed_to_next_position[0] = time
        # The next positions
        for idx in range(len(pon_list) - 1):
            distance, time = self.environment.GetDistanceandTime(pon_list[idx], pon_list[idx+1], type = self.cal_dis_method)
            dis_to_next_position[idx + 1] = distance
            time_needed_to_next_position[idx + 1] = time
        
        # For the current requests, we only need to check the travel time and distance
        for req in cur_reqs:
            # Get the index of the dropoff position in the position list
            pidx = pon_list.index(req.dropoff_position)
            total_travel_time = req.time_on_vehicle + np.sum(time_needed_to_next_position[:pidx+1])
            total_travel_distance = req.distance_on_vehicle + np.sum(dis_to_next_position[:pidx+1])
            
            # Check constraints of travel time and distance
            if total_travel_time > req.max_con_travel_time or total_travel_distance > req.max_con_travel_diatance:
                MEET_CON = False
                return MEET_CON, None
            
            time_delay_to_each_position[pidx] = max(0, (total_travel_time - req.original_travel_time) / req.MAX_DROPOFF_DELAY) # nomalization
            
        # For the next requests, we need to check pickup time, travel time and travel distance
        for req in next_reqs:
            # Get the index of the dropoff and pickup position in the position list
            try:
                pickup_idx = pon_list.index(req.pickup_position)
                dropoff_idx = pon_list.index(req.dropoff_position)
            except:
                return False, None
            # pickup position is front of dropoff position
            if pickup_idx >= dropoff_idx:
                return False, None
                #raise ValueError('pickup position is front of dropoff position')
            
            # Check pickup time
            pickup_time = np.sum(time_needed_to_next_position[:pickup_idx+1])
            if pickup_time > req.max_con_pickup_time:
                MEET_CON = False
                return MEET_CON, None
            time_delay_to_each_position[pickup_idx] = pickup_time / req.max_con_pickup_time  # nomalization
            
            # Check dropoff time
            total_travel_time = np.sum(time_needed_to_next_position[pickup_idx + 1: dropoff_idx + 1])
            total_travel_distance = np.sum(dis_to_next_position[pickup_idx + 1 : dropoff_idx + 1])
            # Check constraints of travel time and distance
            if total_travel_time > req.max_con_travel_time or total_travel_distance > req.max_con_travel_diatance:
                MEET_CON = False
                return MEET_CON, None
            
            time_delay_to_each_position[dropoff_idx] = max(0, (total_travel_time - req.original_travel_time) / req.MAX_DROPOFF_DELAY) # nomalization
        
        path = Path(current_position=cur_pos,
                    next_positions=list(pon_list),
                    time_needed_to_next_position=time_needed_to_next_position,
                    dis_to_next_position=dis_to_next_position,
                    time_delay_to_each_position=time_delay_to_each_position)
        
        return MEET_CON, path
    
    
    # function: Choose the best path of the canditate paths (minimum time delay)
    # params: the current position of the vehicle, all possible paths
    # return: the best path
    # Note: Here each path is represented by a position list
    def ChooseBestPath(self, all_possible_paths):
        # Calculate delay time of all trips
        min_time_delay = 99999
        best_path_idx = None
        for idx, path in enumerate(all_possible_paths):
            time_delay = np.sum(path.time_delay_to_each_position)
            if time_delay < min_time_delay:
                min_time_delay = time_delay
                best_path_idx = idx
        
        assert best_path_idx is not None    
            
        return all_possible_paths[best_path_idx]


    # function: Update the itinerary node list of path and the corresponding travel distance and time
    # params: path
    # return: updated path
    def UpdateItineraryNodes(self, path):
        # Get the itinerary node list
        itinerary_node_list = []
        dis_to_next_node, time_needed_to_next_node = [], []
        
        # First position
        nodes_tmp, dis_tmp, t_tmp = self.environment.GetItinerary(path.current_position, path.next_positions[0], self.itinerary_method)
        # '''
        #     If there is no road between two positions, we just drop it out
        # '''
        # if nodes_tmp is None:
        #     return False
        
        itinerary_node_list.extend(nodes_tmp)
        dis_to_next_node.extend(dis_tmp)
        time_needed_to_next_node.extend(t_tmp)
        
        # next positions
        for idx in range(1, len(path.next_positions)):
            nodes_tmp, dis_tmp, t_tmp = self.environment.GetItinerary(path.next_positions[idx-1], path.next_positions[idx], self.itinerary_method)
            
            # for ni, node in enumerate(nodes_tmp):
            #     if node != itinerary_node_list[-1]: # starting and ending nodes needed to be checked
            itinerary_node_list.extend(nodes_tmp[1:])
            dis_to_next_node.extend(dis_tmp)
            time_needed_to_next_node.extend(t_tmp)
        
        # Pop the first node (same as the current node)
        if path.current_position == itinerary_node_list[0]:
            itinerary_node_list = itinerary_node_list[1:]
        
        assert len(itinerary_node_list) == len(dis_to_next_node) and len(itinerary_node_list) == len(time_needed_to_next_node)
        
        # Update the path
        path.next_itinerary_nodes = itinerary_node_list
        path.time_needed_to_next_node = time_needed_to_next_node
        path.dis_to_next_node = dis_to_next_node

        return True
