from .component.Trip import Trip, Path
from .utils.Reposition import Reposition
from .component.Vehicle import Vehicle
from .component.Request import Request
from .utils.PlanPath import PlanPath
from .component.VirtualRequest import VirtualRequest
import random
import heapq
from tqdm import tqdm
import numpy as np
import pandas as pd

'''
The subsystem of the control center that handles requests, vehicles and trips
'''
class RTVSystem:
    def __init__(self,
                cfg,
                environment
            ):
        self.cfg = cfg
        self.environment = environment
        # simulation time
        self.step_time = self.cfg.SIMULATION.STEP_TIME
        self.start_timepoint = self.cfg.SIMULATION.START
        self.end_timepoint = self.cfg.SIMULATION.END
        self.current_timepoint = self.start_timepoint
        
        # The target area
        self.lng_min, self.lng_max = cfg.ENVIRONMENT.MINLNG, cfg.ENVIRONMENT.MAXLNG
        self.lat_min, self.lat_max = cfg.ENVIRONMENT.MINLAT, cfg.ENVIRONMENT.MAXLAT


        self.consider_itinerary = self.cfg.ENVIRONMENT.CONSIDER_ITINERARY.TYPE
        # To accelerate the the simulation process, we don't consider itinerary nodes when we chenk constraints
        self.check_itinerary = self.cfg.REQUEST.CHECK_ITINERARY

        self.total_steps = int((self.end_timepoint - self.start_timepoint) / self.step_time + 1)

        self.PlanPath = PlanPath(environment=environment,
                                check_itinerary=self.check_itinerary,
                                method=self.cfg.VEHICLE.PlanPathMethod)
        
        self.reposition = Reposition(environment = environment,
                                    method = self.cfg.VEHICLE.REPOSITION.METHOD)
        # If we assume vechiles move from the origin to destination along the straight line, we calculate the Euclidean distance
        if self.consider_itinerary and self.environment.itinerary_method == 'straight':
            self.cal_dis_method = 'Linear'
        # Else we calculate the Manhattan distance
        else:
            self.cal_dis_method = 'Manhattan'
        
        # Initialize requests and vehicles
        # self.requests_all = self.InitializeRequests()
        # self.vehicles_all = self.InitializeVehicles()


    # function: Initialize all requests
    # params: dataset director or database API
    # return: list[request]
    # Note: If no dataset is provided, the function is supposed to generate requests according to a specific process 
    def InitializeRequests(self, request_data_dir = None, pooling_rate = 0.):
        #print('Initialize requests')
        requests_all = [[] for _ in range(self.total_steps)]
        num_requests = 0
        avg_trip_dis = 0
        # We load requests from the pickle file
        # Note: the data is not only used for our ride-pooling simulator, but other simulators.
        # Therefore, we need to convert the data from the pickle file to our request format
        if request_data_dir:
            '''
            request_columns = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                                'trip_distance', 'timestamp','start_time', 'date', 'origin_grid_id','dest_grid_id', 'itinerary_node_list',
                                'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob', 'fare']
            trip_distance: consider itinerary, unit: km
            itinerary_segment_dis_list: distance between each pair of itinerary nodes
            '''
            #requests_raw = pickle.load(open(request_data_dir, 'rb')) # dict
            with open(request_data_dir, 'rb') as f:
                requests_raw = pd.read_csv(f)
            for idx in tqdm(range(len(requests_raw)), desc = 'Initialize requests'):   
                #for request_raw in requests_raw[idx]:
                # Calculate the corresponding step
                # Note: the start_time's unit is second. e.g., 100 means 00:01:40 and 3600 means 01:00:00
                timepoint = requests_raw['start_time'][idx]
                # The request is not within the target tiem interval
                # Note: We assume that there are no new requests in the last 30 mins to complete the simulation
                if timepoint < self.start_timepoint or timepoint > self.end_timepoint:
                    continue
                # we filter the trip whose distance is less 1 km
                if requests_raw['trip_distance'][idx] < 0.5:
                    continue
                
                # We sample requests according to the given sampling rate
                if random.random() > self.cfg.REQUEST.SAMPLE_RATE:
                    continue
                
                # split the road network
                expand = 0
                lng_u, lat_u, lng_v, lat_v = requests_raw['origin_lng'][idx], requests_raw['origin_lat'][idx], requests_raw['dest_lng'][idx], requests_raw['dest_lat'][idx]
                if not (lng_u >= self.lng_min - expand and lng_u <= self.lng_max + expand and lat_u >= self.lat_min - expand and lat_u <= self.lat_max + expand
                and lng_v >= self.lng_min - expand and lng_v <= self.lng_max + expand and lat_v >= self.lat_min - expand and lat_v <= self.lat_max + expand):
                    continue
                
                # Assign each request to a simulation step
                step = round((timepoint - self.start_timepoint) / self.step_time)
                # pick-up and drop-off position (longitude, latitude)
                # Make sure to keep 7 decimal palces
                pickup_position = (round(requests_raw['origin_lng'][idx], 7), round(requests_raw['origin_lat'][idx], 7))
                dropoff_position = (round(requests_raw['dest_lng'][idx], 7), round(requests_raw['dest_lat'][idx], 7))
                # pickup_position = self.environment.GetNearestNode((request_raw[3], request_raw[2]))
                # dropoff_position = self.environment.GetNearestNode((request_raw[6], request_raw[5]))
                
                if self.consider_itinerary:
                    # We use Euclidean distance here
                    if self.cfg.ENVIRONMENT.CONSIDER_ITINERARY.METHOD == 'straight':
                        iti_nodes, iti_dis, iti_t = self.environment.GetItinerary(pickup_position, dropoff_position, method = 'straight') # original trajectory
                        travel_distance, travel_time = sum(iti_dis), sum(iti_t) # original travel distance and time
                    # We consider the itinerary and sum all distances between each pair of nodes
                    else:
                        # Read trajectory from the request file
                        iti_nodes = requests_raw['itinerary_node_list'][idx]
                        iti_nodes = [int(itm) for itm in iti_nodes.strip('[').strip(']').split(', ')]
                        iti_dis = requests_raw['itinerary_segment_dis_list'][idx]
                        iti_dis = [float(itm) * 1000 for itm in iti_dis.strip('[').strip(']').split(', ')]
                        iti_t = [itm / self.cfg.VEHICLE.VELOCITY for itm in iti_dis]
                        travel_distance, travel_time = sum(iti_dis), sum(iti_t) # original travel distance and time
                        
                # We use Manhattan distance here to approximate real itinerary distance
                else:
                    travel_distance, travel_time = self.environment.GetDistanceandTime(pickup_position, dropoff_position, type = 'Manhattan')
                
                # Initialize requests
                request = Request(cfg = self.cfg,
                                id = requests_raw['order_id'][idx],
                                send_request_timepoint = step * self.step_time + self.start_timepoint,
                                pickup_position = pickup_position,
                                dropoff_position = dropoff_position, # We use (lng, lat) to represent position
                                pickup_grid_id = self.environment.node_coord_to_grid[pickup_position],
                                dropoff_grid_id = self.environment.node_coord_to_grid[dropoff_position],
                                iti_nodes = iti_nodes, # original trajectory without considering ride-pooling and traffic congestion
                                iti_dis = iti_dis,
                                iti_t = iti_t,
                                original_travel_time = travel_time,
                                original_travel_distance = travel_distance,
                                num_person = 1)
                '''
                Define the maximal number of passengers that can be toleranted according to the pooling rate.
                If the passenger is willing to share a vehicle with other passengers, 
                the maximal number of passengers that can be toleranted equals the maximal capacity of vehicles
                '''
                if random.random() < pooling_rate:
                    request.max_tol_num_person = self.cfg.VEHICLE.MAXCAPACITY

                requests_all[step].append(request)
                
                num_requests += 1
                avg_trip_dis += travel_distance
                    
                
        
        # If there is no request file, we generate requests randomly
        # Here, we create 2 hot areas (3,3), (6,6)
        else:
            num_requests = 5000 # We will generate 60 requests randomly
            try: 
                # Load the generated requests
                with open('gen-'+ str(num_requests) + '-req.csv', 'rb') as f:
                    reqs = pd.read_csv(f)
                for id in reqs['id']:
                    # Initial the request
                    request = Request(cfg = self.cfg,
                                    id = id,
                                    send_request_timepoint = reqs['send_request_timepoint'][id],
                                    pickup_position = (reqs['pickup_position_lng'][id], reqs['pickup_position_lat'][id]),
                                    dropoff_position = (reqs['dropoff_position_lng'][id], reqs['dropoff_position_lat'][id]),
                                    original_travel_time = reqs['original_travel_time'][id],
                                    original_travel_distance = reqs['original_travel_distance'][id],
                                    num_person = 1)
                    step = int((reqs['send_request_timepoint'][id] - self.start_timepoint) / self.step_time)
                    requests_all[step].append(request)

            except:
                '''We save the generated requests that will be used for training and simulation'''
                reqs = {'id':[], 'send_request_timepoint':[], 'pickup_position_lng':[], 'pickup_position_lat':[], 'dropoff_position_lng':[],
                        'dropoff_position_lat':[], 'original_travel_time':[], 'original_travel_distance':[], 'num_person': []}
                
                for request_id in range(num_requests):
                    # the status of all requests are random
                    step = int(random.random() * (self.end_timepoint - self.start_timepoint) / self.step_time) 
                    send_request_timepoint = self.start_timepoint + step * self.step_time
                    # We assume that there is a high probability (e.g., 0.8) that a request occurs in one of hot aeras
                    prob = random.random()
                    if prob < 0.30:
                        coord_list = self.environment.nodes_coordinate_grid[3,3]
                        pickup_position = coord_list[int(random.random() * len(coord_list))]
                    elif prob < 0.60:
                        coord_list = self.environment.nodes_coordinate_grid[6,6]
                        pickup_position = coord_list[int(random.random() * len(coord_list))]
                    # elif prob < 0.30:
                    #     coord_list = self.environment.nodes_coordinate_grid[1,8]
                    #     pickup_position = coord_list[int(random.random() * len(coord_list))]
                    else:
                        pickup_position = self.environment.nodes_coordinate[int(random.random() * len(self.environment.nodes_coordinate))]
                    # Dropoff positions of requests are assumed randomly
                    dropoff_position =  self.environment.nodes_coordinate[int(random.random() * len(self.environment.nodes_coordinate))]
                    
                    # Filter the requests that have same pickup and dropoff positions
                    while dropoff_position == pickup_position:
                        dropoff_position = self.environment.nodes_coordinate[int(random.random() * len(self.environment.nodes_coordinate))]
                    
                    travel_distance, travel_time = self.environment.GetDistanceandTime(pickup_position, dropoff_position, type = self.cal_dis_method)
                    
                    # Initial the request
                    request = Request(cfg = self.cfg,
                                id = request_id,
                                send_request_timepoint = send_request_timepoint,
                                pickup_position = pickup_position,
                                dropoff_position = dropoff_position,
                                original_travel_time = travel_time,
                                original_travel_distance = travel_distance,
                                num_person = 1)
                    requests_all[step].append(request)

                    # append the data
                    reqs['id'].append(request_id)
                    reqs['send_request_timepoint'].append(send_request_timepoint)
                    reqs['pickup_position_lng'].append(pickup_position[0])
                    reqs['pickup_position_lat'].append(pickup_position[1])
                    reqs['dropoff_position_lng'].append(dropoff_position[0])
                    reqs['dropoff_position_lat'].append(dropoff_position[1])
                    reqs['original_travel_time'].append(travel_time)
                    reqs['original_travel_distance'].append(travel_distance)
                    reqs['num_person'].append(1)
                
                # store the data
                pd.DataFrame(reqs).to_csv('gen-' + str(num_requests) + '-req.csv', index = False, header = True)

            # # Demo case   
            # req1 = Request(id = 0,
            #                 send_request_timepoint = 0,
            #                 pickup_position = 19,
            #                 dropoff_position = 7,
            #                 original_travel_time = self.environment.GetTravelTime(19, 7),
            #                 original_travel_distance = self.environment.GetTravelDistance(19, 7),
            #                 num_person = 1)
            # req2 = Request(id = 1,
            #                 send_request_timepoint = 0,
            #                 pickup_position = 23,
            #                 dropoff_position = 8,
            #                 original_travel_time = self.environment.GetTravelTime(23, 8),
            #                 original_travel_distance = self.environment.GetTravelDistance(23, 8),
            #                 num_person = 1)
            # requests_ll[0].append(req1)
            # requests_ll[0].append(req2)
        
        return requests_all, num_requests, avg_trip_dis/num_requests



    '''
        If we consider traffic congerstion, that is, the traffic speed and travel time change from time to time, we need to initialize requests online,
        which means we need to recalculate the trajectories, travel distance, and travel time of requests in each batch before matching vehicles and requests.
        It should be noted that the maximal detour time in calculated based on the travel time at current time step.
    '''
    # function: Initialize requests in each batch
    # params: requests need to be assigned in this batch, update_all = False means we only update the new requests in this batch
    # return: updated requests
    def IniReqsBatch(self, reqs, update_all = False):
        for i in len(reqs):
            if (reqs[i].send_request_timepoint < self.current_timepoint and update_all) or reqs[i].send_request_timepoint >= self.current_timepoint:
                iti_nodes, iti_dis, iti_t = self.environment.GetItinerary(reqs[i].pickup_position, reqs[i].dropoff_position, method = 'API') # updated trajectory
                # Update requests
                reqs[i].UpdateRoute(iti_nodes, iti_dis, iti_t)
        
        return reqs



    # function: Initialize all vehicles
    # params: dataset director or database API
    # return: list[vehicle]
    # Note: If no dataset is provided, the function is supposed to generate vehicles according to a specific distuibution
    def InitializeVehicles(self, vehicle_data_dir = None, num_vehicles = 1000, requests = None):
        vehicles_all = []

        # We load requests from the pickle file
        # Note: (1) the data is not only used for our ride-pooling simulator, but other simulators.
        # Therefore, we need to convert the data from the pickle file to our vechile format
        # (2) There are so many vehicles in the pickle file (i.e., 20,000 vehicles) that out pc may not be able to run them all.
        # Therefore, we need to downsample vehicles (e.g., 1,000 vehicles)
        if vehicle_data_dir:
            '''
            self.driver_columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising', 'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list',
                               'node_id', 'grid_id']
            '''
            # vehicles_raw = pickle.load(open(vehicle_data_dir, 'rb'))
            with open(vehicle_data_dir, 'rb') as f:
                vehicles_raw = pd.read_csv(f)
            # Downsample vehicles
            num_vehilces_raw = len(vehicles_raw)
            ds_gap = int(num_vehilces_raw / num_vehicles)
            vehicle_id = 0
            for idx in tqdm(range(0, num_vehilces_raw, ds_gap), desc = 'Initialize vehicles'):
                # Initialize vehicles
                # We allocate vehicles to nearest intersections
                #current_position = self.environment.GetNearestNode((vehicles_raw['lng'][idx], vehicles_raw['lat'][idx]))
                current_position = (round(vehicles_raw['lng'][idx], 7), round(vehicles_raw['lat'][idx], 7))
                vehicle = Vehicle(cfg=self.cfg,
                                id = vehicles_raw['driver_id'][idx],
                                current_position = current_position, # We use coordinate to represent position
                                current_grid_id = vehicles_raw['grid_id'][idx],
                                start_time = vehicles_raw['start_time'][idx],
                                end_time = vehicles_raw['end_time'][idx],
                                online = True,
                                open2request = True)
                vehicles_all.append(vehicle)

        # If there is no vehicle file, we generate vehicles according to the distribution of requests
        else:
            try:
                 # Load the generated vehicles
                with open('gen-'+ str(num_vehicles) + '-veh.csv', 'rb') as f:
                    vehs = pd.read_csv(f)
                for id in vehs['id']:
                    # Initial the vehicles
                    current_position = (round(vehs['lng'][id], 7), round(vehs['lat'][id], 7))
                    veh = Vehicle(id = id,
                                cfg = self.cfg,
                                current_position = current_position,
                                start_time = 0,
                                end_time = 999999,
                                online = True,
                                open2request = True)
                    vehicles_all.append(veh)

            except:
                gen_veh = {'id':[], 'lng':[], 'lat':[]}
                # the distribution of requests
                req_dis = []
                for reqs in requests:
                    req_dis.extend(reqs)
                
                for vehicle_id in range(num_vehicles):
                    
                    # current_position = self.environment.nodes_coordinate[int(random.random() * len(self.environment.nodes_coordinate))]
                    '''
                    Generally, We can generate uniformly distributed vehicles in the area of interest among all the grids.
                    However, we can also generate vehciles according to the ditribution of requests
                    '''
                    # Generate vehicles according to the distribution of requests
                    idx = int(random.random() * len(req_dis))
                    current_position = req_dis[idx].pickup_position
                    
                    

                    # # Generate grid id randomly
                    # x, y = random.randint(0, 9), random.randint(0, 9)
                    # nodes_list = self.environment.nodes_coordinate_grid[y, x]
                    # if isinstance(nodes_list, list): # Make sure there exists a node in the grid
                    #     current_position = nodes_list[int(random.random() * len(nodes_list))]
                    #     if current_position[1] < 1.33*current_position[0] + 139.2 and current_position[1] > 1.33*current_position[0] + 139.115:
                    #         break
                    
                    veh = Vehicle(id = vehicle_id,
                                cfg = self.cfg,
                                current_position = current_position,
                                start_time = 0,
                                end_time = self.cfg.SIMULATION.END,
                                online = True,
                                open2request = True)
                    vehicles_all.append(veh)
                    
                    # gen_veh['id'].append(vehicle_id)
                    # gen_veh['lng'].append(current_position[0])
                    # gen_veh['lat'].append(current_position[1])
                
                # store the data
                # pd.DataFrame(gen_veh).to_csv('gen-' + str(num_vehicles) + '-veh.csv', index = False, header = True)
            # # Demo case   
            # veh = Vehicle(id = 0,
            #                 current_position = 19,
            #                 start_time = 0,
            #                 end_time = 999999,
            #                 online = True,
            #                 open2request = True,
            #                 max_capacity = 4)
            # vehicles_all.append(veh)

        return vehicles_all



    # function: Allocate each request to 30 nearest vehicles
    # params: 1) requests of the current step; 2) vehicles of the current step; 
    #         3) The maximum number of vehicles that a request will be allocated; 4) The maximum distance between a vehicle and the allocated request
    # return: list[list[request]], length = num_vehicles
    def AllocateRequest2Vehicles(self, requests_step, vehicles_step, max_num_vehicles = 10, max_match_distance = None):
        if max_match_distance is None:
            max_match_distance = 999999
        
        max_num_vehicles = min(len(vehicles_step), max_num_vehicles)
        requests_for_each_vehicle = [[] for _ in range(len(vehicles_step))] # list[list[request]], length = num_vehicles

        for request in requests_step:
            # Calculte the pickup time and distance for all vehicles to pick the request
            # In order to accelerate calculation, we only assign a request to vehilces nearby
            for vehicle_idx, vehicle in enumerate(vehicles_step):
                # Check if the vehicle is online and open to requests
                if not vehicles_step[vehicle_idx].online or not vehicles_step[vehicle_idx].open2request:
                    continue
                # Requests that are not willing to share a vehicle with others should be assigned to idle vehicles only
                if request.max_tol_num_person == 1 and len(vehicles_step[vehicle_idx].current_requests) + len(vehicles_step[vehicle_idx].next_requests) > 0:
                    continue
                # Check the constraints
                # todo ...
                if self.check_itinerary:
                    pass
                # We use Manhattan distance to check the constraints of maxiaml pickup and detour time to accelarate the process
                else:
                    dis, t = self.environment.GetDistanceandTime(vehicle.current_position, request.pickup_position, type = self.cal_dis_method)

                if dis < max_match_distance and t < request.max_con_pickup_time:
                    requests_for_each_vehicle[vehicle_idx].append(request)
        
        # We associate at most 5 requests for each vehicle to accelerate the simulation   
        # for idx, reqs in enumerate(requests_for_each_vehicle):
        #     if len(reqs) > self.cfg.VEHICLE.MAXCAPACITY + 2:
        #         requests_for_each_vehicle[idx] = requests_for_each_vehicle[idx][:self.cfg.VEHICLE.MAXCAPACITY + 1]

        return requests_for_each_vehicle


    # function: Generate feasible trips
    # params: vehicles for the current time step, request batch generated by the GetRequestBatch function
    # return: list[list[trip]], list[list[path]], length = num_vehicles
    def GenerateFeasibleTrips(self, vehicles_step, requests_for_each_vehicle, MAX_IS_FEASIBLE_CALLS = 150, MAX_TRIPS = 30):
        # Get feasible trips for each vehicle
        feasible_trips = []
        feasible_paths = []
        for requests_for_vehicle, vehicle in zip(requests_for_each_vehicle, vehicles_step):
            trips = []
            paths = []
            tested_trips_requests = []
            num_is_feasible_calls = 0
            
            # append a null trip that means the vehicle is assigned nothing
            trips.append(Trip())
            paths.append(Path())
            
            # If the vehicle is empty and still online, but there is no requests nearby, then we consider repositioning the vehicle
            if len(requests_for_vehicle) == 0 and vehicle.path is None and vehicle.online and vehicle.open2request:
                # Considering repositioning in the RL model
                if self.cfg.MODEL.REPOSITION.TYPE and not self.cfg.VEHICLE.REPOSITION.TYPE:
                    # Repositioning idle vehicles to 8 (or less) grids nearby
                    reposition_locations = self.reposition.GetRepositionLocation(vehicle.current_position, method = self.cfg.MODEL.REPOSITION.METHOD)
                    for rep_loc in reposition_locations:
                        lng, lat, pickup_grid_id, dropoff_grid_id, distance, time = rep_loc
                        # Initialize request
                        virtual_request = VirtualRequest(pickup_position = vehicle.current_position,
                                                        dropoff_position = (lng, lat),
                                                        pickup_grid_id = pickup_grid_id,
                                                        dropoff_grid_id = dropoff_grid_id,
                                                        original_travel_time = time,
                                                        original_travel_distance = -distance * 0.1)
                        reposition_trip = Trip(virtual_request)
                        # Initialize path
                        reposition_path = Path(current_position = vehicle.current_position,
                                                next_positions = [vehicle.current_position, (lng, lat)],
                                                time_needed_to_next_position = np.array([0, time]),
                                                dis_to_next_position = np.array([0, distance]),
                                                time_delay_to_each_position = np.zeros((2)))
                        
                        trips.append(reposition_trip)
                        paths.append(reposition_path)
                    
                    feasible_trips.append(trips)
                    feasible_paths.append(paths)
                    continue
            
            # No trip when repositioning or delivering passengers
            if len(requests_for_vehicle) == 0:
                feasible_trips.append(trips)
                feasible_paths.append(paths)
                continue

            # Check feasibility for individual requests
            for request in requests_for_vehicle:
                # If there exists requests nearby, we stop the repositioning process
                # if vehicle.current_capacity < len(vehicle.current_requests):
                #     vehicle.Status2Idle()
                
                trip = Trip(request)

                path = self.PlanPath.PlanPath(vehicle, trip) # Note: Any parameters of the vehicle should not be changed at this fuction
              
                if path is not None:
                    # print(path.current_position)
                    # print(path.next_itinerary_nodes)
                    trips.append(trip)
                    paths.append(path)

                tested_trips_requests.append(trip.requests)
                num_is_feasible_calls += 1
            
            # Non-ride-pooling
            if self.cfg.VEHICLE.MAXCAPACITY == 1:
                feasible_trips.append(trips)
                feasible_paths.append(paths)
                continue
            
            # We use the average travel distance of each request in the trips to determine the trip priority
            def TripPriority(trip):
                assert len(trip.requests) > 0
                return -sum(request.original_travel_distance for request in trip.requests) / len(trip.requests)
            
            # Get feasible trips of size > 1, with a fixed budget of MAX_IS_FEASIBLE_CALLS
            trips_tobe_combined = [(TripPriority(trip), trip_idx+1) for trip_idx, trip in enumerate(trips[1:])]
            heapq.heapify(trips_tobe_combined) # convert list to heap
            
            while len(trips_tobe_combined) > 0 and num_is_feasible_calls < MAX_IS_FEASIBLE_CALLS:
                _, trip_heap_idx = heapq.heappop(trips_tobe_combined) # pop the trip with maximum average travel distance
                
                for trip_list_idx in range(1, len(trips)):
                    pre_requests = trips[trip_heap_idx].requests
                    new_requests = trips[trip_list_idx].requests
                    combined_trip = Trip(list(set(pre_requests) | set(new_requests)))
                   
                    # We judge if the combined trip has been tested through the requests
                    if combined_trip.requests not in tested_trips_requests:
                        
                        path = self.PlanPath.PlanPath(vehicle, combined_trip)
                        
                        if path is not None:
                            trips.append(combined_trip)
                            paths.append(path)
                            heapq.heappush(trips_tobe_combined, (TripPriority(combined_trip), len(trips) - 1))

                        num_is_feasible_calls += 1
                        tested_trips_requests.append(combined_trip.requests)


                # Create only MAX_ACTIONS actions
                if (MAX_TRIPS >= 0 and len(trips) >= MAX_TRIPS):
                    break

            feasible_trips.append(trips)
            feasible_paths.append(paths)
        
        return feasible_trips, feasible_paths