from .ActionSystem import ActionSystem
from .RTVSystem import RTVSystem
from .EvaluationSystem import EvaluationSystem
from .PostProcessSystem import PostProcessSystem
import numpy as np


'''
The control center is just like a ride-hailing platform
All requests and vehicles are handled here
The control center consists of 4 subsystems:
1. RTV System: handles requests, vehicles and trips
2. Evaluation System: evaluates and chooses trips
3. Action System: simulates actions and manages all vehicles
4. Post Process System: count and visulize results

See each object for detailed information
'''
class ControlCenter:
    def __init__(self,
                cfg,
                environment
                ):
        self.cfg = cfg
        self.environment = environment
        self.step_time = self.cfg.SIMULATION.STEP_TIME
        self.start_timepoint = self.cfg.SIMULATION.START
        self.end_timepoint = self.cfg.SIMULATION.END
        self.total_steps = int((self.end_timepoint + self.cfg.SIMULATION.TIME2FINISH - self.start_timepoint) / self.step_time - 1)
        self.simulation_steps = int((self.end_timepoint - self.start_timepoint) / self.step_time)
        
        self.consider_itinerary = self.cfg.ENVIRONMENT.CONSIDER_ITINERARY.TYPE

        self.RTV_system = RTVSystem(cfg = cfg, environment = self.environment)
        self.evaluation_system = EvaluationSystem(cfg = cfg, environment=self.environment)

        self.current_timepoint = self.start_timepoint
        self.step = 0

        # Initialize requests and vehicles, see class RTVSystem for detailed information
        self.requests_all = None
        self.requests_step = None
        self.vehicles_all = None

        self.action_system = ActionSystem(cfg = self.cfg,
                                        vehicles = None,
                                        requests = None,
                                        environment = self.environment,
                                        current_timepoint = self.current_timepoint,
                                        step_time = self.step_time,
                                        RTV_system = self.RTV_system,
                                        consider_itinerary = self.consider_itinerary)

        self.post_process_system = PostProcessSystem(vehicles = None,
                                                    requests = None,
                                                    environment = self.environment,
                                                    current_timepoint = self.current_timepoint
                                                    )
    # Initialize the requests and vehicles
    def Initialize(self, requests, vehicles):
        self.requests_all = requests
        self.vehicles_all = vehicles
        self.requests_step = requests[self.step]
        self.action_system.vehicles = vehicles
        self.action_system.requests = requests[self.step]
        self.action_system.reposition.past_requests = requests[self.step]
        self.post_process_system.requests = requests[self.step]
        self.post_process_system.vehicles = vehicles


    # Update the system's parameters
    def UpdateParameters(self, timepoint, step):
        self.current_timepoint = timepoint
        self.RTV_system.current_timepoint = timepoint
        self.action_system.current_timepoint = timepoint
        self.post_process_system.current_timepoint = timepoint
        self.step = step
    
    # Update requests at next time step
    # params: Unmatched_requests: requests that haven't been allocated to any vehicles and don't cancel
    def UpdateRequests(self, unmatched_requests):
        if self.step >= self.total_steps-1 or self.step >= len(self.requests_all) - 1:
            new_requests = []
        else:
            new_requests = self.requests_all[self.step + 1] # New requests at next time step
        requests = list(set(unmatched_requests) | set(new_requests)) # Union
        self.action_system.requests = requests
        self.post_process_system.requests = requests
        self.requests_step = requests
        # Update distribution of requests and vehicles that will be used to guide repositioning
        self.environment.UpdateDistributions(new_requests, self.vehicles_all)


    '''RTV System'''
    # Allocate requests to each vehicle, see class RTVSystem for detailed information    
    def AllocateRequest2Vehicles(self, max_num_vehicles = 30, max_match_distance = 3000):
        requests_for_each_vehicle = self.RTV_system.AllocateRequest2Vehicles(self.requests_step, self.vehicles_all, max_num_vehicles, max_match_distance)
        return requests_for_each_vehicle
    
    # Generate feasible trips and the corresponding paths, see class RTVSystem for detailed information
    def GenerateFeasibleTrips(self, requests_for_each_vehicle, MAX_IS_FEASIBLE_CALLS = 150, MAX_TRIPS = 30):
        feasible_trips, feasible_paths = self.RTV_system.GenerateFeasibleTrips(self.vehicles_all, requests_for_each_vehicle, MAX_IS_FEASIBLE_CALLS, MAX_TRIPS)
        return feasible_trips, feasible_paths

    # Initialize requests in each batch, see class RTVSystem for detailed information
    def IniReqsBatch(self, reqs, update_all = False):
        return self.RTV_system.IniReqsBatch(reqs=reqs, update_all=update_all)


    '''Evaluation System'''
    # Score feasible trips, see class EvaluationSystem for detailed information
    def ScoreTrips(self, feasible_trips, feasible_paths, pre_values):
        scored_feasible_trips = self.evaluation_system.ScoreTrips(feasible_trips, feasible_paths, pre_values)
        return scored_feasible_trips
    
    # Score feasible trips based on Reinforcement Learning, see class EvaluationSystem for detailed information
    # todo...
    def ScoreTripsRL(self, feasible_trips):
        return self.evaluation_system.ScoreTripsRL(feasible_trips)

    # Choose a trip and the corresponding path for each vehicle, see class EvaluationSystem for detailed information
    def ChooseTrips(self, scored_feasible_trips, feasible_paths):
        final_trips, final_paths, rewards = self.evaluation_system.ChooseTrips(scored_feasible_trips, feasible_paths)
        return final_trips, final_paths, rewards
    

    '''Action System'''
    # Update the trip and the path of the vehicle, see class ActionSystem for detailed information
    def UpdateVehicles(self, final_trips, final_paths, vehicles = None):
        self.action_system.UpdateVehicles(final_trips, final_paths, vehicles)
    
    # Simulate the action of each vehicle and manage all vehicles, see class ActionSystem for detailed information
    def SimulateVehicleAction(self, vehicles = None):
        self.action_system.SimulateVehicleActions(vehicles)

    # Remove the finished requests, and the unmatched requests are returned and will be merged with the requests at next time step, see class ActionSystem for detailed information
    def ProcessRequests(self):
        unmatched_requests = self.action_system.ProcessRequests()
        return unmatched_requests


    '''PostProcess System'''
    # Draw the time distribution of sending requests, see class PostProcessSystem for detailed information
    def ReqTimeSta(self, ax, requests):
        ax = self.post_process_system.ReqTimeSta(ax = ax, requests = requests)
        return ax
    
    # Draw the distance distribution of requests, see class PostProcessSystem for detailed information
    def ReqDisSta(self, ax, requests, MaxDis = None, nor_fit = True):
        ax = self.post_process_system.ReqDisSta(ax = ax, requests = requests, MaxDis = MaxDis, nor_fit = nor_fit)
        return ax
    
    # Draw road network of New York model, see class PostProcessSystem for detailed information
    def DrawRoadNetwork(self, ax, TIME = False, congestion = False, speed_lim = [0,20], axis_lim = None):
        ax = self.post_process_system.DrawRoadNetwork(ax, TIME = TIME, congestion = congestion, speed_lim = speed_lim, axis_lim = axis_lim)
        return ax
    
    # Draw the distribution of vehicles, see class PostProcessSystem for detailed information
    def DrawVehicles(self, ax, vehicles, v_size = 0.002):
        ax = self.post_process_system.DrawVehicles(ax = ax, vehicles = vehicles, v_size = v_size)
        return ax
    
    # Draw the distribution of requests, see class PostProcessSystem for detailed information
    def DrawRequests(self, ax, requests, type = 'pickup', s = 10, count = False, cmap = 'viridis', cmax = 10, color = 'red', draw_grid = False):
        ax = self.post_process_system.DrawRequests(ax = ax, requests = requests, type = type, s = s, count = count, cmap = cmap, cmax = cmax, color = color, draw_grid = draw_grid)
        return ax
    
    # Draw vehicles and requests of at a specific time point, see class PostProcessSystem for detailed information
    def DrawSnapshot(self, ax, v_size = 0.002, s = 100, colors = [], draw_route = True, draw_road_network = True, speed_lim = [0, 20], axis_lim = None):
        ax = self.post_process_system.DrawSnapshot(ax, v_size = v_size, s = s, colors=colors, draw_route = draw_route, draw_road_netwrod=draw_road_network, speed_lim=speed_lim, axis_lim = axis_lim)
        return ax

    # Calculate the aspect ratio of the figure, see class PostProcessSystem for detailed information
    def FigAspectRatio(self, box = None):
        return self.post_process_system.FigAspectRatio(box = box)

    # Draw road network of toy model, see class PostProcessSystem for detailed information
    def DrawRoadNetworkToyModel(self, ax):
        return self.post_process_system.DrawRoadNetworkToyModel(ax)

    # Draw vehicles and requests of toy model, see class PostProcessSystem for detailed information
    def DrawVehiclesandRequestsToyModel(self, ax):
        ax = self.post_process_system.DrawVehiclesandReuqestsToyModel(ax)
        return ax
    # Integrate all result images to a vedio, see class PostProcessSystem for detailed information
    def MakeVedio(self, imgs = None, img_path = 'Output/tmp', vedio_fps = 30, vedio_path = 'Output', vedio_name = 'result.mp4', del_img = False):
        self.post_process_system.MakeVedio(imgs = imgs, img_path = img_path, vedio_fps = vedio_fps, vedio_path = vedio_path, vedio_name=vedio_name, del_img = del_img)


    ''' 
    Once the simulation finished, we calculate:
    (1) REQUEST
        0.  Service rate (non-ride-pooling)
        1.  Service rate (ride-pooling)
        2.  The average assigning time
        3.  The average waiting time of requests
        4.  The average detour time
        5.  The average detour time ratio
        6.  The average total time ratio
        7.  The average detour distance
        8.  The average detour distance ratio
        9. Cancellation rate (assign)
        10. Cancallation rate (pickup)
        11. ft1
        12. ft2
    (2) VEHICLE
        1. The number of vehicles
        2. The average idle time
        3. The total income of all vehicles
        4. The total travel distance of all vehicles
    '''
    def CalculateResults(self):
        requests_results = np.zeros((13))
        vehicles_results = np.zeros((4))
        # Requests' results
        num_req = 0
        num_req_pool = 0
        num_req_non = 0
        for requests in self.requests_all:
            for request in requests:
                num_req += 1
                if request.max_tol_num_person == 1:
                    num_req_non += 1
                else:
                    num_req_pool += 1
                # The request has been served
                if request.finish_dropoff:
                    if request.max_tol_num_person == 1:
                        requests_results[0] += 1
                    else:
                        requests_results[1] += 1
                    requests_results[2] += request.assign_timepoint - request.send_request_timepoint
                    requests_results[3] += request.pickup_timepoint - request.assign_timepoint
                    requests_results[4] += max(0, request.time_on_vehicle - request.original_travel_time)
                    requests_results[5] += max(0, request.time_on_vehicle - request.original_travel_time) / request.original_travel_time
                    requests_results[6] += (request.time_on_vehicle + request.pickup_timepoint - request.send_request_timepoint) / request.original_travel_time
                    requests_results[7] += max(0, request.distance_on_vehicle - request.original_travel_distance)
                    requests_results[8] += max(0, request.distance_on_vehicle - request.original_travel_distance) / request.original_travel_distance
                    requests_results[11] += request.time_on_vehicle
                    requests_results[12] += request.time_on_vehicle + request.pickup_timepoint - request.assign_timepoint
                    #requests_results[4] += request.dropoff_timepoint - request.pickup_timepoint - request.original_travel_time
                    # requests_results[5] += request.distance_on_vehicle - request.original_travel_distance
                # The request has been cancelled
                # Note: Here, we assume that there is no passenger at any vehicles. In other words, all trips are finished at the end of simulation
                else:
                    if request.finish_assign:
                        requests_results[9] += 1
                    else:
                        requests_results[10] += 1
        
        print('*'*50)
        print('The number of requests: ', num_req)
        print('The number of non-ride-pooling requests: ', num_req_non)
        print('The number of ride-pooling requests: ', num_req_pool)
        print('Service rate: ', (requests_results[0] + requests_results[1]) / num_req)
        print('*'*50)
        
        # mean value
        requests_results[2:9] /= (requests_results[0]+requests_results[1])
        requests_results[11:] /= (requests_results[0]+requests_results[1])
        requests_results[0] /= num_req_non
        requests_results[1] /= num_req_pool
        requests_results[9:11] /= num_req
        # Vehicles' results
        for vehicle in self.vehicles_all:
            vehicles_results[0] += 1
            vehicles_results[1] += vehicle.total_idle_time
            # Here, we calculate the total income and travel distance to evaluate the system's income and energy consumption
            vehicles_results[2] += vehicle.total_income
            vehicles_results[3] += vehicle.total_distance
        vehicles_results[1] /= vehicles_results[0]

        return requests_results, vehicles_results


    # function: calculate the average number of requests in each vehicle
    # params: all vehicles
    # return: the average number of reuqests in a vehicle at this simulation step
    def VehicleUtility(self):
        req_num = 0
        for vehicle in self.vehicles_all:
            req_num += vehicle.current_capacity
            req_num += sum(req.num_person for req in vehicle.next_requests)
        
        ave_req_num = req_num / len(self.vehicles_all)

        return ave_req_num

    