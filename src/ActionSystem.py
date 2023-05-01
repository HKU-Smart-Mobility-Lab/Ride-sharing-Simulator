from os import environ
import random
import numpy as np
from .component.Trip import Trip, Path
from .utils.Reposition import Reposition


'''
The subsystem of the control center that simulates actions and manages all vehicles
The parameters of all vehicles are changed here
'''
class ActionSystem:
    def __init__(self,
                cfg,
                vehicles,
                requests,
                environment,
                current_timepoint,
                step_time,
                RTV_system,
                consider_itinerary
                ):
        self.cfg = cfg
        self.vehicles = vehicles
        self.requests = requests
        self.environment = environment
        self.current_timepoint = current_timepoint
        self.step_time = step_time
        self.RTV_system = RTV_system
        self.consider_itinerary = consider_itinerary
        
        self.consider_congestion = self.cfg.ENVIRONMENT.CONSIDER_CONGESTION
        self.real_time_update = self.cfg.ENVIRONMENT.REAL_TIME_UPDATE

        self.reposition = Reposition(environment = environment,
                                    method = self.cfg.VEHICLE.REPOSITION.METHOD)
    

    # function: Update the trip and the path of the vehicle
    # params: feasible trips and paths returned by RTV system
    # return: None
    # Note: (1) Once the vehicle is associated with a request, it stops repositioning immediately
    # (2) How to update vehicles when using RL is needed to design carefully (todo...)
    def UpdateVehicles(self, final_trips, final_paths, vehicles = None):
        
        if vehicles is None:
            vehicles = self.vehicles
        
        assert len(vehicles) == len(final_trips) and len(vehicles) == len(final_paths)
        
        vehicles_to_reposition = []
        for vehicle, trip, path in zip(vehicles, final_trips, final_paths):
            if vehicle.path is None and len(trip.requests) == 0 and vehicle.online: # No trip
                vehicles_to_reposition.append(vehicle) # vehicles needed to be rebalanced
                continue
            
            if len(trip.requests) == 0: # Null trip
                continue

            # The current position of the vehicle and the path should be the same
            assert vehicle.current_position == path.current_position

            # Update path (including itinerary nodes)
            if self.consider_itinerary: # and len(vehicles) > 1:
                self.RTV_system.PlanPath.UpdateItineraryNodes(path)

            # Update requests
            for request in trip.requests:
                request.finish_assign = True # The control center assigns the request to an vehicle successfully
                request.assign_timepoint = self.current_timepoint
                request.vehicle_id = vehicle.id
                # if the pickup position of the request is the very current position of the vehicle, the vehicle can pick up the request immediately 
                if request.pickup_position == vehicle.current_position:
                    vehicle.current_requests.append(request)
                    vehicle.current_capacity += request.num_person
                    # Update the request
                    request.finish_pickup = True
                    request.pickup_timepoint = self.current_timepoint
                    # Update the travel time, distance, and delay, as the vehicle has picked up the request
                    path.Update(self.consider_itinerary)
                else:
                    vehicle.next_requests.append(request)
            
            if path.current_position == path.next_positions[0]:
                path.next_positions = path.next_positions[1:]

            # Check if the original path of the vehicle is the same as new path
            # if vehicle.path is None or vehicle.path.next_positions[0] == path.next_positions[0]:
            #     vehicle.path = path
            # else:
            vehicle.dis_from_cur_pos = 0
            vehicle.path = path
            
            # The path of the vehicle may change, so we need to update the traffic density for the road
            if self.consider_congestion:
                self.UpdateSpeed(vehicle)
            
            # Chack if the vehicle is (will be) full
            if vehicle.current_capacity + sum([req.num_person for req in vehicle.next_requests]) >= vehicle.max_capacity:
                vehicle.open2request = False
            else:
                vehicle.open2request = True
        
        # repositioning idle vehicles
        if self.cfg.VEHICLE.REPOSITION.TYPE and len(vehicles_to_reposition) > 0 :
            self.reposition.Reposition(vehicles_to_reposition)  
        # Update path (including itinerary nodes)
        if self.consider_itinerary:
            for vehicle in vehicles_to_reposition:
                if vehicle.path is not None:
                    self.RTV_system.PlanPath.UpdateItineraryNodes(vehicle.path)
                    # The path of the vehicle may change, so we need to update the traffic density for the road
                    if self.consider_congestion:
                        self.UpdateSpeed(vehicle)

    

    # function: Update the traffic density and vehicle speed for each road
    # params: None
    # return: None
    def UpdateSpeed(self, vehicle):
        # Update the speed for the orginal road
        if vehicle.road_id is not None:
            self.environment.UpdateSpeed(vehicle.road_id, vehicle.id, JOIN = False)

        # Not just stay at the node
        if vehicle.path.current_position != vehicle.path.next_itinerary_nodes[0]:
            new_road_id = self.environment.GetRodeID(vehicle.path.current_position, vehicle.path.next_itinerary_nodes[0])
            if new_road_id is not None:
                # Update the speed for the new road
                self.environment.UpdateSpeed(new_road_id, vehicle.id, JOIN = True)
            # update the road id for the vehicle
            vehicle.road_id = new_road_id
        else:
            vehicle.road_id = None

            

    # function: Simulate the action of each vehicle and manage all vehicles (to the next position, get offline, and etc.)
    # params: ConsiderIntersections means we consider itinerary of trips (including origin, destination, and median nodes)
    # return: None
    # Note: This function only implements the best route of each vehicle. If you want to change the route of an vehicle, you shoulf use the PlanRoute() function in the RTVSystem object.
    def SimulateVehicleActions(self, vehicles = None, ERR_THRE = 10):
        current_timepoint = self.current_timepoint + self.step_time # Update time
        
        if vehicles is None:
            vehicles = self.vehicles

        for vehicle in vehicles:
            # There doesn't exist passengers in the vehicle means the vehicle is idle
            if vehicle.current_capacity == 0 and sum(req.num_person for req in vehicle.next_requests) == 0:
                if vehicle.online:
                    vehicle.total_idle_time += self.step_time
        
            # Null trip and not repositioning
            if vehicle.path is None:
                continue
           
            # Whether consider intersections between origins and destinations or not
            next_positions, time_needed_to_next_position, dis_to_next_position = vehicle.path.GetPath(self.consider_itinerary)
            
            # Update distance to the next position
            if vehicle.dis_from_cur_pos == 0: # The vehicle has a new path
                vehicle.dis_to_next_pos = dis_to_next_position[0]
            
            # The vehicle moves forward
            vehicle.dis_from_cur_pos += self.step_time * vehicle.speed
            # The passengers stay on the vehicle for another step_time
            for request in vehicle.current_requests:
                request.time_on_vehicle += self.step_time

            # move the vehicle to the next position
            if vehicle.dis_from_cur_pos >= vehicle.dis_to_next_pos or vehicle.dis_to_next_pos - vehicle.dis_from_cur_pos < ERR_THRE:
                # Update the travelled distance from current position to next position
                vehicle.dis_from_cur_pos = 0

                # the vehicle move to the next position, so we need to check which request should be dropped off or picked up
                # Check the current requests
                new_current_requests = []
                for request in vehicle.current_requests:
                    request.distance_on_vehicle += dis_to_next_position[0]
                    
                    # Drop off the current request
                    if request.dropoff_position == next_positions[0]:
                        # Update the request
                        request.finish_dropoff = True
                        request.dropoff_timepoint = current_timepoint
                        # Update the vehicle
                        vehicle.total_income += request.CalculatePrice() # Here, we assume the original (shortest) travel distance of the request represents the price
                        vehicle.current_capacity -= request.num_person
                        
                    else:
                        new_current_requests.append(request)
                # Update the current requests
                vehicle.current_requests = new_current_requests

                # Check the next requests
                new_next_requests = []
                for request in vehicle.next_requests:
                    # Pick up the next request
                    if request.pickup_position == next_positions[0]:
                        # Update the request
                        request.finish_pickup = True
                        request.pickup_timepoint = current_timepoint
                        assert request.pickup_timepoint >= request.assign_timepoint
                        # Update the current requests
                        vehicle.current_requests.append(request)
                        vehicle.current_capacity += request.num_person
                    else:
                        new_next_requests.append(request)
                # Update the next requests
                vehicle.next_requests = new_next_requests

                # Accumulate the distance travelled of the vehicle
                vehicle.total_distance += dis_to_next_position[0]
                # Update the current position of the vehicle
                vehicle.current_position = next_positions[0]
                vehicle.path.current_position = next_positions[0]
                
                # Record the actions of the vehicle
                vehicle.actions_timepoint.append(current_timepoint)
                vehicle.actions_positions.append(vehicle.current_position)
                
                # Update the path
                if len(next_positions) > 1:
                    vehicle.dis_to_next_pos = dis_to_next_position[1]
                    vehicle.path.Update(self.consider_itinerary)
                    '''
                    Update the vehicle's road in real time, that is, the vehicle's road is updated once the vehicle arrives at a new node
                    '''
                    if self.real_time_update:
                        self.RTV_system.PlanPath.UpdateItineraryNodes(vehicle.path)
                    
                    # The vehicle has left the original road and joined a new road, so we need to update the traffic density for the two roads as well as the vehicle's speed
                    if self.consider_congestion:
                        self.UpdateSpeed(vehicle)
                    
                else: # There is only one position in the path, which means the vehicle will be idle after delivering the current request
                    assert len(vehicle.next_requests) == 0
                    # Update the vehicle
                    vehicle.Status2Idle() # Set the vehicle idle
            
            # Chack if the vehicle is (will be) full
            if vehicle.current_capacity + sum([req.num_person for req in vehicle.next_requests]) >= vehicle.max_capacity:
                vehicle.open2request = False
            else:
                vehicle.open2request = True
            
            # Check if the vehicle is still online
            if current_timepoint >= vehicle.end_time:
                vehicle.Offline() # No new trips will be assigned
            

    # function: Remove the finished requests, and the unmatched requests are returned and will be merged with the new requests at next time step
    # params: None
    # return: the unmatched requests, List[requests]
    # Note: we consider behaviors of passengers in the function
    def ProcessRequests(self):
        unmatched_requests = []
        for request in self.requests:
            # the request has been pick up
            if request.finish_pickup:
                continue
            
            # The request has been assigned to a vehicle
            # Note: (1) If the vehicle does not catch passengers for a long time (e.g., 5 mins), the passengers may cancel the request.
            # Then we need to replan the path of the vehicle
            # (2) If the request hasn't been cancelled, we can find them through vehicles
            elif request.finish_assign:
                waiting_pickup_time = self.current_timepoint + self.step_time - request.assign_timepoint
                if waiting_pickup_time > request.max_tol_pickup_time:
                    if random.random() < request.cancel_prob_pickup:
                        vehicle_id = request.vehicle_id
                        assert vehicle_id
                        # Remove the request from the vehicle's trip
                        vehicle = self.vehicles[vehicle_id]
                        assert request in vehicle.next_requests
                        vehicle.next_requests = list(set(vehicle.next_requests) - set(request))
                        # Replan the path of the vehicle
                        vehicle.path = self.RTV_system.PlanPath_CompleteSearch(vehicle, Trip())
                
            
            # The request hasn't been assigned to any vehicles
            # Note: (1) Passengers may cancel their requests if the system doesn't assign them to any vehicles for a long time (e.g., 3 mins)
            # (2) The request will be cancelled automatically if it hasn't been assigned to any vehicles for 10 mins
            else:
                waiting_assign_time = self.current_timepoint + self.step_time - request.send_request_timepoint 
                if waiting_assign_time < request.max_tol_assign_time:
                    unmatched_requests.append(request) # Unmatched and not cancelled requests will be added to requests at next tiem step
                elif waiting_assign_time < request.max_con_assign_time:
                    if random.random() > request.cancel_prob_assign:
                        unmatched_requests.append(request)
                else:
                    continue
                    
        return unmatched_requests

    
    