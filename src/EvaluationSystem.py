from docplex.mp.model import Model  # type: ignore
from docplex.mp.linear import Var  # type: ignore
from typing import List, Dict, Tuple, Set, Any, Optional, Callable
import numpy as np


'''
The subsystem of the control center that evaluates and chooses trips
'''
class EvaluationSystem:
    def __init__(self,
                cfg,
                environment):
        self.cfg = cfg
        self.environment = environment
        
    
    # function: Score each trip, which will be the criteria for choosing trips subsequently
    # params: feasible trips returned by RTV system: list[list[vehicle]], length = num_vehicles
    # return: Scored trips: list[list[(vehicel, score)]], length = num_vehicles
    def ScoreTrips(self, feasible_trips, feasible_paths, pre_values = None, alpha = 0.25):
        scored_feasible_trips = []
        value_cnt = 0
        for trips, paths in zip(feasible_trips, feasible_paths):
            scored_vehicle_trips = []
            
            # There is only one trip (Null trip) means the vehicle is total determined by the simulation other than the RL model.
            # Therefore, we don't need to score this action
            # if len(trips) == 1:
            #     scored_vehicle_trips.append((trips[0], 0, None))
            # else:
            for trip, path in zip(trips, paths):
                reward = 0
                # We use the original total travel distance of the trip to measure the trips 
                # Note: if the original travel distance of all trips are the same for diffreent vehicles, we assume that the vehicle's score is higher if it is closer to the first request
                if len(trip.requests) > 0:
                    #reward += sum(request.original_travel_distance for request in trip.requests) / 1000
                    reward -= alpha * np.sum(path.time_delay_to_each_position)
                    reward += sum(request.CalculatePrice() for request in trip.requests)
                    #reward -= sum(self.environment.GetDistanceandTime(req.pickup_position, path.current_position, type = 'Manhattan')[1] / 600 for req in trip.requests)
                    #reward += len(trip.requests)
                if pre_values is not None:
                    score = reward + self.cfg.MODEL.DISCOUNT_FACTOR * pre_values[value_cnt, 0]
                else:
                    score = reward
                
                scored_vehicle_trips.append((trip, score, reward))
                value_cnt += 1

            scored_feasible_trips.append(scored_vehicle_trips)
        
        if pre_values is not None:
            assert value_cnt == len(pre_values) # the pre_values and trips should be one-to-one correspondence

        return scored_feasible_trips
    
    
    '''
    Here we record the distribtion of vehicles and requests in the previous 30 mins and all time,
    and combine the two distributions to guide repositioning and matching.
    Specifically, we regard the diifference between the average number of requests and vehicles of each grid as the future reward.
    '''
    # function: Predict the future reward according the previous distribution of vehicles and requests
    # params: the trip
    # return: the predicted future reward of the trip
    def FutureReward(self, path, alpha = 0.5):
        req_dis, req_dis_all = self.environment.requests_distribution.GetDistribution()
        veh_dis, veh_dis_all = self.environment.vehicles_distribution.GetDistribution()
        # the relevantly hot areas
        r_dis = alpha * (req_dis - veh_dis) + (1 - alpha) * (req_dis_all - veh_dis_all)
        
        # We only measure the final position of the path, arriving which the vehicle will be idle
        final_pos = path.next_positions[-1]
        try:
            px, py = self.environment.node_coord_to_grid[final_pos]
        except:
            px, py = self.environment.Coord2Grid(final_pos)
        
        future_reward = min(0.99, r_dis[py, px])

        return future_reward
        


    # function: Score each trip based on Reinforcement Learning (todo...)
    # params: feasible trips returned by RTV system
    # return: Scored trips: list[vehicle]
    def ScoreTripsRL(feasible_trips):
        pass


    # function: Choose a trip for each vehicle that maximize the whole score
    # params: scored feasible trips returned by function EvaluateTrips(): list[list[(vehicel, score)], and feasible paths, length = num_vehicles
    # return: List[Tuple(vehicle, score)] (in which each vehicle has a specific trip) --> length = num_vehicles
    # Note: 1. A vehicle may be associated with a Null trip, which means the vehicle does not need to response to any requests at the current time step
    #       2. Trips are chosen by solving an Integer Linear Program (ILP) problem:
    #           object: maximize the whole score
    #           constraints: (1) Each vehicle is assigned to a singe trip at most；(2) Each request is assigned to a vehicle at most or ignored.
    def ChooseTrips(self, scored_feasible_trips, feasible_paths):

        # Model as ILP
        model = Model()

        # For converting trip -> trip_id and back
        trip_to_id = {}
        id_to_trip = {}
        current_trip_id = 0

        # For constraint 2
        requests = set()

        # Create decision variables and their coefficients in the objective
        # There is a decision variable for each (Trip, vehicle).
        # The coefficient is the value associated with the decision variable
        decision_variables: Dict[int, Dict[int, Tuple[Any, float]]] = {}
        
        for vehicle_idx, scored_trips in enumerate(scored_feasible_trips):
            for trip, score, reward in scored_trips:
                # Convert trip -> id if it hasn't already been done
                if trip not in trip_to_id:
                    trip_to_id[trip] = current_trip_id
                    id_to_trip[current_trip_id] = trip
                    current_trip_id += 1

                    trip_id = current_trip_id - 1
                    decision_variables[trip_id] = {}
                else:
                    trip_id = trip_to_id[trip]

                # Update set of requests in trips
                for request in trip.requests:
                    if request not in requests:
                        requests.add(request)

                # Create variable for (trip_id, vehicle_id)
                variable = model.binary_var(name='x{},{}'.format(trip_id, vehicle_idx))

                # Save to decision_variable data structure
                decision_variables[trip_id][vehicle_idx] = (variable, score)

        # Create Constraint 1: Almost one trip per vehicle
        for vehicle_idx in range(len(scored_feasible_trips)):
            vehicle_specific_variables: List[Any] = []
            for trip_dict in decision_variables.values():
                if vehicle_idx in trip_dict:
                    vehicle_specific_variables.append(trip_dict[vehicle_idx])
            model.add_constraint(model.sum(var for var, _ in vehicle_specific_variables) == 1)

        # Create Constraint 2: Almost one trip per Request
        for request in requests:
            relevent_trip_dicts: List[Dict[int, Tuple[Any, float]]] = []
            for trip_id in decision_variables:
                if request in id_to_trip[trip_id].requests:
                    relevent_trip_dicts.append(decision_variables[trip_id])
            model.add_constraint(model.sum(var for trip_dict in relevent_trip_dicts for var, _ in trip_dict.values()) <= 1)

        # Create Objective
        object_score = model.sum(sco * var for trip_dict in decision_variables.values() for (var, sco) in trip_dict.values())
        model.maximize(object_score)

        # Solve ILP
        solution = model.solve(agent='local')
        if not solution:
            print('#'*50)
            print()
            print('Warning: No solution at this step!')
            print()
            print('#'*50)
            return None, None, None
        #assert solution  # making sure that the model doesn't fail

        # Get vehicle specific trips from ILP solution
        assigned_trips: Dict[int, int] = {}
        for trip_id, trip_dict in decision_variables.items():
            for vehicle_idx, (var, _) in trip_dict.items():
                if (solution.get_value(var) == 1):
                    assigned_trips[vehicle_idx] = trip_id
                    break

        
        #assert len(assigned_trips) == len(scored_feasible_trips) # Only one trip per vehicle
        if len(assigned_trips) != len(scored_feasible_trips):
            print('*'*20)
            print('Warning: choosing trips error !!!')
            print('*'*20)


        # Choose tha final trip and path for each vehicle
        final_trips = []
        scores = []
        final_paths = []

        for vehicle_idx in range(len(scored_feasible_trips)):
            # There may exist bugs in the solution, e.g., the assigned trips are less than the number of vehicles.
            # However, we ignore the bugs here, and assign a Null trip to the vehicle
            try:
                assigned_trip_id = assigned_trips[vehicle_idx]
                assigned_trip = id_to_trip[assigned_trip_id]
            except:
                assigned_trip = scored_feasible_trips[vehicle_idx][0][0]
            scored_final_trip = None # The final trip is None if there are no assigned trips
            for trip_idx, (trip, score, reward) in enumerate(scored_feasible_trips[vehicle_idx]):
                if (trip == assigned_trip):
                    scored_final_trip = trip
                    final_score = score
                    final_reward = reward
                    final_path = feasible_paths[vehicle_idx][trip_idx]
                    break

            assert scored_final_trip is not None
            final_trips.append(scored_final_trip)
            if final_reward is not None:
                scores.append(final_score)
            final_paths.append(final_path)

        return final_trips, final_paths, scores




    # All the items can be evaluated in the system, but these functions may be realized in the next development stage
    # todo...
    def EvaluateRequestPrice(self):
        pass

    # todo...
    def EvaluatePassengerComfort(self):
        pass

    # todo...
    def EvaluateVehicleIncome(self):
        pass

    # todo...
    def EvaluateTravelledDistance(self):
        pass
