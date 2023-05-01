import copy
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import copy

from src.RL.states import States

running_time_for_print = 20

'''
    (1) agent=None and train=False mean running the simulation without RL model
    (2) One can find anything in the control_centrol
'''
def RunEpisode(requests, vehicles, control_center, agent = None, train = False, train_step = 0, draw_veh_req = False, draw_fre = 60, img_path = 'exp'):
    
    # Initialization
    control_center.Initialize(requests, vehicles)
    if agent:
        states = States(cfg = control_center.cfg,
                        environment= control_center.environment,
                        requests_record_time = 1800) # We record requests in the previous 30 mins 
    imgs = []
    img_cnt = 0
    req_num = 0

    # Run the simulation
    for step in tqdm(range(control_center.total_steps), desc = 'Running simulation steps: '):
        # Upadate parameters of the control center
        current_timepoint = control_center.start_timepoint + step * control_center.step_time
        control_center.UpdateParameters(current_timepoint, step)
        if agent:
            # Update the vehicles' distribution
            states.vehicles_distribution.Update(vehicles)
       
        # Allocate each rquest to the vehicles nearby
        requests_for_each_vehicle = control_center.AllocateRequest2Vehicles()
        # Filter requests that don't meet the system's contraints and combine requests together (ride-pooling) of each vehicle
        feasible_trips, feasible_paths = control_center.GenerateFeasibleTrips(requests_for_each_vehicle)
        

        # For each vehicle, simulate action of each trip to get post-decision states
        if agent:
            next_vehicles = []
            cur_vehicles = []
            for (vehicle, trips, paths) in zip(vehicles, feasible_trips, feasible_paths):
                # There is only one trip (Null trip) means the vehicle is total determined by the simulation other than RL model.
                # Therefore, we don't need to save this action
                # if len(trips) == 1:
                #     continue
                
                cur_vehicles.append(vehicle) # The vehicle's actions will be evaluated by RL model 

                for (trip, path) in zip(trips, paths):
                    # Here we should deepcopy the vehicle for each feasible trip
                    next_vehicle = copy.deepcopy(vehicle)
                    next_trip, next_path = copy.deepcopy(trip), copy.deepcopy(path)
                    # For each trip, simulate the action
                    control_center.UpdateVehicles([next_trip], [next_path], [next_vehicle])
                    control_center.SimulateVehicleAction([next_vehicle])
                    
                    next_vehicles.append(next_vehicle)
            
            
            # Make sure there exists vehicles needed to be evaluated by RL Model
            if len(next_vehicles) > 0:
                # Get the current states
                cur_states = states.GetStates(cur_vehicles, step)
                # Update requests' distribution
                states.requests_distribution.Update(requests[step])
                # Get the post-decision states
                post_states = states.GetStates(next_vehicles, step+1)
                
                # Score each decision by RL model
                pre_value = agent.get_value(post_states)
                pre_value_target = agent.get_value_target(post_states)
            else:
                pre_value = None
                pre_value_target = None
        else:
            pre_value = None


        # Get the final score of each decision
        scored_feasible_trips = control_center.ScoreTrips(feasible_trips, feasible_paths, pre_value)
        # Choose a trip for aeach vehicle
        final_trips, final_paths, scores = control_center.ChooseTrips(scored_feasible_trips, feasible_paths)
        # If no solution
        if final_trips is None:
            continue
        
        if agent:
            '''Here, we calculate the target scores that will be used when training the model ahead of time'''
            # Get the final score of each decision
            scored_feasible_trips_target = control_center.ScoreTrips(feasible_trips, feasible_paths, pre_value_target)
            # Choose a trip for aeach vehicle
            _, _, scores_target = control_center.ChooseTrips(scored_feasible_trips_target, feasible_paths)
        
        
        # Update the vehicles according to the final trips and paths
        control_center.UpdateVehicles(final_trips, final_paths)
        # Simulate actions 
        control_center.SimulateVehicleAction()

        
        if agent:
            # Judge if it's the final step
            done = np.zeros((len(vehicles), 1)) if step < control_center.total_steps -1 else np.ones((len(vehicles), 1)) 
            # Save the experience to the memory
            if len(cur_vehicles) > 0:
                assert len(cur_states[0]) == len(scores_target)
                agent.append_sample(cur_states, scores_target, done)
       
        # Process the requests that unassigned: cancel or wait
        unmatched_requests = control_center.ProcessRequests()
        # Update requests
        control_center.UpdateRequests(unmatched_requests)

        # Train the model
        if agent is not None and train and (train_step + 1) % agent.train_frequency == 0:
            agent.train_model()
        
        train_step += 1
        
        # Draw vehicles and requests at each simulation epoch
        if draw_veh_req and step * control_center.step_time % draw_fre == 0:
            fig_aspect_ratio = control_center.FigAspectRatio()
            fig = plt.figure(figsize=(15/fig_aspect_ratio*1.2,15), dpi=100)
            ax = fig.add_subplot(111)
            axis_lim = copy.deepcopy(control_center.environment.area_box)
            axis_lim[2] -= 0.01
            control_center.post_process_system.cmap = 'RdYlGn'
            control_center.post_process_system.legend_loc = (0.08, 1.01)
            ax = control_center.DrawSnapshot(ax, v_size = 0.004, s = 100, draw_route = False, draw_road_network = True, speed_lim = [5,10], axis_lim = axis_lim)
            plt.subplots_adjust(left=0.15, right=0.95)
            plt.savefig(os.path.join(img_path, str(img_cnt).zfill(6) + '.png'))
            plt.close('all')
            img_cnt += 1
        
        # the average number of reuqests in a vehicle at this simulation ste
        if step <= control_center.simulation_steps:
            req_num += control_center.VehicleUtility()
    
    req_num /= control_center.simulation_steps
    
    # every episode update the target model to be same with model
    if train:
        #agent.update_target_model()
        return train_step, req_num
    
    return req_num