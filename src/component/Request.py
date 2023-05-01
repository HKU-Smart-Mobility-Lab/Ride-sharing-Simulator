'''
The object of a single request
The whole requests are operated in the Contral Center
''' 
class Request:
    def __init__(self,
                cfg,
                id = 0,
                send_request_timepoint = 0,
                pickup_position = 0,
                dropoff_position = 0,
                pickup_grid_id = 0,
                dropoff_grid_id = 0,
                iti_nodes = [],
                iti_dis = [],
                iti_t = [],
                original_travel_time = 0,
                original_travel_distance = 0,
                num_person = 1):
        self.cfg = cfg
        self.id = id
        self.send_request_timepoint = send_request_timepoint
        # origin and destination node id
        self.pickup_position = pickup_position
        self.dropoff_position = dropoff_position
        # grid id : (x_num, y_num)
        self.pickup_grid_id = pickup_grid_id
        self.dropoff_grid_id = dropoff_grid_id
        # original trajector, travel distance, and travel time without ride-pooling
        self.iti_nodes = iti_nodes
        self.iti_dis = iti_dis
        self.iti_t = iti_t
        self.original_travel_distance = original_travel_distance
        self.original_travel_time = original_travel_time
        
        # todo... (we assume that there is only one person in each request)
        self.num_person = num_person # There may be more than 1 person in some requests
        self.max_tol_num_person = 1
        

        
        ################################  These perameters can be redefined  ###################################
        # behaviors
        self.max_tol_assign_time = self.cfg.REQUEST.BEHAVIORS.max_assign_time
        self.cancel_prob_assign = self.cfg.REQUEST.BEHAVIORS.cancel_prob_assign
        self.max_tol_pickup_time = self.cfg.REQUEST.BEHAVIORS.max_pickup_time
        self.cancel_prob_pickup =  self.cfg.REQUEST.BEHAVIORS.cancel_prob_pickup
        self.max_tol_vehicle_capacity =  self.cfg.REQUEST.BEHAVIORS.max_tol_vehicle_capacity # Some passengers can not stand full capacity of a vehicle
        
        # constraints
        self.max_con_assign_time = self.cfg.REQUEST.CONSTRAINTS.max_assign_time
        self.max_con_pickup_time = self.cfg.REQUEST.CONSTRAINTS.max_pickup_time
        self.max_con_travel_time = self.cfg.REQUEST.CONSTRAINTS.max_travel_time_mul * self.original_travel_time
        self.max_con_travel_diatance = self.cfg.REQUEST.CONSTRAINTS.max_travel_dis_mul * self.original_travel_distance
        ################################  These perameters can be redefined  ###################################

        self.MAX_DROPOFF_DELAY = self.max_con_travel_time - self.original_travel_time


        # Record the status of the request
        self.finish_assign = False
        self.finish_pickup = False
        self.finish_dropoff = False
        self.assign_timepoint = 0
        self.pickup_timepoint = 0
        self.dropoff_timepoint = 0
        self.vehicle_id = None

        # Record the time and distance of the request on the vehicle
        self.time_on_vehicle = 0
        self.distance_on_vehicle = 0
        
        # todo...
        self.max_tol_price = 0
        self.comfortable_value = 0
    

    # We update the shortest route of the request according to the traffic congestion
    def UpdateRoute(self, iti_nodes, iti_dis, iti_t):
        self.iti_nodes = iti_nodes
        self.iti_dis = iti_dis
        self.iti_t = iti_t
        self.original_travel_distance = sum(iti_dis)
        self.original_travel_time = sum(iti_t)
        self.max_con_travel_time = self.cfg.REQUEST.CONSTRAINTS.max_travel_time_mul * self.original_travel_time
        self.max_con_travel_diatance = self.cfg.REQUEST.CONSTRAINTS.max_travel_dis_mul * self.original_travel_distance


    # Calculate the price of request
    # We refer to: https://www.introducingnewyork.com/taxis#:~:text=These%20are%20the%20general%20rates%3A%201%20Base%20fare%3A,%28from%204%20pm%20to%208%20pm%29%3A%20US%24%201
    def CalculatePrice(self, DISCOUNT = 0.7):
        initial_charge = 3.0
        mileage_charge = 0.7  # per 0.2 mile
        waiting_charge = 0.5  # per 60 seconds, which may be used when considering congestion
        # 8 p.m. - 6 a.m.  --> 7 nights
        night_surcharge = 1.0 if self.send_request_timepoint > 20*3600 or self.send_request_timepoint < 6*3600 else 0.0
        # 4 - 8 p.m.   --> weekdays only, excluding holidays
        peak_hour_price = 1.0 if self.send_request_timepoint > 16*3600 and self.send_request_timepoint < 20*3600 else 0.0
        
        total_price =  initial_charge + self.original_travel_distance / (1609 / 5) * mileage_charge + self.original_travel_time / 60 * waiting_charge + night_surcharge + peak_hour_price
        
        # No discount without ride-pooling
        if self.max_tol_num_person == 1:
            discount = 1.0
        else:
            discount = DISCOUNT
        
        return discount * total_price
    
    
    
    # function: Calculate the maximum price that passenger(s) can accept
    # params: todo...(May be travel time, travel distance, the number of passengers, etc.)
    # return: maximum tolerant price
    # Note: The maximum price may also be estimated from a specific distribution function or assigned ahead of time
    def MaxTolPrice(self):
        pass

    # function: Calculate the comfortable value of passenger(s) and evaluate comfort
    # params: todo...(May be travel time, travel distance, the number of passengers, etc.)
    # return: comfortable value
    def ComfortableValue(self):
        pass