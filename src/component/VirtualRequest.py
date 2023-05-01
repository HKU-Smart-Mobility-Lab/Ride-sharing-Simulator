
'''
The object of a single virtual request that will be used for repositioning idle vehicles
''' 
class VirtualRequest:
    def __init__(self,
                pickup_position = 0,
                dropoff_position = 0,
                pickup_grid_id = 0,
                dropoff_grid_id = 0,
                original_travel_time = 0,
                original_travel_distance = 0):
        
        # origin and destination node id
        self.pickup_position = pickup_position
        self.dropoff_position = dropoff_position
        # grid id
        self.pickup_grid_id = pickup_grid_id
        self.dropoff_grid_id = dropoff_grid_id
        # travel distance and time
        self.original_travel_distance = original_travel_distance
        self.original_travel_time = original_travel_time  

        self.num_person = 0 # no person

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
        
    def CalculatePrice(self):
        return 0


    
    