'''
The object of a single vehicle
The whole vehicles are operated in the Contral Center
'''
class Vehicle:
    def __init__(self,
                cfg,
                id = 0,
                current_position = None,
                current_grid_id = 0,
                start_time = 0,
                end_time = 9999999,
                online = True,
                open2request = True,
                speed = 15):
        self.cfg = cfg
        self.id = id
        self.current_position = current_position
        self.current_grid_id = current_grid_id
        # Used to control the status of the vehicle (online or offline)
        self.start_time = start_time
        self.end_time = end_time
        
        self.online = online
        self.open2request = open2request # The vehicle is close to requests if current_capacity == maximum_capacity or the vehicle is offline

        self.max_capacity = self.cfg.VEHICLE.MAXCAPACITY
        self.current_capacity = 0

        self.speed = speed
        self.road_id = None # The current road id of the vehicle

        # Requests are devided into 2 types: 1) already on the vehicle; 2) waiting to be picked up (allocated at last time steps)
        self.current_requests = []
        self.next_requests = []
        
        self.path = None # Note: the order of next positions represents the route of the vehicle

        # used to dispatch vehicles
        self.remaining_time_for_current_node = 0
        self.dis_to_next_pos = 0
        self.dis_from_cur_pos = 0
        
        # Record the action time and the corresponding pisition of the vehicle, which will be used to visualize the vehicle actions
        self.actions_timepoint = [self.start_time]
        self.actions_positions = [self.current_position]

        # Record the total idle time of the vehicle        
        self.total_idle_time = 0

        # May be used to calculate the income of the vehicle
        self.total_income = 0
        # May be used to calculate the distance travelled of the vehicle, which can also represent the energy consumption and carbon dioxide (CO2) emissions
        self.total_distance = 0


    def Offline(self):
        self.online = False
        self.open2request = False

    # Set the vehicle idle
    def Status2Idle(self):
        self.current_capacity = 0
        self.current_requests = []
        self.next_requests = []
        self.path = None
        self.road_id = None
        self.remaining_time_for_current_node = 0
        self.dis_to_next_pos = 0
        self.dis_from_cur_pos = 0

