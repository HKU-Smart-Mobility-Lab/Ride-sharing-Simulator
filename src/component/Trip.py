'''
The object of a trip (one or more requests)
''' 
class Trip:
    def __init__(self, requests = []):
        self.requests = [] # For coding convinience, we set reqeusts a list
        
        if isinstance(requests, list):
            self.requests = requests
        else:
            self.requests.append(requests) # only one request

'''
The object of a route including the current position and next positions
'''
class Path:
    def __init__(self,
                current_position = 0,
                next_positions = [],
                time_needed_to_next_position = None,
                dis_to_next_position  = None,
                time_delay_to_each_position = 0,
                next_itinerary_nodes = [],
                time_needed_to_next_node = None,
                dis_to_next_node = None):

        self.current_position = current_position
        
        self.next_positions = [] # including pickup and dropoff positions only
        if isinstance(next_positions, list):
            self.next_positions = next_positions
        else:
            self.next_positions.append(next_positions) # only one position
        # Array
        self.time_needed_to_next_position = time_needed_to_next_position
        self.dis_to_next_position = dis_to_next_position # Used to update vehicles
        self.time_delay_to_each_position = time_delay_to_each_position # Represent the state when using RL

        self.next_itinerary_nodes = next_itinerary_nodes # including all cross intersections but current position
        # Array
        self.time_needed_to_next_node = time_needed_to_next_node
        self.dis_to_next_node = dis_to_next_node # Used to update vehicles when considering itinerary nodes
    
    # Move to next position (node)
    def Update(self, consider_itinerary = False):
        # Consider itinerary nodes
        if consider_itinerary:
            self.next_itinerary_nodes = self.next_itinerary_nodes[1:]
            self.time_needed_to_next_node = self.time_needed_to_next_node[1:]
            self.dis_to_next_node = self.dis_to_next_node[1:]
            # Update positions
            if self.next_itinerary_nodes[0] == self.next_positions[0]:
                self.next_positions = self.next_positions[1:]
                self.time_needed_to_next_position = self.time_needed_to_next_position[1:]
                self.dis_to_next_position = self.dis_to_next_position[1:]
                self.time_delay_to_each_position = self.time_delay_to_each_position[1:]
        
        # Only consider OD of requests
        else:
            self.next_positions = self.next_positions[1:]
            self.time_needed_to_next_position = self.time_needed_to_next_position[1:]
            self.dis_to_next_position = self.dis_to_next_position[1:]
            self.time_delay_to_each_position = self.time_delay_to_each_position[1:]
    
    # Get the variables
    def GetPath(self, consider_itinerary = False):
        if consider_itinerary:
            return self.next_itinerary_nodes, self.time_needed_to_next_node, self.dis_to_next_node
        else:
            return self.next_positions, self.time_needed_to_next_position, self.dis_to_next_position

