'''
The object of a single road

params:
id    : the id of the road (all roads will be saved in a list, so the id is also the order of road in the list)
n12_id: the Open Street Map id of two nodes
n12_coord: the longtitude and latitude of two nodes
lanes : the number of lanes of the road that will be used to calculate the traffic density
length: the length of the road that will be used to calculate the travel time on the road and the total distance of trip
'''
import random

class Road:
    def __init__(self,
                 id,
                 n1_id,
                 n1_coord,
                 n2_id,
                 n2_coord,
                 lanes = 2,
                 length = 0,
                 speed = 20
                 ):
        self.id = id
        self.n1_id = n1_id
        self.n1_coord = n1_coord
        self.n2_id = n2_id
        self.n2_coord = n2_coord
        self.lanes = lanes
        self.length = length # unit: m

        self.speed = speed  # unit: m/s
        self.time = self.length / self.speed # unit: s
        
        '''
            parameters of Greenshield model
        '''
        self.uf = 24
        self.kj = 0.2
        self.alpha = 0.75
        self.beta = 5
        self.basic_k = 0.1 + (random.random() - 0.5) * 0.06 # basic traffic flow excluding ride-sourcing 
        
        # Count the number of vehicles on the road that can be used to calculate the traffic density and flow
        # Note: all the vehicles on each lane should be specified if conducting micro-simulations (car-following, traffic lights, etc.)
        self.num_vehs = 0
        self.vehicles = {}


    # function: Calculate the real-time average speed of vehicles on the road according to the Greenshield model
    # params: uf (speed when free flow), kj (density when traffic jam), alpha (used to calibrate the free speed), beta (used to calibrate density)
    # return: the average speed of vehicles
    # Note: the units of density and speed are vehicle/m/lane and m/s, respectively.
    def UpdateSpeed(self):
        k = self.num_vehs / self.length / self.lanes
        speed = self.alpha * self.uf * (1 - (self.beta * k + self.basic_k) / self.kj) # Greenshield model
        
        # We assume the minimal traffic speed is 1 m/s
        self.speed = max(speed, 1)
        self.time = self.length / self.speed
        