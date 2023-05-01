from cmath import cos, sin
import numpy as np
import osmnx as ox
import math
import random
import copy

from .utils.HotDistribution import HotDistribution
from .component.Road import Road

'''
The object of the environment of interest
The object only provides the static physical information
The dynamic process is realized in the control cnter
'''
class ENVIRONMENT:
    def __init__(self,
                cfg = None,
                vehicles = None
                ):
        self.cfg = cfg
        
        # vehicles
        self.vehicle_velocity = self.cfg.VEHICLE.VELOCITY
        self.vehicles = vehicles
        
        # get file path
        self.network_file_path = self.cfg.ENVIRONMENT.CITY.RoadFile
        
        # split grids
        self.x_grid_num = self.cfg.ENVIRONMENT.CITY.X_GRID_NUM
        self.y_grid_num = self.cfg.ENVIRONMENT.CITY.Y_GRID_NUM
        # simulation time
        self.step_time = self.cfg.SIMULATION.STEP_TIME
        self.start_time = self.cfg.SIMULATION.START
        self.end_time = self.cfg.SIMULATION.END
        
        # The target area
        self.lng_min, self.lng_max = cfg.ENVIRONMENT.MINLNG, cfg.ENVIRONMENT.MAXLNG
        self.lat_min, self.lat_max = cfg.ENVIRONMENT.MINLAT, cfg.ENVIRONMENT.MAXLAT

        # Note: Using osmnx' API to generate itinerary nodes may be too slow to run the simulation for many epochs
        # We can generate the itinerary nodes between each pair of nodes ahead of time and save the results at mongodb
        # that we can call when we want to generate itinerary nodes, which will accelerate the simulation
        self.consider_itinerary = self.cfg.ENVIRONMENT.CONSIDER_ITINERARY.TYPE
        self.itinerary_method = self.cfg.ENVIRONMENT.CONSIDER_ITINERARY.METHOD
        self.consider_congestion = self.cfg.ENVIRONMENT.CONSIDER_CONGESTION

        if self.itinerary_method == 'database':
            import pymongo
            """
            Here, we build the connection to mongodb, which will be used to speed up access to road network information.
            """
            myclient = pymongo.MongoClient("mongodb://localhost:27018/")
            mydb = myclient["route_network"]
            self.mycollect = mydb['route_list']

        # Initialize road network
        self.road_network, self.node_coord_to_id, self.node_id_to_coord, self.nodes_coordinate, self.roads, self.nodes_to_road = self.InitializeEnvironment()
        self.node_coord_to_grid, self.grid_center_node, self.nodes_coordinate_grid, self.area_box = self.SplitGrids()
        self.node_lnglat_to_xy = self.LngLat2xy_all()

        # Record hot areas (including vehicles and requests) in the previous 30 mins and all time
        self.requests_distribution = HotDistribution(type = 'requests',
                                                    step_time = self.step_time,
                                                    record_time = 1800,
                                                    x_grid_num = self.x_grid_num,
                                                    y_grid_num = self.y_grid_num,
                                                    area_box=self.area_box)
        self.vehicles_distribution = HotDistribution(type = 'vehicles',
                                                    step_time = self.step_time,
                                                    record_time = 1800,
                                                    x_grid_num = self.x_grid_num,
                                                    y_grid_num = self.y_grid_num,
                                                    area_box=self.area_box)
        self.past_requests = []
    
    # function: Update distribution of requests and vehicles
    # params: requests and vheilces at the current step
    # return: None
    def UpdateDistributions(self, requests, vehicles):
        #self.requests_distribution.Update(requests)
        #self.vehicles_distribution.Update(vehicles)
        self.past_requests.extend(requests)


    # function: Initialize road network, including nodes and roads
    # params: data director
    # return: the road network graph, mutual converting between node id and coordinates, roads, and node id to road id
    def InitializeEnvironment(self):
        # Load road network: nodes and edges from graphml file
        if self.network_file_path:
            G = ox.load_graphml(self.network_file_path)
        # Directly download road network from Open Street Map according to the given box
        else:
            G = ox.graph.graph_from_bbox(north = self.lat_max, south = self.lat_min, east = self.lng_max, west = self.lng_min, network_type = "drive")
        
        # Load nodes' id and coordinate
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
        nodes_id = gdf_nodes.index.tolist()
        nodes_lng = gdf_nodes['x'].tolist()
        nodes_lat = gdf_nodes['y'].tolist()
        
        # mutual conversion
        node_id_to_coord, node_coord_to_id = {}, {}
        nodes_coordinate = []
        for idx in range(len(nodes_lng)):
            node_id_to_coord[nodes_id[idx]] = (nodes_lng[idx], nodes_lat[idx])
            node_coord_to_id[(nodes_lng[idx], nodes_lat[idx])] = nodes_id[idx]
            
            # split road network and record the coordiantes of all nodes
            lng, lat = nodes_lng[idx], nodes_lat[idx]
            if lng >= self.lng_min and lng <= self.lng_max and lat >= self.lat_min and lat <= self.lat_max:
                nodes_coordinate.append((lng, lat))
       
        # Roads
        expand = 0.1
        roads = []
        road_id = 0
        nodes_to_road = {}
        for e in G.edges():
            lng_u, lat_u = node_id_to_coord[e[0]]
            lng_v, lat_v = node_id_to_coord[e[1]]
            # We only consider the road network in the area of interest
            if (lng_u >= self.lng_min - expand and lng_u <= self.lng_max + expand and lat_u >= self.lat_min - expand and lat_u <= self.lat_max + expand
                and lng_v >= self.lng_min - expand and lng_v <= self.lng_max + expand and lat_v >= self.lat_min - expand and lat_v <= self.lat_max + expand):
                # Define the road
                try:
                    lanes = float(G[e[0]][e[1]][0]['lanes'])
                # If there is no 'lanes' attribute, we approximate the lanes using the road type
                except:
                    fclass = G[e[0]][e[1]][0]['highway']
                    if fclass == 'primary' or fclass == 'trunk':
                        lanes = 3
                    elif fclass == 'secondary':
                        lanes = 2
                    else:
                        lanes = 1
                lanes = max(lanes, 1)
                
                length = G[e[0]][e[1]][0]['length'] # unit : m
                road = Road(id = road_id,
                            n1_id = e[0],
                            n1_coord = (lng_u, lat_u),
                            n2_id = e[1],
                            n2_coord = (lng_v, lat_v),
                            lanes = lanes,
                            length = length,
                            speed=self.vehicle_velocity)
                roads.append(road)
                road_id += 1
                # Add the average travel time on the road to the graph that will be used to calculate shortest route
                G[e[0]][e[1]][0]['time'] = road.time # unit : h
                # Establish the conversion from two nodes' id and the road id
                nodes_to_road[(e[0], e[1])] = road_id
                
                '''
                    If the road is bi-directional, we record it as two individual raods
                '''
                if not G[e[0]][e[1]][0]['oneway']:
                    road = Road(id = road_id,
                            n1_id = e[1],
                            n1_coord = (lng_v, lat_v),
                            n2_id = e[0],
                            n2_coord = (lng_u, lat_u),
                            lanes = lanes,
                            length = length,
                            speed=self.vehicle_velocity)
                    roads.append(road)
                    road_id += 1
                    # Add the average travel time on the road to the graph that will be used to calculate shortest route
                    G[e[1]][e[0]][0]['time'] = road.time # unit : h
                    # Establish the conversion from two nodes' id and the road id
                    nodes_to_road[(e[1], e[0])] = road_id
                
        return G, node_coord_to_id, node_id_to_coord, nodes_coordinate, roads, nodes_to_road

    
    # function: Get road id according to the given two nodes
    # params: two nodes
    # return: the corresponding road id
    def GetRodeID(self, node1, node2):
        if isinstance(node1, tuple):
            node1 = self.node_coord_to_id[node1]
        if isinstance(node2, tuple):
            node2 = self.node_coord_to_id[node2]
        try:
            road_id = self.nodes_to_road[(node1, node2)]
        except:
            road_id = None
        
        return road_id

    
    
    # function: Update traffic speed and travel time for the road and the corresponding graph
    # params: road id
    # return: None
    def UpdateSpeed(self, road_id, veh_id, JOIN = True):
        # Update the vehicles on the road
        if JOIN:
            self.roads[road_id].num_vehs += 1
            self.roads[road_id].vehicles[veh_id] = 1
        else:
            self.roads[road_id].num_vehs -= 1
            self.roads[road_id].vehicles.pop(veh_id)
        # update traffic speed for the road
        self.roads[road_id].UpdateSpeed()
        # Update the speed for all vehicles on the road
        for veh_id in self.roads[road_id].vehicles:
            self.vehicles[veh_id].speed = self.roads[road_id].speed
        
        # Update travel time for Graph
        self.road_network[self.roads[road_id].n1_id][self.roads[road_id].n2_id][0]['time'] = self.roads[road_id].time



    # function: Split the environment to grids
    # params: The horizontal and vertical grid number of the environment
    # return: node coordinate to grid N.o. & nodes' coordinate in each grid
    def SplitGrids(self):
        node_coord_to_grid = {}
        grid_center_node = {}
        nodes_coordinate_grid = np.zeros((self.y_grid_num, self.x_grid_num), dtype = list)
        nodes_coord_np = np.array(self.nodes_coordinate)
        # The border of the environment
        lng_max, lng_min = np.max(nodes_coord_np[:,0]), np.min(nodes_coord_np[:,0])
        lat_max, lat_min = np.max(nodes_coord_np[:,1]), np.min(nodes_coord_np[:,1])
        # scale of each grid
        delta_x = (lng_max - lng_min) / self.x_grid_num
        delta_y = (lat_max - lat_min) / self.y_grid_num
        # associate each node of the environment to a grid
        for (lng, lat) in self.nodes_coordinate:
            x_num = math.floor((lng - lng_min) / delta_x)
            y_num = math.floor((lat - lat_min) / delta_y)
            
            # Nodes on the border
            if x_num == self.x_grid_num:
                x_num -= 1
            if y_num == self.y_grid_num:
                y_num -= 1
            
            node_coord_to_grid[(lng, lat)] = (x_num, y_num)
            # connect each node with grid
            if nodes_coordinate_grid[self.y_grid_num - y_num - 1, x_num] == 0:
                nodes_coordinate_grid[self.y_grid_num - y_num - 1, x_num] = [(lng,lat)]
            else:
                nodes_coordinate_grid[self.y_grid_num - y_num - 1, x_num].append((lng, lat))
        
        # calculate the center node of each grid, which will be used to guide reposition
        for x in range(self.x_grid_num):
            for y in range(self.y_grid_num):
                nodes_list = nodes_coordinate_grid[y, x]
                if not isinstance(nodes_list, list): # Make sure there exists a node in the grid
                    grid_center_node[y, x] = None
                else:
                    # calculate the mean value
                    mean_node = np.mean(np.array(nodes_list), axis = 0)
                    # the minimal Manhattan distance to the mean node
                    center_node = min(nodes_list, key = lambda x: abs(x[0] + mean_node[0]) + abs(x[1] + mean_node[1]))
                    grid_center_node[y,x] = center_node

        return node_coord_to_grid, grid_center_node, nodes_coordinate_grid, [lng_min, lng_max, lat_min, lat_max]

    
    # function: Convert coordinate system from longtitude & latitude to x & y
    # params: None
    # return: dict of converting (lng, lat) to (x,y)
    # Note: It's computing expensive to calculate distance using longtitude and latitude, so we convert nodes' coordinate to xy
    def LngLat2xy_all(self):
        node_lnglat_to_xy = {}
        ori = np.mean(np.array(self.nodes_coordinate), axis = 0)
        
        for des in self.nodes_coordinate:
            x, y = self.LngLat2xy(ori, des)
            node_lnglat_to_xy[des] = (x, y)
            
        return node_lnglat_to_xy



    # function: Convert coordinate system from longtitude & latitude to x & y
    # params: origin and destination
    # return: (x,y)
    def LngLat2xy(self, ori, des):
        ori_lng, ori_lat = ori[0] * math.pi / 180., ori[1] * math.pi / 180.
        des_lng, des_lat = des[0] * math.pi / 180., des[1] * math.pi / 180.
        Earth_R = 6371393 # unit: meter
        # Distance
        dis_EW = Earth_R * math.acos(min(1, math.cos(ori_lat)**2 * math.cos(ori_lng - des_lng) + math.sin(ori_lat)**2))
        dis_NS = Earth_R * abs(ori_lat - des_lat)
        x = int(dis_EW) * np.sign(des_lng - ori_lng)
        y = int(dis_NS) * np.sign(des_lat - ori_lat)

        return (x, y)


    '''
    If we assume drivers move from origin to destination along the straight line, then itinerary nodes are not located at intersections.
    Therefore, we need to recalculate grid N.O. of each position
    '''
    def Coord2Grid(self, coord):
        lng_min, lng_max, lat_min, lat_max = self.area_box
        delta_x = (lng_max - lng_min) / self.x_grid_num
        delta_y = (lat_max - lat_min) / self.y_grid_num
        lng, lat = coord
        x_num = math.floor((lng - lng_min) / delta_x)
        y_num = math.floor((lat - lat_min) / delta_y)
        
        if x_num == self.x_grid_num:
            x_num -= 1
        if y_num == self.y_grid_num:
            y_num -= 1
        
        return x_num, y_num



    # function: Given a position, find the nearest road node
    # params: The position's coordinate (lng, lat)
    # return: the nearest road node' coordinate (lng, lat)
    def GetNearestNode(self, node):
        nearest_dis = 99999999
        nearest_node = None
        for road_noad in self.nodes_coordinate:
            dis, _ = self.GetDistanceandTime(node, road_noad)
            if dis < nearest_dis:
                nearest_dis = dis
                nearest_node = road_noad
        assert nearest_node is not None
        
        return nearest_node
    
    
    # function: Calculate the travel distance and time between origin and destination according to the type
    # params: The origin and destination position, type: 'Linear', 'Manhattan' or 'Itinerary'
    # return: the travel distance and time
    ''' 
        Note: Since it is not necessary to consider itinerary nodes, i.e., Manhattan distance is good enough (e.g., check constraints),
        we allow users to choose whether considering itinerary nodes or not when calculating distance.
    '''
    def GetDistanceandTime(self, origin, destination, type = 'Linear', congestion_factor = 1.0):
        # if the input origin and destination are node id, then we convert them to coordinate
        if not isinstance(origin, tuple):
            origin = self.node_id_to_coord[origin]
        if not isinstance(destination, tuple):
            destination = self.node_id_to_coord[destination]
        
        # convert to xy coordinate system
        try:
            x1, y1 = self.node_lnglat_to_xy[origin]
            x2, y2 = self.node_lnglat_to_xy[destination]
        except:
            x1, y1 = 0, 0
            x2, y2 = self.LngLat2xy(origin, destination)
        
        # convert lat and lng to distance
        if type == 'Linear': # linear distance
            dis = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        elif type == 'Manhattan': # Manhattan distance
            dis = abs(x1 - x2) + abs(y1 - y2)
        else:
            raise NotImplementedError
        # Here congestion_factoe = 1 means we don't consider congestion
        time = dis / self.vehicle_velocity

        return dis, time

        

    # function: Calculate or export the travel path between the origin and the destination
    # params: The origin and destination position
    # return: The travel path (intersections) between two positions
    # Note: method = 'straight' or 'API' or 'database'
    def GetItinerary(self, origin, destination, method = 'straight'):
        if origin == destination: # The origin is same as destination so that distance = 0, time = 0
            return [origin, destination], [0], [0]
       
        itinerary = None

        # choose a method
        # We use osmnx's APIs
        if method == 'API':
            if isinstance(origin, tuple):
                origin = self.node_coord_to_id[origin]
            if isinstance(destination, tuple):
                destination = self.node_coord_to_id[destination]
            '''
                We should use travel time to calculate the shortest route when considering congestion
            '''
            if self.consider_congestion:
                itinerary = ox.distance.shortest_path(self.road_network, origin, destination, weight='time', cpus=16)
            else:
                itinerary = ox.distance.shortest_path(self.road_network, origin, destination, weight='length', cpus=16)
        
        # In order to accelerate the simulation when considering itinerary nodes, we caculate itinerary nodes between each pair of nodes
        # and save them in the database, so that we can search itinerary nodes from the database when we need. 
        elif method == 'database':
            if isinstance(origin, tuple):
                origin = self.node_coord_to_id[origin]
            if isinstance(destination, tuple):
                destination = self.node_coord_to_id[destination]

            data = {
                'node': str(origin) + str(destination)
            }
            re = self.mycollect.find_one(data)
            if re:
                itinerary = [int(item) for item in re['itinerary_node_list'].strip('[').strip(']').split(', ')]
            else:
                itinerary = ox.distance.shortest_path(self.road_network, origin, destination, weight='length', cpus=16)
                
        # It's so computationally expensive to use osmnx's APIs to get the itinerary nodes. Therefore, we can simplify the problem as follows: 
        # the vehicles move from the origin to the destination along the straight line. And we split the line into several short lines.
        elif method == 'straight':
            dis, _ = self.GetDistanceandTime(origin, destination)
            split_num = int(dis / 300) # we set each short line about 300 meters
            itinerary = [] # include the origin and destination
            itinerary.append(origin)
            for i in range(split_num - 1):
                lng = origin[0] + (destination[0] - origin[0]) / split_num * (i+1)
                lat = origin[1] + (destination[1] - origin[1]) / split_num * (i+1)
                itinerary.append((lng, lat))
            itinerary.append(destination)
        
        else:
            raise NotImplementedError
        
        if itinerary is None: # We only consider the origin and destination if the API can't find the itinerary
            #return None, None, None
            itinerary = [origin , destination]
        
        # Calculate distance and time
        dis , time = [], []
        for node_idx in range(len(itinerary) - 1):
            # We need to calculate the time according to the traffic density of road if we consider traffic congestion
            if self.consider_congestion:
                try:
                    road_id = self.nodes_to_road[(itinerary[node_idx], itinerary[node_idx + 1])]
                    d, t = self.roads[road_id].length, self.roads[road_id].time
                except:
                    d, t = self.GetDistanceandTime(itinerary[node_idx], itinerary[node_idx + 1])
            else:
                d, t = self.GetDistanceandTime(itinerary[node_idx], itinerary[node_idx + 1])
            dis.append(d)
            time.append(t)
            # convert node id to (lng, lat)
            if not isinstance(itinerary[node_idx], tuple):
                itinerary[node_idx] = self.node_id_to_coord[itinerary[node_idx]]
        if not isinstance(itinerary[-1], tuple):
            itinerary[-1] = self.node_id_to_coord[itinerary[-1]]
        
        return list(itinerary), dis, time



    # function: Simulate or export road congestion 
    # params: todo...
    # return: todo...
    # Note: it may be too complicated to consider the road congestion, so that we may not consider the congestion at the first development stage
    def GetCongestion(self):
        raise NotImplementedError



'''
The object of the environment of interest
The object only provides the static physical information
The dynamic process is realized in the control cnter
'''
class EnvironmentToyModel:
    def __init__(self,
                num_nodes = 10,
                distance_per_line = 1000,
                vehicle_velocity = 20/3.6,
                consider_congestion = False
                ):
        self.num_nodes = num_nodes
        self.distance_per_line = distance_per_line
        self.vehicle_velocity = vehicle_velocity
        self.consider_congestion = consider_congestion
        self.nodes_coordinate, self.nodes_connection = None, None
        # Initialize network
        self.nodes_coordinate, self.nodes_connection = self.InitializeEnvironment()

    
    # function: Initialize road network, including shortest path, traval time, travel distance, etc.
    # params: data director or database API
    # return: num_nodes * 3: [node_id, x, y]; List[(node_id, node_id)]: connection
    def InitializeEnvironment(self):
        total_num_nodes = self.num_nodes ** 2
        nodes_coordinate = np.zeros((total_num_nodes, 3))
        nodes_connection = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                nodes_coordinate[i*self.num_nodes + j, 0] = i * self.num_nodes + j + 1 # node_id
                nodes_coordinate[i*self.num_nodes + j, 1] = j * self.distance_per_line # x coordinate (positive to right)
                nodes_coordinate[i*self.num_nodes + j, 2] = i * self.distance_per_line # y coordinate (positive to down)
        for i in range(total_num_nodes-1):
            for j in range(i+1, total_num_nodes):
                if self.GetTravelDistance(nodes_coordinate[i,0], nodes_coordinate[j,0]) <= self.distance_per_line:
                    nodes_connection.append((i,j))
        return nodes_coordinate, nodes_connection


    # function: Calculate or export the shortest travel time between the origin and the destination
    # params: The origin and destination position
    # return: The shortest travel time between two positions
    def GetTravelTime(self, origin, destination, consider_itinerary = None, dis = None, congestion_factor = 1.0):
        # assert origin >0 and origin <= self.num_nodes **2 and  destination >0 and destination <= self.num_nodes **2
        # if origin == destination:
        #     return 0
        if dis is not None:
            total_distance = dis
        else:
            ori_row, des_row = np.ceil(origin / self.num_nodes), np.ceil(destination / self.num_nodes)
            ori_col, des_col = origin - (ori_row - 1) * self.num_nodes , destination - (des_row - 1) * self.num_nodes
            total_distance = (abs(ori_row - des_row) + abs(ori_col - des_col)) * self.distance_per_line
            
        return total_distance / self.vehicle_velocity



    # function: Calculate or export the shortest travel distance between the origin and the destination
    # params: The origin and destination position
    # return: The shortest travel distance between two positions
    def GetTravelDistance(self, origin, destination, consider_itinerary = None):
        assert origin >0 and origin <= self.num_nodes **2 and  destination >0 and destination <= self.num_nodes **2
        if origin == destination:
            return 0
        
        ori_row, des_row = np.ceil(origin / self.num_nodes), np.ceil(destination / self.num_nodes)
        ori_col, des_col = origin - (ori_row - 1) * self.num_nodes , destination - (des_row - 1) * self.num_nodes
        total_distance = (abs(ori_row - des_row) + abs(ori_col - des_col)) * self.distance_per_line

        return total_distance



    # function: Calculate or export the travel path between the origin and the destination
    # params: The origin and destination position
    # return: The travel path (intersections) between two positions
    def GetItineraryNodeList(self, origin, destination):
        assert origin >0 and origin <= self.num_nodes **2 and  destination >0 and destination <= self.num_nodes **2
        assert origin != destination

        # Calculate row and column of origin and destination
        ori_row, des_row = int(np.ceil(origin / self.num_nodes)), int(np.ceil(destination / self.num_nodes))
        ori_col, des_col = int(origin - (ori_row - 1) * self.num_nodes) , int(destination - (des_row - 1) * self.num_nodes)

        itinerary_node_list = [] # Note: We skip the current node (the first node)
        
        if ori_row == des_row: # If the rows of the origin and the destination are same, we only calculate the column difference
            delta_col = ori_col - des_col
            for i in range(1, abs(delta_col)+1):
                node_id = origin - delta_col / abs(delta_col) * i
                itinerary_node_list.append(node_id)
        
        else: # We calculate the row difference first
            delta_row = ori_row - des_row
            for i in range(1, abs(delta_row) + 1):
                node_id_row = origin - delta_row / abs(delta_row) * i * self.num_nodes
                itinerary_node_list.append(node_id_row)
            if ori_col != des_col: # Then we calculate the column difference
                delta_col = ori_col - des_col
                for i in range(1, abs(delta_col)+1):
                    node_id_col = node_id_row - delta_col / abs(delta_col) * i
                    itinerary_node_list.append(node_id_col)
        
        return itinerary_node_list

        

    # function: Simulate or export road congestion 
    # params: todo...
    # return: todo...
    # Note: it may be too complicated to consider the road congestion, so that we may not consider the congestion at the first development stage
    def GetCongestion(self):
        pass