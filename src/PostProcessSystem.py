import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.pyplot import MultipleLocator
import cv2
import os
import copy
from tqdm import tqdm
import numpy as np
from scipy.stats import norm

'''
The subsystem of the control center that visualizes the results
'''
class PostProcessSystem:
    def __init__(self,
                vehicles,
                requests,
                environment,
                current_timepoint
                ):
        self.vehicles = vehicles
        self.requests = requests
        self.environment = environment
        self.current_timepoint = current_timepoint

        self.img_num = 0

        ########### figure settings ###############
        self.title_fs = 35
        self.label_fs = 35
        self.tick_fs = 30
        self.legend_fs = 20
        self.colorbar_fs = 35
        
        # self.fonttype = 'Times New Roman'
        self.fonttype = 'Arial'
        
        self.legendfont = {'family': self.fonttype,
                            'weight': 'normal',
                            'size': self.legend_fs
            }
        self.legend_ms = 1 # legend marker scale
        self.legend_loc = 'upper center'
        self.legend_loc = (0.01, 1.01)
        self.legend_ncol = 2

        self.cb_font = {'family': self.fonttype,
                            'weight': 'normal',
                            'size': self.colorbar_fs
            }
        
        self.cmap = 'Reds'
        
        ###########################################


    # function: statistics of requests' time
    # params: requests
    # return: picture of distribution
    def ReqTimeSta(self, ax, requests):
        time = []
        for reqs in requests:
            for req in reqs:
                time.append(req.send_request_timepoint / 3600)
        # time range
        tmin, tmax = int(self.environment.start_time / 3600), int(self.environment.end_time / 3600)
        plt.hist(time, bins = tmax-tmin, range = [tmin, tmax], density = False, facecolor = 'blue', alpha = 0.5)
        plt.xlabel('Time (h)')
        plt.ylabel('The number of requests')
        
        return ax


    # function: statistics of requests' distance and normal distribution fitting
    # params: requests
    # return: picture of distribution
    def ReqDisSta(self, ax, requests, MaxDis = None, nor_fit = True):
        dis = []
        for reqs in requests:
            for req in reqs:
                dis.append(req.original_travel_distance / 1000)
        dis = np.array(dis)
        # hist diagram       
        if nor_fit:
            _,bins,_ = plt.hist(dis, bins = 100, density = True, facecolor = 'blue', alpha = 0.5) 
            # the mean value and standrad error 
            miu, sigma = np.mean(dis), np.std(dis)
            y = norm.pdf(bins, miu, sigma)
            # plot
            plt.plot(bins, y, 'r--')
            plt.ylabel('Probability Density')
            plt.title('Histogram of Normal Distribution: $\mu = %.3f$, $\sigma=%.3f$'%(miu,sigma))
        else:
            plt.hist(dis, bins = 100, density = False, facecolor = 'blue', alpha = 0.5) 
            plt.ylabel('The number of requests')
        plt.xlabel('Distance (km)')
        
        if MaxDis is not None:
            plt.xlim(0, MaxDis)
        
        return ax

    
    # function: Draw the intial road network
    # params: the predefined picture
    # return: picture
    # Note: the coordinates of nodes are (lng, lat)
    def DrawRoadNetwork(self, ax, TIME = False, congestion = False, speed_lim = [0, 20], axis_lim = None):
        # set axis
        if axis_lim:
            lng_min, lng_max, lat_min, lat_max = axis_lim
        else:
            lng_min, lng_max, lat_min, lat_max = self.environment.area_box
            lng_min = max(lng_min, self.environment.lng_min)
            lat_min = max(lat_min, self.environment.lat_min)
            lng_max = min(lng_max, self.environment.lng_max)
            lat_max = min(lat_max, self.environment.lat_max)
        ax.set_xlim(lng_min, lng_max)
        ax.set_ylim(lat_min, lat_max)

        x_major_locator=MultipleLocator(0.015)
        ax.xaxis.set_major_locator(x_major_locator)

        plt.xlabel('Longitude', fontsize = self.label_fs, fontproperties = self.fonttype)
        plt.ylabel('Latitude', fontsize = self.label_fs, fontproperties = self.fonttype)
        plt.xticks(fontsize = self.tick_fs, fontproperties = self.fonttype)
        plt.yticks(fontsize = self.tick_fs, fontproperties = self.fonttype)
        # ax.xaxis.set_label_position('top')
        # ax.xaxis.set_ticks_position('top') 
        # ax.invert_yaxis() # positive to down
        
        # Initialize roads and speed
        expand = 0.002
        lines, speeds = [], []
        for road in self.environment.roads:
            lng_u, lat_u = road.n1_coord
            lng_v, lat_v = road.n2_coord
            lng_min, lng_max, lat_min, lat_max = self.environment.area_box
            '''
                We only draw the roads in the area of interest
            '''
            if (lng_u >= lng_min - expand and lng_u <= lng_max + expand and lat_u >= lat_min - expand and lat_u <= lat_max + expand
                and lng_v >= lng_min - expand and lng_v <= lng_max + expand and lat_v >= lat_min - expand and lat_v <= lat_max + expand):
                lines.append([road.n1_coord, road.n2_coord])
                speeds.append(road.speed)
        lines = np.array(lines)
        speeds = np.array(speeds)
        # We use color map to visualize the traffic speed (congestion)
        if congestion:
            # Project speed [0 m/s, 20 m/s] to color
            norm = plt.Normalize(speed_lim[0], speed_lim[1])
            lc = LineCollection(lines, norm = norm, linewidths=1, linestyles='solid', cmap = self.cmap)
            lc.set_array(speeds)
            # set the color bar
            cb = plt.colorbar(lc)
            cb.ax.tick_params(labelsize = self.tick_fs)
            cb.set_label('Average traffic speed (m/s)', fontdict = self.cb_font)
        else:
            lc = LineCollection(lines, color = 'gray', linewidths=1, linestyles='solid') # Draw the road network in gray when we do not consider traffic congestion
        
        # Draw the road network
        ax.add_collection(lc)
        
        # Show the time
        if TIME:
            ax.set_title(self.GetTime(), y=0, fontsize = self.title_fs, fontproperties = self.fonttype, fontweight ="bold")

        return ax
    
    
    # Draw the distribution of vehicles
    def DrawVehicles(self, ax, vehicles, v_size, colors = ['darkolivegreen', 'darkviolet']):

        # Draw the vehicles' distribution
        for vehicle in vehicles:
            if not isinstance(vehicle.current_position, tuple):
                xv, yv = self.environment.node_id_to_coord[vehicle.current_position]
            else:
                xv, yv = vehicle.current_position
            # We use rectangle to represent vehicles
            color = colors[0] if vehicle.current_capacity + sum([req.num_person for req in vehicle.next_requests]) == 0 else colors[1]
            rec = plt.Rectangle((xv-v_size/2, yv-v_size/2), v_size, v_size, facecolor = color, alpha = 1)
            ax.add_patch(rec)
        
        return ax


    # function: Draw the distribution of all requests
    # params: The predefined picture, all requests, position type, area of circles, etc.
    # return: The picture of the distribution of all requests
    '''
        (1) It's so complicated to draw the vehicles and requests at each time step that we can draw the distribution of all requests, from
        which we can have a general view of the simulation.
        (2) There may be more than one request at each node, so we use different colors to represent the number of the requests at a node, 
        i.e., green: [1,2]; bule: [3,4]; orange: [5,6]; yellow: [7,8]; red: [9, +]
    '''
    def DrawRequests(self, ax, requests, type = 'pickup', s = 10, count = False, cmap = 'viridis', cmax = 10, color = 'red', draw_grid = False):
        
        lngs, lats, nums, areas = [], [], [], []
        
        # record all nodes
        coords_to_num = {}
        for node in self.environment.nodes_coordinate:
            coords_to_num[node] = 0
        
        # Count the number of requests at each node
        for reqs in requests:
            if isinstance(reqs, list):
                for req in reqs:
                    if type == 'pickup': # distribution of pickup positions
                        lng, lat = req.pickup_position
                    else: # distribution of dropoff positions
                        lng, lat = req.dropoff_position
                    coords_to_num[(lng, lat)] += 1
            else:
                if type == 'pickup': # distribution of pickup positions
                    lng, lat = reqs.pickup_position
                else: # distribution of dropoff positions
                    lng, lat = reqs.dropoff_position
                coords_to_num[(lng, lat)] += 1
        
        # Filter nodes without requests
        for coord in coords_to_num:
            if coords_to_num[coord] > 0:
                lngs.append(coord[0])
                lats.append(coord[1])
                nums.append(coords_to_num[coord])
                areas.append(coords_to_num[coord] * 3)
        
        # Show the number of requests
        if count:          
            plt.scatter(lngs, lats, c = nums, cmap = cmap, s = areas)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize = self.tick_fs)
            cb.set_label('The number of requests', fontdict = self.cb_font)
            # ticks of colorbar
            ticks = [1]
            for i in range(2, cmax):
                if i % 5 == 0:
                    ticks.append(i)
            ticks.append(cmax)
            cb.set_ticks(ticks)
            plt.clim(1,cmax)
        else:
            plt.scatter(lngs, lats, c = color, s = s)
            
        # plot the grids
        if draw_grid:
            lng_min, lng_max, lat_min, lat_max = self.environment.area_box
            for i in range(0, self.environment.x_grid_num + 1):
                x = lng_min + i * (lng_max - lng_min) / self.environment.x_grid_num
                plt.plot((x, x), (lat_min, lat_max), color = 'linen', linestyle = 'dashed', linewidth = 0)
            for i in range(0, self.environment.y_grid_num + 1):
                y = lat_min + i * (lat_max - lat_min) / self.environment.y_grid_num
                plt.plot((lng_min, lng_max), (y, y), color = 'linen', linestyle = 'dashed', linewidth = 0)   
        
        return ax
        
    
    
    # function: Draw the snapshot of vehicles and requests
    # params: The predefined picture, vehicle size, etc.
    # return: picture
    def DrawSnapshot(self, ax, v_size = 0.004, s = 100, colors = ['lightslategray', 'darkviolet', 'red', 'orange', 'lime'], draw_route = True, draw_road_netwrod = True, speed_lim = [0, 20], axis_lim = None):
        # Draw the road network first
        if draw_road_netwrod:
            ax = self.DrawRoadNetwork(ax, TIME = True, congestion = True, speed_lim = speed_lim, axis_lim = axis_lim)
        # It takes so much time to draw the road network, so we can only draw the vehicles and requests to visualize the results
        else:
            ax.set_title(self.GetTime(), y=0, fontsize = self.title_fs, fontproperties = self.fonttype, fontweight ="bold")
        
        # set colors
        if len(colors) == 5:
            c_veh_cru, c_veh_ser, c_req_w, c_req_pic, c_req_del = colors
        else:
            c_veh_cru, c_veh_ser, c_req_w, c_req_pic, c_req_del = 'lightslategray', 'darkviolet', 'red', 'orange', 'lime'
        
        # Draw the vehicles
        ax = self.DrawVehicles(ax, self.vehicles, v_size, [c_veh_cru, c_veh_ser])
        
        # Draw requests that have been assigned or picked up
        for vehicle in self.vehicles:
            xv,yv = vehicle.current_position
            # Draw routes of vehicles
            if draw_route:
                if vehicle.path is not None:
                    nodes = copy.deepcopy(vehicle.path.next_itinerary_nodes)
                    if len(nodes) == 0:
                        nodes = copy.deepcopy(vehicle.path.next_positions)
                    
                    nodes.insert(0, vehicle.current_position) # insert the current node
                    for idx in range(len(nodes) - 1):
                        n1, n2 = nodes[idx], nodes[idx+1]
                        if not isinstance(n1, tuple):
                            x1, y1 = self.environment.node_id_to_coord[n1]
                        else:
                            x1, y1 = n1
                        if not isinstance(n2, tuple):
                            x2, y2 = self.environment.node_id_to_coord[n2]
                        else:
                            x2, y2 = n2
                        plt.plot((x1,x2), (y1,y2), color = 'grey', linestyle = 'dashed', linewidth = 1)
                    
            # We use circles to represent requests
            # lime represents requests have been picked up
            for i, request in enumerate(vehicle.current_requests):
                cir = plt.Circle(self.GetCircleCenter(xv,yv,v_size,i), radius = v_size/4, color=c_req_del, fill=True, alpha=1)
                ax.add_patch(cir)
                #plt.scatter(lng, lat, s = s, c = c_req_del, alpha=1)
                
                # If we draw the route of vehicles, we also the destination of the request
                if draw_route:
                    if not isinstance(request.dropoff_position, tuple):
                        xr_d, yr_d = self.environment.node_id_to_coord[request.dropoff_position]
                    else:
                        xr_d,yr_d = request.dropoff_position
                    plt.scatter(xr_d, yr_d, s = s, c = c_req_del, marker = '*', alpha=0.7)
            
            # orange represents request has been assigned to a vehicle but has not been picked up
            for i,request in enumerate(vehicle.next_requests):
                # The coordinate of the request
                if not isinstance(request.pickup_position, tuple):
                    xr_p,yr_p = self.environment.node_id_to_coord[request.pickup_position]
                else:
                    xr_p,yr_p = request.pickup_position
                #plt.scatter(xr_p, yr_p, s = s, c = c_req_pic, alpha=1)
                cir = plt.Circle((xr_p, yr_p), radius = v_size/4, color=c_req_pic, fill=True, alpha=1)
                ax.add_patch(cir)
                
                # If we draw the route of vehicles, we also the destination of the request
                if draw_route:
                    if not isinstance(request.dropoff_position, tuple):
                        xr_d, yr_d = self.environment.node_id_to_coord[request.dropoff_position]
                    else:
                        xr_d,yr_d = request.dropoff_position
                    plt.scatter(xr_d, yr_d, s = s, c = c_req_del, marker = '*', alpha=0.7)
        
        # Draw the new requests
        # red represents new requests
        for request in self.requests:
            # The coordinate of the request
            if not isinstance(request.pickup_position, tuple):
                xr_p,yr_p = self.environment.node_id_to_coord[request.pickup_position]
            else:
                xr_p,yr_p = request.pickup_position
            # non-ride-pooling
            if request.max_tol_num_person == 1:
                cir = plt.Circle((xr_p, yr_p), radius = v_size/4, color=c_req_w, fill=True, alpha=1)
                ax.add_patch(cir)
            # ride-poolings
            else:
                plt.scatter(xr_p, yr_p, s = 4*s, c = c_req_w, marker = '*', alpha=1)
            
        # ax = self.DrawRequests(ax, self.requests, color = c_req_w)
        
        '''
            Draw the legend
        '''
        plt.scatter(1000, 1000, s = s, c = c_req_del, alpha=1, label = 'Passengers in vehicle')
        plt.scatter(1000, 1000, s = s, c = c_req_w, marker = '*', alpha=1, label = 'Passengers waiting to be assigned (pooling)')
        plt.scatter(1000, 1000, s = s, c = c_req_w, alpha=1, label = 'Passengers waiting to be assigned (non-pooling)')
        plt.scatter(1000, 1000, s = s, c = c_req_pic, alpha=1, label = 'Passengers waiting to be picked up')
        
        rec = plt.Rectangle((1000, 1000), v_size, v_size, facecolor = c_veh_cru, alpha = 1, label = 'Vehicles on cruise')
        ax.add_patch(rec)
        rec = plt.Rectangle((1000, 1000), v_size, v_size, facecolor = c_veh_ser, alpha = 1, label = 'Vehicles in service')
        ax.add_patch(rec)
        
        if draw_route:
            plt.scatter(1000, 1000, s = s, c = c_req_del, marker = '*', alpha=0.7, label = 'Destinations of passengers')
        
        plt.legend(loc = self.legend_loc,
                    ncol = self.legend_ncol,
                    prop = self.legendfont,
                    markerscale = self.legend_ms)
        
        return ax


    # function: Draw the intial road network of toy model
    # params: None
    # return: picture
    def DrawRoadNetworkToyModel(self, ax):
        nodes_coordinate = self.environment.nodes_coordinate
        nodes_connection = self.environment.nodes_connection

        ax.set_xlim(-self.environment.distance_per_line / 1000, max(nodes_coordinate[:,1])/1000 + 1)
        ax.set_ylim(-self.environment.distance_per_line / 1000, max(nodes_coordinate[:,2])/1000 + 1)
        ax.set_xlabel('km')
        ax.set_ylabel('km')
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top') 
        ax.invert_yaxis() # positive to down


        for (i,j) in nodes_connection:
            x1, x2 = nodes_coordinate[i,1] / 1000, nodes_coordinate[j,1] / 1000
            y1, y2 = nodes_coordinate[i,2] / 1000, nodes_coordinate[j,2] / 1000 # unit: km
            plt.plot((x1,x2), (y1,y2), color = 'gray')
        
        
        ax.set_title(self.GetTime(), y=0, fontsize = 22)

        return ax
    

    # function: Draw the vehicles and requests
    # params: None
    # return: picture
    def DrawVehiclesandReuqestsToyModel(self, ax):
        #ax = self.DrawRoadNetworkToyModel(ax) # Draw the road network first

        nodes_coordinate = self.environment.nodes_coordinate
        v_size = self.environment.distance_per_line / 1000 # the size of the vehicle

        # Draw the vehicles and request therein
        for vehicle in self.vehicles:
            xv, yv = nodes_coordinate[int(vehicle.current_position - 1), 1:] / 1000
            # We use rectangle to represent vehicles
            rec = plt.Rectangle((xv-v_size/2, yv-v_size/2), v_size, v_size, facecolor = 'slategrey')
            ax.add_patch(rec)

            # We use circles to represent requests
            # green represents requests have been pick up
            for i, request in enumerate(vehicle.current_requests):
                cir = plt.Circle(self.GetCircleCenter(xv,yv,v_size,i), radius = v_size/4, color="green", fill=True, alpha=1)
                ax.add_patch(cir)
                # We also draw the destination of the request
                xr_d,yr_d = nodes_coordinate[int(request.dropoff_position - 1), 1:] / 1000
                cir = plt.Circle((xr_d, yr_d), radius = v_size/4, color="green", fill=False, alpha=0.5)
                ax.add_patch(cir)
            
            # orange represents request has been allocated to a vehicle but has not been pick up
            for i,request in enumerate(vehicle.next_requests):
                xr_p,yr_p = nodes_coordinate[int(request.pickup_position - 1), 1:] / 1000 
                xr_d,yr_d = nodes_coordinate[int(request.dropoff_position - 1), 1:] / 1000 # The coordinate of the request
                cir = plt.Circle((xr_p, yr_p), radius = v_size/4, color="orange", fill=True, alpha=0.5)
                ax.add_patch(cir)
                cir = plt.Circle((xr_d, yr_d), radius = v_size/4, color="orange", fill=False, alpha=0.5)
                ax.add_patch(cir)
        
        # Draw the new requests
        # red represents new requests
        for request in self.requests:
            xr_p,yr_p = nodes_coordinate[int(request.pickup_position - 1), 1:] / 1000 
            xr_d,yr_d = nodes_coordinate[int(request.dropoff_position - 1), 1:] / 1000 # The coordinate of the request
            cir = plt.Circle((xr_p, yr_p), radius = v_size/4, color="red", fill=True, alpha=0.5)
            ax.add_patch(cir)
            cir = plt.Circle((xr_d, yr_d), radius = v_size/4, color="red", fill=False, alpha=0.5)
            ax.add_patch(cir)
        return ax

    
    # function: convert current timepoint to real time
    # params: None
    # return: real time: string
    def GetTime(self):
        hour = int(self.current_timepoint / 3600)
        min = int((self.current_timepoint - hour * 3600) / 60)
        sec = self.current_timepoint - hour * 3600 - min * 60

        return f'{hour} : {str(min).zfill(2)} : {str(sec).zfill(2)}'

    
    # function: Calculate the position of requests in the vehicle
    # params: the coordinate of the vehicle, the size of the vehicle, and the index of the request
    # return: the circle center of the request
    def GetCircleCenter(self, xv, yv, v_size, i):
        if i == 0:
            return (xv - v_size/4, yv - v_size/4)
        elif i == 1:
            return (xv + v_size/4, yv - v_size/4)
        elif i == 2:
            return (xv + v_size/4, yv + v_size/4)
        else:
            return (xv - v_size/4, yv + v_size/4)

    
    '''
        The distances from south to north and from east to west are not same, so we need to adjust the aspect ratio of the figure
    '''
    # Calculate the aspect ratio of the figure
    def FigAspectRatio(self, box = None):
        if box is None:
            lng_min, lng_max, lat_min, lat_max = self.environment.area_box
        else:
            lng_min, lng_max, lat_min, lat_max = box
        # distance
        x, y = self.environment.LngLat2xy((lng_min, lat_min), (lng_max, lat_max))
        aspect_ratio = abs(y / x)

        return aspect_ratio


    # function: Make all result images a vedio
    # params: the image path
    # return: None
    def MakeVedio(self, imgs = None, img_path = 'output/tmp', vedio_fps=20, vedio_path = 'output', vedio_name = 'result.mp4', del_img = False):
        # read images
        if imgs is not None:
            imgs = imgs
        else:
            imgs = []
            img_names = os.listdir(img_path)
            for idx in range(len(img_names)):
                img_name = str(idx).zfill(6) + '.png'
                img = cv2.imread(os.path.join(img_path, img_name))
                if img is None:
                    continue
                imgs.append(img)
                if del_img:
                    os.remove(os.path.join(img_path, img_name)) # remove the images
        
        # make vedio
        height, width = imgs[0].shape[:2]
        i2v = image2video(width, height)    
        i2v.start(os.path.join(vedio_path, vedio_name), vedio_fps)
        for i in tqdm(range(len(imgs)), desc = 'Making video: '):
            img = imgs[i]
            i2v.record(img)
                
        i2v.end()



class image2video():
    def __init__(self, img_width, img_height):
        self.video_writer = None
        self.is_end = False
        self.img_width = img_width
        self.img_height = img_height 

    def start(self, file_name, fps):
        four_cc = cv2.VideoWriter_fourcc(*'mp4v')
        img_size = (self.img_width, self.img_height)

        self.video_writer = cv2.VideoWriter()
        self.video_writer.open(file_name, four_cc, fps, img_size, True)

    def record(self, img):
        if self.is_end is False:
            self.video_writer.write(img)

    def end(self):
        self.is_end = True
        self.video_writer.release()


class video2image():
    def __init__(self, file, start_frame = 1330, end_frame = 2000):
        video = cv2.VideoCapture(file)
        self.n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)+0.5)
        self.fps = int(video.get(cv2.CAP_PROP_FPS)+0.5)
        self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
        self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
        
        print('start frame extraction ...')
        self.images = []
        if start_frame is None or end_frame is None:
            start_frame, end_frame = 0, self.n_frames
        for frame in range(end_frame):
            if (frame+1) % 50 == 0 and frame >= start_frame:
                print(f'complete {frame+1}/{self.n_frames}')
                #break
            _, image = video.read()
            if image is not None and frame >= start_frame:
                self.images.append(image)
        