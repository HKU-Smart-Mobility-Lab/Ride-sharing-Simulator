#####################################################
######         Written by Wang CHEN            ######
######     E-mail: wchen22@connect.hku.hk      ######
######     Copyright @ Smart Mobility Lab      ######
######    Department of Civil Engineering      ######
######      Thu University of Hong Kong        ######
#####################################################


from msilib.schema import Environment
from src.Environment import EnvironmentToyModel, ENVIRONMENT
from src.ControlCenter import ControlCenter
from run_episode import RunEpisode

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import logging
import argparse
import yaml
from easydict import EasyDict as edict
import numpy as np
import copy



def parse_args():
    parser = argparse.ArgumentParser(description='Ride-pooling simulator')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--OutputDir',
                        help='output directory',
                        type=str,
                        default='./exp')
    parser.add_argument('--device',
                        help='GPU or CPU',
                        type=str,
                        default='cuda')
    parser.add_argument('--DrawResult',
                        help='Draw the result image of each step',
                        type=bool,
                        default=False)                  
    parser.add_argument('--DrawDistribution',
                        help='Draw the distribution of vehicles and requests',
                        type=bool,
                        default=False)  
    args = parser.parse_args()

    return args



def main():
    args = parse_args()
    # config file
    with open(args.cfg) as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    # log file
    logger = logging.getLogger('')

    # New output filefold
    if not os.path.exists(args.OutputDir):
        os.makedirs(args.OutputDir)
    # New the output filefold of the current experiment
    cfg_file_name = os.path.basename(args.cfg).split('.')[0]
    if not os.path.exists(os.path.join(args.OutputDir, cfg_file_name)):
        os.makedirs(os.path.join(args.OutputDir, cfg_file_name))
    # New the image path
    img_path = os.path.join(args.OutputDir, cfg_file_name, 'tmp')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    
    # set the log file path
    filehandler = logging.FileHandler(os.path.join(args.OutputDir, cfg_file_name, 'simulation.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    
    # Write config information
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')
    
    # For convenience, we load the configrations ahead of time
    # For control center
    start_timepoint = cfg.SIMULATION.START
    end_timepoint = cfg.SIMULATION.END
    step_time = cfg.SIMULATION.STEP_TIME
    # For environment
    velocity = cfg.VEHICLE.VELOCITY
    consider_itinerary = cfg.ENVIRONMENT.CONSIDER_ITINERARY.TYPE
    env_type = cfg.ENVIRONMENT.TYPE
    
    # Initialize environment
    if env_type == 'CITY':
        # Initialize the environment
        environment = ENVIRONMENT(cfg = cfg)
    elif env_type == 'TOY':
        # Initilize the Toy Model
        environment = EnvironmentToyModel(num_nodes = cfg.ENVIRONMENT.TOY.NumNode,
                                        distance_per_line = cfg.ENVIRONMENT.TOY.DisPerLine,
                                        vehicle_velocity = velocity,
                                        consider_congestion = False)
    else:
        raise NotImplementedError

    # Initilize the control center
    control_center = ControlCenter(cfg=cfg, environment = environment)

    # Record the number of requests and vehicles
    total_steps = int((end_timepoint - start_timepoint) / step_time - 1)
    total_grids = int(cfg.ENVIRONMENT.CITY.X_GRID_NUM * cfg.ENVIRONMENT.CITY.Y_GRID_NUM)
    logger.info('The number of steps: {}'.format(total_steps))
    logger.info('The number of grids: {}'.format(total_grids))
    logger.info('******************************')
    

    # Record the results
    def LogResults(logger, requests_results, vehicles_results):
        # Requests
        logger.info('Service rate (non-ride-pooling):  {}'.format(requests_results[0]))
        logger.info('Service rate (ride-pooling):      {}'.format(requests_results[1]))
        logger.info('The average assigning time (s):   {}'.format(requests_results[2]))
        logger.info('The average pick-up time (min):   {}'.format(requests_results[3] / 60))
        logger.info('The average detour time (min):    {}'.format(requests_results[4] / 60))
        logger.info('The average detour time ratio:    {}'.format(requests_results[5]))
        logger.info('The average total time ratio:     {}'.format(requests_results[6]))
        logger.info('The average detour distance (km): {}'.format(requests_results[7] / 1000))
        logger.info('The average detour distance ratio:{}'.format(requests_results[8]))
        logger.info('Cancellation rate (pickup):       {}'.format(requests_results[9]))
        logger.info('Cancellation rate (assign):       {}'.format(requests_results[10]))
        logger.info('Ratio of delivering time to shortest time(ft1):{}'.format(requests_results[11]))
        logger.info('Ratio of delivering time to shortest time(ft2):{}'.format(requests_results[12]))
        logger.info('******************************')
        
        # Vehicles
        logger.info('The average idle time(min):                     {}'.format(vehicles_results[1] / 60))
        logger.info('The total income of all vehicles (USD):         {}'.format(vehicles_results[2]))
        logger.info('The total travel distance of all vehicles (km): {}'.format(vehicles_results[3] / 1000))


    pooling_rates = cfg.REQUEST.POOLING_RATE
    # simulation
    for pooling_rate in pooling_rates:
        # Load requests for test
    
        test_data_path = cfg.REQUEST.DATA.TEST
        requests, req_num, avg_trip_dis = control_center.RTV_system.InitializeRequests(test_data_path, pooling_rate=pooling_rate)
        logger.info('The number of test requests: {} '.format(req_num))

        veh_num = cfg.VEHICLE.NUM
        # Load vehicles
        vehicles = control_center.RTV_system.InitializeVehicles(cfg.VEHICLE.DATA, num_vehicles = veh_num, requests=requests)
        environment.vehicles = vehicles
        
        q = req_num / (end_timepoint - start_timepoint)
        logger.info('******************************')
        logger.info('The number of vehicles (N):       {}'.format(veh_num))
        logger.info('Request rate (q):                 {}'.format(q))
        logger.info('Average trip distance (L):        {}'.format(avg_trip_dis))
        logger.info('Average vehicle velocity (v):     {}'.format(velocity))
        logger.info('Dimensionless parameter (Lq / vB):{}'.format(avg_trip_dis * q / velocity / veh_num))
        logger.info('******************************')
        
        # Draw the distribution of vehicles and requests
        if args.DrawDistribution:
            
            logger.info('Draw the distribution of distance and time of requests ...')
            # time distribution
            fig = plt.figure(figsize=(5,5), dpi=400)
            ax = fig.add_subplot(111)
            ax = control_center.post_process_system.ReqTimeSta(ax, requests=requests)
            plt.subplots_adjust(left=0.15)
            plt.savefig(os.path.join(args.OutputDir, cfg_file_name, 'ReqTimeDist.png'))
            plt.close('all')
            
            # distance distribution
            fig = plt.figure(figsize=(5,5), dpi=400)
            ax = fig.add_subplot(111)
            ax = control_center.post_process_system.ReqDisSta(ax, requests=requests,MaxDis = 25, nor_fit = False)
            plt.subplots_adjust(left=0.15)
            plt.savefig(os.path.join(args.OutputDir, cfg_file_name, 'ReqDisDist.png'))
            plt.close('all')
            logger.info('******************************')


            logger.info('Draw the distribution of vehicles and requests...')
            fig_aspect_ratio = control_center.FigAspectRatio()

            # requests (pickup positions)
            fig = plt.figure(figsize=(15/fig_aspect_ratio*1.25,15), dpi=200)
            ax = fig.add_subplot(111)
            ax = control_center.DrawRoadNetwork(ax, TIME = False, congestion = False)
            ax = control_center.DrawRequests(ax, requests, type = 'pickup', s = 15, count = True, cmap = 'Reds', cmax = 40, color = 'red', draw_grid = False)
            plt.subplots_adjust(left=0.15)
            plt.savefig(os.path.join(args.OutputDir, cfg_file_name, 'requests.png'))
            plt.close('all')

            # vehicles
            fig = plt.figure(figsize=(15/fig_aspect_ratio,15), dpi=200)
            ax = fig.add_subplot(111)
            ax = control_center.DrawRoadNetwork(ax, TIME = False, congestion = False)
            ax = control_center.DrawVehicles(ax, vehicles, v_size = 0.002)
            plt.subplots_adjust(left=0.15)
            
            plt.savefig(os.path.join(args.OutputDir, cfg_file_name, 'vehicles.png'))
            plt.close('all')
            # requests (dropoff positions)
            # fig = plt.figure(figsize=(12,12))
            # ax = fig.add_subplot(111)
            # ax = control_center.DrawRequests(ax, requests, type = 'dropoff', radius = 0.0005)
            # plt.savefig(os.path.join(args.OutputDir, cfg_file_name, 'requests_day' + str(day + 1) + '_dropoff.png'))
            # plt.close('all')
            # logger.info('done!')
            logger.info('******************************')

            break


        epoch_num = 1
        req_num_avg = 0
        requests_results_all = []
        vehicles_results_all = []
        for i in range(epoch_num):
            vehicles_tmp = copy.deepcopy(vehicles)
            requests_tmp = copy.deepcopy(requests)
            if args.DrawResult:
                img_path = os.path.join(args.OutputDir, cfg_file_name,'tmp')
                req_num = RunEpisode(requests_tmp, vehicles_tmp, control_center, draw_veh_req = True, draw_fre = 60, img_path=img_path)
                # visualize the results
                # control_center.MakeVedio(img_path=img_path, vedio_fps=5, vedio_path=os.path.join(args.OutputDir, cfg_file_name), vedio_name='Manhattan-v50-day'+ str(day) +'-fps5.mp4')
                control_center.MakeVedio(img_path=img_path, vedio_fps=10, vedio_path=os.path.join(args.OutputDir, cfg_file_name), vedio_name='Chengdu-v800-pooling'+ str(pooling_rate) +'-fps10.mp4', del_img=False)
            else:
                req_num = RunEpisode(requests_tmp, vehicles_tmp, control_center, draw_veh_req = False)
            
            req_num_avg += req_num
            # Record the results
            requests_results, vehicles_results = control_center.CalculateResults()
            requests_results_all.append(requests_results)
            vehicles_results_all.append(vehicles_results)

            logger.info('****************** Simulation Polling rate: {} *********************'.format(pooling_rate))
            LogResults(logger, requests_results, vehicles_results)
            logger.info('The average number of requests in each vehicle: {}'.format(req_num))
            logger.info('******************************')

            # Reset control center
            control_center.UpdateParameters(timepoint=start_timepoint, step=0)

    
        # Average results
        requests_results_all = np.array(requests_results_all).mean(axis = 0)
        vehicles_results_all = np.array(vehicles_results_all).mean(axis = 0)
        logger.info('****************** Simulation Average Results *********************')
        LogResults(logger, requests_results_all, vehicles_results_all)
        logger.info('The average number of requests in each vehicle: {}'.format(req_num_avg / epoch_num))
        logger.info('******************************')
        


if __name__ == '__main__':
    main()