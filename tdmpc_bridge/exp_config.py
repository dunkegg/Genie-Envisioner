import os
import sys
import math
import argparse

IF_START_WITH_ROSLAUNCH = True

# 25x25x4 world config: threshold_collision: 0.8, img_range: 128, img_width: 128, img_height: 64, pixel_per_meter: 64 / 8.0, traj_img_range: 400, traj_img_scale: 400/20, norm_threshold_collision_distance: 4
#                      src/navigation_research-master/src/robot_agent/kobuki_description/urdf/kobuki_gazebo.urdf.xacro line 221 <min>0.56</min>
# 10x10x4 world config: threshold_collision: 0.4, img_range: 256, img_width: 256, img_height: 128, pixel_per_meter: 128 / 8.0, traj_img_range: 800, traj_img_scale: 800/20, norm_threshold_collision_distance: 2.8
#                      src/navigation_research-master/src/robot_agent/kobuki_description/urdf/kobuki_gazebo.urdf.xacro line 221 <min>0.2</min>

def get_algo_config(parser):

    # ========================================> Policy <========================================
    # Algorithm Parameters
    parser.add_argument('--rl_type', type=str, default='off_policy',
        help="the type of the reinforcement learning  :  ['off_policy', 'on_policy']  ")
    parser.add_argument('--rl_algo_type', type=str, default='SAC',
        help="the type of the reinforcement learning algorithm  :  ['TD3', 'SAC', 'DDPG', 'DQN', 'PPO']  ")
    parser.add_argument('--encoder_type', type=str, default='cnn',
        help="the type of encoder network  :  ['fc', 'cnn', 'gnn', 'lstm', 'deepsets', 'set_transformer']  ")



def get_config():

    # TODO: algo_config, task_config, env_config, using_yaml
    parser = argparse.ArgumentParser(description='navigation_research')
    get_algo_config(parser)
    # ========================================> Basic <========================================
    parser.add_argument('--train', action='store_true', default=True,
        help="if toggled, the training procedure will be started")
    parser.add_argument('--parallel', action='store_true', default=False,
        help="if toggled, the parallel training procedure will be started")
    parser.add_argument('--num_process', type=int, default=1,
        help="the number of parallel process")
    parser.add_argument('--root', default=os.path.dirname(sys.path[0]),
        help="root directory")
    parser.add_argument('--eval_mode', type=bool, default=False,
        help="evaluation mode")
    parser.add_argument('--world_size', type=str, default='10x10x4',
        help="the size of the environment : ['25x25x4', '10x10x4']")
    
    # ========================================> Simulation <========================================
    parser.add_argument('--T', type=float, default=0.1,
        help="the time interval of the simulation")
    parser.add_argument('--num_agent', type=int, default=4,
        help="the number of the agent robots")
    parser.add_argument('--robot_type', type=str, default='turtlebot',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    parser.add_argument('--info_type', type=str, default='obs_based',
        help="the type of the information feedback from the simulation environment  :  ['obs_based', 'state_based']  ")
    parser.add_argument('--state_seq_type', type=str, default='independent',
        help="the type of the state sequence  :  ['independent', 'id', 'random', 'sort']  ")
    parser.add_argument('--connection_type', type=str, default='local',
        help="the type of the connecton between the robot agents  (used in graph-based task)  :  ['global', 'local']")
    parser.add_argument('--action_type', type=str, default='continuous',
        help="the type of the action space in the simulation  :  ['continuous', 'discrete']  ")
    parser.add_argument('--exp_notes', type=str, default='algorithm_0214',
        help="the additional experiment notes")
    
    # ========================================> Task <========================================
    # Auxiliary Task
    parser.add_argument('--auxiliary_control', action='store_true', default=False,
        help="if toggled, the auxiliary control will be initialized")
    parser.add_argument('--manual_control', action='store_true', default=False,
        help="if toggled, the manual control will be initialized")
    parser.add_argument('--reference_control_frequency', type=int, default=5,
        help="the frequency of the reference control")
    # Formation Task
    parser.add_argument('--formation_control', action='store_true', default=False,
        help="if toggled, the formation control will be initialized")
    parser.add_argument('--centralized_control', action='store_true', default=False,
        help="if toggled, the centralized control will be initialized")
    parser.add_argument('--max_formation_seperation', type=int, default=40,
        help="the max distance between formation members, task fails if any member distance get beyond this")
    
    # ========================================> Environment <========================================
    # Basic
    parser.add_argument('--max_distance', type=float, default=40.0,
        help="the max distance considered in simulation, used for normalization")
    
    # Visualization
    parser.add_argument('--vis_agent_info', action='store_true', default=True,
        help="if toggled, each agent's state will be visualized")
    parser.add_argument('--traj_img_range', type=int, default=400*2,
        help="the width and height of the trajactory visualization image") # *2
    parser.add_argument('--traj_img_scale', type=int, default=int(400*2/20),
        help="the scale level of the trajactory visualization image") # *2
    
    # Robot Params
    parser.add_argument('--laser_range', type=float, default=8.0, # 8.0
        help="the max range of laser scan for draw") # 1.5
    parser.add_argument('--true_laser_range', type=float, default=8.0,
        help="real laser range") # 4.0
    parser.add_argument('--max_action', type=float, default=1.0,
        help="the max action")
    parser.add_argument('--min_linear_vel', type=float, default=0.01,
        help="the minimum linear velocity, to enforce movement, in case of staying put")
    parser.add_argument('--max_linear_vel', type=float, default=2.0, # 4.0 irsim simulate / 1.5 real robot / 0.8 smaller / 0.6 7.4realrobot
        help="the maximum linear velocity, 1.0 m/s for simulation, 1.5 m/s for XHC, 1.0m/s for gazebo") # 0.7
    parser.add_argument('--max_angular_vel', type=float, default=1.0, # 3.0 irsim simulate / 3.0 real robot / 0.3 7.4realrobot failed /change to 0.6 7.5
        help="the maximum angular velocity, 1.5 rad/s for simulation, 3.0 rad/s for XHC, 0.9 rad/s for gazebo") # 0.7
    parser.add_argument('--max_linear_acc', type=float, default=0.3, # 0.4 irsim simulate 4 m/s^2 / 0.1 real robot / 0.1 7.4realrobot
        help="the maximum linear acceleration in 0.1s (not 1s !!!), 0.2 m/s^2 for simulation, 0.5 m/s^2 for XHC, 0.2 m/s^2 for gazebo") # 0.15
    parser.add_argument('--max_angular_acc', type=float, default=0.3, # 0.3 irsim simulate 3 rad/s^2 / 0.3 real robot / 0.04 7.4realrobot
        help="the maximum angular acceleration in 0.1s (not 1s !!!), 0.5 rad/s^2 for simulation, 1.0 rad/s^2 for XHC, 0.1 rad/s^2 for gazebo") # 0.08
    
    # Information History Record
    parser.add_argument('--history_length', type=int, default=8, # 64 8
        help="the length of history state record")
    parser.add_argument('--sample_interval', type=int, default=1, # 8 1
        help="the length of sampled interval in history states")
    parser.add_argument('--sample_length', type=int, default=8, # 8 1
        help="the length of history states after sampled")
    
    # Information Dimensions
    parser.add_argument('--vel_dim', type=int, default=2,
        help="the dimension of velocity state")
    parser.add_argument('--goal_dim', type=int, default=2,
        help="the dimension of goal state")
    parser.add_argument('--action_dim', type=int, default=2,
        help="the dimension of action state")
    parser.add_argument('--pose_dim', type=int, default=3,
        help="the dimension of pose state")
    parser.add_argument('--laser_dim', type=int, default=180,
        help="the dimension of laser state")
    parser.add_argument('--observation_dim', type=int, default=180+3,
        help="the dimension of observation")
    parser.add_argument('--state_dim', type=int, default=(180+3)*8+2+2,
        help="the dimension of the final state") # (180+3)*8+2+2
    
    # Discrete Action Space
    parser.add_argument('--discrete_action_dim', type=int, default=1,
        help="the dimension of discrete actions")
    parser.add_argument('--discrete_actions', nargs='+', 
        default=[[-1., 0.], [0., -1.], [0., -0.5], [0., 0.], [0., 0.5], [0., 1.],[1., -1.], [1., -0.5], [1., 0.], [1., 0.5], [1., 1.]],
        help="example list of discrete actions")
    parser.add_argument('--discrete_action_v', nargs='+', 
        default=[-1., -0.5, 0., 0.5, 1.],
        help="example list of discrete linear velocity")
    parser.add_argument('--discrete_action_w', nargs='+', 
        default=[-1., -0.5, 0., 0.5, 1.],
        help="example list of discrete angular velocity")
    
    # Env Setup
    # 25x25x4_little_hard: 5 # 25x25x4_very_hard: 3 (diff_sep or train) # 25x25x4_simple: 4 
    # 20x20wall: 0 # 20x20wallsimple: 6 # 25x25_single_simple: 7 # eval_world_0: 8
    # 25x25x4_hard: 9 # 25x25x4_kind_hard: 10 # 25x25x4_bit_hard: 11 # 10x10x4_simple: 12
    # 10x10x4_very_simple: 13 # 10x10x4_diff_levels: 14 # 10x10x4_diff_levels_1: 15
    # 10x10x4_diff_levels_2: 16 (easier than 14 and 15) # 10x10x4_diff_levels_hard: 17
    # 12x12x1_dyn: 18 # 20x20x1_dyn_center: 19 # 10x10x1_dyn_smp_center: 20
    # hard  > kind hard > bit hard > little hard > simple > very simple
    parser.add_argument('--world_idx', type=int, default=20,
        help="the index of world_idx, if -1, world idx is the same as idx") # 25x25x4_little_hard: 5 # 25x25x4_very_hard: 3 (diff_sep or train) # 25x25x4_simple: 4 # 20x20wall: 0 # 20x20wallsimple: 6 # 25x25_single_simple: 7 # eval_world_0: 8
    parser.add_argument('--target_layout', type=str, default='random',
        help="the layout type of the targets  :  ['random', 'side']  ")
    parser.add_argument('--init_target_position_range', type=int, default=7,
        help="the range of initial target positions")# 6 for 25x25x4_littlehard, 8 for 20x20wall
    parser.add_argument('--init_robot_position_range', nargs='+', default=[9.5, 9.],
        help="the range of initial robot positions  :  [11, 9], [9, 8], [8, 6]") 
    parser.add_argument('--box_pos_list_0', nargs='+', 
        default=[7.12423, 7.11104,
                 6.83398, 6.0556,
                 0.000, 5.95005, 
                 4.22177, 1.913, 
                 -5.91047, 6.87356, 
                 -6.41181, 5.42234,  
                 7.67834, -4.28773, 
                 -7.04507, -1.17418, 
                 -6.09518, -2.28239])
    parser.add_argument('--box_pos_list_1', nargs='+',
        default=[6.54099, 6.66623,
                 5.34412, 6.22089,
                 6.01214, 0.598434,
                 4.593, 1.18294,
                 7.96091, -6.24871,
                 -1.6418, 5.85905,
                 -0.918129, -1.71178,
                 -6.03958, 2.2128,
                 -7.23644, -7.91875])
    parser.add_argument('--box_pos_list_2', nargs='+',
        default=[5.74238, 5.45796,
                 -7.77387, -6.21638]) 
    parser.add_argument('--box_pos_list_3', nargs='+', 
        default=[-9.89, 16.49,
                 -20.12, 10.50,
                 -8.48, 8.51,
                 -19.5, -6.21,
                 -8.04, -6.52,
                 -19.06, -13.26,
                 -9.34, -16.38,
                 10.41, -4.31,
                 16.83, -5.37,
                 7.74, -10.69,
                 11.39, -12.88,
                 11.21, -17.17,
                 6.61, -19.48,
                 15.42, -20.04,
                 6.09, 20.6,
                 12.64, 20.67,
                 16.99, 16.46,
                 4.31, 5.28,
                 5.62, 5.77,
                 20.29, 6.36,
                 -15.74, -8.92,
                 -7.82, -15.89,
                 12.04, -4.06,
                 4.52, -11.7,
                 8.44, -19.43,
                 9.90, 3.62]) # 20 small boxes(1mx1m) 6 big boxes(2mx2m)
    parser.add_argument('--box_pos_list_4', nargs='+',
        default=[-17.31, 15.81,
                 -9.37, 8.17,
                 -14.96, -12.38,
                 9.71, -16.76]) 
    parser.add_argument('--box_pos_list_5', nargs='+',
        default=[-18.46, 10.91,
                 -10.73, 15.90,
                 12.22, 12.60,
                 -16.45, -9.36,
                 8.23, -14.62, 
                 8.01, -7.67]) 
    parser.add_argument('--box_pos_list_6', nargs='+',
        default=[-3.99, 3.17,
                 3.33, -2.73]) 
    parser.add_argument('--box_pos_list_7', nargs='+',
        default=[-5.23, 4.11,
                 -2.94, -4.09,
                 4.84, 5.53]) 
    parser.add_argument('--box_pos_list_8', nargs='+',
        default=[7.35, 10.71,
                 17.08, 17.45,
                 -17.65, 10.71,
                 -7.92, 17.45,
                 -17.65, -14.29,
                 -7.92, -7.55,
                 7.35, -14.29,
                 17.08, -7.55])
    parser.add_argument('--box_pos_list_9', nargs='+',
        default=[-17.74, 8.61,
                 -14.71, 15.51,
                 -8.71, 6.60,
                 -9.99, 11.90,
                 -17.79, 18.79,
                 11.21, 15.23,
                 6.14, 8.19,
                 12.97, 9.91,
                 6.53, 16.54,
                 -17.62, -18.38,
                 -9.16, -5.80,
                 -18.97, -7.09,
                 -12.89, -11.00,
                 -11.85, -12.53,
                 -8.16, -18.48,
                 7.17, -9.25,
                 16.40, -8.46,
                 8.76, -9.01,
                 6.81, -14.18,
                 13.91, -14.46,
                 10.47, -19.14,
                 17.47, -15.29])
    parser.add_argument('--box_pos_list_10', nargs='+',
        default=[16.27, 18.09,
                 15.08, 17.34,
                 13.39, 9.86,
                 8.72, 15.95,
                 8.07, 14.66,
                 16.37, -7.93,
                 8.67, -10.76,
                 15.92, -17.51,
                 12.20, -12.15,
                 -8.94, 16.96,
                 -9.83, 15.67,
                 -17.01, 13.04,
                 -13.73, 18.73,
                 -9.22, -9.40,
                 -10.58, -15.65,
                 -14.57, -8.56,
                 -18.57, -8.93])
    parser.add_argument('--box_pos_list_11', nargs='+',
        default=[15.64, 17.91,
                 16.54, 16.66,
                 7.26, 10.37,
                 -10.10, 16.28,
                 -7.07, 7.74,
                 -18.79, 10.27,
                 -7.93, -9.41,
                 -10.20, -16.45,
                 -11.85, -16.98,
                 8.35, -10.24,
                 9.89, -11.11,
                 13.27, -17.90])
    parser.add_argument('--box_pos_list_12', nargs='+',
        default=[])
    parser.add_argument('--box_pos_list_13', nargs='+',
        default=[])
    parser.add_argument('--box_pos_list_14', nargs='+',
        default=[])
    parser.add_argument('--box_pos_list_15', nargs='+',
        default=[])
    parser.add_argument('--box_pos_list_16', nargs='+',
        default=[])
    parser.add_argument('--box_pos_list_17', nargs='+',
        default=[])
    parser.add_argument('--box_pos_list_18', nargs='+',
        default=[2.64, -2.40,
                 -0.07, 0.73,
                 -2.56, -0.49,
                 -0.71, -1.96,
                 -2.68, 3.47,
                 1.06, -3.28,
                 3.17, 2.97,
                 1.31, -0.04])
    parser.add_argument('--box_pos_list_19', nargs='+',
        default=[-4.16, -6.08,
                 1.80, -6.39,
                 5.27, -4.56,
                 5.77, -0.04,
                 5.85, 4.29,
                 -7.40, 1.72,
                 -6.90, 3.47,
                 -6.28, -3.93,
                 -4.90, 5.81,
                 -2.31, 5.20,
                 1.13, 6.07,
                 -1.86, 1.69,
                 1.74, -1.88,
                 1.57, 2.13,
                 -2.10, -1.08])
    parser.add_argument('--box_pos_list_20', nargs='+',
        default=[1.61, 1.41])
    
    parser.add_argument('--circle_pos_list_0', nargs='+',
        default=[3.27187, -5.07931,
                    2.4539, -4.3405,
                    -1.95257, -7.13743,
                    -4.96057, 4.84184,
                    -4.56478, 3.65447])
    parser.add_argument('--circle_pos_list_1', nargs='+',
        default=[2.83946, -5.44154,
                    -8.09929, -2.51897,
                    -7.06943, 7.44559]) 
    parser.add_argument('--circle_pos_list_2', nargs='+',
        default=[-3.41292, 4.42867,
                    1.59811, -0.203152,
                    8.45103, -4.23906])
    parser.add_argument('--circle_pos_list_3', nargs='+',
        default=[-17.24, 17.73,
                 -15.45, 5.23,
                 -12.61, -4.52,
                 -7.54, -11.39,
                 -17.08, -18.82,
                 5.61, -4.97,
                 5.67, -7.77,
                 17.07, 19.27,
                 16.16, 15.27,
                 12.37, 13.83,
                 16.88, 4.76,
                 17.63, 3.78,
                 -15.96, -17.61,
                 18.01, -11.25,
                 4.6, 13.39,
                 14.00, 6.23,
                 -12.69, -12.25,
                 -11.06, -20.87,
                 13.35, -10.75,
                 4.92, -18.01,
                 17.89, -16.81,
                 9.28, 18.83,
                 5.75, 17.71,
                 9.59, 15.45,
                 7.83, 12.92,
                 19.59, 14.03,
                 10.07, 10.74,
                 14.86, 11.85,
                 17.47, 10.12,
                 5.42, 8.66,
                 12.0, 8.0]) # 12 small circles(1m) 4 big circles(2m) 15 mini circles(0.3m)
    parser.add_argument('--circle_pos_list_4', nargs='+',
        default=[])
    parser.add_argument('--circle_pos_list_5', nargs='+',
        default=[-10.45, -14.59,
                 -19.03, -17.40,
                 9.72, 5.13,
                 7.2, 16.0,
                 -9.0, 8.2,
                 17.42, -10.35,
                 13.19, -19.65,
                 14.97, 16.25,
                 -13.97, 16.06,
                 -9.18, -6.38,
                 -9.56, -19.6])
    parser.add_argument('--circle_pos_list_6', nargs='+',
        default=[-5.98, -3.72,
                 4.79, 7.64,
                 -0.02, -5.96])
    parser.add_argument('--circle_pos_list_7', nargs='+',
        default=[5.27, -6.52,
                 -2.4, 1.5,
                 -6.44, -6.48])
    parser.add_argument('--circle_pos_list_8', nargs='+',
        default=[15.28, 9.15,
                 8.07, 18.72,
                 -9.72, 9.15,
                 -16.93, 18.72,
                 -9.72, -15.85,
                 -16.93, -6.28,
                 15.28, -15.85,
                 8.07, -6.28])
    # circle 0 1 2 6 16 17 box 3 7 9 13
    parser.add_argument('--circle_pos_list_9', nargs='+',
        default=[-14.55, 16.95,
                 -18.53, 11.34,
                 -14.75, 5.57,
                 -8.98, 17.04,
                 -7.59, 17.76,
                 17.84, 11.89,
                 17.65, 10.52,
                 12.90, 6.09,
                 5.79, 17.64,
                 13.42, 16.17,
                 -18.72, -8.60,
                 -12.94, -19.35,
                 -7.65, -10.80,
                 -19.21, -14.52,
                 7.80, -7.58,
                 6.00, -15.71,
                 13.10, -5.62,
                 15.66, -18.30,
                 6.90, 6.82])
    parser.add_argument('--circle_pos_list_10', nargs='+',
        default=[14.99, 9.76,
                 17.88, 12.52,
                 6.83, 10.12,
                 12.47, -10.82,
                 17.94, -14.90,
                 5.69, -16.94,
                 -9.81, 9.34,
                 -17.16, 11.50,
                 -18.36, 17.74,
                 -9.24, -14.30,
                 -15.72, -17.06,
                 -13.38, -6.86])
    parser.add_argument('--circle_pos_list_11', nargs='+',
        default=[9.83, 16.85,
                 14.52, 6.47,
                 18.06, 11.13,
                 -8.65, 7.54,
                 -15.51, 18.30,
                 -6.90, 17.91,
                 -17.41, -8.21,
                 -8.95, -15.15,
                 -16.76, -13.96,
                 16.47, -14.21,
                 15.06, -7.05,
                 6.79, -16.88])
    parser.add_argument('--circle_pos_list_12', nargs='+',
        default=[-6.94, 6.98,
                 -4.19, 6.44,
                 -3.71, 6.52,
                 -4.44, 3.15,
                 -6.27, 4.55,
                 -3.50, -3.31,
                 -4.51, -4.11,
                 -5.79, -4.38,
                 -6.10, -3.89,
                 -7.04, -6.23,
                 -6.67, -6.17,
                 -4.46, -6.15,
                 -4.01, -6.27,
                 5.04, -2.94,
                 3.22, -3.91,
                 6.93, -4.05,
                 4.57, -5.39,
                 4.98, -6.66,
                 5.45, -7.25,
                 5.91, 7.30,
                 3.79, 6.54,
                 6.94, 5.66,
                 2.66, 4.41,
                 5.41, 4.13,
                 5.91, 3.81])
    parser.add_argument('--circle_pos_list_13', nargs='+',
        default=[-6.93, 5.54,
                 -3.72, 6.20,
                 -7.06, -3.8,
                 -6.69, -6.03,
                 3.6, -3.04,
                 5.34, -6.69,
                 5.74, 6.67,
                 3.79, 5.87])
    parser.add_argument('--circle_pos_list_14', nargs='+',
        default=[-7.24, 6.93,
                 -5.62, 3.61,
                 -5.44, 4.12,
                 3.32, 6.35,
                 -4.79, 6.16,
                 6.36, 5.26,
                 6.19, 5.50,
                 -3.44, 3.39,
                 -6.75, -4.57,
                 -4.90, -2.89,
                 -5.26, -3.31,
                 -6.39, -6.94,
                 -5.87, -6.72,
                 -5.75, -6.17,
                 -7.14, -2.85,
                 -3.63, -5.29,
                 3.5, -4.09,
                 6.62, -3.26,
                 4.21, -6.35,
                 5.16, -4.82,
                 3.15, -2.7,
                 6.59, -5.25,
                 6.96, -6.80,
                 2.63, -7.34])
    parser.add_argument('--circle_pos_list_15', nargs='+',
        default=[5.35, 5.51,
                 6.53, 3.97,
                 2.95, 6.97,
                 4.8, 6.9,
                 3.5, 3.0,
                 5.12, -4.49,
                 3.49, -6.60,
                 6.45, -6.65,
                 3.63, -3.13,
                 -4.32, 6.01,
                 -4.86, 5.36,
                 -4.54, 3.24,
                 -6.73, 6.81,
                 -6.67, 6.19,
                 -6.89, 3.63,
                 -3.94, -2.97,
                 -2.66, -6.32,
                 -4.46, -5.27,
                 -7.09, -3.85,
                 -6.68, -4.49,
                 -5.48, -7.14,
                 -6.18, -6.77,
                 -2.89, -2.72,
                 -3.27, 7.02,
                 -2.61, 3.97,
                 -2.27, -4.46,
                 -2.84, -6.60])
    parser.add_argument('--circle_pos_list_16', nargs='+',
        default=[-5.83, 5.15,
                 -4.19, 6.99,
                 2.83, 5.01,
                 7.12, 4.86,
                 5.08, 6.06,
                 -6.83, -3.09,
                 -6.51, -3.45,
                 -7.13, -6.12,
                 -6.64, -5.84,
                 -4.65, -6.41,
                 -4.62, -5.63,
                 -3.77, -2.72,
                 3.60, -4.08,
                 7.04, -2.82,
                 5.27, -6.80,
                 2.98, -6.02,
                 6.62, -4.81])
    parser.add_argument('--circle_pos_list_17', nargs='+',
        default=[5.62, 7.14,
                 4.28, 6.40,
                 2.97, 4.17,
                 5.27, 4.54,
                 6.99, 5.24,
                 7.09, 4.61,
                 6.58, 5.09,
                 2.36, 7.19,
                 4.40, 2.91,
                 6.68, 6.55,
                 6.55, -2.92,
                 5.20, -3.66,
                 4.63, -4.63,
                 3.40, -5.62,
                 2.61, -2.37,
                 6.80, -5.70,
                 4.94, -7.24,
                 4.77, -6.79,
                 3.24, -7.04,
                 4.17, -2.76,
                 -2.64, 7.49,
                 -3.81, 5.81,
                 -5.99, 6.98,
                 -6.07, 4.78,
                 -3.82, 2.95,
                 -2.43, 4.52,
                 -6.58, 3.12,
                 -7.49, 6.11,
                 -7.39, 5.28,
                 -2.62, 2.70,
                 -4.62, 7.64,
                 -2.43, -2.66,
                 -3.73, -4.84,
                 -5.18, -2.86,
                 -2.62, -6.13,
                 -5.39, -4.35,
                 -6.13, -5.99,
                 -5.41, -6.34,
                 -7.82, -2.91,
                 -7.62, -7.35])
    
    parser.add_argument('--circle_pos_list_18', nargs='+',
        default=[1.50, 2.65,
                 -2.12, 1.67,
                 0.81, -1.47,
                 0.12, 2.71,
                 3.02, 0.59,
                 -2.26, -2.03])
    
    parser.add_argument('--circle_pos_list_19', nargs='+',
        default=[-5.98, 1.93,
                 -6.79, -0.53,
                 -5.7, -4.88,
                 -0.88, -6.00,
                 3.36, -5.27,
                 5.36, -2.66,
                 5.58, 2.06,
                 2.84, 5.48,
                 -0.46, 5.88,
                 -3.83, 5.6,
                 -6.36, 4.20,
                 4.56, 6.30,
                 7.41, 1.20,
                 5.33, -6.55,
                 -0.08, 0.13])
    
    parser.add_argument('--circle_pos_list_20', nargs='+',
        default=[-1.11, 1.56,
                 0.94, -1.82])
    
    

    # state_num = 5
    parser.add_argument('--state_pos_list', nargs='+',
        default=[13.06, 13.29,
                 14.30, 15.27,
                 12.74, 16.72,
                 16.14, 16.55,
                 15.86, 13.77,
                 -16.92, 14.04,
                 -15.16, 14.22,
                 -14.00, 13.00,
                 -13.71, 16.50,
                 -17.22, 16.49,
                 -16.78, -15.16,
                 -16.43, -13.68,
                 -17.10, -16.85,
                 -14.56, -16.52,
                 -13.52, -14.66,
                 12.97, -13.56,
                 14.08, -13.72,
                 16.77, -13.87,
                 12.95, -16.53,
                 15.64, -17.02])
    
    parser.add_argument('--state_pos_list_img', nargs='+',
        default=[3.06, 3.29,
                 4.30, 5.27,
                 2.74, 6.72,
                 6.14, 6.55,
                 5.86, 3.77,
                 -6.92, 4.04,
                 -5.16, 4.22,
                 -4.00, 3.00,
                 -3.71, 6.50,
                 -7.22, 6.49,
                 -6.78, -5.16,
                 -6.43, -3.68,
                 -7.10, -6.85,
                 -4.56, -6.52,
                 -3.52, -4.66,
                 2.97, -3.56,
                 4.08, -3.72,
                 6.77, -3.87,
                 2.95, -6.53,
                 5.64, -7.02])
    
    # state_num = 3
    parser.add_argument('--state_pos_list_1', nargs='+',
        default=[14.30, 15.27,
                 16.14, 16.55,
                 15.86, 13.77,
                 -16.92, 14.04,
                 -13.71, 16.50,
                 -17.22, 16.49,
                 -16.78, -15.16,
                 -16.43, -13.68,
                 -17.10, -16.85,
                 16.77, -13.87,
                 12.95, -16.53,
                 15.64, -17.02])
    
    parser.add_argument('--state_eval_pos_list', nargs='+',
        default=[3.0, 7.0,
                 6.0, 6.0,
                 4.0, 5.0,
                 3.0, 3.0,
                 7.0, 4.0,
                 -7.0, 7.0,
                 -4.0, 6.0,
                 -6.0, 5.0,
                 -7.0, 3.0,
                 -3.0, 4.0,
                 -7.0, -3.0,
                 -4.0, -4.0,
                 -6.0, -5.0,
                 -7.0, -7.0,
                 -3.0, -6.0,
                 3.0, -3.0,
                 6.0, -4.0,
                 4.0, -5.0,
                 3.0, -7.0,
                 7.0, -6.0])

    
    parser.add_argument('--difficulty_steps', nargs='+',
        default=[3, 6, 9, 260, 340])    # 4 difficulty levels (episode num) each level: 7 1x1 boxes 4 2x2 boxes 8 r1 cylinders 6 r03 cylinders 5 r02 cylinders
    
    parser.add_argument('--box_1x1_list_l0', nargs='+',
                        default=['box_1x1_l0_0', 8.98, 16.13,
                                 'box_1x1_l0_1', 13.80, 7.80, 
                                 'box_1x1_l0_2', -17.22, -6.87,
                                 'box_1x1_l0_3', 13.79, -17.22,
                                 'box_1x1_l0_4', -8.84, 17.21,
                                 'box_1x1_l0_5', 30, 8.5,
                                 'box_1x1_l0_6', 30, 10])      # 1x1 boxs l0
    
    parser.add_argument('--box_1x1_list_l1', nargs='+',
                        default=['box_1x1_l1_0', -30, 1,
                                 'box_1x1_l1_1', -30, 2.5, 
                                 'box_1x1_l1_2', -30, 4,
                                 'box_1x1_l1_3', -30, 5.5,
                                 'box_1x1_l1_4', -30, 7,
                                 'box_1x1_l1_5', -30, 8.5,
                                 'box_1x1_l1_6', -30, 10])     # 1x1 boxs l1
    
    parser.add_argument('--box_1x1_list_l2', nargs='+',
                        default=['box_1x1_l2_0', -30, -1,
                                 'box_1x1_l2_1', -30, -2.5, 
                                 'box_1x1_l2_2', -30, -4,
                                 'box_1x1_l2_3', -30, -5.5,
                                 'box_1x1_l2_4', -30, -7,
                                 'box_1x1_l2_5', -30, -8.5,
                                 'box_1x1_l2_6', -30, -10])     # 1x1 boxs l2
    
    parser.add_argument('--box_1x1_list_l3', nargs='+',
                        default=['box_1x1_l3_0', 30, 1,
                                 'box_1x1_l3_1', 30, 2,5, 
                                 'box_1x1_l3_2', 30, 4,
                                 'box_1x1_l3_3', 30, 5.5,
                                 'box_1x1_l3_4', 30, 7,
                                 'box_1x1_l3_5', 30, 8.5,
                                 'box_1x1_l3_6', 30, 10])     # 1x1 boxs l3
    
    parser.add_argument('--box_2x2_list_l0', nargs='+',
                        default=['box_2x2_l0_0', 35, 2,
                                 'box_2x2_l0_1', 35, 5,
                                 'box_2x2_l0_2', 35, 8,
                                 'box_2x2_l0_3', 35, 11
                                 ])     # 2x2 boxs l0
    
    parser.add_argument('--box_2x2_list_l1', nargs='+',
                        default=['box_2x2_l1_0', -17.92, -17.85,
                                 'box_2x2_l1_1', 15.15, -7.31,
                                 'box_2x2_l1_2', 8.38, 12.52,
                                 'box_2x2_l1_3', -35, 11])     # 2x2 boxs l1
    
    parser.add_argument('--box_2x2_list_l2', nargs='+',
                        default=['box_2x2_l2_0', -35, -2,
                                 'box_2x2_l2_1', -35, -5,
                                 'box_2x2_l2_2', -35, -8,
                                 'box_2x2_l2_3', -35, -11])     # 2x2 boxs l2
    
    parser.add_argument('--box_2x2_list_l3', nargs='+',
                        default=['box_2x2_l3_0', 35, -2,
                                 'box_2x2_l3_1', 35, -5,
                                 'box_2x2_l3_2', 35, -8,
                                 'box_2x2_l3_3', 35, -11])     # 2x2 boxs l3
    
    parser.add_argument('--circle_r1_list_l0', nargs='+',
                        default=['cylinder_r1_l0_0', 40, 1,
                                 'cylinder_r1_l0_1', 40, 2.5,
                                 'cylinder_r1_l0_2', 40, 4,
                                 'cylinder_r1_l0_3', 40, 5.5,
                                 'cylinder_r1_l0_4', 40, 7,
                                 'cylinder_r1_l0_5', 40, 8.5,
                                 'cylinder_r1_l0_6', 40, 10,
                                 'cylinder_r1_l0_7', 40, 11.5
                                 ])     # circle r1 l0
    
    parser.add_argument('--circle_r1_list_l1', nargs='+',
                        default=['cylinder_r1_l1_0', -9.32, -9.26,
                                 'cylinder_r1_l1_1', -16.49, 7.22,
                                 'cylinder_r1_l1_2', 16.89, 17.18,
                                 'cylinder_r1_l1_3', 5.14, -15.86,
                                 'cylinder_r1_l1_4', -40, 40,
                                 'cylinder_r1_l1_5', -40, 40,
                                 'cylinder_r1_l1_6', -40, 40,
                                 'cylinder_r1_l1_7', -40, 40
                                 ])     # circle r1 l1
    
    parser.add_argument('--circle_r1_list_l2', nargs='+',
                        default=['cylinder_r1_l2_0', -40, 40,
                                 'cylinder_r1_l2_1', -40, 40,
                                 'cylinder_r1_l2_2', -40, 40,
                                 'cylinder_r1_l2_3', -40, 40,
                                 'cylinder_r1_l2_4', -40, 40,
                                 'cylinder_r1_l2_5', -40, 40,
                                 'cylinder_r1_l2_6', -40, 40,
                                 'cylinder_r1_l2_7', -40, 40
                                 ])     # circle r1 l2
    
    parser.add_argument('--circle_r1_list_l3', nargs='+',  
                        default=['cylinder_r1_l3_0', 40, 40,
                                 'cylinder_r1_l3_1', 40, 40,
                                 'cylinder_r1_l3_2', 40, 40,
                                 'cylinder_r1_l3_3', 40, 40,
                                 'cylinder_r1_l3_4', 40, 40,
                                 'cylinder_r1_l3_5', 40, 40,
                                 'cylinder_r1_l3_6', 40, 40,
                                 'cylinder_r1_l3_7', 40, 40
                                 ])     # circle r1 l3
    
    parser.add_argument('--circle_r03_list_l0', nargs='+',
                        default=['cylinder_r03_l0_0', 45, 45,
                                 'cylinder_r03_l0_1', 45, 45,
                                 'cylinder_r03_l0_2', 45, 45,
                                 'cylinder_r03_l0_3', 45, 45,
                                 'cylinder_r03_l0_4', 45, 45,
                                 'cylinder_r03_l0_5', 45, 45
                                 ])     # circle r3 l0
    
    parser.add_argument('--circle_r03_list_l1', nargs='+', 
                        default=['cylinder_r03_l1_0', -45, 45,
                                 'cylinder_r03_l1_1', -45, 45,
                                 'cylinder_r03_l1_2', -45, 45,
                                 'cylinder_r03_l1_3', -45, 45,
                                 'cylinder_r03_l1_4', -45, 45,
                                 'cylinder_r03_l1_5', -45, 45
                                 ])     # circle r3 l1
    
    parser.add_argument('--circle_r03_list_l2', nargs='+',
                        default=['cylinder_r03_l2_0', -12.27, 12.12,
                                 'cylinder_r03_l2_1', -10.88, -16.24,
                                 'cylinder_r03_l2_2', 7.28, -5.85,
                                 'cylinder_r03_l2_3', 10.84, 6.51,
                                 'cylinder_r03_l2_4', 15.73, 11.82,
                                 'cylinder_r03_l2_5', -10.33, 5.57
                                 ])     # circle r3 l2
    
    parser.add_argument('--circle_r03_list_l3', nargs='+',
                        default=['cylinder_r03_l3_0', 45, -45,
                                 'cylinder_r03_l3_1', 45, -45,
                                 'cylinder_r03_l3_2', 45, -45,
                                 'cylinder_r03_l3_3', 45, -45,
                                 'cylinder_r03_l3_4', 45, -45,
                                 'cylinder_r03_l3_5', 45, -45
                                 ])     # circle r3 l3
    
    
    # ========================================> Reward <========================================
    parser.add_argument('--eval_scale', type=float, default=1.2,
        help="the scale level used in evaluation")
    parser.add_argument('--threshold_arrival', type=float, default=0.4,
        help="the threshold used for arrival detection")
    parser.add_argument('--threshold_collision', type=float, default=0.4,
        help="the threshold used for collision detection") # 0.8 # irsim 0.4
    parser.add_argument('--norm_threshold_goal_distance', type=float, default=5.0,
        help="the normalization threshold used for goal distance input") # 5
    parser.add_argument('--norm_threshold_agent_distance', type=float, default=10,
        help="the normalization threshold used for agent distance input")
    parser.add_argument('--norm_threshold_collision_distance', type=float, default=4.0,
        help="the normalization threshold used for collision reward calculation") # 4 # irsim 2.8
    parser.add_argument('--norm_threshold_CBF_collision_distance', type=float, default=0.5,
        help="the normalization threshold used for CBF collision reward calculation")

    parser.add_argument('--reward_goal', type=float, default=6.0,
        help="the reward ratio of goal distance delta") # 9.0
    parser.add_argument('--reward_collision', type=float, default=1.2,
        help="the reward ratio of collision distance") # 0.6 # 1.2 0728before
    parser.add_argument('--reward_done', type=float, default=1.0,
        help="the reward of arrival / collision") # 100.0
    parser.add_argument('--reward_CBF', type=float, default=0.5,
        help="the reward ratio of CBF collision distance")
    parser.add_argument('--reward_CBF_refine_ratio', type=float, default=0.1,
        help="the reward ratio of CBF refine cost")
    parser.add_argument('--reward_auxiliary', type=float, default=1.0,
        help="the reward ratio of reference control difference in auxiliary task")

    # ========================================> Policy <========================================
    # Algorithm Parameters
    parser.add_argument('--lr_actor', type=float, default=1e-4,
        help="the learning rate of actor network")
    parser.add_argument('--lr_critic', type=float, default=3e-4,
        help="the learning rate of critic network")
    parser.add_argument('--network_dim', type=int, default=256,
        help="the dimension of the latent layer in the network")
    parser.add_argument('--lr_scheduler_interval', type=int, default=1e3,
        help="the interval of the learning rate scheduler step in training")
    
    # Training Parameters
    parser.add_argument('--gpu', type=int, default=1,
        help="the index of GPU device")
    parser.add_argument('--seed', type=int, default=99,
        help="the seed for torch, numpy, random etc.") # 77 99
    parser.add_argument('--pre_policy', type=int, default=501,
        help="pre-train policy model index")
    parser.add_argument('--init_exploration_noise', type=float, default=0.2,
        help="the initial action nosie in RL training for exploration")
    parser.add_argument('--batch_size', type=int, default=16,
        help="the batch size in training") #64
    parser.add_argument('--buffer_size', type=int, default=1e6,
        help="the buffer size in training")
    parser.add_argument('--iter_per_step', type=int, default=1,
        help="the number of iteration per step in training")
    parser.add_argument('--model_save_frequency', type=int, default=50,
        help="the frequency(episodes) of model saving in training")
    parser.add_argument('--episode_length', type=int, default=100,
        help="the length of the episode, max step number") # 600   use 300 for tdmpc2
    parser.add_argument('--random_exploration_length', type=int, default=5000, # 5000
        help="the length of the random exploration procedure, max random exploration step number")
    
    # On-Policy Training Parameters
    # FIXME : PPO
    parser.add_argument('--on_policy_train_episode', type=int, default=2000,
        help="the episode length in on-policy algorithm training, with multiple agents, the actual buffer size is larger")
    parser.add_argument('--num_update_ppo', type=int, default=8,
        help="the number of updates in PPO, [2,8), \
        smaller update num is better for 1 : larger clip range, 2 : lamda entropy(more exploration), 3 : unstable multiagent envs")
    parser.add_argument('--gae', type=bool, default=True,
        help="use gae or not in PPO")
    parser.add_argument('--norm_adv', type=bool, default=True,
        help="use norm_adv or not in PPO")
    parser.add_argument('--clip_vloss', type=bool, default=True,
        help="clip value loss or not in PPO")
    parser.add_argument('--gamma', type=float, default=0.99,
        help="gamma in PPO")
    parser.add_argument('--gae_lamda', type=float, default=0.98,
        help="gae_lamda in PPO, larger => higher variance, lower bias")
    # parser.add_argument('--entropy_coef', type=float, default=0.01,
    #     help="coefficient of entropy in PPO, larger means more exploration")
    # parser.add_argument('--value_coef', type=float, default=0.5,
    #     help="coefficient of value function in PPO")
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
        help="max gradient norm in PPO")
    parser.add_argument('--clip_coef', type=float, default=0.25,
        help="clip_coef in PPO")
    parser.add_argument('--target_kl', type=float, default=0.1,
        help="target_kl in PPO")
    
    # State Processing => Image
    parser.add_argument('--img_range', type=int, default=128*2,
        help="the range of the visualization of each agent") # *2 # 128
    parser.add_argument('--img_width', type=int, default=128*2,
        help="the width of the visualization image of each agent") # *2 # 128
    parser.add_argument('--img_height', type=int, default=64*2,
        help="the height of the visualization image of each agent") # *2 # 64
    parser.add_argument('--pixel_per_meter', type=float, default=64*2.0/1.5,
        help="the number of pixels that represent 1 meter in real scale") # *2 #64/8.0 img_height / laser_range
    parser.add_argument('--highlight_iterations', type=int, default=2,
        help="the num of iterations of erode / dilate in visualization image")
    parser.add_argument('--merge_vis', type=bool, default=True,
        help="merge history observations or not in visualization image")
    
    # ========================================> Model-based RL <========================================
    # World Model Training
    parser.add_argument('--pre_model', type=int, default=0,
        help="pre-train world model index")
    parser.add_argument('--model_train_frequency', type=int, default=30,
        help="the number of epochs between each round of model training")
    parser.add_argument('--model_evaluation_frequency', type=int, default=3,
        help="the number of model training epochs between each round of model evaluation")
    parser.add_argument('--model_train_epoch', type=int, default=10,
        help="the number of epochs in each round of model training")
    parser.add_argument('--model_train_batchsize', type=int, default=32,
        help="the batch size in model training")
    parser.add_argument('--model_ready_epoch', type=int, default=3,
        help="the minimum epochs of model training before using it for prediction")


    # ========================================> Safe RL <========================================
    # CBF Refine
    parser.add_argument('--alpha_cbf', type=float, default=1.0,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--num_refine_loop', type=int, default=10,
        help="the number of refine loops in CBF refine")
    parser.add_argument('--lr_cbf_refine', type=float, default=0.25,
        help="the learning rate in cbf refine")    
    
    # ========================================> TDMPC2 <==========================================
    parser.add_argument('--obs', type=str, default='state',
        help="the type of observation")
    
    # enc_dyn_lr_change_0: enc_lr_scale: 0.3, enc_dyn_lr_change_1: enc_lr_scale: 0.1, enc_dyn_lr_change_2: enc_lr_scale: 0.1, pi_lr_scale: 1.5
    # enc_dyn_lr_change_3: enc_lr_scale: 0.1, pi_lr_scale: 0.5 enc_dyn_lr_change_4: enc_lr_scale: 0.1, pi_lr_scale: 5.0
    # enc_dyn_lr_change_5: enc_lr_scale: 0.03, pi_lr_scale: 10.0 enc_dyn_lr_change_6: enc_lr_scale: 0.1, pi_lr_scale: 0.03
    # enc_dyn_lr_change_7: lr: 1e-5 enc_lr_scale: 1.0, pi_lr_scale: 20
    parser.add_argument('--steps', type=int, default=10_000_000,
        help="the param of tdmpc2 training")
    parser.add_argument('--reward_coef', type=float, default=0.5,
        help="the param of tdmpc2 training") # 0.1
    parser.add_argument('--value_coef', type=float, default=0.5,
        help="the param of tdmpc2 training") # 0.1
    parser.add_argument('--consistency_coef', type=float, default=20,
        help="the param of tdmpc2 training") # 20
    parser.add_argument('--rho', type=float, default=0.5,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--lr', type=float, default=3e-4,
        help="the ratio of deriv in CBF refine") # 3e-4
    parser.add_argument('--enc_lr_scale', type=float, default=0.3,
        help="the ratio of deriv in CBF refine") # 0.3
    parser.add_argument('--dyn_lr_scale', type=float, default=0.3,
        help="the ratio of deriv in CBF refine") # 1.0
    parser.add_argument('--pi_lr_scale', type=float, default=1.0,
        help="the ratio of deriv in CBF refine") 
    parser.add_argument('--grad_clip_norm', type=int, default=20,
        help="the ratio of deriv in CBF refine") # 20
    parser.add_argument('--tau', type=float, default=0.01,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--discount_denom', type=int, default=5,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--discount_min', type=float, default=0.95,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--discount_max', type=float, default=0.995,
        help="the ratio of deriv in CBF refine")
    
    parser.add_argument('--mpc', type=bool, default=True,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--iterations', type=int, default=6,
        help="the ratio of deriv in CBF refine") # 6
    parser.add_argument('--num_samples', type=int, default=512,
        help="the ratio of deriv in CBF refine") # 512
    parser.add_argument('--num_elites', type=int, default=64,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--num_pi_trajs', type=int, default=6,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--horizon', type=int, default=6,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--min_std', type=float, default=0.05,
        help="the ratio of deriv in CBF refine") # 0.05
    parser.add_argument('--max_std', type=float, default=2,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--temperature', type=float, default=0.5,
        help="the ratio of deriv in CBF refine") # 0.5
    
    parser.add_argument('--log_std_min', type=float, default=-10,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--log_std_max', type=float, default=2,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--entropy_coef', type=float, default=1e-4,
        help="the ratio of deriv in CBF refine")
    
    parser.add_argument('--num_bins', type=float, default=101,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--vmin', type=float, default=-10,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--vmax', type=float, default=10,
        help="the ratio of deriv in CBF refine")
    
    parser.add_argument('--model_size', type=int, default=5,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--num_enc_layers', type=int, default=3,
        help="the ratio of deriv in CBF refine") # 2 need test
    parser.add_argument('--num_dyn_layers', type=int, default=2,
        help="the ratio of deriv in CBF refine") # 2 org
    parser.add_argument('--num_pi_layers', type=int, default=2,
        help="the ratio of deriv in CBF refine") # 2 org
    parser.add_argument('--num_reward_layers', type=int, default=2,
        help="the ratio of deriv in CBF refine") # 2 org
    parser.add_argument('--num_q_layers', type=int, default=2,
        help="the ratio of deriv in CBF refine") # 2 org
    parser.add_argument('--enc_dim', type=int, default=256,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--num_channels', type=int, default=32,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--mlp_dim', type=int, default=512,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--latent_dim', type=int, default=512,
        help="the ratio of deriv in CBF refine") # 512
    parser.add_argument('--task_dim', type=int, default=0,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--num_q', type=int, default=5,
        help="the ratio of deriv in CBF refine") # 5 
    parser.add_argument('--dropout', type=float, default=0.01,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--simnorm_dim', type=int, default=8,
        help="the ratio of deriv in CBF refine")
    
    parser.add_argument('--multitask', type=bool, default=False,
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--obs_shape', type=dict, default={"state": [184]},
        help="the ratio of deriv in CBF refine")
    parser.add_argument('--bin_size', type=float, default=0.2,
        help="the ratio of deriv in CBF refine")
    
    parser.add_argument('--use_one', type=bool, default=False,
        help="use one action or all actions")
    parser.add_argument('--h3u3', type=str, default='train_turtlebot_tdmpc2_multi_laser_eval_traj_hor3_algorithm_904/201_9_6',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    parser.add_argument('--h3u1', type=str, default='train_turtlebot_tdmpc2_multi_laser_eval_traj_hor3_use1algorithm_904/201',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    parser.add_argument('--h5u5', type=str, default='train_turtlebot_tdmpc2_multi_laser_eval_traj_hor5_use5algorithm_905/201',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    parser.add_argument('--h5u1', type=str, default='train_turtlebot_tdmpc2_multi_laser_eval_traj_hor5_use1algorithm_907/201',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    
    parser.add_argument('--h3u3_lhard', type=str, default='train_turtlebot_tdmpc2_multi_laser_eval_vw_lhard_hor3_use3_algorithm_927/401',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    parser.add_argument('--h3u1_lhard', type=str, default='train_turtlebot_tdmpc2_multi_laser_eval_vw_lhard_hor3_use1_algorithm_927/401',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    parser.add_argument('--h5u5_lhard', type=str, default='train_turtlebot_tdmpc2_multi_laser_eval_vw_lhard_hor5_use5_1_algorithm_927/401',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    parser.add_argument('--h5u1_lhard', type=str, default='train_turtlebot_tdmpc2_multi_laser_eval_vw_lhard_hor5_use1_algorithm_927/251',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    parser.add_argument('--h3_traj_follow_change', type=str, default='train_turtlebot_tdmpc2_multi_laser_eval_traj_action_follow_improve_diff_axis_simple_hor3_use3_algorithm_1009/1302',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ") # lhard world follow change (allow v=0 if dirdiff is too large) pretrain in simple world 
    parser.add_argument('--h3_traj', type=str, default='train_turtlebot_tdmpc2_multi_laser_use_traj_diff_axis_hor3_use3algorithm_926/752', 
        help="the type of the robots  :  ['turtlebot', 'jackle']  ") # lhard world it is hard to calculate a v which is 0, pretrain in simple world

    parser.add_argument('--h3u3_lhard_use_sp_model', type=str, default='train_turtlebot_tdmpc2_multi_laser_eval_vw_lhard_hor3_use3_use_simple_model_algorithm_1026/452',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    parser.add_argument('--h3u3_bhard_use_lh_model', type=str, default='train_turtlebot_tdmpc2_multi_laser_eval_vw_bhard_hor3_use3_use_lhard_model_algorithm_1027/853',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    
    parser.add_argument('--h3u3_barn_diff_levels_1', type=str, default='train_tdmpc2_vw_barn_diff_levels_1_h3u3_cnn_change_6_use_pre_model_1107algorithm_1111/1252',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    
    parser.add_argument('--h3u1_barn_diff_levels_mlp', type=str, default='train_tdmpc2_vw_barn_diff_levels_h3u1_mlp_obs_no_a_encoder_enc_d_lr_change_smaller_algorithm_1121/751',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    
    parser.add_argument('--h1u1_barn_diff_levels_mlp', type=str, default='train_tdmpc2_vw_barn_diff_levels_h1u1_mlp_obs_no_a_encoder_enc_d_lr_layer_change_algorithm_1120/601',
        help="the type of the robots  :  ['turtlebot', 'jackle']  ")
    
    parser.add_argument('--SAC_barn_diff_levels_cnn', type=str, default='train_SAC_vw_barn_diff_levels_cnn_change_6_algorithm_1113/451',
        help="the type of the robots   :  ['turtlebot', 'jackle']  ")
    
    parser.add_argument('--h3u1_barn_diff_levels_mlp_obs_fix_lg_cat_same_obs_same_goal', type=str, default='train_tdmpc2_vw_barn_diff_levels_no_norm_cossin_obs_fix_same_obs_goal_lgcat_2enc_512latent_h3u1org_mlp_lr_0_enc_3_dyn_2_algorithm_1221/401',
        help="h3u1 vw obs_fix mlp_obs same_obs same_goal laser_goal_cat multi_enc one_dyn dis_theta_goal")
    
    parser.add_argument('--h3u1_barn_diff_levels_mlp_obs_fix_lg_cat_same_obs_diff_goal', type=str, default='train_tdmpc2_vw_barn_diff_levels_no_norm_cossin_obs_fix_same_obs_diff_goal_lgcat_2enc_512latent_h3u1org_mlp_lr_0_enc_3_dyn_2_algorithm_1221/351',
        help="h3u1 vw obs_fix mlp_obs same_obs diff_goal laser_goal_cat multi_enc one_dyn dis_theta_goal")


    # update change (Q PI)
    parser.add_argument('--h1_nompc_state_mlp', type=str, default='train_tdmpc2_traj_state_mlp_diff_laser_goal_xy_with_norm_with_simnorm_reward_nonorm_done_100_goal8_dtheta_obs_goal_nompc_updatesaclike_usedone_withdynr_userealz_usedec_1enc_1dyn_h1u1org_mlp_lr_0_enc_3_dyn_2_algorithm_0214/751',
        help="h=1 nompc use_done")
    
    parser.add_argument('--h1_mpch1_state_mlp', type=str, default='train_tdmpc2_traj_state_mlp_diff_laser_goal_xy_with_norm_with_simnorm_reward_nonorm_done_100_goal8_dtheta_obs_goal_mpch1_updatesaclike_usedone_withdynr_userealz_usedec_1enc_1dyn_h1u1org_mlp_lr_higher_enc_3_dyn_2_algorithm_0214/451',
        help="h=1 mpch=1 use_done")
    
    parser.add_argument('--h1_mpch3_state_mlp', type=str, default='train_tdmpc2_traj_state_mlp_diff_laser_goal_xy_with_norm_with_simnorm_reward_nonorm_done_100_goal8_dtheta_obs_goal_mpch3_updatesaclike_usedone_withdynr_userealz_usedec_1enc_1dyn_h1u1org_mlp_lr_higher_enc_3_dyn_2_algorithm_0214/451',
        help="h=1 mpch=3 use_done")
    
    parser.add_argument('--sac_state_mlp', type=str, default='train_SAC_traj_state_mlp_with_simnorm_with_norm_use_theta_obs_goal_reward_nonorm_done100_algorithm_0211/751',
        help="sac state")
    
    # keep run mode
    parser.add_argument('--policy_h3_nompc_kr', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/nodec_srange8_keeprun_nodone_mse_qr_nonormq_h3u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2/1101',
        help="keep running h=3 no mpc")
    parser.add_argument('--policy_sac_kr', type=str, default='train_SAC_traj_keeprun_reward_nonorm_done20_usestatecolli_algorithm_0214/1551',
        help="keep running sac")
    parser.add_argument('--keeprun_laser_vw_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/uselaser_vw_nodec_drawlaserrange1.5_keeprun_nodone_mse_qr_nonormq_vismpch6u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/501',
        help="keep running laser h=6")
    parser.add_argument('--keeprun_laser4_vw_difflevels_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/uselaser_vw_barndifflevel_nodec_drawlaserrange1.5_truelaser4_keeprun_nodone_mse_qr_nonormq_vismpch6u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/1001',
        help="keep running laser range 4 h=6 difflevels")
    parser.add_argument('--keeprun_laser4_vw_difflevels_h10rdact', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/uselaser_vw_barndifflevel_rndactlen_nodec_drawlaserrange1.5_truelaser4_keeprun_nodone_mse_qr_nonormq_vismpch10u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/751',
        help="kepp running laser range 4 h=10 use1~10 difflevels")
    
    parser.add_argument('--keeprun_laser4_vw_difflevels_h6u6', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/uselaser_vw_barndifflevel_nodec_drawlaserrange1.5_truelaser4_keeprun_nodone_mse_qr_nonormq_vismpch6u6_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/951',
        help="kepp running laser range 4 h=6 use6 difflevels")
    parser.add_argument('--keeprun_laser4_vw_difflevels_h6rdact1_3', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/uselaser_vw_barndifflevel_nodec_drawlaserrange1.5_truelaser4_keeprun_nodone_mse_qr_nonormq_vismpch6rdact1_3_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/951',
        help="kepp running laser range 4 h=6 use1~3 difflevels")
    parser.add_argument('--keeprun_laser4_vw_difflevels_h10rdact1_3', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/uselaser_vw_barndifflevel_nodec_drawlaserrange1.5_truelaser4_keeprun_nodone_mse_qr_nonormq_vismpch10rdact1_3_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/951',
        help="kepp running laser range 4 h=10 use1~3 difflevels")
    parser.add_argument('--keeprun_laser4_vw_difflevels_h10u6', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/uselaser_vw_barndifflevel_nodec_drawlaserrange1.5_truelaser4_keeprun_nodone_mse_qr_nonormq_vismpch10u6_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/951',
        help="kepp running laser range 4 h=10 use6 difflevels")
    parser.add_argument('--keeprun_laser4_vw_difflevels_h6rdact1_6', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/uselaser_vw_barndifflevel_nodec_drawlaserrange1.5_truelaser4_keeprun_nodone_mse_qr_nonormq_vismpch6rdact1_6_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/951',
        help="kepp running laser range 4 h=6 use1~6 difflevels") 
    parser.add_argument('--keeprun_laser4_vw_20x20dyn_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/uselaser_vw_20x20dynnoif_withdyninitmode_gscale3_withgui_B64_nodec_drawlaserrange1.5_truelaser4_keeprun_nodone_mse_qr_nonormq_vismpch6u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/951',
        help="keep running laser range 4 h=6 use1 20x20dyn")
    parser.add_argument('--keeprun_lasercnn_vw_10x10dyn_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/uselasercnnorg_vw_10x10dyncenternoif_gscale3_withgui_B32_nodec_drawlaserrange1.5_truelaser2_normg2_keeprun_nodone_mse_qr_nonormq_vismpch6u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/251')
    parser.add_argument('--keeprun_laser4_vw_emptydyncenter_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/uselaser_vw_emptydyncenternoif_gscale3_withgui_B64_nodec_drawlaserrange1.5_truelaser4_normg5_keeprun_nodone_mse_qr_nonormq_vismpch6u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/151')
    parser.add_argument('--keeprun_laser4cnnorg_vw_emptydyncenter_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/uselasercnnorg_vw_id17_emptydyncenternoif_gscale3_withgui_B32_nodec_drawlaserrange1.5_truelaser4_normg5_keeprun_nodone_mse_qr_nonormq_vismpch6u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/751')
    parser.add_argument('--keeprun_laser4_vw_irsim_dyn_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/irsim_uselaser_vw_dynobscenter_withgui_B64_nodec_drawlaserrange2.0_truelaser4_normg5_keeprun_nodone_mse_qr_nonormq_vismpch6u1_6_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/301')
    parser.add_argument('--keeprun_fast_laser4_vw_irsim_dyn_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/irsim_uselaser_vw_fast_dynobscenter_withgui_B64_nodec_drawlaserrange2.0_truelaser4_normg5_keeprun_nodone_mse_qr_nonormq_vismpch6u1_6_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/501')
    parser.add_argument('--keeprun_fast_laser8cnnhis8sp1_vw_irsim_dyn_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/irsim_uselasercnnorg_vw_fastfix_his8sp1_again_dynobscenter_normreward_withgui_B32_nodec_drawlaserrange8_truelaser8_normg5_keeprun_nodone_mse_qr_nonormq_vismpch6u1_6_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/201')
    parser.add_argument('--keeprun_fast_laser4cnnhis8sp1_vw_irsim_dyn_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/irsim_uselasercnnorg_vw_fastfix_his8sp1_dynobscenter_normreward_withgui_B32_nodec_drawlaserrange4_truelaser4_normg5_keeprun_nodone_mse_qr_nonormq_vismpch6u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/351') # 351
    parser.add_argument('--keeprun_fast_laser4cnnhis8sp1_vw_irsim_dyn_h6u1_6', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/irsim_uselasercnnorg_vw_fastfix_his8sp1_dynobscenter_normreward_withgui_B32_nodec_drawlaserrange4_truelaser4_normg5_keeprun_nodone_mse_qr_nonormq_vismpch6urd1_6_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/351') # 351
    parser.add_argument('--keeprun_fast_laser4cnnhis8sp1_vw_irsim_dyn_h6u6', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/irsim_uselasercnnorg_vw_fastfix_his8sp1_dynobscenter_normreward_withgui_B32_nodec_drawlaserrange4_truelaser4_normg5_keeprun_nodone_mse_qr_nonormq_vismpch6u6_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/301')  # 301
    parser.add_argument('--keeprun_fast_obsfast_steppunish_laser4cnnhis8sp1_vw_irsim_dyn_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/irsim_uselasercnnorg_vw_fastfix_fobs_his8sp1_dynobswothwall_normreward_steppunish_withgui_B32_nodec_drawlaserrange4_truelaser4_normg5_keeprunarrivestop_nodone_mse_qr_nonormq_vismpch6u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/251')  # 371
    parser.add_argument('--keeprun_fast_obsfast_steppunish2_normreward36_laser4cnnhis8sp1_vw_irsim_dyn_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/irsim_uselasercnnorg_vw_fastfix_fobs_his8sp1_dynobswithwallfarg_normreward9*0.4_steppunish2.0_withgui_B32_nodec_drawlaserrange4_truelaser4_eps150_normg5_keeprunarrivestop_nodone_mse_qr_nonormq_vismpch6u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/451')   # 391
    parser.add_argument('--keeprun_fast_obsfast_steppunish2_normreward36_laser8cnnhis8sp1_vw_irsim_dyn_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/irsim_uselasercnnorg_vw_fastfix_fobs_his8sp1_dynobswithwallfarg_normreward9*0.4_steppunish2.0_rdc0.6_withgui_B32_nodec_drawlaserrange8_truelaser8_eps150_normg5_keeprunarrivestop_nodone_mse_qr_nonormq_vismpch6u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/451')   # 381
    parser.add_argument('--keeprun_fast_obsfastdash_steppunish2_normreward36_laser5cnnhis8sp1_arrivenostop_vw_irsim_dyn_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/irsim_uselasercnnorg_vw_fastfix_6fobs_dash_his8sp1_dynobswithwallfarg_normreward9*0.4_steppunish2.0_rdc0.6_withgui_B32_nodec_drawlaserrange4_truelaser4_eps150_normg5_keeprunarrivenostop_nodone_mse_qr_nonormq_vismpch6u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/351')    # 397
    parser.add_argument('--keeprun_fast_obsfastdash_normrewardimprove_g9rdc06dev05tp02v025_laser4cnnhis8sp1_arrivenostop_vw_irsim_dyn_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/irsim_laser4cnnorg_vw_fastfix_6fobs_dash_his8sp1_dynobswithwallfarg_normrewardimprove_dev_vel_mode_g9rdc0.6dev0.5tp0.2v0.25_B32_nodec_eps150_normg5_keeprunarrivenostop_nodone_mse_qr_nonormq_vismpch6u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/451')     # 388
    parser.add_argument('--keeprun_fast_obsfastdash_normreward36_g9rdc2dev0tp0v0_laser4cnnhis8sp1_arrivenostop_vw_irsim_dyn_h6u6', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_normreward3.6_g9rdc2dev0tp0v0pr0_B32_nodec_eps150_normg5_krarrivenostop_nodone_mse_qr_nonormq_vismpch6u6_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/601')
    parser.add_argument('--keeprun_fast_obsfastdash_normreward36_g9rdc06dev0tp0v0_laser4cnnhis8sp1_arrivenostop_vw_irsim_dyn_h6u1', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_normreward3.6_g9rdc0.6dev0tp0v0pr0_B32_nodec_eps150_normg5_krarrivenostop_nodone_mse_qr_nonormq_vismpch6u1_lr3e-4enc0.3dyn0.3pi1.0_numpi2_nogui/551')
    parser.add_argument('--keeprun_circle8_nonormreward_g6exprdc12dev0tp0v0pr0trd0_laser4cnnhis8sp1_arrivenostop_vw_irsim_h6u6', type=str, default='train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_nonormreward_8robot_circle_g6exprdc12dev0tp0v0pr0trd0_vismpch6u6/301')
    parser.add_argument('--keeprun_sep_corridor_diffdash_nonormreward_noaccclip_merge_g6exprdc12dev0tp0v0pr0trd0_h3u3', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_sep_obsdash_nonormreward_resetabuffer_noaccclip_merge_g6exprdc12dev0tp0v0pr0trd0_vismpch3u3/1101")
    parser.add_argument('--keeprun_sep_square_omnidash_nonormreward_noaccclip_merge_g6exprdc12dev0tp0v0pr0trd0_h3u3', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_sep_square_obsomnidash_nonormreward_kr_resetabuffer_noaccclip_merge_laser4_g6exprdc12dev0tp0v0pr0trd0_vismpch3u3/1101")
    parser.add_argument('--keeprun_sep_corridor_omnidash_nonormreward_noaccclip_merge8_g6exprdc12dev0tp0v0pr0trd0_h6u6_dyn', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_sep_corridor_omnidash_nonormreward_kr_resetabuffer_noaccclip_merge_laser8_g6exprdc12dev0tp0v0pr0trd0_vismpch6u6/1601")
    parser.add_argument('--keeprun_sep_corridor_omnipatrol_nonormreward_noaccclip_merge8_g6exprdc12dev0tp0v0pr0trd0_h6u6', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_sep_corridor_omnipatrol_nonormreward_kr_resetabuffer_noaccclip_merge_laser8_g6exprdc12dev0tp0v0pr0trd0_vismpch6u6/1501")
    parser.add_argument('--_sac_sep_corridor_omnipatrol_nonormreward_withdone100_cnnlaser8', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_sac_sep_corridor_omnipatrol_nonormreward_withdone100_cnnlaser8/2101")
    parser.add_argument('--sac_sep_corridor_omnipatrol_nonormreward_withdonereal100_cnnlaser8', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_sac_sep_corridor_omnipatrol_nonormreward_withdonereal100_step100_cnnlaser8/2101")

    #================> eval irsim contract models <=================
    parser.add_argument('--tdmpc_1', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_sep_corridor_omnipatrol_nonormreward_kr_resetabuffer_noaccclip_merge_laser8_g6exprdc12dev0tp0v0pr0trd0_vismpch6u6/1501",
        help="tdmpc_sep_corridor_patrol_rd0_smooth0_laser8_noclip_h6u6")
    parser.add_argument('--tdmpc_2', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_nomdn_sep_corridor_omnipatrol_rd03_nonormreward_kr_resetabuffer_smooth0_clip_actorF_rollF_sampleF_valueF_merge_laser8_g6exprdc12dev0tp0v0pr0trd0_vismpch6u6/1101",
        help="tdmpc_sep_corridor_patrol_rd03_smooth0_laser8_noclip_h6u6")
    parser.add_argument('--tdmpc_3', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_nomdn_sep_corridor_omnipatrol_rd03_nonormreward_kr_resetabuffer_smooth1_clip_actorF_rollF_sampleF_valueF_merge_laser8_g6exprdc12dev0tp0v0pr0trd0_vismpch6u6/1101",
        help="tdmpc_sep_corridor_patrol_rd03_smooth1_laser8_noclip_h6u6")
    parser.add_argument('--tdmpc_4', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_nomdn_sep_square_omnipatrol_rd03_bigvw_nonormreward_kr_resetabuffer_smooth1_noclip_merge_laser8_g6exprdc12dev0tp0v0pr0trd0_vismpch6u6/2201",
        help="tdmpc_sep_square_patrol_rd03_smooth1_laser8_noclip_h6u6")
    parser.add_argument('--tdmpc_5', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_nomdn_sep_corridor_omnipatrol_rd03_bigvw_nonormreward_kr_resetabuffer_smooth1_social0.3_noclip_merge_laser8_g6exprdc12dev0tp0v0pr0trd0_vismpch3u3/451",
        help="tdmpc_sep_corridor_patrol_rd03_smooth1_laser8_noclip_h3u3")
    
    # social reward contract
    parser.add_argument('--tdmpc_6', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_nomdn_sep_corridor_omnipatrol_rd03_bigvw_nonormreward_kr_resetabuffer_smooth1_social0.0_t2_noclip_merge_laser8_g6exprdc12dev0tp0v0pr0trd0_vismpch6u6/1501",
        help="tdmpc_sep_corridor_patrol_rd03_smooth1_social0_laser8_noclip_h6u6")
    parser.add_argument('--tdmpc_7', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_nomdn_sep_corridor_omnipatrol_rd03_bigvw_nonormreward_kr_resetabuffer_smooth1_social1_t2_noclip_merge_laser8_g6exprdc12dev0tp0v0pr0trd0_vismpch6u6/1501",
        help="tdmpc_sep_corridor_patrol_rd03_smooth1_social1_laser8_noclip_h6u6")
    parser.add_argument('--tdmpc_8', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_nomdn_sep_corridor_omnipatrol_rd03_bigvw_nonormreward_kr_resetabuffer_smooth1_social15_t2_noclip_merge_laser8_g6exprdc12dev0tp0v0pr0trd0_vismpch6u6/1501",
        help="tdmpc_sep_corridor_patrol_rd03_smooth1_social15_laser8_noclip_h6u6")
    
    # social reward contract in open corridor
    parser.add_argument('--tdmpc_9', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_nomdn_sep_corridor_nowall_omnipatrol_rd03_v4w3step150_nonormreward_kr_resetabuffer_smooth1_social0_t2_noclip_merge_laser8_g6exprdc12dev0tp0v0pr0trd0_vismpch6u6/651",
        help="tdmpc_sep_corridornowall_patrol_rd03_smooth1_social0_laser8_noclip_h6u6")
    parser.add_argument('--tdmpc_10', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_nomdn_sep_corridor_nowall_omnipatrol_rd03_v4w3step150_nonormreward_kr_resetabuffer_smooth1_social1_t2_noclip_merge_laser8_g6exprdc12dev0tp0v0pr0trd0_vismpch6u6/651",
        help="tdmpc_sep_corridornowall_patrol_rd03_smooth1_social1_laser8_noclip_h6u6")
    parser.add_argument('--tdmpc_11', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_nomdn_sep_corridor_nowall_omnipatrol_rd03_v4w3step150_nonormreward_kr_resetabuffer_smooth1_social2_t2_noclip_merge_laser8_g6exprdc12dev0tp0v0pr0trd0_vismpch6u6/651",
        help="tdmpc_sep_corridornowall_patrol_rd03_smooth1_social2_laser8_noclip_h6u6")
    
    # smooth reward for video 0713
    parser.add_argument('--tdmpc_smooth_1', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_nomdn_sep_corridor_omnipatrol_rd03_v4w2step100_orgtdmpc_nonormreward_kr_resetabuffer_smooth5_social1_t2_noclip_collision04_4_k5_merge_laser8_g6exprdc12smooth1_vismpch6u6/1001",
        help="tdmpc_sep_corridor_patrol_rd03_smooth5_reward_smooth1")
    parser.add_argument('--tdmpc_smooth_2', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_nomdn_sep_corridornowall_omnipatrol_rd03_v4w2step100_orgtdmpc_nonormreward_kr_resetabuffer_smooth5_social1_t2_noclip_collision04_4_k5_merge_laser8_g6exprdc12smooth1_vismpch6u6/1001",
        help="tdmpc_sep_corridornowall_patrol_rd03_smooth5_reward_smooth1")
    
    
    parser.add_argument('--sac_1', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_sac_sep_corridor_omnipatrol_nonormreward_withdonereal100_step100_cnnlaser8/2101",
        help="sac_sep_corridor_patrol_rd03_laser8_noclip")
    parser.add_argument('--sac_2', type=str, default="train_tdmpc2_irsim_laser4cnnorg_vw_fastfix_6fobsallrange_dash_his8sp1_dynobswithwallfarg_rewardexplore/irsim_sac_sep_square_omnidash_nonormreward_withdonereal100_step100_cnnlaser8/1601",
        help="sac_sep_square_dash_laser8_noclip")
    #==================================================================

    parser.add_argument('--sac_withdone_fast_laser4cnnhis8sp1_vw_irsim_dyn', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/train_SAC_irsim_dyn_uselaser4cnnorg_vw_withdone_rewardnorm_algorithm_0214/901')
    parser.add_argument('--sac_withanoise_withdone_fast_laser4cnnhis8sp1_vw_irsim_dyn', type=str, default='train_tdmpc2_traj_state_mlp_with_norm_lnrelumlp_reward_nonorm_done_20_dtheta_obs_goal_nompc_updateorg_usestatecolli/train_SAC_irsim_dyn_uselaser4cnnorg_vw_withdone_withanoise_rewardnorm_algorithm_0214/751')

    parser.add_argument('--use_vw_action', type=bool, default=True,
        help="use vw action or traj action")
    parser.add_argument('--max_traj', type=float, default=0.2,
        help="the max length of trajectories")
    parser.add_argument('--max_traj_time', type=float, default=5.0,
        help="the max time for trajectories")
    parser.add_argument('--follow_safe', type=bool, default=False,
        help="use follow safe mode")
    parser.add_argument('--barn_eval', type=bool, default=False,
        help="use barn world eval mode")
    parser.add_argument('--action_embedding_dim', type=int, default=64,
        help="The dim of action latent for tdmpc2 dynamic reward and Q model")
    parser.add_argument('--use_action_embedding', type=bool, default=False,
        help="use action embedding or not")
    parser.add_argument('--diff_action_embedding', type=bool, default=False,
        help="use diff action embedding or not (d q r)")
    parser.add_argument('--mlp_obs', type=bool, default=False,
        help="use cnn or mlp for obs encoder")
    parser.add_argument('--update_all', type=bool, default=True,
        help="update cons r q model together or seperately")
    parser.add_argument('--traj_fix_dt', type=bool, default=True,
        help='if fix dt or not, fix -> action_dim = 2, not fix -> action_dim = 3')
    parser.add_argument('--turn_easy', type=bool, default=False,
        help='if use turn easy or not (v = 0 easy)')
    parser.add_argument('--obs_low_level', type=bool, default=False,
        help="if lower the lidar levels, False: obs_shape:[184], True: obs_shape:[14]")
    parser.add_argument('--obs_more_feature', type=bool, default=False,
        help="if use more features to describe lidar data, False: obs_shape:[184], True: obs_shape:[44]")
    parser.add_argument('--state_mlp', type=bool, default=False,                                         
        help="if use state mlp for tdmpc2 (distance and theta)")
    parser.add_argument('--state_num', type=int, default=5,
        help="num of obs for each agent")
    parser.add_argument('--state_range', type=float, default=8.0,
        help="max distance for state calculate") # 8
    parser.add_argument('--state_global', type=bool, default=False,
        help="if use global state for task (dx dy in global frame)")
    parser.add_argument('--sac_mlp_enc', type=bool, default=False,
        help="if sac use mlp encoder")
    parser.add_argument('--lstm_dyn', type=bool, default=False,
        help="if use LSTM for dyn model")
    parser.add_argument('--lstm_hidden_dim', type=int, default=512,
        help="the LSTM hidden dim")
    parser.add_argument('--lstm_hidden_num', type=int, default=2,
        help="the LSTM layer num")
    parser.add_argument('--lstm_seq_len', type=int, default=1,
        help="LSTM seq length")
    parser.add_argument('--save_mode', type=bool, default=False,
        help="if save buffer or not")
    parser.add_argument('--save_eps', type=int, default=1800,
        help="the eps len for save buffer")
    parser.add_argument('--save_buffer_path', type=str, default='/home/wheel-arm/catkin_ws/src/navigation_research-master/src/navigation_research/buffer_saved_1216',
        help="buffer save path")
    parser.add_argument('--obs_fix', type=bool, default=False,
        help="if fix obs(laser) and only predict robot state")
    parser.add_argument('--max_traj_obs_fix', type=float, default=0.4, 
        help="the len of traj in obs fix mode")
    parser.add_argument('--normalization_state', type=bool, default=True,
        help="if normalize the state")
    parser.add_argument('--use_cos_sin', type=bool, default=True,
        help="obs fix mode use theta or cos and sin (theta)")
    parser.add_argument('--same_obs', type=bool, default=False,
        help="if use same obs in the obs fix mode")
    parser.add_argument('--same_goal', type=bool, default=False,
        help="if use same goal information in the obs fix mode")
    parser.add_argument('--no_laser', type=bool, default=False,
        help="if use laser information in the obs fix mode")
    parser.add_argument('--use_multi_mlp_enc', type=bool, default=False ,
        help="if use two mlp for laser and state")
    parser.add_argument('--enc_num', type=int, default=1,
        help="use_multi_mlp_enc:2 else 1")
    parser.add_argument('--dyn_num', type=int, default=1,
        help="use_multi_mlp_enc and obs_state_cat:2 else 1")
    parser.add_argument('--laser_goal_cat', type=bool, default=False,
        help="if goal take part in the dyn model")
    parser.add_argument('--obs_dim', type=int, default=180,
        help="the obs dim")
    parser.add_argument('--obs_state_cat', type=bool, default=False,
        help="if obs and state take part in the dyn model together")
    parser.add_argument('--use_multi_dyn', type=bool, default=False,
        help="if use two dyn encoder")
    parser.add_argument('--use_xy_goal', type=bool, default=False,
        help="if use g_x g_y in t0 frame")
    parser.add_argument('--sac_rand_xyyaw', type=bool, default=False,
        help="if use random x y yaw in sac")
    parser.add_argument('--sac_rand_dis_theta', type=bool, default=False,
        help="if use dis theta to replace x y")
    parser.add_argument('--state_obs_xy', type=bool, default=False,
        help="if use obs x y instead dis and theta")
    parser.add_argument('--use_fix_traj_a', type=bool, default=False,
        help="if use fix frame traj for action")
    parser.add_argument('--fix_max_traj', type=float, default=1.0,
        help="the max traj in fix frame")
    parser.add_argument('--use_org_action', type=bool, default=False,
        help="if use org action instead action in t0 frame for dyn")
    parser.add_argument('--same_obs_hand', type=bool, default=False,
        help="if same obs one by one")
    parser.add_argument('--abandon_yaw', type=bool, default=False,
        help="abandon cos sin in train")
    parser.add_argument('--zero_reward', type=bool, default=False,
        help="if reward function use zero adn one mode")
    parser.add_argument('--dis_reward', type=bool, default=False,
        help="if use - dis goal reward")
    parser.add_argument('--empty_goal_dis', type=int, default=8,
        help="goal distance in empty scene") # org 8
    parser.add_argument('--tdmpc_rand_xyyaw', type=bool, default=False,
        help="if add random x y yaw to tdmpc")
    parser.add_argument('--check_traj', type=bool, default=False,
        help="if check traj")
    parser.add_argument('--check_goal', type=bool, default=False,
        help="if check goal")
    parser.add_argument('--detach_state_z', type=bool, default=False,
        help="if detach state z to stop q and r loss backward to dyn and enc_state model")
    parser.add_argument('--use_obs_multi_enc', type=bool, default=False,
        help="if use 2 enc for laser and goal")
    parser.add_argument('--same_goal_hand', type=bool, default=False,
        help="if same goal one by one")
    parser.add_argument('--use_dxdydtheta', type=bool, default=False,
        help="if use dx dy dtheta instead of v w")
    parser.add_argument('--state_far_obs', type=bool, default=False,
        help="if put obs far away from agent in state mlp mode")
    parser.add_argument('--random_laser_goal', type=bool, default=False,
        help="if add random num to laser and goal information when obs fixed")
    parser.add_argument('--rand_obs_range', type=float, default=0.1,
        help="the random num range added to laser and goal")
    parser.add_argument('--use_multi_sep_dyn', type=bool, default=False,
        help="if use 2 dyn model for laser goal and state")
    parser.add_argument('--sac_fix_obs', type=bool, default=False,
        help="if use fix state")
    parser.add_argument('--update_2enc', type=bool, default=False,
        help="update 2enc mode")
    parser.add_argument('--teacher_forcing', type=bool, default=False,
        help="if use teacher forcing")
    parser.add_argument('--p_end', type=float, default=0.05,
        help="end p for teacher forcing")
    parser.add_argument('--p_start', type=float, default=0.8,
        help="start p for teacher forcing")
    parser.add_argument('--p_decay_rate', type=float, default=0.2,
        help="decay rate for p in teacher forcing")    
    parser.add_argument('--ideal_motion', type=bool, default=False,
        help="if use ideal motion dx dy action")
    parser.add_argument('--ideal_dis', type=float, default=0.3,
        help="the ideal action dx dy range")
    parser.add_argument('--done_collision_rate', type=float, default=1.0,
        help="the rate to limit done collision punishment")
    parser.add_argument('--no_collision', type=bool, default=False,
        help="if ignore collision in the ideal motion mode")
    parser.add_argument('--use_dtheta_action', type=bool, default=False,
        help="if use dtheta action in ideal motion mode")
    parser.add_argument('--use_decoder', type=bool, default=False,
        help="if use decoder to decode z")
    parser.add_argument('--visualize_dyn_obs', type=bool, default=True,
        help="if visulaize the pred and true obs")
    parser.add_argument('--norm_reward', type=bool, default=False,
        help="if norm the reward")
    parser.add_argument('--use_one_data', type=bool, default=False,
        help="if use one robot data")
    parser.add_argument('--mppi_horizon', type=int, default=6,
        help="the horizon in mppi")
    parser.add_argument('--update_sac_pi', type=bool, default=False,
        help="if use sac inspired update pi function")
    parser.add_argument('--norm_q', type=bool, default=False,
        help="if norm q")
    parser.add_argument('--sac_pi', type=bool, default=False,
        help="if use sac inspired pi function")
    parser.add_argument('--update_nomodel', type=bool, default=False,
        help="if use no model upadte function")
    parser.add_argument('--use_done', type=bool, default=False,
        help="if use no model upadte function")
    parser.add_argument('--dyn_state', type=bool, default=False,
        help="if use dyn state mode")
    parser.add_argument('--use_base_mlp', type=bool, default=False,
        help="if use base mlp")
    parser.add_argument('--use_real_z', type=bool, default=False,
        help="use real z")
    parser.add_argument('--use_act_dyn', type=bool, default=True,
        help="if use act for dyn (default simnorm)")
    parser.add_argument('--use_act_enc', type=bool, default=True,
        help="if use act for enc (default simnorm)")
    parser.add_argument('--use_mish', type=bool, default=True,
        help="if use mish as act for dyn and enc")
    parser.add_argument('--use_state_collision', type=bool, default=False,
        help="if use min state dis for collision dis")
    parser.add_argument('--weight_reward_loss', type=bool, default=False,
        help="if use weight to deal with big reward")
    parser.add_argument('--init_at_center', type=bool, default=False,
        help="if init robots at center in state mode")
    parser.add_argument('--reward_model_reduce', type=bool, default=False,
        help="if reduce reward sampled from buffer")
    parser.add_argument('--use_dec_reward', type=bool, default=False,
        help="if use decoder to pred reward")
    parser.add_argument('--use_mse_r', type=bool, default=True,
        help="if use mse loss for reward (update estimate)")
    parser.add_argument('--use_hot_q', type=bool, default=False,
        help="if use two hot inv for q (update world model)")
    parser.add_argument('--soft_ce_q', type=bool, default=False,
        help="if use soft ce loss for q")
    parser.add_argument('--q_discount_scale', type=float, default=1.0,
        help="q discount scale param in mppi")
    parser.add_argument('--dec_detach', type=bool, default=False,
        help="if detach decode s")
    parser.add_argument('--update_org', type=bool, default=False,
        help="if use org tdmpc update func")
    parser.add_argument('--use_lnrelu', type=bool, default=True,
        help="if use ln+lkrelu for all model (enc dyn q reward pi)")
    parser.add_argument('--pi_normln', type=bool, default=False,
        help="if use normln+mish for pi model")
    parser.add_argument('--keep_run', type=bool, default=True,
        help="if keep running (no done) sac use false tdmpc use true")
    parser.add_argument('--use_sep_state_0', type=bool, default=False,
        help="if use sep state 0 world 10x10x4")
    parser.add_argument('--use_s2l', type=bool, default=False,
        help="if add mlp (state dim --> laser dim) in state_mlp mode")
    parser.add_argument('--vis_mpc', type=bool, default=True,
        help="if visualize mppi trajs")
    parser.add_argument('--random_act_len', type=bool, default=False,
        help="if act random steps in one plan (use one = false)")
    parser.add_argument('--use_horizon', type=int, default=6,
        help="true act steps (use one = false)")
    parser.add_argument('--rand_horizon', type=int, default=6,
        help="rand act steps range (use one = false rand act len = true)")
    parser.add_argument('--dyn_laser', type=bool, default=False,
        help="if use dyn train mode with laser in 12x12x1 world")
    parser.add_argument('--check_12x12_sep', type=bool, default=False,
        help="if check 12x12 world in sep mode"    )
    parser.add_argument('--dyn_eval', type=bool, default=False,
        help="if eval dyn")
    parser.add_argument('--use_irsim', type=bool, default=True,
        help="if use irsim instead of gazebo")
    parser.add_argument('--irsim_yaml_path', type=str, default="irsim_env_0.yaml",
        help="irsim env yml path")
    parser.add_argument('--irsim_test_path', type=str, default="irsim_corridor_test_env.yaml",
        help="test irsim env, irsim_corridor_test_env, irsim_test_dynanonobs")
    parser.add_argument('--irsim_test_multi_path', type=str, default="irsim_env_multi_8robot_20x20_eval.yaml",
        help="test irsim env")
    parser.add_argument('--keep_run_arrive_stop', type=bool, default=False,
        help="if stop robot and lidar when arrive goal on keep run mode")
    parser.add_argument('--reward_with_time_punish', type=bool, default=False,
        help="if use step punishment in reward function")
    parser.add_argument('--reward_time', type=float, default=0.0,
        help="time punish")
    parser.add_argument('--far_goal', type=bool, default=True,
        help="if use far goal for train in irsim")
    parser.add_argument('--save_traj_data', type=bool, default=False,
        help="if save tdmpc eval traj data (false in train mode)")
    parser.add_argument('--deviation_mode', type=bool, default=False,
        help="if use path deviation in reward function")
    parser.add_argument('--reward_deviation', type=float, default=0.0,
        help="the reward ratio of path deviation")
    parser.add_argument('--path_dim', type=int, default=0,
        help="path deviation mode min_distance + angle")
    parser.add_argument('--norm_deviation_distance', type=float, default=3.0,
        help="norm deviation distance") 
    parser.add_argument('--reward_vel_mode', type=bool, default=False,
        help="if use reward vel")
    parser.add_argument('--reward_vel', type=float, default=1.0,
        help="the reward ratio of vel")   
    parser.add_argument('--no_goal', type=bool, default=False,
        help="if use no goal in the net input (deviation mode = True)")
    parser.add_argument('--forward_N', type=int, default=0,
        help="forward steps in deviation calculation")
    parser.add_argument('--reward_vel_cos', type=bool, default=False,
        help="if use cos angle in reward vel")
    parser.add_argument('--deviation_difference', type=bool, default=False,
        help="if use path deviation difference to calculate reward deviation")
    parser.add_argument('--use_deviation_angle', type=bool, default=False,
        help="if input deviation angle to the net")
    parser.add_argument('--use_waypoints_guide', type=bool, default=False,
        help="if use waypoints guide")
    parser.add_argument('--waypoints_distance', type=float, default=0.4,
        help="waypoints distance")
    parser.add_argument('--waypoints_threshold', type=float, default=5.6,
        help="waypoints threshold")
    parser.add_argument('--waypoints_start_idx', type=int, default=15,
        help="waypoints start idx")
    parser.add_argument('--waypoints_idx_mode', type=bool, default=False,
        help="if add waypoint idx to the net")
    parser.add_argument('--waypoints_idx_dim', type=int, default=1,
        help="waypoint idx dim")
    parser.add_argument('--progress_ratio_mode', type=bool, default=False,
        help="if put progress ratio and traj unit into the net")
    parser.add_argument('--progress_dim', type=int, default=3,
        help="progress dim")
    parser.add_argument('--reward_progress', type=float, default=80.0,
        help="the reward ratio of progress")

    parser.add_argument('--exp_reward_collision', type=bool, default=True,
        help="if use exp reward collision")
    parser.add_argument('--collision_trend', type=bool, default=False,
        help="if use collision trend")
    parser.add_argument('--reward_trend', type=float, default=0.0,
        help="the reward ratio of trend")
    parser.add_argument('--collision_k', type=float, default=5.0,
        help="exp collision k")
    parser.add_argument('--circle_init', type=bool, default=False,
        help="if use circle init state and goal")
    parser.add_argument('--cnn_slide_T', type=bool, default=False,
        help="if slide T in cnn")
    parser.add_argument('--img_channel', type=int, default=16,
        help="img channel in cnn")
    parser.add_argument('--sep_corridor_env', type=bool, default=True,
        help="if use static env")
    parser.add_argument('--vel_clip', type=bool, default=False,
        help="vel clip")
    parser.add_argument('--sep_square_env', type=bool, default=False,
        help="if use sep square env")
    parser.add_argument('--with_done_low_reward', type=bool, default=True,
        help="if use with done low reward")
    parser.add_argument('--use_mdn', type=bool, default=False,
        help="if use mdn")
    parser.add_argument('--use_mdn_loss', type=bool, default=False,
        help="if use mdn loss in mdn mode")
    parser.add_argument('--num_mixtures', type=int, default=5,
        help="num of mixtures in mdn mode")
    parser.add_argument('--clip_vel_rollout', type=bool, default=False,
        help="if clip vel in mppi rollout pi action")
    parser.add_argument('--clip_vel_sample', type=bool, default=False,
        help="if clip vel in mppi sample random action")
    parser.add_argument('--estimate_clip', type=bool, default=False,
        help="if clip vel in estimate value")
    parser.add_argument('--warm_start', type=bool, default=False,
        help="if warm start in mppi")
    parser.add_argument('--vis_mpc_trajs_irsim', type=bool, default=False,
        help="if vis mpc trajs in irsim")
    parser.add_argument('--use_smooth_penalty', type=bool, default=False,
        help="if use smooth penalty in mppi")
    parser.add_argument('--smooth_weight', type=float, default=3.0,
        help="smooth weight in mppi")
    parser.add_argument('--reward_social', type=float, default=2.0,
        help="social reward")
    parser.add_argument('--social_mode', type=bool, default=True,
        help="if use social reward")
    parser.add_argument('--social_pred_t', type=float, default=2.0,
        help="social pred t")
    parser.add_argument('--close_goal', type=bool, default=False,
        help="if close goal")    
    parser.add_argument('--small_square', type=bool, default=False,
        help="if small square")
    parser.add_argument('--clip_k', type=float, default=0.5,
        help="clip k")
    parser.add_argument('--actor_mode', type=str, default='sac',
        help="TDM(PC)^2")
    parser.add_argument('--prior_coef', type=float, default=1.0, 
        help="prior coef")
    parser.add_argument('--reward_smooth', type=float, default=2.0, 
        help="smooth weight")
    parser.add_argument('--smooth_reward_mode', type=bool, default=True,
        help="smooth reward mode")
    parser.add_argument('--aggressive_social', type=bool, default=False,
        help="if use aggressive social")
    parser.add_argument('--vis_extra_trajs', type=bool, default=False,
        help="if vis extra trajs")

    # 2026.01.16 ensemble_dyn + risk aware + CBF
    parser.add_argument('--use_ensemble_dyn', type=bool, default=False,
        help="if use ensemble dynamics model") # org change
    parser.add_argument('--ensemble_num', type=int, default=5,
        help="number of ensemble model") # 5
    parser.add_argument('--dyn_logvar_min', type=float, default=-6.0,
        help="if use ensemble dynamics model")
    parser.add_argument('--dyn_logvar_max', type=float, default=2.0,
        help="if use ensemble dynamics model")
    parser.add_argument('--dyn_rollout_mode', type=str, default="mean",
        help="ensemble dyn rollout mode")
    parser.add_argument('--use_risk_aware', type=bool, default=False,
        help="if use risk_aware MPPI")   # org change
    parser.add_argument('--num_particles', type=int, default=4,
        help="num_particles in risk aware MPPI")
    parser.add_argument('--risk_type', type=str, default="cvar",
        help="type of risk aware (cavr or mean_var or mean)")
    parser.add_argument('--cvar_alpha', type=float, default=0.9,
        help="cvar_alpha")
    parser.add_argument('--risk_lambda', type=float, default=0.3,
        help="risk_lambda")
    parser.add_argument('--dyn_plan_mode', type=str, default="mean",
        help="ensemble dyn plan mode (ts, mean, sample_mean, ts_mean, ts_mean_fixed)")
    parser.add_argument('--risk_debug', type=bool, default=False,
        help="if debug mode")
    parser.add_argument('--temp_ratio', type=float, default=1.0,
        help="temp ratio")
    parser.add_argument('--value_zscore', type=bool, default=False,
        help="if use value zscore") # org change
    parser.add_argument('--value_zscore_eps', type=float, default=1e-6,
        help="if use value zscore")
    parser.add_argument('--value_zscore_clip', type=float, default=5.0,
        help="if use value zscore")
    parser.add_argument('--dyn_noise_scale', type=float, default=0.0,
        help="dyn_noise_scale in ts mode")
    parser.add_argument('--use_unc_penalty', type=bool, default=True,
        help="if use unc penalty")
    parser.add_argument('--unc_penalty', type=float, default=0.1,
        help="if use unc penalty")
    parser.add_argument('--unc_mode', type=str, default="epistemic",
        help="if use unc penalty")
    parser.add_argument('--unc_reduce', type=str, default="mean",
        help="if use unc penalty")
    parser.add_argument('--unc_clip_max', type=float, default=5.0,
            help="if use unc penalty")

    parser.add_argument('--ret_reduce', type=str, default="mean",
        help="type of risk aware (cavr or mean_var or mean) mean stable")
    parser.add_argument('--use_unc_risk', type=bool, default=False,
        help="if use unc penalty") # org change
    parser.add_argument('--unc_beta', type=float, default=0.1,
        help="if use unc penalty")
    parser.add_argument('--unc_risk_type', type=str, default="cvar",
        help="if use unc penalty")
    parser.add_argument('--unc_cvar_alpha', type=float, default=0.9,
        help="if use unc penalty")
    parser.add_argument('--unc_lam', type=float, default=1.0,
        help="if use unc penalty")
    parser.add_argument('--unc_per_step', type=bool, default=True,
        help="if use unc penalty")
    parser.add_argument('--use_unc_penalty_in_reward', type=bool, default=False,
        help="if use unc penalty")
    parser.add_argument('--unc_sample_by_member', type=bool, default=True,
        help="if sample by member in unc mode")
    parser.add_argument('--tb_log_interval', type=int, default=1,
        help="tb_log_interval")
    parser.add_argument('--tb_log_figures', type=bool, default=False,
        help="if record fig to TB")
    parser.add_argument('--tb_fig_interval', type=int, default=10,
        help="save fig in tb times")
    parser.add_argument('--use_all_unc', type=bool, default=False,
        help="if use epi+ale unc") #
    parser.add_argument('--unc_cover_all_members', type=bool, default=False,
        help="if use all members to cover unc in MPPI") #
    parser.add_argument('--risk_rollout_by_member', type=bool, default=False,
        help="if use member to rollout in MPPI")
    parser.add_argument('--risk_rollout_sample_noise', type=bool, default=False,
        help="if add sample noise to rollout in MPPI")
    parser.add_argument('--show_traj_ret_realtime', type=bool, default=False,
        help="if show traj ret realtime")
    parser.add_argument('--use_env_dyn_weight', type=bool, default=False,
        help="if use costmap to calculate env dyn weight") #

    # ======================new project eval=====================
    # parser.add_argument('--base_line', type=str, default="new_project/corridor1_org_nosmoothpenalty_test/901",
    #     help="base_line no ensemble no risk-aware")
    parser.add_argument('--B4', type=str, default="new_project/corridor1_ensemble_riskaware_s4_alpha09_mean_uncnotinreward_unccvar09_sample_noise0_riskmean_lmd03_temp1_vzscore_nllmean_6_2_test/901",
        help="ensemble5_S4_meanrollout_uncnotinreward_uncriskcvar09_samplebynumber")
    parser.add_argument('--B3', type=str, default="new_project/corridor1_ensemble_riskaware_s4_alpha09_mean_uncinreward_noise0_riskmean_lmd03_temp1_vzscore_nllmean_6_2_test/901",
        help="ensemble5_S4_meanrollout_uncinreward_riskmean")

    parser.add_argument('--use_hybrid_attn_encoder', type=bool, default=True,
        help="if use hybrid encoder")
    parser.add_argument('--use_dual_branch_encoder', type=bool, default=True, 
        help='if use dual branch encoder')
    parser.add_argument('--hybrid_temporal_layers', type=int, default=1, 
        help='number of temporal self-attention layers')
    parser.add_argument('--hybrid_num_heads', type=int, default=4, 
        help='number of attention heads')
    parser.add_argument('--hybrid_map_dim', type=int, default=16, 
        help='feature dimension after CNN stem')
    parser.add_argument('--hybrid_cross_layers', type=int, default=1, 
        help='number of cross-attention layers')
    parser.add_argument('--hybrid_cond_dim', type=int, default=16, 
        help='dimension for condition tokens (vel/goal embedding)')
    parser.add_argument('--hybrid_temporal_pos_emb', type=bool, default=True, 
        help='whether to add temporal position embedding')
    parser.add_argument('--hybrid_temporal_pos_type', type=str, default='sin', 
        help='type of temporal position encoding: sin or learnable')
    parser.add_argument('--hybrid_disable_cross_attn', type=bool, default=True, 
        help='disable cross-attention in hybrid encoder')
    parser.add_argument('--hybrid_fusion_mode', type=str, default='gated_residual', 
        help='dual-branch fusion mode: combine_only, temporal_only, concat_mlp, residual, gated_residual')
    parser.add_argument('--hybrid_residual_alpha', type=float, default=0.05, 
        help='initial residual strength for residual/gated residual fusion')
    parser.add_argument('--hybrid_learnable_alpha', type=bool, default=True, 
        help='whether residual alpha is learnable')
        
    # ================== VIT ARGS ==================
    parser.add_argument('--image_encoder_backbone', type=str, default='cnn', 
        help='image backbone: cnn or vit')
    parser.add_argument('--vit_input_mode', type=str, default='temporal', 
        help='ViT input mode: temporal or combine')
    parser.add_argument('--vit_patch_size', type=int, default=16, 
        help='ViT patch size')
    parser.add_argument('--vit_embed_dim', type=int, default=128, 
        help='ViT embedding dimension')
    parser.add_argument('--vit_depth', type=int, default=4, 
        help='ViT number of transformer layers')
    parser.add_argument('--vit_num_heads', type=int, default=4, 
        help='ViT number of attention heads')
    parser.add_argument('--vit_mlp_ratio', type=float, default=4.0, 
        help='ViT MLP expansion ratio')
    parser.add_argument('--vit_dropout', type=float, default=0.0, 
        help='ViT dropout rate')
    parser.add_argument('--vit_use_cls_token', type=bool, default=True, 
        help='use class token in ViT')
    parser.add_argument('--vit_pos_emb_type', type=str, default='learnable', 
        help='position encoding type: learnable or sin')
    parser.add_argument('--vit_use_condition_tokens', type=bool, default=False, 
        help='concat vel/goal as tokens in ViT')    

    parser.add_argument('--hybrid_log_gate_stats', type=bool, default=True, 
        help='whether to log gate statistics')
    parser.add_argument('--hybrid_freeze_combine_warmup_steps', type=int, default=0, 
        help='number of warmup steps to freeze combine branch (0 = disabled)')
    parser.add_argument('--hybrid_use_env_complexity_gate', type=bool, default=True, 
        help='use environment complexity to gate the temporal branch')
    parser.add_argument('--hybrid_env_gate_scale', type=float, default=1.0, 
        help='scaling factor for environment complexity gate')
    parser.add_argument('--hybrid_env_front_ratio', type=float, default=0.5, 
        help='horizontal ratio of ROI for environment complexity (center width)')
    parser.add_argument('--hybrid_env_near_ratio', type=float, default=0.35, 
        help='vertical ratio of ROI for environment complexity (bottom height)')

    parser.add_argument('--use_sac', type=bool, default=False,
        help='if use sac to train model')

    parser.add_argument('--hybrid_disable_temporal_attn', type=bool, default=False, 
        help='if True, completely skip temporal attention in hybrid encoder')
    parser.add_argument('--hybrid_temporal_pool', type=str, default='mean', 
        help='temporal pooling method: mean or last')
    parser.add_argument('--hybrid_log_attn_stats', type=bool, default=True, 
        help='whether to cache attention statistics for debugging/tensorboard')

    parser.add_argument('--hybrid_force_temporal_t1', type=bool, default=False, 
        help='force temporal branch to have time length 1')
    parser.add_argument('--hybrid_temporal_t1_source', type=str, default='last', 
        help='which frame to use when T=1: last, first, or middle')
    parser.add_argument('--hybrid_log_temporal_t1', type=bool, default=True, 
        help='log the actual temporal frame count used')

    # --------------------------------------------------
    # temporal branch backbone 模式
    # 可选:
    #   '2d_stem_attn'        -> 你当前默认 single-branch hybrid
    #   '3d_stem_only'        -> H1
    #   '3d_stem_attn'        -> H2
    #   '3d_stem_flatten_mlp' -> H3
    #   'orig_3dcnn'          -> H0（严格复现原始 3D CNN temporal path）
    # --------------------------------------------------
    parser.add_argument('--hybrid_temporal_encoder_mode', type=str, default='2d_stem_attn', 
        help='temporal branch backbone: 2d_stem_attn, 3d_stem_only, 3d_stem_attn')
    parser.add_argument('--hybrid_3d_temporal_stride', type=int, default=1, 
        help='whether to reduce temporal dimension in 3D stem (1 = keep, >1 = stride)')

    # --------------------------------------------------
    # dual-branch temporal branch 选择
    # 可选:
    #   'hybrid'     -> 当前默认 hybrid temporal branch
    #   'orig_3dcnn' -> H4a
    #   'h3'         -> H4b（3d_stem_flatten_mlp）
    #   'vit'        -> dual-branch + ViT temporal correction
    # --------------------------------------------------
    parser.add_argument('--dual_temporal_branch_mode', type=str, default='hybrid', 
        help='dual-branch temporal branch selection: hybrid, orig_3dcnn, h3, vit')

    # --------------------------------------------------
    # complexity-guided gate 增强版
    # 可选:
    #   'scalar'    -> 当前已有的单一 scalar
    #   'multi_roi' -> 新增，多区域 complexity feature
    # --------------------------------------------------
    parser.add_argument('--hybrid_env_feature_mode', type=str, default='multi_roi', 
        help='complexity gate feature mode: scalar or multi_roi')

    # multi_roi 模式下，gate 输入维度（代码里自动决定，一般不用手动改）
    # 这里放着只是便于你记录实验
    parser.add_argument('--hybrid_env_gate_hidden_dim', type=int, default=64, 
        help='hidden dimension for complexity gate (multi_roi mode)')

    # ==================================================
    # training-only: temporal future auxiliary
    # ==================================================
    parser.add_argument('--use_temporal_future_aux', type=bool, default=True, 
        help='whether to use temporal future auxiliary loss')
    parser.add_argument('--temporal_future_aux_weight', type=float, default=0.05, 
        help='weight for temporal future auxiliary loss')
    parser.add_argument('--temporal_future_aux_normalize', type=bool, default=True, 
        help='whether to normalize temporal future auxiliary loss')
    parser.add_argument('--temporal_future_aux_horizon', type=int, default=1, 
        help='prediction horizon for temporal future auxiliary loss')

    # ==================================================
    # training-only: temporal interp auxiliary
    # ==================================================
    parser.add_argument('--use_temporal_interp_aux', type=bool, default=True,
                        help='Enable temporal interpolation auxiliary loss')
    parser.add_argument('--temporal_interp_aux_weight', type=float, default=0.03,
                        help='Weight for temporal interpolation auxiliary loss')
    parser.add_argument('--temporal_interp_aux_normalize', type=bool, default=True,
                        help='Disable normalization in temporal interpolation auxiliary (default: enabled)')
    parser.add_argument('--temporal_interp_aux_horizon', type=int, default=1,
                        help='Horizon for temporal interpolation auxiliary')

    # ==================================================
    # training-only: risk proxy auxiliary
    # ==================================================
    parser.add_argument('--use_risk_proxy_aux', type=bool, default=True,
                        help='Enable risk proxy auxiliary loss')
    parser.add_argument('--risk_proxy_aux_weight', type=float, default=0.03,
                        help='Weight for risk proxy auxiliary loss')
    parser.add_argument('--risk_proxy_aux_horizon', type=int, default=1,
                        help='Horizon for risk proxy auxiliary')

    # ================ eval loop =====================
    parser.add_argument('--irsim_display', type=bool, default=True,
        help="show irsim GUI window; set False for headless eval loop")
    parser.add_argument('--eval_model_key', type=str, default='base_line',
                    help='Key for evaluation model (default: base_line)')
    parser.add_argument('--eval_preset', type=str, default='baseline',
                    help='Preset for evaluation (default: baseline)')
    parser.add_argument('--base_line', type=str, default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_base_cnn_laser8_nospenalty_sm2_h6u6_gpu0_518/801",
        help="base_line no ensemble no risk-aware")        
    parser.add_argument('--D3', type=str, default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_multi_roi_laser8_nospenalty_sm2_h6u6_gpu1_518/801",
        help="base_line no ensemble no risk-aware")   
    parser.add_argument('--D3_aux1', type=str, default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux1_laser8_nosp_sm2_h6u6_gpu0_518/801",
        help="base_line no ensemble no risk-aware")
    parser.add_argument('--D3_aux2', type=str, default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux2_laser8_nosp_sm2_h6u6_gpu0_518/801",
        help="base_line no ensemble no risk-aware")
    parser.add_argument('--D3_aux3', type=str, default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux3_laser8_nosp_sm2_h6u6_gpu0_518/801",
        help="base_line no ensemble no risk-aware")                                                                                
    parser.add_argument('--D3_seed77', type=str, default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_multi_roi_seed77_laser8_nosp_sm2_h6u6_gpu0_518/801",
        help="base_line no ensemble no risk-aware")   
    parser.add_argument('--D3_seed66', type=str, default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_multi_roi_seed66_laser8_nosp_sm2_h6u6_gpu0_518/801",
        help="base_line no ensemble no risk-aware") 
    # ============
    parser.add_argument('--D3_aux1_seed77', type=str, default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux1_seed77_laser8_nosp_sm2_h6u6_gpu0_518/801",
        help="base_line no ensemble no risk-aware")   
    parser.add_argument('--D3_aux1_seed66', type=str, default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux1_seed66_laser8_nosp_sm2_h6u6_gpu0_518/801",
        help="base_line no ensemble no risk-aware")   
    parser.add_argument('--D3_aux12_seed66', type=str, default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux12_laser8_nosp_sm2_h6u6_gpu0_518/801",
        help="base_line no ensemble no risk-aware")   
    parser.add_argument('--D3_aux123_seed99', type=str, default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux123_laser8_nosp_sm2_h6u6_gpu0_518/801",
        help="base_line no ensemble no risk-aware")  
    parser.add_argument('--D3_scalar', type=str, default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_scalar_laser8_nospenalty_sm2_h6u6_gpu1_518/801",
        help="base_line no ensemble no risk-aware")   
    parser.add_argument('--D3_vit', type=str, default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_vit_ps16_multi_roi_seed99_laser8_nosp_sm2_h6u6_gpu0_518/801",
        help="base_line no ensemble no risk-aware")

    # ===== Phase B: P_safe A/B/C 两个新 checkpoint =====
    parser.add_argument('--D3_aux123_nopsafe', type=str,
        default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux123_seed99_laser8_nosp_sm2_h6u6_gpu0_626/801",
        help="best main line (hybrid+multi_roi+aux123), NO psafe head")
    parser.add_argument('--D3_aux123_psafe', type=str,
        default="new_project/hybrid_encode_exp_4_13/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux123_psafe_seed99_laser8_nosp_sm2_h6u6_gpu0_626/801",
        help="best main line (hybrid+multi_roi+aux123) + psafe head")
    
    parser.add_argument('--D3_optim', type=str,
        default="new_project/nav_WM/simrobot_v4w3_corridor1_dual_branch_multi_roi_noaux_optim_seed99_laser8_nosp_sm2_h6u6_gpu0_630/901",
        help="no aux + no psafe")
    parser.add_argument('--D3_aux1_optim', type=str,
        default="new_project/nav_WM/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux1_optim_seed99_laser8_nosp_sm2_h6u6_gpu0_630/901",
        help="aux1 in optim + no psafe")
    parser.add_argument('--D3_aux12_optim', type=str,
        default="new_project/nav_WM/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux12_optim_seed99_laser8_nosp_sm2_h6u6_gpu0_630/901",
        help="aux12 in optim + no psafe")
    parser.add_argument('--D3_aux123_optim', type=str,
        default="new_project/nav_WM/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux123_optim_seed99_laser8_nosp_sm2_h6u6_gpu0_630/901",
        help="aux123 in optim + no psafe")
    parser.add_argument('--D3_aux123_psafe_optim', type=str,
        default="new_project/nav_WM/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux123_psafe_optim_seed99_laser8_nosp_sm2_h6u6_gpu1_630/901",
        help="aux123 in optim + psafe head in optim")
    parser.add_argument('--D3_aux123_psafe_detach_optim', type=str,
        default="new_project/nav_WM/simrobot_v4w3_corridor1_dual_branch_multi_roi_aux123_psafe_detach_optim_seed99_laser8_nosp_sm2_h6u6_gpu1_630/901",
        help="aux123 in optim + psafe head in optimtrained with stop-gradient (detached z)")

    # ===== Phase B: deterministic P_safe (survival) head =====
    parser.add_argument('--use_psafe_head', type=bool, default=False,
        help="build & train the deterministic collision/survival head")
    parser.add_argument('--psafe_head_weight', type=float, default=0.05,
        help="BCE training weight for the survival head")
    parser.add_argument('--psafe_label_soft', type=bool, default=False,
        help="True: soft target = near-occupancy; False: binary by threshold")
    parser.add_argument('--psafe_label_thr', type=float, default=0.5,
        help="near-occupancy threshold for binary unsafe label")
    parser.add_argument('--use_psafe', type=bool, default=False,
        help="use logP_safe (log-survival) in _estimate_value scoring")
    parser.add_argument('--psafe_lambda', type=float, default=1.0,
        help="weight of log-survival term in posterior score")
    parser.add_argument('--psafe_tail_beta', type=float, default=0.0,
        help="weight of worst-step severity term (max_t p_unsafe)")
    parser.add_argument('--psafe_additive_ablation', type=bool, default=False,
        help="ABLATION: use additive -sum(p) instead of multiplicative log-survival")   
    parser.add_argument('--psafe_zscore', type=bool, default=True,
        help="z-score the P_safe term across K candidates before adding to return")
    parser.add_argument('--psafe_p_cap', type=float, default=0.9,
        help="per-step clamp upper bound on p_unsafe (prevents log(1-p) explosion)")
    parser.add_argument('--psafe_lambda_cli', type=float, default=None,
        help="CLI override for psafe_lambda (None = use preset value)")
    parser.add_argument('--psafe_head_detach', type=bool, default=True,
        help="train collision head on detached z (head as read-out, encoder untouched)")

    # use_multi_enc? laser_goal_cat? obs_state_cat?

    
    # ========================================> Parse Args <========================================
    # args = parser.parse_args(sys.argv[1:-2]) if IF_START_WITH_ROSLAUNCH else parser.parse_args()
    _is_roslaunch = IF_START_WITH_ROSLAUNCH and len(sys.argv) >= 2 \
        and any(a.startswith('__') and ':=' in a for a in sys.argv[-2:])
    args = parser.parse_args(sys.argv[1:-2]) if _is_roslaunch else parser.parse_args()
    
    # ========================================> Conditional Params <========================================
    args.buffer_size = 8e5 if args.info_type == "state_based" else args.buffer_size
    args.threshold_collision = 1.4 if args.info_type == "state_based" else args.threshold_collision
    args.state_dim = args.sample_length * (args.num_agent - 1) * 4 + args.vel_dim + args.goal_dim if args.info_type == "state_based" else args.state_dim  
    args.sample_length = math.ceil(args.history_length / args.sample_interval)
    
    args.temperature = args.temperature * args.temp_ratio

    args.path_dim = 2 if args.deviation_mode else 0
    args.goal_dim = 0 if args.no_goal else 2
    args.progress_dim = 2 if args.progress_ratio_mode else 0
    if args.no_goal:
        args.reward_goal = 0
    args.state_dim = args.sample_length * (180+3) + args.vel_dim + args.goal_dim + args.path_dim + args.progress_dim
    
    if args.circle_init:
        args.irsim_yaml_path = "irsim_env_circle_train.yaml"
        args.num_agent = 8
    if args.sep_corridor_env:
        args.irsim_yaml_path = "irsim_sep_corridor_env_1.yaml"
    elif args.sep_square_env:
        if args.small_square:
            args.irsim_yaml_path = "irsim_sep_square_env_4.yaml"
        else:
            args.irsim_yaml_path = "irsim_sep_square_env_1.yaml"
    # args.state_dim = args.sample_length * (180+3) + args.vel_dim + args.goal_dim
    args.batch_size = 256 if args.parallel else args.batch_size
    args.action_embedding_dim = 64 if args.use_action_embedding else args.action_dim

    if args.obs_low_level:
        args.obs_dim = 40 if args.obs_more_feature else 10
    elif args.state_mlp:
        if args.dyn_state:
            args.obs_dim = (args.num_agent - 1) * 4
        else:
            args.obs_dim = 2 * args.state_num
    elif args.no_laser:
        args.obs_dim = 0

    if not args.deviation_mode:
        args.reward_deviation = 0.0
    # if args.update_nomodel:
    #     args.visualize_dyn_obs = False

    if args.obs_fix:
        if args.use_cos_sin:
            if args.use_dxdydtheta:
                args.obs_shape = {"state": [args.obs_dim + args.goal_dim + args.vel_dim + 1 + 4]} if not args.abandon_yaw else {"state": [args.obs_dim + args.goal_dim + args.vel_dim + 1 + 2]}
            
            else:
                args.obs_shape = {"state": [args.obs_dim + args.goal_dim + args.vel_dim + 4]} if not args.abandon_yaw else {"state": [args.obs_dim + args.goal_dim + args.vel_dim + 2]}
        else:
            args.obs_shape = {"state": [args.obs_dim + args.goal_dim + args.vel_dim + 3]} if not args.abandon_yaw else {"state": [args.obs_dim + args.goal_dim + args.vel_dim + 2]}
    elif args.ideal_motion:
        if args.no_laser:
            args.obs_shape = {"state": [args.goal_dim]}
        else:
            args.obs_shape = {"state": [args.obs_dim + args.goal_dim]}
    else:
        args.obs_shape = {"state": [args.obs_dim + args.goal_dim + args.vel_dim]}

    
    args.enc_num = 2 if args.use_multi_mlp_enc else 1
    args.dyn_num = 2 if args.use_multi_mlp_enc and args.obs_state_cat else 1

    if args.use_sac:
        args.keep_run = False
        args.with_done_low_reward = False

    if args.keep_run or args.with_done_low_reward:
        args.reward_done = 1.0 
        if not args.eval_mode:
            args.episode_length = 100 if args.use_sep_state_0 else 200
            if args.world_idx == 19:
                args.episode_length = 300
            if args.use_irsim:
                if args.max_linear_vel > 1.8:
                    args.episode_length = 150 if not (args.sep_corridor_env or args.sep_square_env) else 100
                else:
                    args.episode_length = 150 if not (args.sep_corridor_env or args.sep_square_env) else 200
    else:
        args.reward_done = 100.0
        if args.use_irsim:
                if args.max_linear_vel > 1.8:
                    args.episode_length = 150 if not (args.sep_corridor_env or args.sep_square_env) else 100
                else:
                    args.episode_length = 150 if not (args.sep_corridor_env or args.sep_square_env) else 200
        else:
            args.episode_length = 300
    
    args.use_state_collision = args.state_mlp

    if args.mpc:
        args.mppi_horizon = args.horizon

    if (not args.use_one) and (not args.random_act_len):
        args.use_horizon = args.horizon

    if not args.state_mlp:
        args.use_s2l = False

    if args.random_exploration_length < 200:
        args.episode_length = 8

    if args.world_idx == 19:
        args.init_robot_position_range = [9.5, 9.5]
    elif args.world_idx == 18:
        args.init_robot_position_range = [5.0, 5.0]
    elif args.world_idx == 20:
        args.init_robot_position_range = [6.0, 3.0] # [3.5, 3.5]

    args.pixel_per_meter = args.img_height / args.laser_range
    # if args.obs_low_level:
    #     args.obs_shape = {"state":[44]} if args.obs_more_feature else {"state":[14]}
    # else:
    #     if args.state_mlp and args.state_global:
    #         # args.obs_shape = {"state":[int(3*args.state_num + 2 + 4)]}
    #         args.obs_shape = {"state":[int(2*args.state_num + 2 + 3)]}
    #     elif args.state_mlp:
    #         if args.obs_fix:
    #             args.obs_shape = {"state":[int(2*args.state_num + 4 + 3)]} if not args.use_cos_sin else {"state":[int(2*args.state_num + 4 + 4)]}
    #         else:    
    #             args.obs_shape = {"state":[int(2*args.state_num + 4)]}
    #     elif args.obs_fix:
    #         if args.no_laser:
    #             args.obs_shape = {"state": [7]} if not args.use_cos_sin else {"state":[8]}    
    #         else:
    #             args.obs_shape = {"state": [187]} if not args.use_cos_sin else {"state":[188]}
    #     else:
    #         args.obs_shape = {"state": [184]}
    
    return args

    
if __name__ == '__main__':
    args = get_config()
    print(args)
