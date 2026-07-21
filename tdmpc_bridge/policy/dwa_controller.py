"""
dwa package from: https://github.com/goktug97/DynamicWindowApproach
"""

import dwa
import cv2
import math
import numpy as np
from termcolor import cprint

from .. import params
from tools.utils import cv2_show_img, map2points, CoarseSimulator

    
class DWA_Control():
    def __init__(self):
        super(DWA_Control, self).__init__()
        
        self.points = []
        
        self.vel = (0.0, 0.0)
        self.pose = (0.0, 0.0, 0.0)
        self.goal = None
        # self.width = 1.6
        # self.length = 1.6
        
        self.width = 1.6
        self.length = 1.6
        

        self.scale = 1.0
        self.base = [-self.length / 2 * self.scale, -self.width / 2 * self.scale,
                      self.length / 2 * self.scale,  self.width / 2 * self.scale]
        self.config = dwa.Config(
            max_speed = 1.0,
            min_speed = 0.1,
            max_yawrate = np.radians(90.0),
            max_accel = 15.0,
            max_dyawrate = np.radians(300.0),
            velocity_resolution = 0.1,
            yawrate_resolution = np.radians(1.0), 
            dt = 0.1,
            predict_time = 1.0,
            heading = 0.2,
            clearance = 0.2, 
            velocity = 0.2,
            base = self.base
            )


    def select_action(self, vel, goal, laser):
        
        self.state_interpreter(vel, goal, laser)
    
        cmd_vel = dwa.planning(self.pose, self.vel, self.goal, self.points, self.config)
        
        if cmd_vel[0] == self.goal[0] or cmd_vel[1] == self.goal[1]:
            cmd_vel = np.array([0, 0])
            cprint("==================    Can not find a path !!!    ==================", color='red', attrs=['reverse', 'bold'])
        elif np.sqrt(self.goal[0] ** 2 + self.goal[1] ** 2) < 0.15:
            cmd_vel = np.array([0, 0])
        else:
            # TODO: params
            cmd_vel = np.array([
                np.clip(cmd_vel[0],  -0.5, 1.5),
                np.clip(cmd_vel[1], -1, 1)
                ])
            print("DWA velocity : ", cmd_vel)
        # _ = self.dwa_visualize(cmd_vel, True)

        return cmd_vel.copy()


    def state_interpreter(self, vel, goal, laser):
        # self.vel=(vel[0], vel[1])
        self.vel=(0.0, 0.0)
        self.pose = (0.0, 0.0, 0.0)
        self.goal = (goal[0].astype(np.float32), goal[1].astype(np.float32))
        # self.points = laser
        
        self.points = - np.ones([1,2]).astype(np.float32) * 99
        
