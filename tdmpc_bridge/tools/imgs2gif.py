import os
import imageio
from termcolor import cprint
from pathlib import Path
import cv2
import numpy as np


def imgs2gif(img_paths, img_nums, text_list, out_path, duration=None, loop=0, fps=None, gif_size=400):
    """
    Function:
        Images => GIF
    Input:
        img_paths : list of the directory path of images in multiple experiments
        img_nums  : list of the number of iamges in multiple experiments
        text_list : list of image comment texts
        out_path  : GIF saving path
        duration  : time duration (s) between adjacent frames in GIF, 1 / fps if None
        fps       : frames per second
        loop      : loop number, 0 means endless loop
    """

    if fps:
        duration = 1 / fps
    
    gif_length = max(img_nums) + 25
    images = []
    v_bar_width = int(gif_size / 20)
    v_bar = cv2.cvtColor(np.zeros((gif_size, v_bar_width)).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    for i in range(gif_length):
        imgs = []
        for exp, path in enumerate(img_paths):
            idx = i if i < img_nums[exp] else img_nums[exp] - 1
            img = imageio.imread(str(path + str(idx) + '.png'))
            img = cv2.resize(img, (gif_size, gif_size)).astype(np.uint8) 
            
            # exp name
            text_exp_name = text_list[exp].replace("_", " ")
            pos = (np.array([gif_size * 0.03, gif_size * 0.97])).astype(np.int32)
            img = cv2.putText(img, text_exp_name, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # step num
            text_num_step = str(idx + 1)
            pos = (np.array([gif_size * 0.9, gif_size * 0.97])).astype(np.int32)
            img = cv2.putText(img, text_num_step, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
              
            imgs.extend([v_bar, img])
        imgs.append(v_bar)
        merge_img = cv2.hconcat(imgs)
        images.append(merge_img)
        
    imageio.mimsave(out_path, images, "gif", duration=duration, loop=loop)
    

def main():
    # 31,34,89, 108, 119, 120, 124, 146, 168, 170, 175
    gif_size = 1280 * 2
    img_root = "/home/cyx/navigation_research/src/navigation_research/scripts/env/graph_data/img/"  
    
    exp_name = 'train_graph_navigation_frontier_GAT_size_30_large_sparse_env_env_0_Graph_Exploration_175_30_eval'
    exp_list = ['GOAL_GREEDY', 'RL', 'UTILITY'] # 'RL', 'GOAL_GREEDY', 'UTILITY'
    
    img_paths = [img_root + exp_name + '_' + exp + '/' for exp in exp_list]
    out_path = "/home/cyx/navigation_research/src/navigation_research/scripts/env/graph_data/gifs/" + exp_name + '_all.gif'
    
    img_nums = []
    cprint("======>  Experiments :  ", color='yellow', attrs=['bold'])
    for idx, path in enumerate(img_paths):
        tmp_num = 0
        for file in os.listdir(path):
            if file[-4:] == '.png':
                tmp_num += 1
        img_nums.append(tmp_num)
        cprint("{} : {}".format(exp_list[idx], tmp_num), color='yellow', attrs=['bold'])
    cprint("======>  Converting {} images into {} x {} GIF : \n {}".format(max(img_nums), gif_size, gif_size, exp_name), color='yellow', attrs=['bold'])
    
    # Record GIF
    imgs2gif(img_paths, img_nums, exp_list, out_path, 0.2, 0, gif_size)
    cprint("======>  GIF Saved !!! ", color='green', attrs=['bold'])
                
        
if __name__ == '__main__':
    main()
    