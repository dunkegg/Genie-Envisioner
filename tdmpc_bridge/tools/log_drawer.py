import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from termcolor import cprint
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def parse_data(records, experiment_name, end_step, key, key_rename, smooth_window, is_average_data=False):
    
    x = []
    y = []
    for record in records:
        ea = event_accumulator.EventAccumulator(record)
        ea.Reload()
        # print(ea.scalars.Keys())
        episode_data = ea.scalars.Items(key)

        total_step = 0
        sum_data = 0
        temp_window = []
        
        history_total_value= 0
        
        for data in episode_data:
            
            if total_step <= end_step:
                
                total_step += 1
                x.append(data.step)
                
                if is_average_data:
                    value = data.value * total_step - history_total_value
                    history_total_value =  data.value * total_step
                else:
                    value = data.value
                
                # Original
                # y.append(value)
                
                # Average
                # sum_data += value
                # average_data = sum_data / total_step
                # y.append(average_data)
                
                # Smooth
                if total_step <= (smooth_window + 1):
                    temp_window.append(value)
                else:
                    temp_window.pop(0)
                    temp_window.append(value)
                y.append(sum(temp_window) / len(temp_window))

        
    data = pd.DataFrame({
        'episode': x,
        key_rename : y,
        'Methods': experiment_name
    })
    return data


if __name__ == '__main__':
    
    end_step = 200 # 300
    smooth_window = 20
    
    success_data_list = []
    collision_data_list = []
    step_data_list = []
    
    log_dir = os.getcwd() + '/src/navigation_research/log/policy/'
    vis_dir = os.getcwd() + '/src/navigation_research/scripts/visualize/'
    
    experiment_list = ['train_rl', 'train_cbf_joint', 'train_cbf_not_joint', 'FDMCA', 'MNDS']
    experiment_name_list = ['Ours/RL', 'Ours/RL_CBF_joint', 'Ours/RL_CBF_refine', 'FDMCA', 'MNDS']
    # 200 : 'Ours/RL' : 3, 'Ours/RL_CBF_joint' : 2 + (seed 777 to be done), 'Ours/RL_CBF_refine' : 3, 'FDMCA' : 3, 'MNDS' : 3
    
    cprint('Plot Data ==> Length : ' + str(end_step) + '  Smooth Window : ' + str(smooth_window), color='green', attrs=['reverse', 'bold'])

    name_index = 0
    # ===============================================   Criteria/episode_success_rate  ===============================================
    t1_ = time.time()
    for experiment in experiment_list:
        num = len(os.listdir(log_dir + experiment + '/'))
        for i in range(num):
            t1 = time.time()
            experiment_dir = [log_dir + experiment + '/' + os.listdir(log_dir + experiment + '/')[i]]
            success_data = parse_data(experiment_dir, experiment_name_list[name_index], end_step, 'Criteria/episode_success_rate', 'success rate', smooth_window)
            collision_data = parse_data(experiment_dir, experiment_name_list[name_index], end_step, 'Criteria/episode_collision_rate', 'collision rate', smooth_window)
            step_data = parse_data(experiment_dir, experiment_name_list[name_index], end_step, 'Criteria/episode_average_step', 'total step', smooth_window, is_average_data=True)
            collision_data_list.append(collision_data)
            success_data_list.append(success_data)
            step_data_list.append(step_data)
            cprint("Experiment " + experiment_name_list[name_index] + " Data " + str(i) + " Done!!!", color='yellow', attrs=['reverse', 'bold'])
            t2 = time.time()
            cprint("Data Time : {}  ms".format(round(1000*(t2-t1),2)), color='yellow', attrs=['reverse', 'bold'])
        name_index += 1
    t2_ = time.time()
    cprint("All_Data Time : {}  ms".format(round(1000*(t2_-t1_),2)), color='green', attrs=['reverse', 'bold'])
    # ===============================================   Plot  ===============================================
    
    plot_success_data = pd.concat(success_data_list, ignore_index=True)
    plot_collision_data = pd.concat(collision_data_list, ignore_index=True)
    plot_step_data = pd.concat(step_data_list, ignore_index=True)
    
    
    # plt.subplot(1, 3, 1)
    plt.rcParams['figure.figsize'] = (22.0, 8.0)
    # plt.rc('legend', fontsize=24)
    plt.rc('font', size=20)
    plt.figure()
    sns.lineplot(data=plot_success_data, x='episode', y='success rate', hue='Methods', style="Methods", err_style="band", ci=95).set_title("success rate")
    plt.savefig(vis_dir + 'training_curves_test_success_rate' + str(end_step) + '_' + str(smooth_window) + '.png')
    # plt.subplot(1, 3, 2)
    plt.rcParams['figure.figsize'] = (22.0, 8.0)
    # plt.rc('legend', fontsize=20)
    plt.rc('font', size=20)
    plt.figure()
    sns.lineplot(data=plot_collision_data, x='episode', y='collision rate', hue='Methods', style="Methods", err_style="band", ci=95).set_title("collision rate")
    plt.savefig(vis_dir + 'training_curves_test_collision_rate' + str(end_step) + '_' + str(smooth_window) + '.png')
    # plt.subplot(1, 3, 3)
    plt.rcParams['figure.figsize'] = (22.0, 8.0) 
    # plt.rc('legend', fontsize=20)   
    plt.rc('font', size=20)
    plt.figure()
    sns.lineplot(data=plot_step_data, x='episode', y='total step', hue='Methods', style="Methods", err_style="band", ci=95).set_title("total step")
    plt.savefig(vis_dir + 'training_curves_test_step' + str(end_step) + '_' + str(smooth_window) + '.png')
    
    # plt.savefig(vis_dir + 'training_curves_' + str(smooth_window) + '.svg')
    cprint("Image Saved !!!", color='green', attrs=['reverse', 'bold'])