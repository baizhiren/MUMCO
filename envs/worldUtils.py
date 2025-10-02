import math
from envs import settings
from envs.BigTask import BigTask

from envs.utils import distance

import numpy as np


def get_total_energy(world):
    total_trans_energy = sum(uav.trans_energy for uav in world.uavs)
    total_take_energy = sum(uav.take_energy for uav in world.uavs)
    total_execute_energy = sum(uav.execute_energy for uav in world.uavs)
    total_fly_energy = sum(uav.fly_energy for uav in world.uavs)

    total_energy = total_trans_energy + total_take_energy + total_execute_energy + total_fly_energy
    return total_energy, {
        "total_trans_energy": total_trans_energy,
        "total_take_energy": total_take_energy,
        "total_execute_energy": total_execute_energy,
        "total_fly_energy": total_fly_energy
    }


def get_object(world):
    
    delay_O = 0
    V0 = 2
    V1 = 5
    for task in world.done_task:
        if isinstance(task, BigTask):
            continue
        delay_O += math.log2(1 + task.remain_time) + V0
    for task in world.fail_task:
        if isinstance(task, BigTask):
            continue
        delay_O -= V1
    energy_O, other_info = get_total_energy(world)
    avg_energy_O = 0

    task_lengths = [len(uav.taskList) for uav in world.uavs]  
    std_val = np.std(task_lengths)  
    mean_val = np.mean(task_lengths)
    
    max_tasks = max(task_lengths)

    
    min_tasks = min(task_lengths)



    other_info = {
        "task_list_std_val": std_val,
        "max_task_num":max_tasks,
        "min_task_num": min_tasks,
        "mean_val": mean_val,
        "task_lengths": task_lengths  
    }

    final_O =  delay_O, energy_O, (delay_O * settings.w1 - energy_O * settings.w2), avg_energy_O, other_info
    
    
    return final_O
def get_available_action(world, uav: 'UAV'):
    available_actions = []

    
    trans_action = np.ones(len(world.uavs) + 1)
    for id, other_uav in enumerate(world.uavs):
        if len(uav.taskList) == 0:
            trans_action[id] = 0
            continue
        task = uav.get_min_remain_task()
        
        if len(task.trans_path) - 1 >= settings.hop_limit:
            trans_action[id] = 0
            continue
        if other_uav == uav:
            if uav.cold_execute_time > 1:
                
                trans_action[id] = 0
            continue
        if uav.cold_trans_time > 1:
            trans_action[id] = 0
            continue
        dist = distance(other_uav, uav)
        if dist >= world.uav_to_uav_dist \
                or len(other_uav.taskList) - len(uav.taskList) >= -1:
            trans_action[id] = 0


    available_actions.append(trans_action)

    return trans_action
