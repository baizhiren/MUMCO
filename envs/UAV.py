from enum import Enum
from typing import List

from envs import Map, env_settings
from random import random as rand
import  random

from envs.Entity import Entity


class UAV_Action:
    def __init__(self):
        
        self.u = None

        
        self.c = None

        self.trans_power_ratio = 1
        self.computing_ratio = 1

class UAV(Entity):
    def __init__(self, name:str):
        super().__init__(name)
        self.color = (rand(), rand(), rand())
        
        self.id = int(self.name.split('-')[1])

        self.index = 0

        self.movable = True
        self.silent = False

        self.computing_rate = env_settings.uav_computing_rate_from
        self.trans_power_ratio = 1
        self.computing_ratio = 1

        self.init()


    @property
    def cur_and_future_task(self):
        return self.taskList + self.uav_receive_task_list + self.uav_take_tasks_list


    def init(self):
        self.taskList = []
        self.action:UAV_Action = UAV_Action()

        # 记录上一次位置，用于计算飞行能耗
        self.prev_x = self.x
        self.prev_y = self.y

        self.cold_execute_time = 0
        self.cold_trans_time = 0
        self.cold_take_time = 0

        self.last_trans_time = 0
        self.last_compute_time = 0

        self.trajectory = []
        self.current_step = 0

        self.old_tasks_list = []
        self.uav_take_tasks_list = []
        self.uav_receive_task_list = []
        self.clear_status()

    def update_position_history(self):
        """更新位置历史，用于计算飞行能耗"""
        if self.x is not None and self.y is not None:
            self.prev_x = self.x
            self.prev_y = self.y


    def clear_status(self):
        
        
        

        self.uav_execute_task_list = []
        self.uav_trans_task_list = []
        self.uav_fail_task_num = 0

        self.not_move_update = 0
        self.not_execute_update = 0

        self.cold_execute_update = 0
        self.cold_trans_update = 0


        self.too_far_update = 0
        self.invalid_position_update = 0
        self.trans_time_out = 0

        self.take_energy = 0
        self.trans_energy = 0
        self.execute_energy = 0

        self.fly_energy = 0

    def get_min_remain_task(self):
        target_task = None
        for task in self.taskList:
            if target_task is None or target_task.remain_time > task.remain_time:
                target_task = task
        return target_task

    @property
    def task_names(self):
        return [task.name for task in self.taskList]


    def fly(self, delta_x: int, delta_y: int):
        self.x += delta_x
        self.y += delta_y
    
    def record_position(self, step):

        if not self.trajectory or self.trajectory[-1][:2] != (self.x, self.y):
            self.trajectory.append((self.x, self.y, step))
        self.current_step = step
    
    def get_trajectory(self):
        
        return self.trajectory.copy()
    
    def clear_trajectory(self):
        
        self.trajectory = []
        self.current_step = 0

    def get_take_num(self):
        return sum(task.small_task_num for task in self.uav_take_tasks_list)


    def merge(self):
        self.old_tasks_list = self.taskList.copy()
        
        for task in self.uav_take_tasks_list:
            assert task.remain_take_time != None
            if task.is_fail:
                self.uav_take_tasks_list.remove(task)
                continue
            if task.remain_take_time >= 0:
                
                task.remain_take_time -= 1

                if task.remain_take_time <= 1e-6:
                    
                    self.uav_take_tasks_list.remove(task)
                    self.taskList.append(task)
                    task.belong_uav = self


        for task in reversed(self.uav_receive_task_list):
            assert task.remain_trans_time != None
            if task.is_fail:
                self.uav_receive_task_list.remove(task)
                continue


            if task.remain_trans_time >= 0:
                task.remain_trans_time -= 1
                if task.remain_trans_time <= 1e-6:
                    
                    self.uav_receive_task_list.remove(task)
                    self.taskList.append(task)
                    task.belong_uav = self
