from typing import List

from envs import env_settings
from envs.Entity import Entity
from envs.UAV import UAV
import envs.Manager as Manager

class Task(Entity):
    def __init__(self, remain_time:int, name: str):
        super().__init__(name)

        self.remain_time = remain_time

        
        self.is_take = False
        self.take_time = None

        
        self.is_done = False

        
        self.is_fail = False

        self.status = []

        
        self.belong_uav:UAV = None

        
        self.trans_path:List[UAV] = []

        self.color = (0.5, 1, 0.5)
        self.size = 0.3

        self.small_task_num = 1

        self.belong_big_task = None

        self.remain_take_time = None
        self.remain_trans_time = None

        self.total_trans_time = 0
        self.take_consume_time = 0

        self.last_trans_time = None
        
        self.wait_time = None

        self.ok = env_settings.ok_to
        self.fk = env_settings.fk_to
        
        self.delta_i = remain_time 
        self.t_gen = 0 

        self.gain = None

    
    
    
    
    
    

    def isExpire(self):
        if self.remain_time <= Manager.get_min_execute_time():
            return True
        return False

    def take(self, uav: UAV):
        if not self.is_take:
            self.is_take = True
            self.take_time = self.remain_time
            remain_take_time, take_enegry = Manager.get_take_task_time_and_energy(uav, self)
            self.status.append(f'剩余时间{self.remain_time}, 首次传输给:{uav.name}, take_time:{int(remain_take_time)}, take_energy:{take_enegry:.2f}')

            self.remain_take_time = int(remain_take_time)
            self.take_consume_time = int(remain_take_time)
            uav.take_energy += take_enegry
            uav.cold_take_time = int(remain_take_time)
            self.bind(uav)
            uav.uav_take_tasks_list.append(self)
        else:
            raise NotImplementedError("当前任务已被认领，请勿重复take")

    
    def bind(self, uav: UAV):
        self.x = uav.x
        self.y = uav.y


        

        
        

        self.trans_path.append(uav)


    def status_info(self):
        result = f"{self.name}, [" + "; ".join(self.status) + "]"
        return result

    def trans(self, uav: UAV):
        
        remain_trans_time, trans_enegry = Manager.get_trans_task_time_and_energy(self.belong_uav, uav, self)
        if remain_trans_time >= self.remain_time:
            self.status.append(
                f'剩余时间{self.remain_time}, 尝试从{self.belong_uav.name}中继给: {uav.name}超时, trans_time:{int(remain_trans_time)}, trans_energy:{trans_enegry:.2f}')
            return False, remain_trans_time, trans_enegry
        self.status.append(f'剩余时间{self.remain_time}, 从{self.belong_uav.name}中继给: {uav.name}, trans_time:{int(remain_trans_time)}, trans_energy:{trans_enegry:.2f}')

        self.belong_uav.uav_trans_task_list.append(self)
        uav.uav_receive_task_list.append(self)


        self.remain_trans_time = int(remain_trans_time)
        self.total_trans_time += int(remain_trans_time)

        self.last_trans_time = self.remain_time

        self.belong_uav.trans_energy += trans_enegry
        
        self.belong_uav.cold_trans_time = int(remain_trans_time)

        self.bind(uav)
        if self.belong_uav != None:
            
            if self in self.belong_uav.taskList:
                self.belong_uav.taskList.remove(self)
            else:
                raise NotImplementedError("belong_uav 不包含当前任务")
        self.belong_uav = '传输中'
        return True, remain_trans_time, trans_enegry

    def execute(self, uav: UAV):
        
        remain_execute_time, execute_energy = Manager.get_uav_execute_task_time_and_energy(uav, self)

        
        if self.remain_time - remain_execute_time < 0:
            self.status.append(
                f'剩余时间{self.remain_time}, {self.belong_uav.name} 试图执行, 执行时间:{remain_execute_time}, 本次执行失败')
            return False, remain_execute_time, execute_energy


        self.x = uav.x
        self.y = uav.y
        self.is_done = True
        self.belong_uav.uav_execute_task_list.append(self)
        self.belong_uav.taskList.remove(self)

        uav.cold_execute_time += int(remain_execute_time)
        uav.execute_energy += execute_energy

        if self.last_trans_time is None:
            self.wait_time = self.take_time - self.remain_time
        else:
            self.wait_time = self.last_trans_time - self.remain_time

        self.remain_time -= remain_execute_time

        self.status.append(f'剩余时间{self.remain_time}, {self.belong_uav.name} 执行完成, 执行时间:{remain_execute_time}, execute_energy:{execute_energy:.2f}')
        if self.belong_big_task is not None:
            self.belong_big_task.done_small_tasks += 1
            if self.belong_big_task.done_small_tasks == self.belong_big_task.small_task_num:
                self.belong_big_task.is_done = True
                
                self.belong_big_task.remain_time = self.remain_time
        return True,remain_execute_time, execute_energy

    def fail(self):
        self.status.append(f'剩余时间{self.remain_time}, 执行失败')
        if self.belong_uav is not None and not isinstance(self.belong_uav, str):
            self.belong_uav.uav_fail_task_num += 1
            try:
                self.belong_uav.taskList.remove(self)
            except Exception as e:
                print(e)
        self.is_fail = True
        if self.belong_big_task is not None:
            self.belong_big_task.done_small_tasks += 1
            self.belong_big_task.is_fail = True
            if self.belong_big_task.done_small_tasks == self.belong_big_task.small_task_num:
                self.belong_big_task.is_done = True


    def daze(self):
        self.status.append(f'剩余时间{self.remain_time}, 什么也没做')

