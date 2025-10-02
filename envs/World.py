from random import  randint
from typing import List

import numpy as np

from envs.BigTask import BigTask
from envs.UAV import UAV
from envs.Task import Task
from envs import settings, utils, Manager, ans
from envs.Map import DiscreteMap
from envs import worldUtils
from envs.utils import distance


class World:
    def __init__(self, act_map:DiscreteMap, **kwargs):
        self.uavs:List[UAV] = []

        
        self.alive_task = []

        
        self.trans_task = 0

        
        self.take_task = 0



        self.current_step_fail_num = 0

        
        self.done_task = []

        self.fail_task = []

        
        self.total_tasks = []
        

        
        
        

        
        self.dim_p = 2

        
        self.dim_color = 3

        self.print_log_num = 0


        
        self.world_step = 0
        self.task_id = 0

        self.world_length = None
        self.task_num = None
        self.uav_num = None

        
        self.collaborative = False

        self.act_map:DiscreteMap = act_map

        self.uav_to_task_dist = kwargs.get('uav_to_task_dist', settings.uav_to_task_dist)
        self.uav_to_uav_dist = kwargs.get('uav_to_uav_dist',  settings.uav_to_uav_dist)

        self.init()
        self.cycle_init_var()

        self.reset_count = 0
        self.seed = None

    def cycle_init_var(self):
        self.success_tasks_num = 0
        self.total_fail_tasks_num = 0

        self.total_task_take_num = 0
        self.take_but_fail_tasks_num = 0
        self.not_take_fail_tasks_num = 0

        self.total_task_start_take_time = 0
        self.total_task_take_consume = 0
        self.total_task_trans_time = 0

        self.history_total_big_task_num = 0
        self.history_total_small_task_num = 0
        self.success_big_task_num = 0
        self.success_small_task_num = 0

        self.sucess_trans_path_num = 0
        self.tot_success_small_task_trans_num = 0

        self.success_task_done_time = 0
        self.success_task_start_take_time = 0
        self.success_task_take_time = 0
        self.success_task_wait_time = 0
        self.success_task_trans_time = 0

        self.total_update = 0
        self.invalid_position_update = 0
        self.collision_position_update = 0
        self.not_move_update = 0
        self.not_execute_update = 0
        self.cold_execute_update = 0
        self.cold_trans_update = 0
        self.too_far_update = 0

        self.beyond_hop_limit = 0

        
        self.trans_power_info = [0, 0, 0, 0, 0, 0]
        
        self.trans_power_info_map = {}

        
        self.compute_power_info = [0, 0, 0, 0, 0]

        
        self.trans_time_out = 0
        self.execute_time_out = 0

        
        self.utility_info = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.fail_task_info = [0, 0]
        
        
        self.task_lengths_history = []  

    
    def init(self):

        self.tot_big_task_num = 0
        self.tot_small_task_num = 0
        self.total_tasks_len = 0

        self.print_id = 0
        self.init_seconds_before = True

        self.success_tasks_num_step_one = 0


    @property
    def dim_c(self):
        return len(self.uavs) + 1

    @property
    def silent_dim_c(self):
        return 2

    
    @property
    def entities(self):
        return self.uavs + self.not_take_tasks

    @property
    def not_take_tasks(self):
        return [task for task in self.alive_task if not task.is_take]

    
    @property
    def policy_agents(self):
        return self.uavs

    
    @property
    def scripted_agents(self):
        return None


    def print_info(self):
        self.print_uav()
        print('------')
        self.print_task()

    def print_uav(self):
        if self.print_id == 0:
            print(f"\nstep: {self.world_step}------------- info -------------")
        for uav in self.uavs:
            print(f'name:{uav.name}, pos:{uav.x, uav.y}, taskList:{uav.task_names}')

    def print_task(self):
        print('alive task:')
        for task in self.alive_task:
            print(f'  name:{task.name}, pos:{task.x, task.y}, task status:{task.status}, remain_time:{task.remain_time}')

        if self.done_task:
            print('done task:) ')
        for task in self.done_task:
            if isinstance(task, Task):
                print(f'  name:{task.name}, pos:{task.x, task.y}, task status:{task.status}, '
                      f'接受时间:{task.take_time} 剩余:{task.remain_time}, 差值:{task.take_time - task.remain_time}')
            elif isinstance(task, BigTask):
                task.show_small_tasks_info()


        if self.fail_task:
            print('fail task:(')
        for task in self.fail_task:
            if isinstance(task, Task):
                if (task.belong_uav != None):
                   msg = f'领取时间{task.take_time} 你就不能执行一下吗'
                else:
                   msg = f'执行失败, 可怜的孩子都没人认领'
                print(f'  name:{task.name}, pos:{task.x, task.y}, task status:{task.status}, remain_time:{task.remain_time}, {msg}')
            elif isinstance(task, BigTask):
                task.show_small_tasks_info()

    def calculate_distances(self):
        self.dists = np.zeros((len(self.alive_task), len(self.uavs)))
        for ia, entity_a in enumerate(self.alive_task):
            for ib in range(ia + 1, len(self.uavs)):
                entity_b = self.entities[ib]
                tot_dist = utils.distance(entity_a, entity_b)
                self.dists[ia, ib] = tot_dist
                self.dists[ib, ia] = tot_dist

    def get_print_id(self):
        if self.print_id == 0:
            print(f"\nstep: {self.world_step}------------- info -------------")
        self.print_id += 1
        return self.print_id

    
    def step(self):
        self.world_step += 1
        self.init_seconds_before = False

        self.trans_task = 0
        self.take_task = 0
        self.fail_task = []
        self.done_task = []
        self.print_id = 0
        self.uav_current_positions = set()

        for uav in self.uavs:
            self.total_update += 1
            uav.clear_status()
        

        for uav in self.uavs:
            # 处理动作类型转换
            if hasattr(settings, 'use_continuous_movement') and settings.use_continuous_movement:
                uav.action.c = int(uav.action.c)
            else:
                uav.action.u = int(uav.action.u)
                uav.action.c = int(uav.action.c)
            if not (hasattr(settings, 'use_continuous_movement') and settings.use_continuous_movement):
                if uav.action.u == 4:
                    self.uav_current_positions.add((uav.x, uav.y))
                    
            if not settings.fix_trans_and_compute_when_train:
                uav.trans_power_ratio = min(uav.action.trans_power_ratio + 0.1, 1)
                uav.computing_ratio = min(uav.action.computing_ratio + 0.1, 1)
                pass

        for uav in self.uavs:
            uav.cold_execute_time = max(uav.cold_execute_time - 1, 0)
            uav.cold_take_time = max(uav.cold_take_time - 1, 0)
            uav.cold_trans_time = max(uav.cold_trans_time - 1, 0)
            self.update_uav_position(uav)
            
            uav.record_position(self.world_step)


        remove_task = []
        remove_bigTasks = []
        for task in self.alive_task:
            
            if task.isExpire():
                    if isinstance(task, Task):
                        if task.belong_uav is not None:
                            self.take_but_fail_tasks_num += 1
                        else:
                            self.not_take_fail_tasks_num += 1
                        self.total_fail_tasks_num += 1
                    elif isinstance(task, BigTask):
                        
                        self.not_take_fail_tasks_num += task.small_task_num
                        self.total_fail_tasks_num += task.small_task_num

                    if settings.print_success_or_fail:
                        if(task.belong_uav != None):
                            print(f'{self.get_print_id()}. 任务{task.name}执行失败, 领取时间{task.take_time} 你就不能执行一下吗')
                        else:
                            print(f'{self.get_print_id()}. 任务{task.name}执行失败, 可怜的孩子都没人认领')
                    task.fail()

                    
                    if isinstance(task, BigTask):
                        self.fail_task.extend(task.small_task_list)

                    self.fail_task.append(task)
                    
                    remove_task.append(task)
                    continue
            else:
                task.remain_time -= 1
                if task.remain_time <= settings.task_alive_time // 2:
                    task.color = (1, 1, 0.5)
                if task.remain_time <= settings.task_alive_time // 4:
                    task.color = (1, 0.5, 0.5)

            if not task.is_take:
                
                min_dist = 10000000
                min_uav = None
                for uav in self.uavs:
                    
                    if uav.cold_take_time > 0: continue
                    dist = utils.distance(uav, task)
                    if dist <= self.uav_to_task_dist and dist < min_dist:
                        min_dist = dist
                        min_uav = uav
                if min_uav != None:
                    task.take(min_uav)
                    if isinstance(task, BigTask):
                        
                        remove_bigTasks.append(task)
                    self.take_task += task.small_task_num

        
        self.alive_task = [task for task in self.alive_task if
                           (task not in remove_task and task not in remove_bigTasks)]


        
        for task in remove_task:
            if isinstance(task, Task):
                self.generate_by_task(task)
            elif isinstance(task, BigTask):
                
                self.generate_task(1, True)

        for bigtask in remove_bigTasks:
            self.alive_task += bigtask.small_task_list

        
        for uav in self.uavs:
            self.update_uav_task(uav)

        
        for uav in self.uavs:
            uav.merge()

        
        
        self.cal_statistic_info_one_step()

    def check_and_update_position(self, x, y, uav):
        if y >= settings.width or y < 0:
            self.invalid_position_update += 1
            uav.invalid_position_update += 1
            self.uav_current_positions.add((uav.x, uav.y))
            return
        if x >= settings.length or x < 0:
            self.invalid_position_update += 1
            uav.invalid_position_update += 1
            self.uav_current_positions.add((uav.x, uav.y))
            return
        if (x, y) in self.uav_current_positions:
            self.collision_position_update += 1
            uav.invalid_position_update += 1
            self.uav_current_positions.add((uav.x, uav.y))
            return

        self.uav_current_positions.add((x, y))
        
        # 在更新位置前记录历史位置
        uav.update_position_history()
        uav.x, uav.y = x, y
        uav.fly_energy = Manager.get_fly_energy(uav)

    def update_uav_position(self, uav: UAV):
        from envs import env_settings
        
        if hasattr(settings, 'use_continuous_movement') and settings.use_continuous_movement:
            # 连续移动模式：极坐标系统
            self._continuous_move_polar(uav, uav.action.u, env_settings.step_len)
        else:
            # 离散移动模式
            d = uav.action.u
            if d == 0:
                self.check_and_update_position(uav.x, uav.y + 1, uav)  
            elif d == 1:
                self.check_and_update_position(uav.x, uav.y - 1, uav)  
            elif d == 2:
                self.check_and_update_position(uav.x - 1, uav.y, uav)  
            elif d == 3:
                self.check_and_update_position(uav.x + 1, uav.y, uav)  
            elif d == 4:
                self.not_move_update += 1
                uav.not_move_update += 1
                pass  
            else:
                raise ValueError("Invalid direction: d must be between 0 and 4")

    def _continuous_move_polar(self, uav, action, delta_t):
        """
        连续移动：使用极坐标系统
        action[0]: 速度 [0,1] -> [0, v_max]
        action[1]: 方向 [0,1] -> [0, 360度)
        """

        from envs import env_settings

        if isinstance(action, (int, float)):
            action = [0.0, 0.5]  # 默认静止，方向朝北

        speed = action[0] * env_settings.v_max
        angle = action[1] * 2 * np.pi

        delta_t = 0.1;

        # 计算位移向量
        dx = speed * np.cos(angle) * delta_t
        dy = speed * np.sin(angle) * delta_t
        
        # 更新位置（四舍五入取整）
        new_x = uav.x + round(dx)
        new_y = uav.y + round(dy);

        # 边界约束
        new_x = max(0, min(settings.length * settings.size_length, new_x))
        new_y = max(0, min(settings.width * settings.size_length, new_y))
        
        # 检查碰撞（使用四舍五入的整数位置进行碰撞检测）
        grid_x, grid_y = int(round(new_x)), int(round(new_y))
        if (grid_x, grid_y) in self.uav_current_positions:
            self.collision_position_update += 1
            uav.invalid_position_update += 1
            self.uav_current_positions.add((int(round(uav.x)), int(round(uav.y))))
            return
        
        # 更新位置（支持浮点数坐标）
        self.uav_current_positions.add((grid_x, grid_y))
        
        # 在更新位置前记录历史位置
        uav.update_position_history()
        uav.x, uav.y = new_x, new_y
        uav.fly_energy = Manager.get_fly_energy(uav)


    def update_uav_task(self, uav: UAV):
        
        next_uav_id = uav.action.c
        
        if next_uav_id >= len(self.uavs):
            self.not_execute_update += 1
            uav.not_execute_update += 1
            return

        next_uav = self.uavs[next_uav_id]
        taskList = uav.taskList

        if len(taskList) == 0:
            
            return
        else:
            task = uav.get_min_remain_task()
            if next_uav == uav:
                if uav.cold_execute_time > 0:
                    self.cold_execute_update += 1
                    uav.cold_execute_update += 1
                    return

                self.compute_power_info[4] += self.world_step - uav.last_compute_time
                uav.last_compute_time = self.world_step
                
                success_flag, execute_time, execute_energy = task.execute(uav)

                self.compute_power_info[0] += 1
                self.compute_power_info[1] += uav.computing_ratio
                self.compute_power_info[2] += execute_time
                self.compute_power_info[3] += execute_energy


                if not success_flag:
                    self.execute_time_out += 1
                    return

                if settings.print_success_or_fail:
                    try:
                        print(f'{self.get_print_id()}. 任务{task.name}执行成功， 剩余时间{task.remain_time}, 差值:{task.take_time - task.remain_time}')
                    except:
                        print('错误')
                self.done_task.append(task)
                
                assert task.take_time != task.remain_time
                self.success_tasks_num += 1
                self.success_tasks_num_step_one += 1
                self.success_task_done_time += (settings.task_alive_time -  task.remain_time)
                self.success_task_wait_time += task.wait_time
                
                self.success_task_start_take_time += (settings.task_alive_time - task.take_time)
                self.success_task_trans_time += task.total_trans_time
                self.success_task_take_time += task.take_consume_time
                self.sucess_trans_path_num += len(task.trans_path) - 1


                if task.belong_big_task is not None:
                    self.success_small_task_num += 1
                    self.tot_success_small_task_trans_num += len(task.trans_path) - 1

                if task in self.alive_task:
                    self.alive_task.remove(task)
                else:
                    raise NotImplementedError("错误，当前任务不存在于alive_task")
                self.generate_by_task(task)
            else:
                dist = distance(next_uav, task.belong_uav)
                flag = False
                if uav.cold_trans_time > 0:
                    self.cold_trans_update += 1
                    uav.cold_trans_update += 1
                    flag = True

                if dist > self.uav_to_uav_dist:
                    
                    self.too_far_update += 1
                    uav.too_far_update += 1
                    flag = True

                if len(task.trans_path) - 1 >= settings.hop_limit:
                    self.beyond_hop_limit += 1
                    flag = True

                if flag:
                    return

                self.trans_power_info[5] += self.world_step - task.belong_uav.last_trans_time
                task.belong_uav.last_trans_time = self.world_step
                trans_success, trans_time, trans_engery = task.trans(next_uav)

                self.trans_power_info[0] += 1
                self.trans_power_info[1] += uav.trans_power_ratio
                self.trans_power_info[2] += trans_time
                self.trans_power_info[3] += trans_engery
                self.trans_power_info[4] += dist
                
                dist_info = self.trans_power_info_map.get(dist, [0, 0])
                dist_info[0] += 1
                dist_info[1] += uav.trans_power_ratio
                self.trans_power_info_map[dist] = dist_info
                if trans_success:
                    self.trans_task += 1
                else:
                    self.trans_time_out += 1
                    uav.trans_time_out += 1

    def generate_by_task(self, task):
        if isinstance(task, Task):
            belong_big_task:BigTask = task.belong_big_task
            if belong_big_task is not None:
                
                if belong_big_task.is_done and not belong_big_task.renew:
                    belong_big_task.renew = True
                    if belong_big_task.is_success:
                        self.success_big_task_num += 1
                        
                        self.done_task.append(belong_big_task)
                    else:
                        self.fail_task.append(belong_big_task)
                    self.generate_task(1, True)

            else:
                self.generate_task(1, False)

    def get_load_balance_factor(self):
        uav_task_len = [len(uav.taskList) for uav in self.uavs]
        return (max(uav_task_len) - min(uav_task_len))



    def generate(self, n):
        big_task_num = int(n * settings.bigTask_ratio)
        small_task_num = n - big_task_num
        self.generate_task(big_task_num, big_task=True)
        self.generate_task(small_task_num, big_task=False)

    def generate_task(self, n, big_task=False, fix=None):
        for i in range(n):
            if big_task:
                small_task_num = randint(settings.small_task_num_range[0], settings.small_task_num_range[1])
                task:BigTask = BigTask(f'BigTask_{self.task_id}', small_task_num)

                self.tot_small_task_num += task.small_task_num
                self.tot_big_task_num += 1
            else:
                task:Task = Task(remain_time=settings.task_alive_time, name=f'task_{self.task_id}')

            if fix != None and i < len(fix):
                task.x, task.y = fix[i]
            else:
                
                if settings.use_gaussian_task_generation:
                    task.x, task.y = self.act_map.multi_gaussian_generate(
                        centers=settings.gaussian_centers,
                        weights=settings.gaussian_weights,
                        sigma=settings.gaussian_sigmas
                    )
                else:
                    task.x, task.y = self.act_map.generate()

            if big_task:
                task.set_small_tasks_position(task.x, task.y)


            self.alive_task.append(task)
            if settings.debug:
                self.total_tasks.append(task)
                self.total_tasks_len += task.small_task_num
            self.task_id += 1

    @property
    def current_total_big_task_num(self):
        size = self.tot_big_task_num
        not_done_big_tasks = set()
        for task in self.alive_task:
            if isinstance(task, BigTask):
                not_done_big_tasks.add(task)
            elif isinstance(task, Task) and task.belong_big_task != None:
                not_done_big_tasks.add(task.belong_big_task)
        size -= len(not_done_big_tasks)
        assert size >= 0
        return size

    @property
    def current_total_small_task_num(self):
        len = self.tot_small_task_num
        len -= sum(task.small_task_num for task in self.alive_task if isinstance(task, BigTask) or task.belong_big_task != None)
        assert len >= 0
        return len

    @property
    def history_task_num(self):
        len = self.total_tasks_len
        len -= sum(task.small_task_num for task in self.alive_task)
        assert len >= 0
        return len


    def print_dist_info(self):
        
        
        
        pass
    
    def save_trajectories_to_ans(self):
        
        
        if not hasattr(ans, 'uav_trajectories'):
            ans.uav_trajectories = {}
        
        for uav in self.uavs:
            ans.uav_trajectories[uav.id] = uav.get_trajectory()
    
    def clear_trajectories(self):
        
        for uav in self.uavs:
            uav.clear_trajectory()

    def cal_statistic_info_one_step(self):
        delay, energy, object, avg_energy, info = worldUtils.get_object(self)
        self.utility_info[0] += 1
        self.utility_info[1] += delay
        self.utility_info[2] += energy
        self.utility_info[3] += object
        self.utility_info[4] += avg_energy
        self.utility_info[5] += len(self.done_task)
        self.utility_info[6] += info["task_list_std_val"]
        self.utility_info[7] += info["max_task_num"]
        self.utility_info[8] += info["min_task_num"]
        self.utility_info[9] += info["mean_val"]
        
        
        self.task_lengths_history.append(info["task_lengths"].copy())

        for task in self.fail_task:
            if isinstance(task, BigTask):
                continue
            self.fail_task_info[0] += 1
            self.fail_task_info[1] += settings.fail_delay


    
    
    def cal_statistic_info(self):
        
        for task in self.total_tasks:
            if task in self.alive_task: continue
            if isinstance(task, Task):
                if task.is_take:
                    self.total_task_take_consume += task.take_consume_time
                    self.total_task_start_take_time += (settings.task_alive_time - task.take_time)
                    self.total_task_trans_time += task.total_trans_time
                    self.total_task_take_num += 1
            else:
                for small_task in task.small_task_list:
                    if small_task not in self.alive_task and small_task.is_take:
                        self.total_task_take_consume += small_task.take_consume_time
                        self.total_task_start_take_time += (settings.task_alive_time - small_task.take_time)
                        self.total_task_trans_time += small_task.total_trans_time
                        self.total_task_take_num += 1

        






        self.history_total_big_task_num += self.current_total_big_task_num
        self.history_total_small_task_num += self.current_total_small_task_num



    def print_log(self, calOnly=False):

        self.save_trajectories_to_ans()
        ans.current_run += 1

        self.print_log_num += 1

        import math
        def safe_divide(numerator, denominator):
            if denominator == 0:
                return math.nan
            else:
                return numerator / denominator




        total_task_num = self.success_tasks_num + self.total_fail_tasks_num
        assert total_task_num != 0
        print(f"\n--------------------统计信息{self.print_log_num} steps:{self.world_step}, {settings.current_run}")

        ans.success_rate.append(safe_divide(self.success_tasks_num, total_task_num))
        ans.object.append(safe_divide(self.utility_info[3], self.utility_info[0]))
        ans.delay.append(safe_divide(self.success_task_done_time + self.fail_task_info[1], self.success_tasks_num + self.fail_task_info[0]))
        ans.energy.append(safe_divide(self.utility_info[2], self.utility_info[0]))

        ans.avg_energy.append(safe_divide(self.utility_info[2], self.utility_info[5]))
        ans.std.append(safe_divide(self.utility_info[6], self.utility_info[0]))
        ans.max.append(safe_divide(self.utility_info[7], self.utility_info[0]))
        ans.min.append(safe_divide(self.utility_info[8], self.utility_info[0]))
        ans.mean.append(safe_divide(self.utility_info[9], self.utility_info[0]))
        


        flattened_task_lengths = []
        for step_lengths in self.task_lengths_history:
            flattened_task_lengths.extend(step_lengths)
        ans.task_distributions.append(flattened_task_lengths)

        print(
              f"任务总成功率: {safe_divide(self.success_tasks_num, total_task_num):.4f}"
              f" 相对成功率率: {safe_divide(self.success_tasks_num, self.total_task_take_num):.4f}"
              f" 单次成功率: {safe_divide(self.success_tasks_num_step_one, self.history_task_num):.4f}"
              f" 任务总失败率: {safe_divide(self.total_fail_tasks_num, total_task_num):.4f}"
              
              f" 任务接受率: {safe_divide(self.total_task_take_num, total_task_num):.4f}" 
              f" 任务接受但是失败率: {safe_divide(self.take_but_fail_tasks_num, total_task_num):.4f}"
              f" 任务没有被认领失败率:{safe_divide(self.not_take_fail_tasks_num, total_task_num):.4f}\n"
                
              f" 总时延效用函数: {safe_divide(self.utility_info[1], self.utility_info[0]):.4f}" 
              f" 总能量: {safe_divide(self.utility_info[2], self.utility_info[0]):.4f}"
              f" 每成功任务平均能量:{safe_divide(self.utility_info[2], self.utility_info[5]):.4f}\n"
              f" 平均Object值:{safe_divide(self.utility_info[3], self.utility_info[0]):.4f}\n"
              f" 平均队列长度std:{safe_divide(self.utility_info[6], self.utility_info[0]):.4f}\n"
              f" 平均最大任务数:{safe_divide(self.utility_info[7], self.utility_info[0]):.4f}\n"
              f" 平均最小任务数:{safe_divide(self.utility_info[8], self.utility_info[0]):.4f}\n"
                          
              f"任务平均开始接受时间:{safe_divide(self.total_task_start_take_time, self.total_task_take_num):.4f}"
              f" 任务平均接受时间:{safe_divide(self.total_task_take_consume, self.total_task_take_num):.4f}"
              f" 任务平均总传输时间:{safe_divide(self.total_task_trans_time, self.total_task_take_num):.4f}\n"
            
              f"大任务成功率: {safe_divide(self.success_big_task_num, self.history_total_big_task_num):.3f}"
              f" 大任务里小任务成功率: {safe_divide(self.success_small_task_num, self.history_total_small_task_num):.3f}\n"
    
              f"成功任务平均传输次数: {safe_divide(self.sucess_trans_path_num, self.success_tasks_num):.3f}"
              f" 成功小任务平均传输次数: {safe_divide(self.tot_success_small_task_trans_num, self.success_small_task_num):.3f}\n"
              
              f"成功任务平均开始接受时间: {safe_divide(self.success_task_start_take_time, self.success_tasks_num):.3f}"
              f" 成功任务平均接受时间: {safe_divide(self.success_task_take_time, self.success_tasks_num):.3f}"
              f" 成功任务平均传输时间: {safe_divide(self.success_task_trans_time, self.success_tasks_num):.3f}"
              f" 成功任务平均等待时间: {safe_divide(self.success_task_wait_time, self.success_tasks_num):.3f}"
              f" 成功任务平均总完成时间: {safe_divide(self.success_task_done_time, self.success_tasks_num):.3f}"
              f" 总时延: {safe_divide(self.success_task_done_time + self.fail_task_info[1], self.success_tasks_num + self.fail_task_info[0]):.3f}"
              
              f"\n越界位置更新比例: {safe_divide(self.invalid_position_update, self.total_update):.3f} "
              f" 碰撞移动比例: {safe_divide(self.collision_position_update, self.total_update):.3f} "
              f" 未移动比例: {safe_divide(self.not_move_update, self.total_update):.3f} "
              f"\n未执行比例: {safe_divide(self.not_execute_update, self.total_update):.3f} "
              f" 冷冻期执行比例: {safe_divide(self.cold_execute_update, self.total_update):.3f}"
              f" 冷冻期传输比例: {safe_divide(self.cold_trans_update, self.total_update):.3f}"
              f" 太远通信比例: {safe_divide(self.too_far_update, self.total_update):.3f}"
              f" 超出跳数限制比例: {safe_divide(self.beyond_hop_limit, self.total_update):.3f}\n"
              
              f"通信平均分配功率{self.trans_power_info[1] / self.trans_power_info[0]}"
              f" 超时传输比例{self.trans_time_out / self.trans_power_info[0]}"
              f" 平均单次传输时间{self.trans_power_info[2] / self.trans_power_info[0]}"
              f" 平均传输能量{self.trans_power_info[3] / self.trans_power_info[0]}"
              f" 平均传输距离{self.trans_power_info[4] / self.trans_power_info[0]}"
              f" 平均传输时间间隔{self.trans_power_info[5] / self.trans_power_info[0]}\n"  
              
              f"计算平均分配功率{self.compute_power_info[1] / self.compute_power_info[0]}"
              f" 超时执行比例{self.execute_time_out / self.compute_power_info[0]}"
              f" 平均计算时间{self.compute_power_info[2] / self.compute_power_info[0]}"
              f" 平均计算能量{self.compute_power_info[3] / self.compute_power_info[0]}"
              f" 平均计算时间间隔{self.compute_power_info[4] / self.compute_power_info[0]}"
              f"\n完成任务数: {self.success_tasks_num}")



        self.print_dist_info()
    def reset(self):
        self.reset_count += 1

        if settings.debug and settings.env == 'render' and self.reset_count == settings.skip_before_render_num * 2:
            self.cycle_init_var()

        if self.reset_count % 50 == 0 and settings.debug and self.seed < 100 and settings.env == 'train':
            self.print_log()
            self.cycle_init_var()

        self.clear()
        self.generate(self.task_num)
    def clear(self):

        for uav in self.uavs:
            uav.init()

        old_total_task = []
        self.total_tasks = []
        self.alive_task = []
        self.done_task = []
        self.fail_task = []
        self.trans_task = 0
        self.take_task = 0
        self.world_step = 0
        self.task_id = 0
        

        self.clear_trajectories()





        self.init()
        return old_total_task


if __name__ == '__main__':
    print('this is world')
    my_map = DiscreteMap(10, 10)
    world = World(my_map)


