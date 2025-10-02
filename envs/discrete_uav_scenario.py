import numpy as np

import envs.settings as settings, envs.utils as utils
from envs.BigTask import BigTask
from envs.Map import DiscreteMap
from envs.UAV import UAV
from envs.World import World
from envs import worldUtils

from envs.scenario import BaseScenario



class Scenario(BaseScenario):
    def make_world(self, **kargs):
        acMap: DiscreteMap = DiscreteMap(settings.length, settings.width)
        world = World(acMap, **kargs)

        
        world.world_length = settings.world_length


        world.task_num = kargs.get('task_num', settings.task_num)
        world.uav_num =  kargs.get('uav_num', settings.uav_num)

        world.uavs = [UAV(f'uav-{i}') for i in range(world.uav_num)]

        for i, uav in enumerate(world.uavs):
            uav.name = 'uav-%d' % i
            uav.x, uav.y = acMap.generate()

        
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.reset()


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False


    def reward(self, uav, world):
        return worldUtils.get_object(world)[2]

    # shape 版reward
    def shape_reward(self, uav, world):
        done_factor = settings.done_factor
        trans_energy_factor = settings.trans_energy_factor
        compute_energy_factor = settings.compute_energy_factor

        rew = 0
        

        distance_punish = 0
        for task in world.not_take_tasks:
            dists = [utils.distance(task, uav)
                     for uav in world.uavs]
            distance_punish += min(dists) / (world.act_map.length + world.act_map.width)
        rew -= distance_punish

        
        done_reward = 0
        for task in uav.uav_execute_task_list:
            
            done_reward += done_factor * (task.remain_time + 1) / settings.task_alive_time + done_factor

        rew += done_reward
        share_trans_reward = 0
        big_task_done_extra_reward = 0
        for task in world.done_task:
            if isinstance(task, BigTask):
                
                
                continue
            if len(task.trans_path) <= 1:
                continue
            trans_path_set = set(task.trans_path)
            if uav in trans_path_set:
                share_trans_reward += done_factor / 4 * 0.5
            if uav == task.trans_path[0]:
                
                share_trans_reward *= 2

        rew += share_trans_reward
        rew += big_task_done_extra_reward

        take_task_reward = 5 * uav.get_take_num()
        rew += take_task_reward

        fail_task_punish = done_factor * len(world.fail_task)
        rew -= fail_task_punish

        rew -= uav.invalid_position_update
        
        rew -= uav.not_move_update / 5

        
        rew -= uav.cold_execute_update * 2
        rew -= uav.cold_trans_update
        rew -= uav.too_far_update

        not_execute_update_punish = len(uav.old_tasks_list) / 5
        rew -= not_execute_update_punish

        
        rew += len(uav.uav_trans_task_list) / 5

        trans_energy_punish = uav.trans_energy * trans_energy_factor
        execute_energy_punish = uav.execute_energy * compute_energy_factor

        rew -= trans_energy_punish
        rew -= execute_energy_punish

        if settings.print_reward_detail:
            print(f'\n-------------{uav.name}\'s reward info ----------')
            print(f"Reward total:{rew}")
            print(f"  Distance punish: {distance_punish:.2f}\n")
            print(f"  Done reward: {done_reward:.2f}\n")
            print(f"  Share trans reward: {share_trans_reward:.2f}\n")
            
            
            print(f"  Take task reward: {take_task_reward:.2f}\n")
            
            print(f"  Fail task punish: {fail_task_punish:.2f}\n")
            print(f"无效位置更新惩罚: {uav.invalid_position_update:.2f}")
            print(f"未移动更新惩罚: {uav.not_move_update / 5:.2f}")

            print(f"冷静期更新惩罚: {uav.cold_execute_update:.2f}\n")
            print(f"过远传输惩罚: {uav.too_far_update:.2f}")
            print(f"什么都不做更新惩罚: {not_execute_update_punish:.2f}")
            print(f"任务传输奖励: {len(uav.uav_trans_task_list) / 5:.2f}\n")
            print(f"任务传输能量惩罚: {trans_energy_punish:.2f}")
            print(f"无人机执行能量惩罚:{execute_energy_punish:.2f}")

        return rew

    def observation(self, uav, world):
        
        self_info = [np.array([uav.id, len(uav.taskList)])]

        
        task_pos = []
        big_tasks = set()
        obs_tasks = []
        
        for task in world.alive_task:
            if isinstance(task, BigTask):
                big_tasks.add(task)
                continue
            if  task.belong_big_task is not None:
                big_tasks.add(task.belong_big_task)
                continue
            if not task.is_take:
                
                obs_tasks.append(task)

        for big_task in big_tasks:
            for task in big_task.small_task_list:
                if not task.is_take:
                    obs_tasks.append(task)

        sorted_tasks = sorted(obs_tasks, key=lambda task: utils.distance(uav, task))
        idx = 0

        for task in sorted_tasks:
            task_pos.append(np.array([uav.x - task.x, uav.y - task.y]))
            idx += 1
            if idx >= settings.obs_tasks_num:
                break
        while idx < settings.obs_tasks_num:
            task_pos.append(np.array([-1, -1]))
            idx += 1
        other_uav_info = []
        for other in world.uavs:
            if other is uav:
                continue
            other_uav_info.append(np.array([uav.x - other.x, uav.y - other.y, len(uav.taskList) - len(other.taskList)]))
            
    
        return np.concatenate(self_info + task_pos + other_uav_info)

    def observation2(self, uav, world):
        
        X = settings.obs_tasks_num
        
        dtask_j = []
        T_j = []
        
        obs_tasks = []
        big_tasks = set()
        
        for task in world.alive_task:
            if isinstance(task, BigTask):
                big_tasks.add(task)
                continue
            if task.belong_big_task is not None:
                big_tasks.add(task.belong_big_task)
                continue
            if not task.is_take:
                obs_tasks.append(task)
        
        for big_task in big_tasks:
            for task in big_task.small_task_list:
                if not task.is_take:
                    obs_tasks.append(task)
        
        sorted_tasks = sorted(obs_tasks, key=lambda task: utils.distance(uav, task))[:X]
        
        for i in range(X):
            if i < len(sorted_tasks):
                task = sorted_tasks[i]
                dtask_j.extend([uav.x - task.x, uav.y - task.y])
                T_j.extend([task.w_i * task.f_i, task.delta_i + task.t_gen])
            else:
                dtask_j.extend([-1, -1])
                T_j.extend([-1, -1])
        
        dother_j = []
        Iother_j = []
        for other in world.uavs:
            if other is uav:
                continue
            dother_j.extend([uav.x - other.x, uav.y - other.y])
            Iother_j.append(len(uav.taskList) - len(other.taskList))
        
        Gamma_j = []
        
        for task in uav.taskList:
            Gamma_j.extend([task.w_i, task.f_i, task.delta_i, task.t_gen])
        
        max_queue_size = 5
        while len(Gamma_j) < max_queue_size * 4:
            Gamma_j.extend([-1, -1, -1, -1])
        
        Gamma_j = Gamma_j[:max_queue_size * 4]
        
        Gamma_j.extend([len(uav.uav_trans_task_list), len(uav.uav_execute_task_list)])
        
        return np.concatenate([dtask_j, dother_j, T_j, Gamma_j, Iother_j])








if __name__ == '__main__':
    scenario = Scenario()
    world = scenario.make_world()

    world.print_info()

    
    position2 = [1, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    
    executeOrTrans2 = [3, 2, 3, 3, 3, 2, 3, 3, 3, 3]


    position1 = [2, 2, 2, 2, 2, 1, 1, 1, 4, 4]
    executeOrTrans1 = [3, 3, 2, 3, 3, 3, 3, 3, 1, 3]


    position0 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    executeOrTrans0 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

    uav0: UAV = world.uavs[0]
    uav1: UAV = world.uavs[1]
    uav2: UAV = world.uavs[2]

    for i in range(len(position0)):

        uav0.set_actions_sequence(position0, executeOrTrans0)
        uav1.set_actions_sequence(position1, executeOrTrans1)
        uav2.set_actions_sequence(position2, executeOrTrans2)

        world.step()
        world.print_info()

        if i == 1 or i == 0:
            print('obs:', scenario.observation(uav2, world))


        reward2 = scenario.reward(uav2, world)
        reward1 = scenario.reward(uav1, world)
        reward0 = scenario.reward(uav0, world)

        print(f'r0:{reward0: .2f}, r1:{reward1: .2f}, r2:{reward2 :.2f}')














