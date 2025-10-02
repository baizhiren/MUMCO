import time

import gym
from gym import spaces
import numpy as np

from envs import settings
from envs.BigTask import BigTask
from envs.Task import Task
from envs.UAV import UAV
from envs.multi_discrete import MultiDiscrete

from envs.worldUtils import get_available_action, get_object

cam_range = 2


class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True, discrete_action=True):

        self.world = world
        self.world_length = self.world.world_length
        self.current_step = 0
        self.agents = self.world.policy_agents
        
        self.n = len(world.entities)
        
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.post_step_callback = post_step_callback

        
        
        self.discrete_action_space = discrete_action

        
        self.shared_reward = world.collaborative if hasattr(
            world, 'collaborative') else False
        
        
        self.time = 0
        
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []

            
            if not settings.fix_trans_and_compute_when_train:
                total_action_space.append(spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32))
                total_action_space.append(spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32))
            
            if agent.movable:
                if hasattr(settings, 'use_continuous_movement') and settings.use_continuous_movement:
                    # 连续移动: 使用Beta分布 [0,1] - 第一维是速度，第二维是方向
                    u_action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
                elif self.discrete_action_space:
                    u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
                else:
                    u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,),
                                                dtype=np.float32)  
                total_action_space.append(u_action_space)

            
            if not agent.silent:
                if self.discrete_action_space:
                    c_action_space = spaces.Discrete(world.dim_c)
                else:
                    c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)  
                total_action_space.append(c_action_space)
            else:
                c_action_space = spaces.Discrete(world.silent_dim_c)
                total_action_space.append(c_action_space)

            
            if len(total_action_space) > 1:
                
                
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete(
                        [[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            
            
            obs_dim = len(observation_callback(agent, self.world))
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))  
            agent.action.c = np.zeros(self.world.dim_c)

        
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]

        
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
        self.world.seed = seed
        

    def step_by_algorithm(self, algorithm='GH'):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []

        self.agents = self.world.policy_agents
        global action_n
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        self.world.step()
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append([self._get_reward(agent)])
            done_n.append(self._get_done(agent))
            info = {'individual_reward': self._get_reward(agent)}
            env_info = self._get_info(agent)
            if 'fail' in env_info.keys():
                info['fail'] = env_info['fail']
            info_n.append(info)

        return obs_n, reward_n, done_n, info_n, None, None

    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        object_n = []
        self.agents = self.world.policy_agents
        
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        
        
        self.world.step()  

        available_n = []
        
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append([self._get_reward(agent)])
            done_n.append(self._get_done(agent))
            info = {'individual_reward': self._get_reward(agent)}
            env_info = self._get_info(agent)
            if 'fail' in env_info.keys():
                info['fail'] = env_info['fail']
            info_n.append(info)
            available_n.append(self._get_available(agent))
            obj = self._get_object(agent)[2]
            obj = np.expand_dims(obj, axis=-1)  
            object_n.append(obj)
        
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n

        if self.post_step_callback is not None:
            self.post_step_callback(self.world)

        return obs_n, reward_n, done_n, info_n, available_n, object_n

    def reset(self):
        self.current_step = 0
        
        self.reset_callback(self.world)
        
        self._reset_render()
        
        obs_n = []
        self.agents = self.world.policy_agents

        
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        return obs_n

    def cal_statistic(self):
        self.world.cal_statistic_info()

    def print_info(self):
        self.world.print_log()

    def print_step(self):
        self.world.print_info()

    
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    
    
    def _get_done(self, agent):
        if self.done_callback is None:
            if self.current_step >= self.world_length:
                return True
            else:
                return False
        return self.done_callback(agent, self.world)

    
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    def _get_object(self, agent):
        return get_object(self.world)

    def _get_available(self, agent):
        return get_available_action(self.world, agent)

    
    def _set_action(self, action, agent, action_space, time=None):
        if not settings.fix_trans_and_compute_when_train:
            agent.action.trans_power_ratio = action[0]
            action = action[1:]

            agent.action.computing_ratio = action[0]
            action = action[1:]

        if agent.movable:
            # 检查是否为连续移动模式
            if hasattr(settings, 'use_continuous_movement') and settings.use_continuous_movement:
                # 连续移动：取两个值 [速度, 方向]
                agent.action.u = action[0:2]
                action = action[2:]
            else:
                # 离散移动：取一个值
                agent.action.u = action[0]
                action = action[1:]

        if not agent.silent:
            
            agent.action.c = action[0]
            action = action[1:]

        
        assert len(action) == 0

    
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    
    def render(self, mode='human', close=False, gap=None):
        from envs.rendering import TextGeom
        if close:
            
            for i, viewer in enumerate(self.viewers):
                if viewer is not None:
                    viewer.close()
                self.viewers[i] = None
            return []

        for i in range(len(self.viewers)):
            

            if self.viewers[i] is None:
                
                
                from . import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        
        if True:
            
            
            from . import rendering

            self.render_geoms = []
            self.render_geoms_xform = []
            self.comm_geoms = []
            factor = self.world.act_map.length / 10
            for entity in self.world.entities:

                entity_size = entity.size * factor * 0.9

                geom = rendering.make_circle(entity_size)
                xform = rendering.Transform()

                entity_comm_geoms = []

                if isinstance(entity, UAV):
                    geom.set_color(*entity.color, alpha=0.5)
                    cur_and_future_task = entity.cur_and_future_task
                    task_num = len(cur_and_future_task)
                    for ci in range(task_num):
                        task = cur_and_future_task[ci]
                        comm = rendering.make_circle(entity_size / (task_num + 1))
                        comm.set_color(*task.color)
                        comm.add_attr(xform)

                        offset = rendering.Transform()
                        comm_size = (entity_size / task_num)
                        offset.set_translation(ci * comm_size * 2 -
                                               entity_size + comm_size, 0)
                        comm.add_attr(offset)
                        entity_comm_geoms.append(comm)
                        task_name = task.name.split("_")

                        text_geom = TextGeom(f"{task_name[1]}", font_size=20 * factor, x=0, y=0)  
                        text_geom.set_color(0, 0, 0, 1)
                        scale_transform = rendering.Transform(
                            scale=(0.01 / max(task_num // 2, 1), 0.01 / max(task_num // 2, 1)))
                        text_geom.add_attr(scale_transform)
                        text_geom.add_attr(xform)  
                        text_geom.add_attr(offset)  

                        entity_comm_geoms.append(text_geom)


                elif (isinstance(entity, Task) or isinstance(entity, BigTask)) and not entity.is_take:
                    geom.set_color(*entity.color)
                    task_name = entity.name.split("_")
                    text_geom = TextGeom(f"{task_name[1]}", font_size=20 * factor, x=0, y=0)
                    text_geom.set_color(0, 0, 0, 1)
                    scale_transform = rendering.Transform(scale=(0.01, 0.01))
                    text_geom.add_attr(scale_transform)
                    text_geom.add_attr(xform)  
                    entity_comm_geoms.append(text_geom)

                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
                self.comm_geoms.append(entity_comm_geoms)

            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for entity_comm_geoms in self.comm_geoms:
                    for geom in entity_comm_geoms:
                        viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            self.viewers[i].set_bounds(
                -1, self.world.act_map.length, -1, self.world.act_map.width)

            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.p_pos)
                self.render_geoms[e].set_color(*entity.color)

            results.append(self.viewers[i].render(
                return_rgb_array=mode == 'rgb_array'))
        if gap == None:
            gap = settings.render_gap
        time.sleep(gap)

        return results

    
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(
                        distance * np.array([np.cos(angle), np.sin(angle)]))
            
            dx.append(np.array([0.0, 0.0]))
        
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


if __name__ == '__main__':
    env = MultiAgentEnv(None)