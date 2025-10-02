import time
import numpy as np
import torch

from envs import settings
from runner.shared.base_runner import Runner
import wandb
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class UAVRunner(Runner):
    
    def __init__(self, config):
        super(UAVRunner, self).__init__(config)

    def run(self):
        self.warmup()
        start = time.time()
        
        
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            
            if self.use_linear_lr_decay:
                
                self.trainer.policy.lr_decay(episode, episodes)
            
            for step in range(self.episode_length):
                
                
                
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                
                obs, rewards, dones, infos, available_actions, objects = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions, objects

                
                self.insert(data)

            
            self.compute()
            train_infos = self.train()
            
            
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "UAV":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards)
                train_infos["average_objects"] = np.mean(self.buffer.objects)
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                print("average episode objects is {}".format(train_infos["average_objects"]))

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        
        
        obs = self.envs.reset()

        
        
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        
        self.trainer.prep_rollout()


        
        
        
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            available_actions=np.concatenate(self.buffer.available_actions[step]))
        

        
        
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))

        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))

        
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            actions_env = actions
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        elif self.envs.action_space[0].__class__.__name__ == 'Tuple':
            # Tuple动作空间直接传递，让环境处理
            actions_env = actions
        else:
            actions_env = actions

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions, objects = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, available_actions=available_actions, objects=objects)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                eval_actions_env = eval_actions
            elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[eval_actions], 2)
            elif self.envs.action_space[0].__class__.__name__ == 'Tuple':
                # Tuple动作空间直接传递，让环境处理
                eval_actions_env = eval_actions
            else:
                eval_actions_env = eval_actions

            
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render_other_algorithm(self):
        
        envs = self.envs
        total_reward_info = [0, 0]
        cnt = 1

        if settings.print_last_only_enable:
            settings.print_last_only = False


        for episode in range(self.all_args.render_episodes):
            envs.reset()
            if not settings.jump_render:
                if cnt > settings.skip_before_render_num:
                    envs.render('human')

            episode_rewards = []

            if episode == self.all_args.render_episodes - 1:
                settings.print_last_only = True

            for step in range(self.episode_length):
                obs, rewards, dones, infos, _ = envs.step(None)
                episode_rewards.append(rewards)

                if settings.print_step:
                    envs.print_step()
                if not settings.jump_render:
                    if cnt > settings.skip_before_render_num:
                        envs.render('human')

            reward = np.mean(np.sum(np.array(episode_rewards), axis=0))
            total_reward_info[0] += 1
            total_reward_info[1] += reward
            
            cnt += 1

        print(f'total avg reward:{total_reward_info[1] / total_reward_info[0]}')

    
    @torch.no_grad()
    def render(self):
        

        envs = self.envs
        all_frames = []
        total_reward_info = [0, 0]
        cnt = 0

        if settings.print_last_only_enable:
            settings.print_last_only = False

        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if not settings.jump_render:
                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                else:
                    if cnt > settings.skip_before_render_num:
                        envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            avis = np.ones((settings.uav_num, settings.uav_num + 1),
                                             dtype=np.float32)

            if episode == self.all_args.render_episodes - 1:
                settings.print_last_only = True
            
            for step in range(self.episode_length):
                calc_start = time.time()
                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True,
                                                    available_actions=avis)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                
                # 处理不同动作空间类型
                if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    actions_env = actions
                elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
                elif self.envs.action_space[0].__class__.__name__ == 'Tuple':
                    # Tuple动作空间直接传递，让环境处理
                    actions_env = actions
                else:
                    actions_env = actions

                
                obs, rewards, dones, infos, avis = envs.step(actions_env)
                avis = avis.squeeze(0)

                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if settings.print_step:
                    envs.print_step()

                if not settings.jump_render:
                    if self.all_args.save_gifs:
                        image = envs.render('rgb_array')[0][0]
                        all_frames.append(image)
                        calc_end = time.time()
                        elapsed = calc_end - calc_start
                        if elapsed < self.all_args.ifi:
                            time.sleep(self.all_args.ifi - elapsed)
                    else:
                        if cnt > settings.skip_before_render_num:
                            envs.render('human')

            reward = np.mean(np.sum(np.array(episode_rewards), axis=0))
            total_reward_info[0] += 1
            total_reward_info[1] += reward
            
            cnt += 1


        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
        print(f'total avg reward:{total_reward_info[1] / total_reward_info[0]}')