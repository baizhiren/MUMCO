from gym import spaces

from .distributions import Bernoulli, Categorical, DiagGaussian, DiagBeta
import torch
import torch.nn as nn

from ...envs import settings


class ACTLayer(nn.Module):
    

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.mujoco_box = False
        self.action_type = action_space.__class__.__name__

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        
        elif action_space.__class__.__name__ == "Box":
            # 根据连续移动设置选择分布类型
            if hasattr(settings, 'use_continuous_movement') and settings.use_continuous_movement:
                # 连续移动模式使用Beta分布 [0,1]
                action_dim = action_space.shape[0]
                self.action_out = DiagBeta(inputs_dim, action_dim, use_orthogonal, gain)
            else:
                # 传统Box空间使用高斯分布
                self.mujoco_box = True
                action_dim = action_space.shape[0]
                self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
                
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  
            # Tuple动作空间（混合动作）
            self.mixed_action = True
            self.discrete_action_dims = []
            self.continuous_action_dims = []

            for space in action_space:
                if isinstance(space, spaces.Discrete):
                    self.discrete_action_dims.append(space.n)
                else:
                    self.continuous_action_dims.append(space.shape[0])

            
            self.action_outs = nn.ModuleList()

            # 连续动作使用Beta分布（适合[0,1]范围）
            for cont_dim in self.continuous_action_dims:
                self.action_outs.append(
                    DiagBeta(inputs_dim, cont_dim, use_orthogonal, gain)
                )
            
            # 离散动作使用Categorical分布
            for disc_dim in self.discrete_action_dims:
                self.action_outs.append(
                    Categorical(inputs_dim, disc_dim, use_orthogonal, gain)
                )
    
    def forward(self, x, available_actions=None, deterministic=False):
        
        if self.mixed_action :
            actions = []
            action_log_probs = []
            for id, action_out in enumerate(self.action_outs):
                if isinstance(action_out, DiagBeta):
                    action_logit = action_out(x)
                    action = action_logit.mode() if deterministic else action_logit.rsample()
                    
                else:
                    if id == 3:
                        action_logit = action_out(x, available_actions=available_actions)
                    else:
                        action_logit = action_out(x)
                    action = action_logit.mode() if deterministic else action_logit.sample()

                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
        
        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            idx = 0
            for action_out in self.action_outs:
                if idx == 1:
                    action_logit = action_out(x, available_actions)
                else:
                    action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
                idx += 1
            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)

        elif self.mujoco_box:
            action_logits = self.action_out(x)
            actions = action_logits.mode() if deterministic else action_logits.sample() 
            action_log_probs = action_logits.log_probs(actions)
            
        elif self.action_type == "Box" and isinstance(self.action_out, DiagBeta):
            # 连续移动Box空间（Beta分布）
            action_logits = self.action_out(x)
            actions = action_logits.mode() if deterministic else action_logits.rsample()
            action_log_probs = action_logits.log_probs(actions)
        
        else:
            action_logits = self.action_out(x, available_actions)
            actions = action_logits.mode() if deterministic else action_logits.sample() 
            action_log_probs = action_logits.log_probs(actions)
        
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        
        if self.mixed_action or self.multi_discrete:
            action_probs = []
            for id, action_out in enumerate(self.action_outs):
                if id == 3 or settings.fix_trans_and_compute_when_train and id == 1:
                    action_logit = action_out(x, available_actions=available_actions)
                else:
                    action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        
        if self.mixed_action:
            
            total_cont_dim = sum(self.continuous_action_dims)  
            continuous_action = action[..., :total_cont_dim]
            discrete_action = action[..., total_cont_dim:]  

            
            
            cont_splits = torch.split(continuous_action, self.continuous_action_dims, dim=-1)
            

            
            disc_splits = torch.split(discrete_action, 1, dim=-1)
            

            
            disc_splits = [act.long().squeeze(-1) for act in disc_splits]

            
            action_log_probs_list = []
            dist_entropy_list = []

            
            
            
            idx = 0

            
            for i, cont_dim in enumerate(self.continuous_action_dims):
                dist = self.action_outs[idx](x)
                act = cont_splits[i]  
                idx += 1

                
                log_p = dist.log_probs(act)
                ent = dist.entropy()

                
                if active_masks is not None:
                    
                    if ent.shape == active_masks.shape:
                        ent_mean = (ent * active_masks).sum() / active_masks.sum()
                    else:
                        ent_mean = (ent * active_masks.squeeze(-1)).sum() / active_masks.sum()
                else:
                    ent_mean = ent.mean()

                action_log_probs_list.append(log_p)
                dist_entropy_list.append(ent_mean)

            
            for j, disc_dim in enumerate(self.discrete_action_dims):
                if idx == 3:
                    dist = self.action_outs[idx](x, available_actions)
                else:
                    dist = self.action_outs[idx](x)  
                act = disc_splits[j]  
                idx += 1

                log_p = dist.log_probs(act)
                ent = dist.entropy()

                if active_masks is not None:
                    if ent.shape == active_masks.shape:
                        ent_mean = (ent * active_masks).sum() / active_masks.sum()
                    else:
                        ent_mean = (ent * active_masks.squeeze(-1)).sum() / active_masks.sum()
                else:
                    ent_mean = ent.mean()

                action_log_probs_list.append(log_p)
                dist_entropy_list.append(ent_mean)

            
            
            cat_log_probs = torch.cat(
                [lp.unsqueeze(-1) if lp.dim() == 1 else lp for lp in action_log_probs_list],
                dim=-1
            )  
            action_log_probs = torch.sum(cat_log_probs, dim=-1, keepdim=True)  

            
            
            

            dist_entropy = dist_entropy_list[0] / 2 + dist_entropy_list[1] / 2 + dist_entropy_list[2] + dist_entropy_list[3]
            
            return action_log_probs, dist_entropy

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            idx = 0
            for action_out, act in zip(self.action_outs, action):
                if idx == 1:
                    action_logit = action_out(x, available_actions)
                else:
                    action_logit = action_out(x)

                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
                idx += 1

            action_log_probs = torch.cat(action_log_probs, -1) 
            dist_entropy = sum(dist_entropy)/len(dist_entropy)
        
        elif self.mujoco_box:
            action_logits = self.action_out(x)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
                
        elif self.action_type == "Box" and isinstance(self.action_out, DiagBeta):
            # 连续移动Box空间（Beta分布）
            action_logits = self.action_out(x)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        else:
            action_logits = self.action_out(x, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy

    def evaluate_actions_trpo(self, x, action, available_actions=None, active_masks=None):
        

        if self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            mu_collector = []
            std_collector = []
            probs_collector = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                mu = action_logit.mean
                std = action_logit.stddev
                action_log_probs.append(action_logit.log_probs(act))
                mu_collector.append(mu)
                std_collector.append(std)
                probs_collector.append(action_logit.logits)
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
            action_mu = torch.cat(mu_collector,-1)
            action_std = torch.cat(std_collector,-1)
            all_probs = torch.cat(probs_collector,-1)
            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = torch.tensor(dist_entropy).mean()
        
        else:
            action_logits = self.action_out(x, available_actions)
            action_mu = action_logits.mean
            action_std = action_logits.stddev
            action_log_probs = action_logits.log_probs(action)
            if self.action_type=="Discrete":
                all_probs = action_logits.logits
            else:
                all_probs = None
            if active_masks is not None:
                if self.action_type=="Discrete":
                    dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
                else:
                    dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy, action_mu, action_std, all_probs