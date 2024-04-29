#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Multimodal Policy Optimization (MUPO) algorithm
#  Reference: 
#  Update: 2024-04-26, Wenjun Zou: create MUPO algorithm

__all__=["ApproxContainer","MUPO"]
import time
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.act_distribution_type import *
from gops.utils.sn import add_sn

class ApproxContainer(ApprBase):
    """Approximate function container for MUPO.

    Contains one policy and one action value.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs["device"]
        # create q networks
        self.q_args = get_apprfunc_dict("value", kwargs["value_func_type"], **kwargs)
        self.q1: nn.Module = create_apprfunc(**self.q_args).to(self.device)
        self.q2: nn.Module = create_apprfunc(**self.q_args).to(self.device)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)

        # create policy network
        self.sub_policy_num = kwargs["sub_policy_num"]
        self.act_dim = kwargs["action_dim"]
        policy_args = get_apprfunc_dict("policy", kwargs["policy_func_type"], **kwargs)
        policy_args["act_dim"] = policy_args["act_dim"] * self.sub_policy_num
        self.policy: nn.Module = create_apprfunc(**policy_args).to(self.device)
        self.policy_target = deepcopy(self.policy)
        if "grid_num_per_dim" in kwargs:
            self.grid_num_per_dim = kwargs["grid_num_per_dim"]
        else:
            self.grid_num_per_dim = 6

        # create gate policy network
        self.act_high_lim = torch.tensor(kwargs["action_high_limit"]).to(self.device)
        self.act_low_lim = torch.tensor(kwargs["action_low_limit"]).to(self.device)
        gate_policy_args = get_apprfunc_dict("gate_policy", kwargs["gate_policy_func_type"], **kwargs)
        gate_policy_args["act_num"] = self.sub_policy_num
        self.gate_policy: nn.Module = create_apprfunc(**gate_policy_args).to(self.device)
        self.gate_policy_index = None

        if kwargs["spectrum_norm"]:
            self.policy = add_sn(self.policy)
            self.policy_target = add_sn(self.policy_target)

        # set target network gradients
        for p in self.policy_target.parameters():
            p.requires_grad = False
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32, device=self.device))

        # create optimizers
        self.q_lr = kwargs["value_learning_rate"]
        self.policy_lr = kwargs["policy_learning_rate"]
        self.alpha_lr = kwargs["alpha_learning_rate"]
        self.final_lr_decay = kwargs["final_lr_decay"]
        self.max_iteration = kwargs["max_iteration"]

        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["value_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["value_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.gate_policy_optimizer = Adam(
            self.gate_policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])
        self.action_grid = None

    def compute_sub_policy_logits(self, state):
        return self.policy(state)

    def __q_evaluate(self, obs, act, qnet):
        StochaQ = qnet(obs, act)
        mean, log_std = StochaQ[..., 0], StochaQ[..., -1]
        std = log_std.exp()
        normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample()
        z = torch.clamp(z, -3, 3)
        q_value = mean + torch.mul(z, std)
        return mean, std, q_value

    def compute_gate_policy_prob(self, state, alpha=0.04):
        logits = self.gate_policy(state)
        logits = torch.clamp(logits, -10, 10)
        gate_prob = torch.softmax(logits, dim=-1)
        return gate_prob

    def sample_langevin(self, state):
        temperature = 0.01
        self.langevin_sampler.set_temperature(temperature)
        return self.langevin_sampler.sample_batch(state, num_steps=self.langevin_step, use_score=True)

    # When the action dimension is low, use a grid method for sampling and select the
    # action with the highest value as the sampling result. For higher action dimensions,
    # Langevin sampling can be utilized.
    def sampleWithoutDerivative(self, state, discre_num_per_dim=None):
        if discre_num_per_dim is None:
            discre_num_per_dim = self.grid_num_per_dim + 1
        with torch.no_grad():
            if self.action_grid is None:
                act_dim = self.act_dim
                discrete_action_list = []
                for i in range(self.act_dim):
                    action_low = self.act_low_lim[i]
                    action_high = self.act_high_lim[i]
                    discrete_action_list.append(
                        torch.linspace(action_low, action_high, discre_num_per_dim)
                    )
                action_grid = torch.meshgrid(*discrete_action_list, indexing="ij")
                action_grid = torch.stack(action_grid, dim=-1)
                action_grid = action_grid.reshape(-1, act_dim)
                action_grid = action_grid.to(state.device)
                self.action_grid = action_grid
            batch_size = state.shape[0]
            state = state.repeat_interleave(self.action_grid.shape[0], dim=0)
            action_grid = self.action_grid.repeat(batch_size, 1)
            StochaQ = self.q1(state, action_grid)
            mean = StochaQ[..., 0]
            q_grid = mean.unsqueeze(-1)

            q_grid = q_grid.reshape(batch_size, -1)
            action_grid = action_grid.reshape(batch_size, -1, self.act_dim)
            sampled_action = torch.gather(action_grid, 1, torch.argmax(q_grid, dim=1, keepdim=True).unsqueeze(-1).repeat(1, 1, self.act_dim)).squeeze(1)
        
        return sampled_action
    
    def compute_gate_policy_logits(self, state):
        return self.gate_policy(state)

    def compute_action(self, state, deterministic=False, continuous=False):
        dist = self.create_action_distributions(state)
        if continuous:
            action,_,_ = dist.get_sub_mean_std(0)
            return action
        if deterministic:
            action = dist.mode()
            return action
        else:
            action, log_prob = dist.sample()
            return action, log_prob

    def compute_q_values(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        q1, _, _ = self.__q_evaluate(state, action, self.q1)
        q2, _, _ = self.__q_evaluate(state, action, self.q2)
        return torch.min(q1, q2)

    def create_action_distributions(self, state, detach_std=False):
        policy_device = next(self.policy.parameters()).device
        state = state.to(policy_device)
        logits = self.compute_sub_policy_logits(state)
        alpha = torch.exp(self.log_alpha)
        gate_prob = self.compute_gate_policy_prob(state, alpha)
        dist = MixedTanGaussDistribution(logits, gate_prob, detach_std)
        dist.set_action_lim(
            self.policy.act_high_lim, self.policy.act_low_lim
        )
        return dist
    
    def get_sub_mean_std_gate(self, state):
        dist = self.create_action_distributions(state)
        mean, std, gate_prob = dist.get_sub_mean_std_gate()
        return mean, std, gate_prob


class MUPO(AlgorithmBase):
    """MUPO algorithm
    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param bool auto_alpha: whether to adjust temperature automatically.
    :param float alpha: initial temperature.
    :param float delay_update: delay update steps for actor.
    :param Optional[float] target_entropy: target entropy for automatic
        temperature adjustment.
    """

    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = kwargs["gamma"]
        self.tau = kwargs["tau"]
        self.target_entropy = -kwargs["action_dim"]
        self.auto_alpha = kwargs["auto_alpha"]
        self.alpha = kwargs.get("alpha", 0.2)
        self.delay_update = kwargs["delay_update"]
        self.iteration = 0
        self.mean_std1= None
        self.mean_std2= None
        self.tau_b = kwargs.get("tau_b", self.tau)
    @property
    def adjustable_parameters(self):
        return (
            "gamma",
            "tau",
            "auto_alpha",
            "alpha",
            "delay_update",
        )

    def local_update(self, data: DataDict, iteration: int) -> dict:
        self.iteration = iteration
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    def get_remote_update_info(
        self, data: DataDict, iteration: int
    ) -> Tuple[dict, dict]:
        raise NotImplementedError

    def remote_update(self, update_info: dict):
        raise NotImplementedError

    def __get_alpha(self, requires_grad: bool = False):
        if self.auto_alpha:
            alpha = self.networks.log_alpha.exp()
            if requires_grad:
                return alpha
            else:
                return alpha.item()
        else:
            return self.alpha

    def __compute_gradient(self, data: DataDict, iteration: int):
        start_time = time.time()

        obs = data["obs"]
        act_dist = self.networks.create_action_distributions(obs)
        gate_prob = self.networks.compute_gate_policy_prob(obs, self.__get_alpha())
        data.update({"gate_prob": gate_prob, "act_dist": act_dist})

        new_act_list = []
        new_logp_list = []
        entropy = 0
        for i in range(self.networks.sub_policy_num):
            new_act, new_logp = act_dist.rsample(i)
            new_act = new_act.to(gate_prob.device)
            new_logp = new_logp.to(gate_prob.device)
            new_act_list.append(new_act)
            new_logp_list.append(new_logp)
            entropy -= (gate_prob[..., i] * new_logp).mean()

        data.update({
            "new_act_list": new_act_list, 
            "new_logp_list": new_logp_list,
            "entropy": entropy})

        # compute q loss
        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        loss_q, q1, q2, std1, std2, min_std1, min_std2 = self.__compute_loss_q(data)
        loss_q.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = False

        for p in self.networks.q2.parameters():
            p.requires_grad = False

        # compute policy loss
        self.networks.policy_optimizer.zero_grad()
        self.networks.gate_policy_optimizer.zero_grad()
        loss_policy, entropy = self.__compute_loss_policy(data)
        loss_policy.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True

        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            loss_alpha = self.__compute_loss_alpha(data)
            loss_alpha.backward()

        tb_info = {
            "MUPO/critic_avg_q1-RL iter": q1.item(),
            "MUPO/critic_avg_q2-RL iter": q2.item(),
            "MUPO/critic_avg_std1-RL iter": std1.item(),
            "MUPO/critic_avg_std2-RL iter": std2.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_critic"]: loss_q.item(),
            # "MUPO/policy_mean-RL iter": policy_mean,
            # "MUPO/policy_std-RL iter": policy_std,
            "MUPO/entropy-RL iter": entropy.item(),
            "MUPO/alpha-RL iter": self.__get_alpha(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info

    def __q_evaluate(self, obs, act, qnet):
        StochaQ = qnet(obs, act)
        mean, log_std = StochaQ[..., 0], StochaQ[..., -1]
        std = log_std.exp()
        normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample()
        z = torch.clamp(z, -3, 3)
        q_value = mean + torch.mul(z, std)
        return mean, std, q_value

    def __compute_loss_q(self, data: DataDict):
        obs, act, rew, obs2, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        
        act2_dist = self.networks.create_action_distributions(obs2)
        act2, log_prob_act2 = act2_dist.sample()

        q1, q1_std, _ = self.__q_evaluate(obs, act, self.networks.q1)
        q2, q2_std, _ = self.__q_evaluate(obs, act, self.networks.q2)
        if self.mean_std1 is None:
            self.mean_std1 = torch.mean(q1_std.detach())
        else:
            self.mean_std1 = (1 - self.tau_b) * self.mean_std1 + self.tau_b * torch.mean(q1_std.detach())

        if self.mean_std2 is None:
            self.mean_std2 = torch.mean(q2_std.detach())
        else:
            self.mean_std2 = (1 - self.tau_b) * self.mean_std2 + self.tau_b * torch.mean(q2_std.detach())


        q1_next, _, q1_next_sample = self.__q_evaluate(
            obs2, act2, self.networks.q1_target
        )
        
        q2_next, _, q2_next_sample = self.__q_evaluate(
            obs2, act2, self.networks.q2_target
        )
        q_next = torch.min(q1_next, q2_next)
        q_next_sample = torch.where(q1_next < q2_next, q1_next_sample, q2_next_sample)

        target_q1, target_q1_bound = self.__compute_target_q(
            rew,
            done,
            q1.detach(),
            self.mean_std1.detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )
        
        target_q2, target_q2_bound = self.__compute_target_q(
            rew,
            done,
            q2.detach(),
            self.mean_std2.detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )

        q1_std_detach = torch.clamp(q1_std, min=0.).detach()
        q2_std_detach = torch.clamp(q2_std, min=0.).detach()
        bias = 0.1

        q1_loss = (torch.pow(self.mean_std1, 2) + bias) * torch.mean(
            -(target_q1 - q1).detach() / ( torch.pow(q1_std_detach, 2)+ bias)*q1
            -((torch.pow(q1.detach() - target_q1_bound, 2)- q1_std_detach.pow(2) )/ (torch.pow(q1_std_detach, 3) +bias)
            )*q1_std
        )

        q2_loss = (torch.pow(self.mean_std2, 2) + bias)*torch.mean(
            -(target_q2 - q2).detach() / ( torch.pow(q2_std_detach, 2)+ bias)*q2
            -((torch.pow(q2.detach() - target_q2_bound, 2)- q2_std_detach.pow(2) )/ (torch.pow(q2_std_detach, 3) +bias)
            )*q2_std
        )

        return q1_loss +q2_loss, q1.detach().mean(), q2.detach().mean(), q1_std.detach().mean(), q2_std.detach().mean(), q1_std.min().detach(), q2_std.min().detach()

    def __compute_target_q(self, r, done, q,q_std, q_next, q_next_sample, log_prob_a_next):
        target_q = r + (1 - done) * self.gamma * (
            q_next - self.__get_alpha() * log_prob_a_next
        )
        target_q_sample = r + (1 - done) * self.gamma * (
            q_next_sample - self.__get_alpha() * log_prob_a_next
        )
        td_bound = 3 * q_std
        difference = torch.clamp(target_q_sample - q, -td_bound, td_bound)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()
    
    def __compute_loss_policy(self, data: DataDict):
        obs, gate_prob, new_act_list = data["obs"], data["gate_prob"], data["new_act_list"]
        entropy = data["entropy"]
        q_sum = 0
        for i in range(self.networks.sub_policy_num):
            q1, _, _ = self.__q_evaluate(obs, new_act_list[i], self.networks.q1)
            q2, _, _ = self.__q_evaluate(obs, new_act_list[i], self.networks.q2)
            q_mean = torch.min(q1, q2)
            q_sum += gate_prob[..., i] * q_mean

        loss_policy = -self.__get_alpha() * entropy - q_sum.mean()

        # Gate prob regularization to prevent numerical instability 
        gate_logits = self.networks.compute_gate_policy_logits(obs)
        gate_reg_mean = torch.abs(gate_logits.mean())
        gate_reg_L2 = torch.pow(gate_logits, 2).mean()
        loss_policy += 0.5 * gate_reg_mean
        if (gate_reg_L2 > 25.0):
            loss_policy += 1.0 * gate_reg_L2
        
        # Sub policy mean normalization
        sub_policy_logits = self.networks.compute_sub_policy_logits(obs)
        sub_policy_mean, _ = torch.chunk(sub_policy_logits, 2, dim=-1)
        sub_policy_mean = torch.tanh(sub_policy_mean) 
        
        # Instead of minimizing the forward KL divergence by directly maximizing log_prob, this method
        # penalizes by measuring the distance between sub_policy_mean and sampled_action.
        # Approximating sub_policy_mean to sampled_action provides better modal exploration compared to
        # simply maximizing log_prob.
        sampled_action = self.networks.sampleWithoutDerivative(obs)
        # Mapping ref_action to the range [-1, 1]
        sampled_action = 2 * (sampled_action - self.networks.act_low_lim) / (self.networks.act_high_lim - self.networks.act_low_lim) - 1
        actions = sub_policy_mean.reshape(-1, self.networks.act_dim)
        actions_reshaped = actions.reshape(-1, self.networks.sub_policy_num, self.networks.act_dim)
        # In each batch, select the sub-policy closest to ref_action
        # Calculate the distance between each sub-policy action's mean and ref_action
        sampled_action = sampled_action.reshape(-1, 1, self.networks.act_dim)
        dist_pow = torch.pow(actions_reshaped - sampled_action, 2).sum(dim=-1)
        # Select the minimum distance
        min_dist, _ = torch.min(dist_pow, dim=-1)
        dist_threshold = (1/self.networks.grid_num_per_dim) ** 2 * self.networks.act_dim
        # Apply a penalty if the minimum distance exceeds the threshold (grid size)
        min_dist = torch.clamp_min(min_dist - dist_threshold, 0).mean()
        weight_decay = 1 - self.iteration / self.networks.max_iteration
        loss_policy += 1.0 * min_dist * weight_decay

        return loss_policy, entropy.detach()

    def __compute_loss_alpha(self, data: DataDict):
        entropy = data["entropy"]
        loss_alpha = (
            -self.networks.log_alpha * (self.target_entropy - entropy.detach())
        )
        return loss_alpha

    def __update(self, iteration: int):
        # udpate lr
        q_lr = self.networks.q_lr * self.networks.final_lr_decay ** (
            iteration / self.networks.max_iteration
        )
        policy_lr = self.networks.policy_lr * self.networks.final_lr_decay ** (
            iteration / self.networks.max_iteration
        )
        for param_group in self.networks.q1_optimizer.param_groups:
            param_group["lr"] = q_lr
        for param_group in self.networks.q2_optimizer.param_groups:
            param_group["lr"] = q_lr
        for param_group in self.networks.policy_optimizer.param_groups:
            param_group["lr"] = policy_lr
        for param_group in self.networks.gate_policy_optimizer.param_groups:
            param_group["lr"] = policy_lr
        # update parameters
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()
            self.networks.gate_policy_optimizer.step()

            if self.auto_alpha:
                self.networks.alpha_optimizer.step()

            with torch.no_grad():
                polyak = 1 - self.tau
                for p, p_targ in zip(
                    self.networks.q1.parameters(), self.networks.q1_target.parameters()
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(
                    self.networks.q2.parameters(), self.networks.q2_target.parameters()
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(
                    self.networks.policy.parameters(),
                    self.networks.policy_target.parameters(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
