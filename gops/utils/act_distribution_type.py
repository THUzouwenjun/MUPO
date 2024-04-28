#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Action Distributions
#  Update Date: 2021-03-10, Yujie Yang: Revise Codes


import torch
import numpy as np

EPS = 1e-6

# 计算每个batch上的action的最小距离
def cal_action_distance(actions, action_dim):
    # reshape为(batch_size, N, action_dim)
    batch_size = actions.shape[0]
    actions_reshape = actions.reshape(batch_size, -1, action_dim)
    # 分别计算每个batch上的action的最小距离
    action_num = actions_reshape.shape[1]
    distance = torch.zeros([batch_size, int(action_num * (action_num - 1) / 2)])
    for i in range(batch_size):
        # 计算最小距离
        action_dist = torch.pdist(actions_reshape[i])
        # print("action_dist: ", action_dist)
        # 注意batch维度
        distance[i] = action_dist
    
    return distance

class TanhGaussDistribution:
    def __init__(self, logits):
        self.logits = logits
        self.device = logits.device
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        self.act_high_lim = torch.tensor([1.0], device = self.device)
        self.act_low_lim = torch.tensor([-1.0], device = self.device)

    def set_action_lim(self, act_high_lim, act_low_lim):
        self.act_high_lim = act_high_lim
        self.act_low_lim = act_low_lim

    def sample(self):
        action = self.gauss_distribution.sample()
        action_limited = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            action
        ) + (self.act_high_lim + self.act_low_lim) / 2
        log_prob = (
            self.gauss_distribution.log_prob(action)
            - torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)).sum(-1)
            - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)
        )
        return action_limited, log_prob

    def rsample(self):
        action = self.gauss_distribution.rsample()
        action_limited = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            action
        ) + (self.act_high_lim + self.act_low_lim) / 2
        log_prob = (
            self.gauss_distribution.log_prob(action)
            - torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)).sum(-1)
            - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)
        )
        return action_limited, log_prob

    def log_prob(self, action_limited) -> torch.Tensor:
        action = torch.atanh(
            (1 - EPS)
            * (2 * action_limited - (self.act_high_lim + self.act_low_lim))
            / (self.act_high_lim - self.act_low_lim)
        )
        log_prob = self.gauss_distribution.log_prob(action) - torch.log(
            (self.act_high_lim - self.act_low_lim) / 2
            * (1 + EPS - torch.pow(torch.tanh(action), 2))
        ).sum(-1)
        return log_prob

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.mean) + (
            self.act_high_lim + self.act_low_lim
        ) / 2

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )


class MixedTanGaussDistribution:
    def __init__(self, logits, gate_prob, detach_std=False):
        self.logits = logits # [batch_size, N * 2 * action_dim]
        self.batch_size = logits.shape[0]
        self.device = logits.device
        self.gate_prob = gate_prob  # [batch_size, N]
        self.sub_dis_num = gate_prob.shape[-1]
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        if detach_std:
            self.std = self.std.detach()
        self.action_dim = self.mean.shape[-1] / self.sub_dis_num 
        self.act_high_lim = None
        self.act_low_lim = None
        
        gate_prob_sum = torch.sum(gate_prob, dim=-1)
        assert torch.allclose(gate_prob_sum, torch.ones_like(gate_prob_sum)), print("gate_prob_sum", gate_prob_sum)
        self.gate_distribution = torch.distributions.Categorical(probs=self.gate_prob)

    # input: gate, shape: [batch_size], output: logits, shape: [batch_size, 2 * action_dim]
    def get_sub_dist_logits(self, gate):
        gate = gate.to(self.device)
        gate = gate.unsqueeze(-1).long()
        index = torch.arange(self.action_dim, device=self.device).unsqueeze(0).repeat(gate.shape[0], 1)
        index = index + gate * self.action_dim
        index = index.long()
        mean = self.mean.gather(-1, index)
        std = self.std.gather(-1, index)
        logits = torch.cat([mean, std], dim=-1)
        return logits

    def set_action_lim(self, act_high_lim, act_low_lim):
        self.act_high_lim = act_high_lim.to(torch.float32).to(self.device)
        self.act_low_lim = act_low_lim.to(torch.float32).to(self.device)
    
    def log_prob(self, action_limited):
        prob = torch.zeros([self.batch_size, self.sub_dis_num], device=self.device)
        for i in range(self.sub_dis_num):
            gate = torch.ones_like(action_limited[..., 0], device = self.device) * i
            sub_dist_logits = self.get_sub_dist_logits(gate)
            sub_dist = TanhGaussDistribution(sub_dist_logits)
            sub_dist.set_action_lim(self.act_high_lim, self.act_low_lim)
            prob[..., i] =  sub_dist.log_prob(action_limited) + torch.log(self.gate_prob[..., i])
        log_prob = torch.logsumexp(prob, dim=-1)
        return log_prob
    
    def sample(self, index=None):
        if index is None:
            gate = self.gate_distribution.sample()
        else:
            gate = torch.ones([self.batch_size]) * index
        logits = self.get_sub_dist_logits(gate)
        sub_dist = TanhGaussDistribution(logits)
        sub_dist.set_action_lim(self.act_high_lim, self.act_low_lim)
        action,_ = sub_dist.sample()
        log_prob = self.log_prob(action)
        log_prob = log_prob.to(action.device)
        return action, log_prob
    
    # gate: [batch_size]
    def rsample(self, index):
        gate = torch.ones([self.batch_size]) * index
        sub_dist = TanhGaussDistribution(self.get_sub_dist_logits(gate))
        sub_dist.set_action_lim(self.act_high_lim, self.act_low_lim)
        action,_ = sub_dist.rsample()
        log_prob = self.log_prob(action)
        return action, log_prob
    
    # This function is used to obtain the mode of the distribution
    # it megres the two modes of the two sub-distributions when the two modes are close enough
    # it does not affect the training process
    def mode(self):
        # select the gate with the highest probability
        gate_1 = torch.argmax(self.gate_prob, dim=-1)
        gate_1_prob,_ = torch.max(self.gate_prob, dim=-1)
        if self.sub_dis_num == 1:
            gate_2 = gate_1
        else:
            gate_2 = torch.argsort(self.gate_prob, dim=-1)[:, -2]

        sub_dist_1 = TanhGaussDistribution(self.get_sub_dist_logits(gate_1))
        sub_dist_1.set_action_lim(self.act_high_lim, self.act_low_lim)
        action_1 = sub_dist_1.mode()
        sub_dist_2 = TanhGaussDistribution(self.get_sub_dist_logits(gate_2))
        sub_dist_2.set_action_lim(self.act_high_lim, self.act_low_lim)
        action_2 = sub_dist_2.mode()

        # normalize the action difference
        delta_act = action_1 - action_2
        delta_act = delta_act / (self.act_high_lim - self.act_low_lim)
        delta_act_norm = torch.norm(delta_act, dim=-1)

        self.act_sep_bound_1 = 0.2
        self.act_sep_bound_2 = 0.3

        gamma = (delta_act_norm - self.act_sep_bound_1) / (self.act_sep_bound_2 - self.act_sep_bound_1)
        gamma = torch.clamp(gamma, min=0.0, max=1.0)
        # get the merged action
        gate_1_prob = (gate_1_prob + (1 - gate_1_prob) * gamma).unsqueeze(-1)
        action = gate_1_prob * action_1 + (1 - gate_1_prob) * action_2
        return action
    
    def get_sub_mean_std(self, gate: int):
        gate = torch.tensor(gate)
        logits = self.get_sub_dist_logits(gate)
        mean, std = torch.chunk(logits, chunks=2, dim=-1)
        mean = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(mean) + (
            self.act_high_lim + self.act_low_lim
        ) / 2
        gate_prob = self.gate_prob[..., gate]
        return mean, std, gate_prob

class GaussDistribution:
    def __init__(self, logits):
        self.logits = logits
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        self.act_high_lim = torch.tensor([1.0])
        self.act_low_lim = torch.tensor([-1.0])

    def set_action_lim(self, act_high_lim, act_low_lim):
        self.act_high_lim = act_high_lim
        self.act_low_lim = act_low_lim
        
    def sample(self):
        action = self.gauss_distribution.sample()
        log_prob = self.gauss_distribution.log_prob(action)
        return action, log_prob

    def rsample(self):
        action = self.gauss_distribution.rsample()
        log_prob = self.gauss_distribution.log_prob(action)
        return action, log_prob

    def log_prob(self, action) -> torch.Tensor:
        log_prob = self.gauss_distribution.log_prob(action)
        return log_prob

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return torch.clamp(self.mean, self.act_low_lim, self.act_high_lim)

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )


class CategoricalDistribution:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.cat = torch.distributions.Categorical(logits=logits)

    def sample(self):
        action = self.cat.sample()
        log_prob = self.log_prob(action)
        return action, log_prob

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """
        action: [batch_size, 1]
        log_prob: [batch_size]"""
        if action.dim() > 1:
            action = action.squeeze(1)
        return self.cat.log_prob(action)

    def entropy(self):
        return self.cat.entropy()

    def mode(self):
        return torch.argmax(self.logits, dim=-1)

    def kl_divergence(self, other: "CategoricalDistribution"):
        return torch.distributions.kl.kl_divergence(self.cat, other.cat)


class Categorical4MultiDiscrete(CategoricalDistribution):
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """
        action: [batch_size, num_action_dim]
        log_prob: [batch_size]"""
        assert action.size(-1) > 1, "action dim should be larger than 1 with MultiDiscrete-Distribution"
        log_prob = self.cat.log_prob(action)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(-1, keepdim=False)
        return log_prob

class DiracDistribution:
    def __init__(self, logits):
        self.logits = logits

    def sample(self):
        return self.logits, torch.tensor([0.0])

    def mode(self):
        return self.logits


class ValueDiracDistribution:
    def __init__(self, logits):
        self.logits = logits

    def sample(self):
        return torch.argmax(self.logits, dim=-1), torch.tensor([0.0])

    def mode(self):
        return torch.argmax(self.logits, dim=-1)
