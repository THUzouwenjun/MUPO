#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Distributed Soft Actor-Critic (DSAC) algorithm
#  Reference: Duan J, Guan Y, Li S E, et al.
#             Distributional soft actor-critic: Off-policy reinforcement learning
#             for addressing value estimation errors[J].
#             IEEE transactions on neural networks and learning systems, 2021.
#  Update: 2023-12-28, Wenxuan Wang: create DSAC2 algorithm

__all__=["ApproxContainer","DSAC2"]
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
from gops.utils.sn import add_sn

class ApproxContainer(ApprBase):
    """Approximate function container for DSAC.

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

        # create value network
        self.v_args = deepcopy(self.q_args)
        self.v_args["name"] = "StateValue"
        self.value = create_apprfunc(**self.v_args).to(self.device)

        # create policy network
        policy_args = get_apprfunc_dict("policy", kwargs["policy_func_type"], **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        self.policy_target = deepcopy(self.policy)

        if kwargs.get("spectrum_norm", False):
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
        self.value_lr = kwargs["value_learning_rate"]
        self.policy_lr = kwargs["policy_learning_rate"]
        self.alpha_lr = kwargs["alpha_learning_rate"]
        self.final_lr_decay = kwargs["final_lr_decay"]
        self.max_iteration = kwargs["max_iteration"]

        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["value_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["value_learning_rate"])
        self.value_optimizer = Adam(self.value.parameters(), lr=kwargs["value_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
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

    def compute_action(self, state, deterministic=False):
        dist = self.create_action_distributions(state)
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

    def create_action_distributions(self, state):
        state = state.to(self.device)
        logits = self.policy(state)
        return self.policy.get_act_dist(logits)

class DSAC2(AlgorithmBase):
    """Modified DSAC algorithm

    Paper: https://arxiv.org/pdf/2001.02811; https://arxiv.org/pdf/2001.02811

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
        self.iteration = iteration
        tb_info = self.__compute_gradient(data, iteration)

        update_info = {
            "q1_grad": [p._grad for p in self.networks.q1.parameters()],
            "q2_grad": [p._grad for p in self.networks.q2.parameters()],
            "policy_grad": [p._grad for p in self.networks.policy.parameters()],
            "iteration": iteration,
        }
        if self.auto_alpha:
            update_info.update({"log_alpha_grad":self.networks.log_alpha.grad})

        return tb_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        q1_grad = update_info["q1_grad"]
        q2_grad = update_info["q2_grad"]
        policy_grad = update_info["policy_grad"]

        for p, grad in zip(self.networks.q1.parameters(), q1_grad):
            p._grad = grad
        for p, grad in zip(self.networks.q2.parameters(), q2_grad):
            p._grad = grad
        for p, grad in zip(self.networks.policy.parameters(), policy_grad):
            p._grad = grad
        if self.auto_alpha:
            self.networks.log_alpha._grad = update_info["log_alpha_grad"]

        self.__update(iteration)

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
        new_act, new_logp = act_dist.rsample()
        entropy = -new_logp.mean()
        data.update({"new_act": new_act, "new_logp": new_logp, "entropy": entropy})
        data.update({"ite": iteration})

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
            "DSAC2/critic_avg_q1-RL iter": q1.item(),
            "DSAC2/critic_avg_q2-RL iter": q2.item(),
            "DSAC2/critic_avg_std1-RL iter": std1.item(),
            "DSAC2/critic_avg_std2-RL iter": std2.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_critic"]: loss_q.item(),
            # "DSAC2/policy_mean-RL iter": policy_mean,
            # "DSAC2/policy_std-RL iter": policy_std,
            "DSAC2/entropy-RL iter": entropy.item(),
            "DSAC2/alpha-RL iter": self.__get_alpha(),
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
        logits_2 = self.networks.policy_target(obs2)
        act2_dist = self.networks.policy_target.get_act_dist(logits_2)
        act2, log_prob_act2 = act2_dist.rsample()

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
        obs, new_act = data["obs"], data["new_act"]
        entropy = data["entropy"]
        q = self.networks.compute_q_values(obs, new_act)
        loss_policy = - self.__get_alpha() * entropy - q.mean()
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
        # update parameters
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()

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
