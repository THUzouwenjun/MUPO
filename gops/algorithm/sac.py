#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Soft Actor-Critic (SAC) algorithm
#  Reference: Haarnoja T, Zhou A, Abbeel P et al (2018) 
#             Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor. 
#             ICML, Stockholm, Sweden.
#  Update: 2021-03-05, Yujie Yang: create SAC algorithm

__all__ = ["ApproxContainer", "SAC"]

import time
from copy import deepcopy
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.sn import add_sn


class ApproxContainer(ApprBase):
    """Approximate function container for SAC.

    Contains one policy and two action values.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs["device"]
        # create q networks
        self.q_args = get_apprfunc_dict("value", kwargs["value_func_type"], **kwargs)
        self.q1: nn.Module = create_apprfunc(**self.q_args)
        self.q2: nn.Module = create_apprfunc(**self.q_args)

        # create policy network
        policy_args = get_apprfunc_dict("policy", kwargs["policy_func_type"], **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        if kwargs["spectrum_norm"]:
            self.policy = add_sn(self.policy)

        # create target networks
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)

        # set target networks gradients
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # create optimizers
        self.q_lr = kwargs["value_learning_rate"]
        self.value_lr = kwargs["value_learning_rate"]
        self.policy_lr = kwargs["policy_learning_rate"]
        self.alpha_lr = kwargs["alpha_learning_rate"]
        self.final_lr_decay = kwargs["final_lr_decay"]
        self.max_iteration = kwargs["max_iteration"]
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["q_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["q_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])

    def compute_action_logits(self, state):
        return self.policy(state)

    def compute_action(self, state, deterministic=False):
        dist = self.create_action_distributions(state)
        if deterministic:
            action = dist.mode()
            return action
        else:
            action, log_prob = dist.sample()
            return action, log_prob

    def compute_q_values(self, state, action):
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return torch.min(q1, q2)

    def create_action_distributions(self, state):
        logits = self.compute_action_logits(state)
        return self.policy.get_act_dist(logits)


class SAC(AlgorithmBase):
    """Soft Actor-Critic (SAC) algorithm

    Paper: https://arxiv.org/abs/1801.01290

    :param int index: algorithm index.
    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param bool auto_alpha: whether to adjust temperature automatically.
    :param float alpha: initial temperature.
    :param Optional[float] target_entropy: target entropy for automatic
        temperature adjustment.
    """

    def __init__(
        self,
        index: int = 0,
        gamma: float = 0.99,
        tau: float = 0.005,
        auto_alpha: bool = True,
        alpha: float = 0.2,
        target_entropy: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        if target_entropy is None:
            target_entropy = -kwargs["action_dim"]
        self.target_entropy = target_entropy

    @property
    def adjustable_parameters(self):
        return ("gamma", "tau", "auto_alpha", "alpha", "target_entropy")

    def local_update(self, data: DataDict, iteration: int) -> dict:
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    def get_remote_update_info(
        self, data: DataDict, iteration: int
    ) -> Tuple[dict, dict]:
        tb_info = self.__compute_gradient(data, iteration)

        update_info = {
            "q1_grad": [p.grad for p in self.networks.q1.parameters()],
            "q2_grad": [p.grad for p in self.networks.q2.parameters()],
            "policy_grad": [p.grad for p in self.networks.policy.parameters()],
            "iteration": iteration,
        }

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

        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        loss_q, q1, q2 = self.__compute_loss_q(data)
        loss_q.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False

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
            tb_tags["loss_critic"]: loss_q.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            "SAC/critic_avg_q1-RL iter": q1.item(),
            "SAC/critic_avg_q2-RL iter": q2.item(),
            "SAC/entropy-RL iter": entropy.item(),
            "SAC/alpha-RL iter": self.__get_alpha(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info

    def __compute_loss_q(self, data: DataDict):
        obs, act, rew, obs2, done, constraint = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
            data["constraint"],
        )
        # penalty = torch.where(constraint > -0.1, 5, torch.zeros_like(constraint)).squeeze(-1)
        # rew = rew - penalty
        q1 = self.networks.q1(obs, act)
        q2 = self.networks.q2(obs, act)
        with torch.no_grad():
            next_act_dist = self.networks.create_action_distributions(obs2)
            next_act, next_logp = next_act_dist.sample()
            next_q1 = self.networks.q1_target(obs2, next_act)
            next_q2 = self.networks.q2_target(obs2, next_act)
            next_q = torch.min(next_q1, next_q2)
            backup = rew + (1 - done) * self.gamma * (
                next_q - self.__get_alpha() * next_logp
            )
        
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        return loss_q1 + loss_q2, q1.detach().mean(), q2.detach().mean()

    def __compute_loss_policy(self, data: DataDict):
        obs, new_act = data["obs"], data["new_act"]
        entropy = data["entropy"]
        q1 = self.networks.q1(obs, new_act)
        q2 = self.networks.q2(obs, new_act)
        loss_policy = - self.__get_alpha() * entropy - torch.min(q1, q2).mean()
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
        # set lr
        for param_group in self.networks.q1_optimizer.param_groups:
            param_group["lr"] = q_lr
        for param_group in self.networks.q2_optimizer.param_groups:
            param_group["lr"] = q_lr
        for param_group in self.networks.policy_optimizer.param_groups:
            param_group["lr"] = policy_lr
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()

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
