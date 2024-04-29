import torch
import inspect
import numpy as np
import matplotlib.pyplot as plt

class LangevinSampler:
    def __init__(self, energy_fn=None, score_fn=None, neg_eng=False, device='cpu'):
        self.energy_fn = None
        self.device = device
        self.temperature = 1.0
        if energy_fn is not None:
            self.set_energy_function(energy_fn, neg_eng)
        if score_fn is not None:
            self.set_score_function(score_fn)
        self.act_lim_high = None
        self.act_lim_low = None
    
    def set_temperature(self, temperature):
        assert temperature > 0, "Temperature must be greater than 0"
        self.temperature = temperature

    def set_action_limit(self, act_lim_high, act_lim_low):
        if isinstance(act_lim_high, np.ndarray):
            self.act_lim_high = torch.from_numpy(act_lim_high).to(dtype=torch.float32, device=self.device)
        else:
            self.act_lim_high = act_lim_high.clone().detach().to(dtype=torch.float32, device=self.device)

        if isinstance(act_lim_low, np.ndarray):
            self.act_lim_low = torch.from_numpy(act_lim_low).to(dtype=torch.float32, device=self.device)
        else:
            self.act_lim_low = act_lim_low.clone().detach().to(dtype=torch.float32, device=self.device)

    def set_score_function(self, score_fn):
        if not callable(score_fn):
            raise ValueError("score_fn must be callable")
        
        sig = inspect.signature(score_fn)
        num_parameters = len(sig.parameters)
        if num_parameters != 2:
            Warning(f"score_fn must take two argument, but it takes {num_parameters}")
        
        self.score_fn = score_fn

    def set_energy_function(self, energy_fn: torch.nn.Module, neg_eng=False):
        if not callable(energy_fn):
            raise ValueError("energy_fn must be callable")
        
        sig = inspect.signature(energy_fn)
        num_parameters = len(sig.parameters)
        if num_parameters != 2:
            Warning(f"energy_fn must take two argument, but it takes {num_parameters}")
        
        self.energy_fn = energy_fn

        def wrapped_energy_fn(state, action):
            return -energy_fn(state, action)
    
        self.energy_fn = energy_fn
        if neg_eng:
            self.energy_fn = wrapped_energy_fn

    def set_device(self, device):
        self.device = device

    def uniform_initial_action(self, batch_size):
        # sample from uniform distribution
        return torch.rand(batch_size, self.act_lim_high.shape[0], device=self.device) * (self.act_lim_high - self.act_lim_low) + self.act_lim_low
    
    def cal_grad(self, state, action, use_score=False):
        if not use_score:
            action.requires_grad = True
            energy = self.energy_fn(state, action)
            energy.sum().backward()
            with torch.no_grad():
                grad = action.grad
        else:
            with torch.no_grad():
                grad = -self.score_fn(state, action)
        
        return grad

    def sample_batch(self, state:torch.Tensor, num_steps=20, use_score=False):
        batch_size = state.shape[0]
        state = state.to(self.device)
        action = self.uniform_initial_action(batch_size).to(self.device)

        # Initialize Adam-like parameters
        v = torch.zeros_like(action)
        beta2 = 0.95
        epsilon = 0.05

        for ite in range(num_steps):
            # Compute the gradient of the energy function
            grad = self.cal_grad(state, action, use_score=use_score)

            with torch.no_grad():
                # Update biased second raw moment estimate
                v = beta2 * v + (1 - beta2) * (grad ** 2)

                # Compute bias-corrected second raw moment estimate
                v_hat = v / (1 - beta2 ** (ite + 1))

                step_size = 0.04 / (torch.sqrt(v_hat) + epsilon)

                # Proposed new action
                proposed_action = action - step_size * grad + torch.randn_like(action) * torch.sqrt(step_size * 2 * self.temperature)
                
                # Check if the proposed action is within the limits
                within_limits = torch.logical_and(proposed_action <= self.act_lim_high, proposed_action >= self.act_lim_low)
                
                # Update action only if within limits
                action = torch.where(within_limits, proposed_action, action)
        
        return action.detach()
    


