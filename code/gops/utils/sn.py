import torch.nn as nn

def add_sn(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(module)
    else:
        for name, child_module in module.named_children():
            module.add_module(name, add_sn(child_module))
        return module

def remove_sn(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return nn.utils.remove_spectral_norm(module)
    else:
        for name, child_module in module.named_children():
            module.add_module(name, remove_sn(child_module))
        return module