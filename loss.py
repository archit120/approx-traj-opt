import torch

def entropy_temperature_loss(log_prob, alpha, entropy_target):
    return torch.mean(-alpha*log_prob - alpha*entropy_target)

def policy_loss(log_prob, alpha, q_val):
    return torch.mean(alpha*log_prob - q_val)

def qfunc_loss(q_val, r, gamma, v_target):
    return torch.mean(0.5*torch.pow(q_val - (r + gamma*v_target), 2))