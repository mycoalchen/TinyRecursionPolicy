from typing import Tuple, Dict
import torch
import torch.distributions as dist
from einops import rearrange

def get_distribution_params(reasoning_module, hidden_states):
    """ Returns mu, sigma """
    mu = reasoning_module.mu_head(hidden_states)
    log_sigma = reasoning_module.log_sigma_head(hidden_states)
    assert not torch.isnan(mu).any()
    assert not torch.isnan(log_sigma).any()
    log_sigma = torch.clamp(log_sigma, min=-3, max=2)
    sigma = torch.exp(log_sigma)
    assert not torch.isnan(sigma).any()
    return mu, sigma

# JIT CAUSES NAN GRADIENTS FOR REASONS ONLY KNOWN TO GOD
# @torch.jit.script # fuse element-wise operations automatically
def manual_gaussian_log_prob(x, mu, sigma):
    # Constant for 0.5 * log(2pi)
    log_2pi = 1.837877

    # Add epsilon to sigma to prevent DivBackward0 (Gradient explosion on sigma -> 0)
    # Even if sigma is clamped > 0, the gradient 1/sigma can be huge.
    safe_sigma = sigma + 1e-5
    
    var = safe_sigma.pow(2)
    log_scale = torch.log(safe_sigma)
    
    # (x - mu)^2 / 2sigma^2
    # Detach x so gradients don't try to move the sample
    diff = x.detach() - mu    
    exp_term = -(diff.pow(2)) / (2 * var)
    
    log_prob = exp_term - log_scale - 0.5 * log_2pi
    return torch.clamp(log_prob, min=-1000.0) # prevent Inf from propagating to gradients when using jit

def run_forward_step(
    inner_model, 
    z_L, 
    z_H, 
    input_embeddings, 
    seq_info, 
    act_step_num
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dist.Categorical, Dict[str, torch.Tensor]]:
    """
    Runs one forward step. 
    Returns: new_z_L, new_z_H, step_log_prob, halt_dist
    """
    cfg = inner_model.config
    step_log_prob = 0.0
    
    curr_z_L = z_L

    z_L_entropies = []
    z_L_sigmas = []
    z_H_entropies = []
    z_H_sigmas = []
    
    # 1. Latent Reasoning (L_cycles)
    for _L_step in range(cfg.L_cycles):
        module = inner_model.L_level
        h_input = curr_z_L
        input_injection = z_H + input_embeddings
        inner_step_tensor = torch.ones_like(act_step_num) * _L_step
        
        hidden = h_input + input_injection
        if cfg.time_embeddings:
            hidden = hidden + module.inner_step_emb(inner_step_tensor)[:, None, :]
            hidden = hidden + module.act_step_emb(act_step_num)[:, None, :]
            
        for layer in module.layers:
            hidden = layer(hidden_states=hidden, **seq_info)
        
        mu, sigma = get_distribution_params(module, hidden)
        dist_L = dist.Normal(mu, sigma)

        z_L_entropies.append(dist_L.entropy().mean())
        z_L_sigmas.append(sigma.mean())
        
        # RLOO only uses the log_probs of the sampled actions
        # So we sample, calculate log_prob, and that log_prob is the computational graph leaf.
        new_z_L = dist_L.sample()
        
        # Sum log probs over dimensions: (B, L_cycles, D) -> (B,)
        # step_log_prob = step_log_prob + dist_L.log_prob(new_z_L).sum(dim=(-1, -2))
        step_log_prob = step_log_prob + manual_gaussian_log_prob(new_z_L, mu, sigma).sum(dim=(-1,-2))
        curr_z_L = new_z_L

    # 2. H-Level Prediction
    if cfg.H_deterministic_mode == "separate weights":
        h_module = inner_model.H_level
    else:
        h_module = inner_model.L_level

    inner_step_tensor = torch.ones_like(act_step_num) * cfg.L_cycles
    hidden = z_H + curr_z_L 
    
    if cfg.time_embeddings:
        hidden = hidden + h_module.inner_step_emb(inner_step_tensor)[:, None, :]
        hidden = hidden + h_module.act_step_emb(act_step_num)[:, None, :]
        
    for layer in h_module.layers:
        hidden = layer(hidden_states=hidden, **seq_info)
        
    # 3. Action (z_H)
    if h_module.gaussian:
        mu, sigma = get_distribution_params(h_module, hidden)
        if h_module.config.H_deterministic_mode == "skip noise":
            dist_z = dist.Normal(mu, 0)
        else:
            dist_z = dist.Normal(mu, sigma)
            z_H_entropies.append(dist_z.entropy().mean())
            z_H_sigmas.append(sigma.mean())
            
        new_z_H = dist_z.sample()
        log_prob_z = manual_gaussian_log_prob(new_z_H, mu, sigma).sum(dim=(-1,-2))
        step_log_prob = step_log_prob + log_prob_z
    else: 
        new_z_H = hidden

    # 4. Halting
    if inner_model.q_head_input_form == "intermediate output":
        output = inner_model.lm_head(new_z_H)[:, inner_model.puzzle_emb_len:]
        q_in = rearrange(output, "b l v -> b (l v)")
    else: 
        q_in = new_z_H[:, 0, :]
        
    if inner_model.q_head_input_detached:
        q_in = q_in.detach()
    
    q_logits = inner_model.q_head(q_in) # logits for (halt, continue)
    q_logits = torch.nan_to_num(q_logits, nan=-10)
    halt_dist = dist.Categorical(logits=q_logits)

    exploration_metrics = {
        "z_L_entropy": torch.stack(z_L_entropies).mean() if z_L_entropies else torch.tensor(0.0, device=z_L.device),
        "z_L_sigma": torch.stack(z_L_sigmas).mean() if z_L_sigmas else torch.tensor(0.0, device=z_L.device),
        "z_H_entropy": torch.stack(z_H_entropies).mean() if z_H_entropies else torch.tensor(0.0, device=z_L.device),
        "z_H_sigma": torch.stack(z_H_sigmas).mean() if z_H_sigmas else torch.tensor(0.0, device=z_L.device),
    }
    
    return curr_z_L, new_z_H, step_log_prob, halt_dist, exploration_metrics, q_logits