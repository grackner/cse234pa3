import argparse
import json
import math
import numpy as np
from scipy.optimize import minimize

def model_training_cost_analysis_llama(model_config_path):
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    max_position_embeddings = config['max_position_embeddings']
    num_hidden_layers = config['num_hidden_layers']
    intermediate_size = config['intermediate_size']
    num_attention_heads = config['num_attention_heads']
    max_sequence_length = config['max_sequence_length']

    word_embedding = vocab_size * hidden_size
    positional_embedding = max_position_embeddings * hidden_size
    attention_params = 4 * hidden_size * hidden_size 
    mlp = 2 * hidden_size * intermediate_size + intermediate_size * hidden_size
    attention_biases = 0
    mlp_biases = 0
    layer_norm = 2 * hidden_size
    
    params_per_layer = attention_params + attention_biases + mlp + mlp_biases + layer_norm
    transformer = num_hidden_layers * params_per_layer
    final_layer_norm = hidden_size
    lm_head = hidden_size * vocab_size
    
    total_params = word_embedding + positional_embedding + transformer + final_layer_norm + lm_head
    N = max_sequence_length
    D = hidden_size
    qkv = 3 * N * D * D
    attention = N * N * D
    out = N * D * D
    attention_flops = qkv + attention + out
    mlpf = 2 * N * D * intermediate_size
    flops_per_layer = attention_flops + mlpf
    flops_layer_TF = flops_per_layer / 1e12
    batch_size = 1 
    bytes_per_param = 2 
    activation_memory = batch_size * N * D * bytes_per_param
    qkv_memory = 3 * batch_size * N * D * bytes_per_param
    attn_scores_memory = batch_size * num_attention_heads * N * N * bytes_per_param
    context_memory = batch_size * N * D * bytes_per_param
    mlp_intermediate_memory = batch_size * N * intermediate_size * bytes_per_param
    peak_memory = max(
        activation_memory + qkv_memory,
        activation_memory + attn_scores_memory,
        activation_memory + context_memory,
        activation_memory + mlp_intermediate_memory
    )
    peak_memory_GB = peak_memory / (1024**3)
    return total_params, flops_layer_TF, peak_memory_GB


def model_training_cost_analysis_deepseek(model_config_path):
    """Calculate parameters, FLOPS, and memory for a single layer of the DeepSeek model."""
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    max_position_embeddings = config['max_position_embeddings']
    num_hidden_layers = config['num_hidden_layers']
    intermediate_size = config['intermediate_size']
    num_attention_heads = config['num_attention_heads']
    num_key_value_heads = config.get('num_key_value_heads', num_attention_heads)
    max_sequence_length = max_position_embeddings  

    n_routed_experts = config.get('n_routed_experts', 0)
    num_experts_per_tok = config.get('num_experts_per_tok', 0)
    moe_intermediate_size = config.get('moe_intermediate_size', 0)
    moe_layer_freq = config.get('moe_layer_freq', 0)  
    
    num_moe_layers = num_hidden_layers // moe_layer_freq if moe_layer_freq > 0 else 0
    num_dense_layers = num_hidden_layers - num_moe_layers
    word_embedding = vocab_size * hidden_size
    q = hidden_size * hidden_size
    k = hidden_size * hidden_size * (num_key_value_heads / num_attention_heads)
    v = hidden_size * hidden_size * (num_key_value_heads / num_attention_heads)
    o = hidden_size * hidden_size
    attention = q + k + v + o
    mlpd = hidden_size * intermediate_size + intermediate_size * hidden_size
    expert = hidden_size * n_routed_experts
    expert_mlp = hidden_size * moe_intermediate_size + moe_intermediate_size * hidden_size
    total_expert = n_routed_experts * expert_mlp + expert

    norm_params = 2 * hidden_size  
    
    dense_layer = attention + mlpd + norm_params
    moe_layer = attention + total_expert + norm_params
    
    transformer = (dense_layer * num_dense_layers) + (moe_layer * num_moe_layers)
    
    final_norm = hidden_size
    lm_head = hidden_size * vocab_size
    
    total_params = word_embedding + transformer + final_norm + lm_head
    
    N = max_sequence_length
    H = num_attention_heads
    H_kv = num_key_value_heads
    D = hidden_size
    head_dim = D // H
   
    layer_norm_flops = 2 * N * D
    
    qflops = N * D * D
    kflops = N * D * D * (H_kv / H)
    vflops = N * D * D * (H_kv / H)
    qkvflops = qflops + kflops + vflops
    
    attn_compute_flops = 2 * N * N * H * head_dim
    out_proj_flops = N * D * D
    attention_flops = qkvflops + attn_compute_flops + out_proj_flops
    dense_mlp_flops = 2 * N * D * intermediate_size

    expert_gate_flops = N * n_routed_experts
    expert_compute_flops = N * num_experts_per_tok * (2 * D * moe_intermediate_size)
    moe_mlp_flops = expert_gate_flops + expert_compute_flops
    dense_layer_flops = layer_norm_flops + attention_flops + dense_mlp_flops
    moe_layer_flops = layer_norm_flops + attention_flops + moe_mlp_flops
    flops_per_layer = moe_layer_flops if num_moe_layers > 0 else dense_layer_flops
    flops_layer_TF = flops_per_layer / 1e12

    batch_size = 1 
    bytes_per_param = 2 
    activation_memory = batch_size * N * D * bytes_per_param
    qkv_memory = 3 * batch_size * N * D * bytes_per_param
    context_memory = batch_size * N * D * bytes_per_param
    mlp_intermediate_memory = batch_size * N * intermediate_size * bytes_per_param
    peak_memory = max(
        activation_memory + qkv_memory,
        activation_memory + context_memory,
        activation_memory + mlp_intermediate_memory
    )
    peak_memory_GB = peak_memory / (1024**3)
    
    return total_params, flops_layer_TF, peak_memory_GB


def get_optimal_N_D_from_cost(cost_budget):
    """
    Find the optimal number of parameters (N) and training tokens (D)
    given a monetary training budget.
    
    Args:
        cost_budget: Training budget in dollars
    
    Returns:
        N: Optimal total model parameters (in absolute numbers)
        D: Optimal number of training tokens (in absolute numbers)
        training_budget_flops: Effective total training FLOPs (in FLOPs)
        best_gpu: Name of the selected GPU
    """

    gpus = {
        'A100': {'cost_per_hour': 4.0, 'TFLOPs': 312},
        'V100': {'cost_per_hour': 2.5, 'TFLOPs': 125},
        'T4': {'cost_per_hour': 1.0, 'TFLOPs': 65}
    }

    mfu = 0.4

    best_gpu = None
    max_flops_per_dollar = 0
    
    for gpu, specs in gpus.items():
        flops_per_dollar = specs['TFLOPs'] * mfu / specs['cost_per_hour']
        if flops_per_dollar > max_flops_per_dollar:
            max_flops_per_dollar = flops_per_dollar
            best_gpu = gpu
    
    total_hours = cost_budget / gpus[best_gpu]['cost_per_hour']
    total_flops = total_hours * gpus[best_gpu]['TFLOPs'] * 1e12 * mfu
    
    def loss(N, D):
        """Chinchilla scaling law."""
        return 406.4 * (N ** -0.34) + 410.7 * (D ** -0.29) + 1.69
    
    def objective(params):
        """Return loss given parameters [log(N), log(D)]."""
        log_N, log_D = params
        N = np.exp(log_N)
        D = np.exp(log_D)
        return loss(N, D)
    
    def constraint(params):
        """Ensure N*D = total_flops."""
        log_N, log_D = params
        return np.exp(log_N + log_D) - total_flops
    
    initial_guess = [np.log(np.sqrt(total_flops)), np.log(np.sqrt(total_flops))]
    
    constraints = {'type': 'eq', 'fun': constraint}
    bounds = [(None, None), (None, None)]  
    
    result = minimize(objective, initial_guess, method='SLSQP', 
                     constraints=constraints, bounds=bounds)
    
    log_N, log_D = result.x
    N = int(np.exp(log_N))
    D = int(np.exp(log_D))
    
    return N, D, total_flops, best_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training cost analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget')
    args = parser.parse_args()

    if args.model_config:
        if 'deepseek' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(args.model_config)
        elif 'llama' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
        else:
            print('Unknown LLM Type!')
            exit()
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:    
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")