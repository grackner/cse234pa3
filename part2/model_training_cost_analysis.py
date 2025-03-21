import argparse
import json
import math

def model_training_cost_analysis_llama(model_config_path):
    #TODO you code here.
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    max_position_embeddings = config['max_position_embeddings']
    num_hidden_layers = config['num_hidden_layers']
    intermediate_size = config['intermediate_size']
    max_sequence_length = config['max_sequence_length']

    word_embedding = vocab_size * hidden_size
    positional_embedding = max_position_embeddings * hidden_size
    attention_params = 4 * hidden_size * hidden_size  # Q, K, V, and output projection
    mlp = 2 * hidden_size * intermediate_size  # Two dense layers
    layer_norm = 2 * hidden_size
    params_per_layer = attention_params + mlp + layer_norm
    transformer = num_hidden_layers * params_per_layer
    total_params = word_embedding + positional_embedding + transformer
    N = max_sequence_length
    D = hidden_size
    qkv = 3 * N * D * D
    attention = N * N * D
    out = N * D * D
    attention_flops = qkv + attention + out
    mlpf = 2 * N * D * intermediate_size
    flops_per_layer = attention_flops + mlpf
    flops_layer_TF = flops_per_layer / 1e12
    kv_cache = 2 * N * D * 2  
    attn_matrix = N * N * 2  
    activations = N * D * 2 
    
    peak_memory_GB = (kv_cache + attn_matrix + activations) / 1e9
    
    
    return total_params, flops_layer_TF, peak_memory_GB

def model_training_cost_analysis_deepseek(model_config_path):
    #TODO you code here.
    with open(model_config_path, 'r') as f:
        config = json.load(f)
        
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    max_position_embeddings = config['max_position_embeddings']
    num_hidden_layers = config['num_hidden_layers']
    intermediate_size = config['intermediate_size']
    num_attention_heads = config['num_attention_heads']
    max_sequence_length = max_position_embeddings  # Use max_position_embeddings for sequence length
    n_routed_experts = config['n_routed_experts']
    n_shared_experts = config['n_shared_experts']
    num_experts_per_tok = config['num_experts_per_tok']
    moe_intermediate_size = config['moe_intermediate_size']
    moe_layer_freq = config['moe_layer_freq']  # MoE layer frequency
    word_embedding_params = vocab_size * hidden_size
    positional_embedding_params = max_position_embeddings * hidden_size
    
    attention_params = 4 * hidden_size * hidden_size
    mlp_params = 2 * hidden_size * intermediate_size
    layer_norm_params = 2 * hidden_size
    expert_mlp_params = 2 * hidden_size * moe_intermediate_size
    total_expert_params = n_routed_experts * expert_mlp_params
    params_per_layer = attention_params + mlp_params + layer_norm_params
    num_moe_layers = num_hidden_layers * moe_layer_freq
    transformer_params = num_hidden_layers * params_per_layer + num_moe_layers * total_expert_params
    total_params = word_embedding_params + positional_embedding_params + transformer_params
    
    # Calculate TFLOPs for a single forward pass of one layer
    N = max_sequence_length
    D = hidden_size
    H = num_attention_heads
    
    qkv_proj_flops = 3 * N * D * D
    attn_flops = N * N * D
    out_proj_flops = N * D * D
    attention_flops = qkv_proj_flops + attn_flops + out_proj_flops
    
    moe_mlp_flops = num_experts_per_tok * N * (2 * D * moe_intermediate_size)
    flops_per_layer = attention_flops + moe_mlp_flops
    flops_layer_TF = flops_per_layer / 1e12
    
    # Calculate peak memory cost for a single forward pass of one layer
    kv_cache = 2 * N * D * 2  
    attn_matrix = N * N * 2  
    moe_activations = num_experts_per_tok * N * D * 2  
    peak_memory_GB = (kv_cache + attn_matrix + moe_activations) / 1e9
    
    return total_params, flops_layer_TF, peak_memory_GB

def get_optimal_N_D_from_cost(cost_budget):
    """
    cost_budget:  a monetary training budget (in dollars)
    Returns:
        N: Optimal total model parameters (in absolute numbers)
        D: Optimal number of training tokens (in absolute numbers)
        training_budget_flops: Effective total training FLOPs (in FLOPs)
        best_gpu: name of the selected GPU (one of 'A100', 'V100', 'T4')
    """
    #TODO you code here
    
    gpus = {
        'A100': {'cost_per_hour': 4.0, 'TFLOPs': 312},
        'V100': {'cost_per_hour': 2.5, 'TFLOPs': 125},
        'T4': {'cost_per_hour': 1.0, 'TFLOPs': 65}
    }
    
    mfu = 0.4
    def scaling_law(N, D):
        return 406.4 / (N ** 0.34) + 410.7 / (D ** 0.29) + 1.69
    best_gpu = None
    N = None
    D = None
    max_flops = 0
    for gpu, specs in gpus.items():
        flops_per_hour = specs['TFLOPs'] * mfu
        hours = cost_budget / specs['cost_per_hour']
        flops = flops_per_hour * hours
        if flops > max_flops:
            max_flops = flops
            best_gpu = gpu
            best_cost = float('inf')
            for n in range(1000000, 100000000, 1000000):  
                for d in range(1000000, 100000000, 1000000): 
                    cost = scaling_law(n, d)
                    if cost < best_cost:
                        best_cost = cost
                        N = n
                        D = d
    
    return N, D, max_flops, best_gpu



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

    