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
    num_attention_heads = config['num_attention_heads']
    max_sequence_length = config['max_sequence_length']

    word_embedding_params = vocab_size * hidden_size
    positional_embedding_params = max_position_embeddings * hidden_size
    attention_params = 4 * hidden_size * hidden_size  # Q, K, V, and output projection
    mlp_params = 2 * hidden_size * intermediate_size  # Two dense layers
    layer_norm_params = 2 * hidden_size
    params_per_layer = attention_params + mlp_params + layer_norm_params
    transformer_params = num_hidden_layers * params_per_layer
    total_params = word_embedding_params + positional_embedding_params + transformer_params
    # # Number of TFLOPs for a single forward pass of one layer
    # N = max_sequence_length
    # D = hidden_size
    # flops_per_layer = 2 * N * D * D  # FLOPs for one matrix multiplication
    # flops_layer_TF = flops_per_layer / 1e12  # Convert to TFLOPs
    
    # # Peak memory cost for a single forward pass of one layer
    # # Assuming fp16 precision (2 bytes per parameter)
    # peak_memory_GB = (2 * N * D * 2) / 1e9  
    N = max_sequence_length
    D = hidden_size
    H = num_attention_heads
    qkv_proj_flops = 3 * N * D * D
    attn_flops = N * N * D
    out_proj_flops = N * D * D
    attention_flops = qkv_proj_flops + attn_flops + out_proj_flops
    mlp_flops = 2 * N * D * intermediate_size
    flops_per_layer = attention_flops + mlp_flops
    flops_layer_TF = flops_per_layer / 1e12
    kv_cache = 2 * N * D * 2  # 2 for K and V, 2 bytes per fp16 value
    attn_matrix = N * N * 2  # 2 bytes per fp16 value
    activations = N * D * 2  # 2 bytes per fp16 value
    
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
    
    # Calculate total parameters
    word_embedding_params = vocab_size * hidden_size
    positional_embedding_params = max_position_embeddings * hidden_size
    
    attention_params = 4 * hidden_size * hidden_size
    mlp_params = 2 * hidden_size * intermediate_size
    layer_norm_params = 2 * hidden_size
    
    # MoE-specific parameters
    expert_mlp_params = 2 * hidden_size * moe_intermediate_size
    total_expert_params = n_routed_experts * expert_mlp_params
    shared_expert_params = n_shared_experts * expert_mlp_params  # Assuming similar structure
    
    # Total parameters per layer
    params_per_layer = attention_params + mlp_params + layer_norm_params
    params_per_moe_layer = params_per_layer + total_expert_params + shared_expert_params
    
    # Adjust for MoE layer frequency
    num_moe_layers = num_hidden_layers * moe_layer_freq
    
    # Total parameters for all layers
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
    kv_cache = 2 * N * D * 2  # 2 bytes per bfloat16 value
    attn_matrix = N * N * 2  # 2 bytes per bfloat16 value
    moe_activations = num_experts_per_tok * N * D * 2  # 2 bytes per bfloat16 value
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
    # gpus = {
    #     'A100': {'cost_per_hour': 4.0, 'peak_flops': 312e12},
    #     'V100': {'cost_per_hour': 2.5, 'peak_flops': 125e12},
    #     'T4': {'cost_per_hour': 1.0, 'peak_flops': 65e12}
    # }

    # mfu = 0.40
    
    # for gpu in gpus:
    #     gpus[gpu]['flops_per_second'] = gpus[gpu]['peak_flops'] * mfu
    
    # # Calculate total FLOPs per dollar for each GPU
    # for gpu in gpus:
    #     flops_per_second = gpus[gpu]['flops_per_second']
    #     cost_per_second = gpus[gpu]['cost_per_hour'] / 3600
    #     gpus[gpu]['flops'] = flops_per_second / cost_per_second
    
    # # Find the GPU with the best FLOPS per dollar ratio
    # best_gpu = max(gpus.items(), key=lambda x: x[1]['flops'])[0]
    
    # # Calculate the total FLOPs we can get from our budget
    # total_flops = gpus[best_gpu]['flops']
    # training_budget_flops = cost_budget * total_flops
    # gpus = {
    #     'A100': {'cost_per_hour': 4.0, 'peak_flops': 312e12, 'mfu': 0.4},
    #     'V100': {'cost_per_hour': 2.5, 'peak_flops': 125e12, 'mfu': 0.4},
    #     'T4': {'cost_per_hour': 1.0, 'peak_flops': 65e12, 'mfu': 0.4}
    # }
    # best_gpu = None
    # max_flops_per_dollar = 0
    
    # for gpu, specs in gpus.items():
    #     effective_flops_per_second = specs['peak_flops'] * specs['mfu']
    #     flops_per_dollar = effective_flops_per_second * 3600 / specs['cost_per_hour']
        
    #     if flops_per_dollar > max_flops_per_dollar:
    #         max_flops_per_dollar = flops_per_dollar
    #         best_gpu = gpu
    
    # # Calculate total effective FLOPs available for the budget
    # effective_flops_per_second = gpus[best_gpu]['peak_flops'] * gpus[best_gpu]['mfu']
    # total_hours = cost_budget / gpus[best_gpu]['cost_per_hour']
    # total_seconds = total_hours * 3600
    # training_budget_flops = effective_flops_per_second * total_seconds
    # # # Constants for the scaling law
    # # a = 406.4
    # # alpha = 0.34
    # # b = 410.7
    # # beta = 0.29
    # # c = 1.69
    # # exponent = (1+beta)/(1+alpha+beta)
    # # N = ((a*alpha*(1+alpha+beta))/(b*beta*6**beta))**(1/(alpha+beta)) * training_budget_flops**exponent
    
    # # # Calculate D from compute = 6ND
    # # D = training_budget_flops / (6 * N)
    # alpha = 0.34
    # beta = 0.29
    # A = 406.4
    # B = 410.7
    # compute = training_budget_flops
    # flops_per_param_token = 6
    
    # # Initial guess
    # N = (compute / flops_per_param_token) ** 0.5  # Balanced starting point
    
    # # Iterative refinement
    # for _ in range(10):
    #     # Based on the scaling law relationship and compute budget
    #     D = compute / (flops_per_param_token * N)
        
    #     # The optimal ratio from the scaling law
    #     optimal_ratio = (A * alpha / (B * beta)) ** (1/(beta+1)) * N**((alpha+1)/(beta+1))
        
    #     # Adjust N based on current ratio
    #     current_ratio = N / D
    #     adjustment = (optimal_ratio / current_ratio) ** 0.5
    #     N = N * adjustment
    
    # # Final calculation of D
    # D = compute / (flops_per_param_token * N)
    
    # # Round to nearest integers
    # N = round(N)
    # D = round(D)
    gpu_costs = {
        'A100': {'cost_per_hour': 4.0, 'TFLOPs': 312},
        'V100': {'cost_per_hour': 2.5, 'TFLOPs': 125},
        'T4': {'cost_per_hour': 1.0, 'TFLOPs': 65}
    }
    
    # Assume MFU (Machine Fill-Up) is 40% for all GPUs
    mfu = 0.4
    
    # Define the scaling law
    def scaling_law(N, D):
        return 406.4 / (N ** 0.34) + 410.7 / (D ** 0.29) + 1.69
    
    # Initialize optimal values
    best_gpu = None
    optimal_N = None
    optimal_D = None
    max_training_budget_flops = 0
    
    # Iterate over GPUs to find the best one
    for gpu, specs in gpu_costs.items():
        # Calculate effective training FLOPs per hour
        effective_flops_per_hour = specs['TFLOPs'] * mfu
        
        # Calculate total hours that can be afforded with the budget
        hours_affordable = cost_budget / specs['cost_per_hour']
        
        # Calculate total effective training FLOPs
        training_budget_flops = effective_flops_per_hour * hours_affordable
        
        # Check if this GPU provides more FLOPs than the current best
        if training_budget_flops > max_training_budget_flops:
            max_training_budget_flops = training_budget_flops
            best_gpu = gpu
            
            # Find optimal N and D using the scaling law
            # This part requires optimization techniques or brute force search
            # For simplicity, let's assume we are searching over a reasonable range
            best_cost = float('inf')
            for N in range(1000000, 100000000, 1000000):  # Example range for N
                for D in range(1000000, 100000000, 1000000):  # Example range for D
                    cost = scaling_law(N, D)
                    if cost < best_cost:
                        best_cost = cost
                        optimal_N = N
                        optimal_D = D
    
    return optimal_N, optimal_D, max_training_budget_flops, best_gpu

    # return N, D, training_budget_flops, best_gpu


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

    