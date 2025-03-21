import numpy as np
from mpi4py import MPI
from rng import get_rng, rng_context, register_rng
from mpiwrapper import mpi
from moe import SimpleMoE, MoE_EP, MoE_TP
import time
import tracemalloc
import matplotlib.pyplot as plt

# Example usage
def run_moe(
    moe_type="tp", 
    batch_size=8, 
    feature_dim=32, 
    hidden_dim=128, 
    output_dim=64, 
    num_experts=None,
    topk=2
):
    """
    Unified function to run different types of MoE models
    
    Args:
        moe_type: Type of MoE to run ("simple", "ep", or "tp")
        batch_size: Number of samples in the batch
        feature_dim: Dimension of input features
        hidden_dim: Hidden dimension for experts
        output_dim: Output dimension
        topk: Number of experts to route each input to
    """
    # Get number of experts based on MPI world size
    num_experts = mpi.get_size()
    
    # Generate input data
    np.random.seed(0)
    X = np.random.randn(batch_size, feature_dim)

    if moe_type != "simple":
        # Synchronize the input data across all processes
        if mpi.get_rank() == 0:
            X = get_rng().randn(batch_size, feature_dim)
        else:
            X = None
        X = mpi.comm.bcast(X, root=0)
    
    # Create appropriate MoE model
    model_class = {
        "simple": SimpleMoE,
        "ep": MoE_EP,
        "tp": MoE_TP
    }.get(moe_type, MoE_TP)
    
    moe = model_class(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        topk=topk
    )
    
    # Run forward pass
    # Warm up
    _ = moe(X)
    
    # Measure time
    N = 10
    mem = 0
    time_vals = []
    t_put_vals = []

    start_time = time.time()
    tracemalloc.start()
    for _ in range(N):
        start_time = time.time()
        outputs = moe(X)
        end_time = time.time()
        avg_duration_ms = 1000 * (end_time - start_time)# / N
        time_vals.append(avg_duration_ms)
        throughput = batch_size / avg_duration_ms
        t_put_vals.append(throughput)

    # Measure memory
    tracemalloc.start()
    mem_vals = []
    N = 10
    mem = 0
    for _ in range(N):
        outputs = moe(X)
        current, peak = tracemalloc.get_traced_memory()
        mem += current
        mem_vals.append(current / 1024 ** 2)

    tracemalloc.stop()
    avg_memory = mem / N
    avg_memory = avg_memory / 1024 ** 2
    
    # Print timing information
    # if mpi.get_rank() == 0:
    #     print(f"Forward pass time for {moe_type} MoE: {avg_duration_ms} ms")

    # # Print memory information
    # if mpi.get_rank() == 0:
    #     print(f"Memory for {moe_type} MoE: {avg_memory / (1024):.2f} KB")

    # return dict(
    #     outputs=outputs,
    #     avg_duration_ms=avg_duration_ms,
    #     avg_memory=avg_memory,
    #     throughput=throughput
    # )
    return time_vals, t_put_vals, mem_vals
    
def plot(simple_vals, tp_vals, ep_vals, metric):
    plt.figure(figsize=(10, 6))  # Set the figure size


    label1 = f'Simple MoE Average {metric}'
    label2 = f'TP MoE Average {metric}'
    label3 = f'EP MoE Average {metric}'
    
    # Plotting the three lists
    plt.plot(simple_vals, label=label1, color='r', marker='o')
    plt.plot(tp_vals, label=label2, color='g', marker='x')
    plt.plot(ep_vals, label=label3, color='b', marker='^')

    # Adding labels, title, and legend
    title = f'MoE {metric} Comparison'
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(f'{metric}')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig('test.png')


def benchmark_moe():
    # Test simple MoE
    simple_time, simple_t_put, simple_mem = run_moe(moe_type="simple")
    
    # print(f"Simple MoE: {simple_result['avg_duration_ms']} ms")
    # print(f"Simple MoE: {simple_result['avg_memory']} mb")
    # print(f"Simple MoE: {simple_result['throughput']} samples / ms")

    # Test TP MoE
    tp_time, tp_t_put, tp_mem = run_moe(moe_type="tp")
    # print(f"TP MoE: {tp_result['avg_duration_ms']} ms")
    # print(f"TP MoE: {tp_result['avg_memory']} mb")
    # print(f"TP MoE: {tp_result['throughput']} samples / ms")

    # Test EP MoE
    ep_time, ep_t_put, ep_mem = run_moe(moe_type="ep")
    # print(f"EP MoE: {ep_result['avg_duration_ms']} ms")
    # print(f"EP MoE: {ep_result['avg_memory']} mb")
    # print(f"EP MoE: {ep_result['throughput']} samples / ms")
    plot(simple_mem, tp_mem, ep_mem, 'Memory')

if __name__ == "__main__":
    benchmark_moe()

