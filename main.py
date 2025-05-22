import math
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """LLaMA3 model configuration"""
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    
    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads

@dataclass
class ServerConfig:
    """Server hardware configuration"""
    flops_per_second: float  # FLOPS/s (e.g., 312e12 for A100)
    memory_bandwidth: float  # GB/s
    network_bandwidth: float  # GB/s
    num_blocks: int  # Number of transformer blocks on this server

@dataclass
class NetworkConfig:
    """Network configuration"""
    client_to_server_bandwidth: float  # GB/s
    server_to_server_bandwidth: float  # GB/s
    client_to_server_rtt: float  # seconds
    server_to_server_rtt: float  # seconds

class PetalsLatencyEstimator:
    def __init__(self, model_config: ModelConfig, server_configs: List[ServerConfig], 
                 network_config: NetworkConfig):
        self.model_config = model_config
        self.server_configs = server_configs
        self.network_config = network_config
        
        # LLaMA3 model variants
        self.model_variants = {
            "llama3-8b": ModelConfig(
                hidden_size=4096,
                num_attention_heads=32,
                num_key_value_heads=8,
                intermediate_size=14336,
                vocab_size=128256,
                max_position_embeddings=8192
            ),
            "llama3-70b": ModelConfig(
                hidden_size=8192,
                num_attention_heads=64,
                num_key_value_heads=8,
                intermediate_size=28672,
                vocab_size=128256,
                max_position_embeddings=8192
            )
        }
    
    def calculate_transformer_block_flops(self, seq_len: int, batch_size: int) -> float:
        """Calculate FLOPs for one transformer block forward pass"""
        config = self.model_config
        
        # Attention FLOPs
        # Q, K, V projections
        qkv_flops = 3 * batch_size * seq_len * config.hidden_size * config.hidden_size
        
        # Attention computation (QK^T)
        qk_flops = batch_size * config.num_attention_heads * seq_len * seq_len * config.head_dim
        
        # Attention weights * V
        attn_v_flops = batch_size * config.num_attention_heads * seq_len * seq_len * config.head_dim
        
        # Output projection
        out_proj_flops = batch_size * seq_len * config.hidden_size * config.hidden_size
        
        # MLP FLOPs
        # Gate and up projections
        gate_up_flops = 2 * batch_size * seq_len * config.hidden_size * config.intermediate_size
        
        # Down projection
        down_flops = batch_size * seq_len * config.intermediate_size * config.hidden_size
        
        # Layer norm FLOPs (approximate)
        layernorm_flops = 2 * batch_size * seq_len * config.hidden_size * 2  # 2 layer norms per block
        
        total_flops = (qkv_flops + qk_flops + attn_v_flops + out_proj_flops + 
                      gate_up_flops + down_flops + layernorm_flops)
        
        return total_flops
    
    def calculate_activation_size(self, seq_len: int, batch_size: int) -> float:
        """Calculate activation tensor size in bytes"""
        config = self.model_config
        
        # Main activation tensor (hidden states)
        hidden_states_size = batch_size * seq_len * config.hidden_size * 4  # 4 bytes per float32
        
        # Additional intermediate tensors (approximate)
        # Attention intermediate tensors
        attention_intermediate = batch_size * seq_len * config.intermediate_size * 4
        
        # Key-Value cache (if applicable)
        kv_cache_size = (batch_size * seq_len * config.num_key_value_heads * 
                        config.head_dim * 2 * 4)  # K and V tensors
        
        total_size = hidden_states_size + attention_intermediate + kv_cache_size
        return total_size
    
    def estimate_computation_latency(self, seq_len: int, batch_size: int) -> Dict[str, float]:
        """Estimate computation latency for each server"""
        results = {}
        
        for i, server in enumerate(self.server_configs):
            # FLOPs for all blocks on this server
            total_flops = (self.calculate_transformer_block_flops(seq_len, batch_size) * 
                          server.num_blocks)
            
            # Compute-bound latency
            compute_latency = total_flops / server.flops_per_second
            
            # Memory-bound latency (data movement)
            activation_size_gb = self.calculate_activation_size(seq_len, batch_size) / (1024**3)
            memory_latency = activation_size_gb / server.memory_bandwidth
            
            # Take the maximum (bottleneck)
            server_latency = max(compute_latency, memory_latency)
            
            results[f'server_{i}'] = {
                'compute_latency': compute_latency,
                'memory_latency': memory_latency,
                'total_latency': server_latency,
                'flops': total_flops,
                'activation_size_gb': activation_size_gb
            }
        
        return results
    
    def estimate_communication_latency(self, seq_len: int, batch_size: int) -> Dict[str, float]:
        """Estimate communication latency"""
        activation_size_gb = self.calculate_activation_size(seq_len, batch_size) / (1024**3)
        
        # Client to first server
        client_to_server_transfer = activation_size_gb / self.network_config.client_to_server_bandwidth
        client_to_server_total = client_to_server_transfer + self.network_config.client_to_server_rtt
        
        # Server to server communication (for pipeline)
        num_server_hops = len(self.server_configs) - 1
        if num_server_hops > 0:
            server_to_server_transfer = activation_size_gb / self.network_config.server_to_server_bandwidth
            server_to_server_total = ((server_to_server_transfer + self.network_config.server_to_server_rtt) * 
                                    num_server_hops)
        else:
            server_to_server_total = 0
        
        # Server to client (response)
        server_to_client_transfer = activation_size_gb / self.network_config.client_to_server_bandwidth
        server_to_client_total = server_to_client_transfer + self.network_config.client_to_server_rtt
        
        total_communication_latency = (client_to_server_total + server_to_server_total + 
                                     server_to_client_total)
        
        return {
            'client_to_server_latency': client_to_server_total,
            'server_to_server_latency': server_to_server_total,
            'server_to_client_latency': server_to_client_total,
            'total_communication_latency': total_communication_latency,
            'activation_size_gb': activation_size_gb
        }
    
    def estimate_total_latency(self, seq_len: int, batch_size: int) -> Dict[str, float]:
        """Estimate total latency for forward pass"""
        computation_results = self.estimate_computation_latency(seq_len, batch_size)
        communication_results = self.estimate_communication_latency(seq_len, batch_size)
        
        # Total computation latency (sum of all servers, assuming pipeline)
        total_computation_latency = sum([server['total_latency'] for server in computation_results.values()])
        
        # For pipeline execution, servers can work in parallel to some extent
        # But we need to account for the critical path
        max_server_latency = max([server['total_latency'] for server in computation_results.values()])
        
        # Pipeline latency is approximately the max server latency plus communication overhead
        pipeline_computation_latency = max_server_latency + (total_computation_latency - max_server_latency) * 0.1
        
        total_latency = pipeline_computation_latency + communication_results['total_communication_latency']
        
        return {
            'computation_latency_per_forward': pipeline_computation_latency,
            'communication_latency_per_forward': communication_results['total_communication_latency'],
            'total_latency_per_forward': total_latency,
            'computation_details': computation_results,
            'communication_details': communication_results
        }

def create_example_setup():
    """Create example configuration for testing"""
    # LLaMA3-8B configuration
    model_config = ModelConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
        max_position_embeddings=8192
    )
    
    # Example server configurations (3 servers with different capabilities)
    server_configs = [
        ServerConfig(
            flops_per_second=312e12,  # A100 (~312 TFLOPS)
            memory_bandwidth=1935,    # A100 memory bandwidth (GB/s)
            network_bandwidth=25,     # 25 GB/s network
            num_blocks=11             # ~1/3 of 32 blocks
        ),
        ServerConfig(
            flops_per_second=250e12,  # V100 (~250 TFLOPS)
            memory_bandwidth=900,     # V100 memory bandwidth (GB/s)
            network_bandwidth=10,     # 10 GB/s network
            num_blocks=10             # ~1/3 of 32 blocks
        ),
        ServerConfig(
            flops_per_second=200e12,  # RTX 4090 (~200 TFLOPS)
            memory_bandwidth=1000,    # RTX 4090 memory bandwidth (GB/s)
            network_bandwidth=5,      # 5 GB/s network
            num_blocks=11             # ~1/3 of 32 blocks
        )
    ]
    
    # Network configuration
    network_config = NetworkConfig(
        client_to_server_bandwidth=1.0,    # 1 GB/s (8 Gbps)
        server_to_server_bandwidth=1.0,   # 1 GB/s
        client_to_server_rtt=0.100,        # 100ms RTT
        server_to_server_rtt=0.100         # 100ms RTT between servers
    )
    
    return model_config, server_configs, network_config

def main():
    """Main function to demonstrate the estimator"""
    model_config, server_configs, network_config = create_example_setup()
    
    estimator = PetalsLatencyEstimator(model_config, server_configs, network_config)
    
    # Test different configurations
    test_configs = [
        (512, 1),    # Context length: 512, Batch size: 1
        # (1024, 1),   # Context length: 1024, Batch size: 1
        # (2048, 1),   # Context length: 2048, Batch size: 1
        # (512, 4),    # Context length: 512, Batch size: 4
        # (1024, 8),   # Context length: 1024, Batch size: 8
    ]
    
    print("Petals Distributed Inference Latency Estimation")
    print("=" * 60)
    print(f"Model: LLaMA3-like ({model_config.hidden_size} hidden size)")
    print(f"Servers: {len(server_configs)} nodes")
    print(f"Total blocks: {sum(s.num_blocks for s in server_configs)}")
    print()
    
    for seq_len, batch_size in test_configs:
        print(f"Configuration: Context Length = {seq_len}, Batch Size = {batch_size}")
        print("-" * 50)
        
        results = estimator.estimate_total_latency(seq_len, batch_size)
        
        print(f"Communication Latency per Forward: {results['communication_latency_per_forward']:.3f} seconds")
        print(f"Computation Latency per Forward:   {results['computation_latency_per_forward']:.3f} seconds")
        print(f"Total Latency per Forward:         {results['total_latency_per_forward']:.3f} seconds")
        print()
        
        # Detailed breakdown
        print("Detailed Breakdown:")
        comm_details = results['communication_details']
        print(f"  Client -> Server: {comm_details['client_to_server_latency']:.3f}s")
        print(f"  Server -> Server: {comm_details['server_to_server_latency']:.3f}s")
        print(f"  Server -> Client: {comm_details['server_to_client_latency']:.3f}s")
        print(f"  Activation Size: {comm_details['activation_size_gb']:.3f} GB")
        print()
        
        comp_details = results['computation_details']
        for server_name, details in comp_details.items():
            print(f"  {server_name}: {details['total_latency']:.3f}s "
                  f"(compute: {details['compute_latency']:.3f}s, "
                  f"memory: {details['memory_latency']:.3f}s)")
        print()
        print("=" * 60)

if __name__ == "__main__":
    main()