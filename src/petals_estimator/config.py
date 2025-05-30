
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Llama3 like model configuration"""
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int    # GQA
    num_layers: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    weight_precision_bytes: int
    activation_precision_bytes: int
    
    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads

@dataclass
class ServerConfig:
    """Server hardware configuration"""
    ops: float  # OPS  
    # this ops number is theoratical and need to match the data type operations match the specify "weight_preceision" and "activation_precision", for example, in llama.cpp, we can use Q4 for weights and Q8 for activation, it should follow the int8 ops.
    ops_utilization: float # pecentage
    pcie_bandwidth: float # B/s
    vram_bandwidth: float  # B/s
    download_network_bandwidth: float  # bits/s
    upload_network_bandwidth: float  # bits/s
    num_blocks: int  # Number of transformer blocks on this server

@dataclass
class ClientConfig:
    """Client configuration"""
    download_network_bandwidth: float  # bits/s
    upload_network_bandwidth: float  # bits/s
    dram_bandwidth: float # bits/s
    # Only used to calculate the load and store operations for the server's input cache.

@dataclass
class NetworkConfig:
    fixed_rtt: float    # s