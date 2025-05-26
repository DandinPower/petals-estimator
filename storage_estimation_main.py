from src.petals_estimator.config import ModelConfig, ServerConfig, ClientConfig, NetworkConfig
from src.petals_estimator.estimator import PetalsEstimator
from src.petals_estimator.server_config_example import get_2060_example, get_4060_example, get_4070ti_example, get_4090_example
from src.petals_estimator.utils import EstimateResults
import matplotlib.pyplot as plt

def create_configuration() -> tuple[ModelConfig, list[ServerConfig], ClientConfig]:
    download_network_bandwidth=1e8  # 100 MB/s
    upload_network_bandwidth=1e8    # 100 MB/s
    # Llama3.3 70B
    # For "weight_precision_bytes": Assume using int4, since unsloth achieve good performance while using dynamic 4 bit quantization: https://unsloth.ai/blog/dynamic-v2, also there is a lot low bits quant from GGUF (reference for gguf encoding: https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)
    # For "activation_precision_bytes": Assume using int8, since even llama.cpp default use fp32 for intermediate result: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml.c#L2825, but it support q8_0 quantization for intermediate result: https://github.com/ggml-org/llama.cpp/pull/951
    model_config = ModelConfig( 
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        num_layers= 80,
        intermediate_size=28672,
        vocab_size=128256,
        max_position_embeddings=131072,
        weight_precision_bytes=0.5,   
        activation_precision_bytes=1,   
    )
    
    server_configs = [
        get_4090_example(download_network_bandwidth, upload_network_bandwidth, 10),
        get_4090_example(download_network_bandwidth, upload_network_bandwidth, 10),
        get_4090_example(download_network_bandwidth, upload_network_bandwidth, 10),
        get_4090_example(download_network_bandwidth, upload_network_bandwidth, 10),
        get_4090_example(download_network_bandwidth, upload_network_bandwidth, 10),
        get_4090_example(download_network_bandwidth, upload_network_bandwidth, 10),
        get_4090_example(download_network_bandwidth, upload_network_bandwidth, 10),
        get_4090_example(download_network_bandwidth, upload_network_bandwidth, 10),
    ]
    
    client_config = ClientConfig(
        download_network_bandwidth=download_network_bandwidth,
        upload_network_bandwidth=upload_network_bandwidth,
        dram_bandwidth=50e9                 # 50 GB/s
    )
    network_config = NetworkConfig(
        fixed_rtt=50e-3
    )
    
    return model_config, server_configs, client_config, network_config

def main():
    """Main function to demonstrate the latency estimator"""
    model_config, server_configs, client_config, nework_config = create_configuration()
    estimator = PetalsEstimator(model_config, server_configs, client_config, nework_config)

    test_settings = [(pow(2, i+9), 1) for i in range(8)]

    # --- Storage stackplots (in GB) ---
    seq_lens = [s for s,_ in test_settings]
    batch_size = 1

    # Client: break out embedding, LM head, inputs cache
    emb, lm, cache = [], [], []
    for seq in seq_lens:
        client_storage, _ = estimator.estimate_total_storage_for_client_and_servers(seq, batch_size)
        # convert bytes to GB
        emb.append(client_storage['embedding_layer_weights'] / 1e9)
        lm .append(client_storage['lm_head_layer_weights']   / 1e9)
        cache.append(client_storage['inputs_cache']          / 1e9)

    plt.figure()
    plt.stackplot(
        seq_lens,
        emb, lm, cache,
        labels=['Embedding', 'LM Head', 'Inputs Cache']
    )
    plt.xscale('log', base=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Storage (GB)')
    plt.title('Client Storage Breakdown vs Sequence Length')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/client_storage_stackplot.png', dpi=300)

    # Server: total weights vs KV cache (summed over all servers)
    weights, kv = [], []
    for seq in seq_lens:
        _, servers = estimator.estimate_total_storage_for_client_and_servers(seq, batch_size)
        total_w = sum(srv['total_weights(all_layers)'] for srv in servers)
        total_k = sum(srv['kvcache']               for srv in servers)
        weights.append(total_w / 1e9)
        kv     .append(total_k / 1e9)

    plt.figure()
    plt.stackplot(
        seq_lens,
        weights, kv,
        labels=['Weights', 'KV Cache']
    )
    plt.xscale('log', base=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Storage (GB)')
    plt.title('Server Storage Breakdown vs Sequence Length')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/server_storage_stackplot.png', dpi=300)

if __name__ == "__main__":
    main()