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
    
    # Define three servers
    # server_configs = [
    #     get_2060_example(download_network_bandwidth, upload_network_bandwidth, 15),
    #     get_2060_example(download_network_bandwidth, upload_network_bandwidth, 15),
    #     get_2060_example(download_network_bandwidth, upload_network_bandwidth, 15),
    #     get_2060_example(download_network_bandwidth, upload_network_bandwidth, 15),
    #     get_2060_example(download_network_bandwidth, upload_network_bandwidth, 15),
    #     get_2060_example(download_network_bandwidth, upload_network_bandwidth, 5),
    # ]
    
    # server_configs = [
    #     get_4060_example(download_network_bandwidth, upload_network_bandwidth, 20),
    #     get_4060_example(download_network_bandwidth, upload_network_bandwidth, 20),
    #     get_4060_example(download_network_bandwidth, upload_network_bandwidth, 20),
    #     get_4060_example(download_network_bandwidth, upload_network_bandwidth, 20),
    # ]
    
    # server_configs = [
    #     get_4070ti_example(download_network_bandwidth, upload_network_bandwidth, 30),
    #     get_4070ti_example(download_network_bandwidth, upload_network_bandwidth, 30),
    #     get_4070ti_example(download_network_bandwidth, upload_network_bandwidth, 20),
    # ]
    
    server_configs = [
        get_4090_example(download_network_bandwidth, upload_network_bandwidth, 30),
        get_4090_example(download_network_bandwidth, upload_network_bandwidth, 30),
        get_4090_example(download_network_bandwidth, upload_network_bandwidth, 20),
    ]
    
    # server_configs = [
    #     get_4090_example(download_network_bandwidth, upload_network_bandwidth, 40),
    #     get_4090_example(download_network_bandwidth, upload_network_bandwidth, 40),
    # ]
    
    client_config = ClientConfig(
        download_network_bandwidth=download_network_bandwidth,
        upload_network_bandwidth=upload_network_bandwidth,
        dram_bandwidth=50e9                 # 50 GB/s
    )
    network_config = NetworkConfig(
        fixed_rtt=5e-3
    )
    
    return model_config, server_configs, client_config, network_config

def main():
    """Main function to demonstrate the latency estimator"""
    model_config, server_configs, client_config, nework_config = create_configuration()
    estimator = PetalsEstimator(model_config, server_configs, client_config, nework_config)

    # test_settings = [(pow(2, i+8), 1) for i in range(9)]

    # # --- Storage stackplots (in GB) ---
    # seq_lens = [s for s,_ in test_settings]
    # batch_size = 1

    # for seq in seq_lens:
    #     _ = estimator.run_and_print(seq, batch_size)
    
    _ = estimator.run_and_print(2048, 1)
        

if __name__ == "__main__":
    main()