import matplotlib.pyplot as plt
from src.petals_estimator.config import ModelConfig, ClientConfig, NetworkConfig
from src.petals_estimator.estimator import PetalsEstimator
from src.petals_estimator.server_config_example import (
    get_2060_example, get_4060_example,
    get_4070ti_example, get_4090_example
)
from src.petals_estimator.utils import EstimateResults

# Sequence lengths to test (from 2^9 to 2^16)
TEST_SEQ_LENS = [2**(i + 9) for i in range(8)]  # [512, 1024, ..., 65536]
BATCH_SIZE = 1

# 1) Vary number of servers (all 4090)
def plot_by_server_count():
    server_counts = [8, 10, 16, 20]
    plt.figure()
    for count in server_counts:
        # build configuration
        model_cfg, _, client_cfg, network_cfg = create_base_configs()
        server_cfgs = [get_4090_example(client_cfg.download_network_bandwidth,
                                        client_cfg.upload_network_bandwidth,
                                        num_blocks=80 // count)] * count
        estimator = PetalsEstimator(model_cfg, server_cfgs, client_cfg, network_cfg)
        throughputs = []
        for seq in TEST_SEQ_LENS:
            res: EstimateResults = estimator.run(seq, BATCH_SIZE)
            throughputs.append(res.token_per_s)
        plt.plot(TEST_SEQ_LENS, throughputs, marker='o', label=f"{count} servers")
    plt.xscale('log', base=2)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Tokens/sec')
    plt.title('Decoding Throughput vs Seq Length by Server Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/throughput_by_servers.png', dpi=300)

# 2) Vary GPU type (two servers each)
def plot_by_gpu_type():
    gpu_funcs = [get_2060_example, get_4060_example, get_4070ti_example, get_4090_example]
    labels = ['RTX2060', 'RTX4060', 'RTX4070Ti', 'RTX4090']
    plt.figure()
    for fn, label in zip(gpu_funcs, labels):
        model_cfg, _, client_cfg, network_cfg = create_base_configs()
        # two servers of this GPU
        server_cfgs = [fn(client_cfg.download_network_bandwidth,
                          client_cfg.upload_network_bandwidth,
                          num_blocks=model_cfg.num_layers // 8)] * 8
        estimator = PetalsEstimator(model_cfg, server_cfgs, client_cfg, network_cfg)
        throughputs = [estimator.run(seq, BATCH_SIZE).token_per_s for seq in TEST_SEQ_LENS]
        plt.plot(TEST_SEQ_LENS, throughputs, marker='o', label=label)
    plt.xscale('log', base=2)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Tokens/sec')
    plt.title('Decoding Throughput vs Seq Length by GPU Type')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/throughput_by_gpu.png', dpi=300)

# 3) Vary RTT
def plot_by_rtt():
    rtts = [10e-3, 50e-3, 200e-3, 500e-3]
    plt.figure()
    for rtt in rtts:
        model_cfg, _, client_cfg, network_cfg = create_base_configs()
        network_cfg.fixed_rtt = rtt
        server_cfgs = [get_4090_example(client_cfg.download_network_bandwidth,
                                        client_cfg.upload_network_bandwidth,
                                        num_blocks=model_cfg.num_layers // 8)] * 8
        estimator = PetalsEstimator(model_cfg, server_cfgs, client_cfg, network_cfg)
        throughputs = [estimator.run(seq, BATCH_SIZE).token_per_s for seq in TEST_SEQ_LENS]
        plt.plot(TEST_SEQ_LENS, throughputs, marker='o', label=f"RTT={rtt*1e3:.0f}ms")
    plt.xscale('log', base=2)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Tokens/sec')
    plt.title('Decoding Throughput vs Seq Length by RTT')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/throughput_by_rtt.png', dpi=300)

# 4) Vary Network Bandwidth
def plot_by_bandwidth():
    bws = [50e6, 100e6, 200e6, 400e6]  # in B/s
    plt.figure()
    for bw in bws:
        model_cfg, _, client_cfg, network_cfg = create_base_configs()
        client_cfg.download_network_bandwidth = client_cfg.upload_network_bandwidth = bw
        server_cfgs = [get_4090_example(bw, bw, num_blocks=model_cfg.num_layers // 8)] * 8
        estimator = PetalsEstimator(model_cfg, server_cfgs, client_cfg, network_cfg)
        throughputs = [estimator.run(seq, BATCH_SIZE).token_per_s for seq in TEST_SEQ_LENS]
        plt.plot(TEST_SEQ_LENS, throughputs, marker='o', label=f"BW={bw/1e6:.0f}MB/s")
    plt.xscale('log', base=2)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Tokens/sec')
    plt.title('Decoding Throughput vs Seq Length by Network Bandwidth')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/throughput_by_bandwidth.png', dpi=300)

# Helper to create base model, client, network configurations
def create_base_configs():
    download_bw = 1e8
    upload_bw = 1e8
    model_cfg = ModelConfig(
        hidden_size=8192, num_attention_heads=64, num_key_value_heads=8,
        num_layers=80, intermediate_size=28672, vocab_size=128256,
        max_position_embeddings=131072,
        weight_precision_bytes=0.5, activation_precision_bytes=1,
    )
    client_cfg = ClientConfig(
        download_network_bandwidth=download_bw,
        upload_network_bandwidth=upload_bw,
        dram_bandwidth=50e9
    )
    network_cfg = NetworkConfig(fixed_rtt=50e-3)
    return model_cfg, None, client_cfg, network_cfg

# Main entrypoint
def main():
    plot_by_server_count()
    plot_by_gpu_type()
    plot_by_rtt()
    plot_by_bandwidth()

if __name__ == '__main__':
    main()
