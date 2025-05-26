from src.petals_estimator.config import ServerConfig

def get_2060_example(download_network_bandwidth: float, upload_network_bandwidth: float, num_blocks: int) -> ServerConfig:
    return ServerConfig(
        ops=100e12,                         # ~100 TOPS INT8
        ops_utilization=0.9,
        pcie_bandwidth=15.75e9,             # PCIe 3.0 ×16 ≈ 15.75 GB/s
        vram_bandwidth=336e9,               # 14 Gbps GDDR6 on 192-bit bus = 336 GB/s
        download_network_bandwidth=download_network_bandwidth,
        upload_network_bandwidth=upload_network_bandwidth,
        num_blocks=num_blocks
    )


def get_4060_example(download_network_bandwidth: float, upload_network_bandwidth: float, num_blocks: int) -> ServerConfig:
    return ServerConfig(
        ops=121e12,                        # 121 TOPS INT8
        ops_utilization=0.9,
        pcie_bandwidth=15.75e9,            # PCIe 4.0 ×8 ≈ ½ of 31.5 GB/s = 15.75 GB/s
        vram_bandwidth=272e9,              # 17 Gbps GDDR6 on 128-bit bus = 272 GB/s
        download_network_bandwidth=download_network_bandwidth,
        upload_network_bandwidth=upload_network_bandwidth,
        num_blocks=num_blocks
    )

    
def get_4070ti_example(download_network_bandwidth: float, upload_network_bandwidth: float, num_blocks: int) -> ServerConfig:
    return ServerConfig(
        ops=641e12,                        # 641 TOPS INT8
        ops_utilization=0.9,
        pcie_bandwidth=31.5e9,             # PCIe 4.0 ×16 ≈ 31.5 GB/s
        vram_bandwidth=504.2e9,            # 21 Gbps GDDR6X on 192-bit bus = 504.2 GB/s
        download_network_bandwidth=download_network_bandwidth,
        upload_network_bandwidth=upload_network_bandwidth,
        num_blocks=num_blocks
    )
    

def get_4090_example(download_network_bandwidth: float, upload_network_bandwidth: float, num_blocks: int) -> ServerConfig:
    return ServerConfig(
        ops=1321e12,                       # 1321 TOPS INT8 (with sparsity)
        ops_utilization=0.9,
        pcie_bandwidth=31.5e9,             # PCIe 4.0 ×16 ≈ 31.5 GB/s
        vram_bandwidth=1.01e12,            # 21 Gbps GDDR6X on 384-bit bus = 1.01 TB/s
        download_network_bandwidth=download_network_bandwidth,
        upload_network_bandwidth=upload_network_bandwidth,
        num_blocks=num_blocks
    )
