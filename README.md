# Petals-Estimator

A Python toolkit for estimating latency, throughput, and storage requirements of distributed LLM inference using the [Petals](https://github.com/bigscience-workshop/petals/) approach. This project enables researchers and engineers to analyze performance trade‑offs across different hardware configurations, network conditions, and model parameters.

## Features

* **Latency & Throughput Estimation**: Generate plots showing decoding throughput versus sequence length by varying server count, GPU type, RTT, and network bandwidth.
* **Storage Breakdown**: Visualize client and server storage requirements for model weights and KV caches.
* **Configurable**: Easily customize model, server, client, and network parameters via dataclasses in `src/petals_estimator/config.py`.
* **Reusable Examples**: Predefined GPU configurations for common cards (RTX2060, RTX4060, RTX4070Ti, RTX4090) in `server_config_example.py`.

## Prerequisites

* Python 3.9 or higher
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Directory Structure

```plaintext
dandinpower-petals-estimator/
├── README.md                       # This file
├── latency_estimation_main.py      # Plot throughput vs. sequence length
├── storage_estimation_main.py      # Plot storage breakdown stackplots
├── main.py                         # CLI demo printing latency breakdowns
├── requirements.txt                # Python package requirements
├── figures/                        # Output folder for generated plots
└── src/
    └── petals_estimator/
        ├── __init__.py
        ├── config.py               # Dataclasses for model/server/client/network configs
        ├── estimator.py            # Core logic for latency & storage estimation
        ├── server_config_example.py# Helper functions for standard GPU setups
        └── utils.py                # Utilities: result dataclass & pretty-printing
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/DandinPower/petals-estimator.git
   cd petals-estimator
   ```
2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Latency & Throughput Estimation

Generate decoding throughput plots for various scenarios:

```bash
python latency_estimation_main.py
```

Plots will be saved under the `figures/` directory:

* `theoratical_throughput_by_servers.png`
* `throughput_by_servers.png`
* `throughput_by_gpu.png`
* `throughput_by_rtt.png`
* `theoratical_throughput_by_rtt.png`
* `throughput_by_bandwidth.png`

### 2. Storage Estimation

Generate storage stackplots showing client vs. server storage breakdowns:

```bash
python storage_estimation_main.py
```

Outputs:

* `client_storage_stackplot.png`
* `server_storage_stackplot.png`

### 3. Interactive CLI Demo

Run a simple demo printing latency breakdowns in the console:

```bash
python main.py
```

## Configuration

* **ModelConfig**: Edit `src/petals_estimator/config.py` to adjust hidden size, number of layers, precision, etc.
* **ServerConfig Examples**: In `src/petals_estimator/server_config_example.py`, customize `get_*_example` functions or add new GPU profiles.
* **Client & Network**: Modify `ClientConfig` and `NetworkConfig` parameters in each script's `create_configuration()`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
