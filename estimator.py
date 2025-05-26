import random
from collections import defaultdict

from config import ModelConfig, ServerConfig, ClientConfig, NetworkConfig
from utils import pretty_print_storage, EstimateResults

class PetalsEstimator:
    def __init__(self, model_config: ModelConfig, server_configs: list[ServerConfig], client_config: ClientConfig, network_config: NetworkConfig):
        # This estimator follows the implementation from [https://arxiv.org/abs/2312.08361](https://arxiv.org/abs/2312.08361), 
        # specifically adhering to the design of Algorithms 1 and 2. It does not address failed server replacement and 
        # currently focuses on a single-client perspective rather than overall decentralized ystem throughput.
        # Ensure blocks sum to layers
        assert sum(s.num_blocks for s in server_configs) == model_config.num_layers, (
            f"Total server blocks ({sum(s.num_blocks for s in server_configs)}) "
            f"!= model layers ({model_config.num_layers})"
        )
        
        self.model_config = model_config
        self.server_configs = server_configs
        self.client_config = client_config
        self.network_config = network_config
    
    def _get_random_rtt(self):
        # 1. 
        # return random.uniform(50e-3, 300e-3)
        # 2. triangular with a fat tail for satellites
        # return random.triangular(20e-3, 550e-3, 100e-3)
        # 3. fix
        return self.network_config.fixed_rtt
    
    def _get_one_transformer_block_datamovements_for_decoding_phase_with_kvcache(self, seq_len: int, batch_size: int) -> float:
        """
        1. load past kv cache
        2. load weight 
        3. store current kv cache
        """
        # need to read the past kv cache from cache, i currently assume the kvcache is in GPU, if want to assume kvcache is stored on CPU, then i need to modify here by adding pcie transfer overhead
        past_kvcache_bytes = 2 * batch_size * (seq_len - 1) * self.model_config.num_key_value_heads * self.model_config.head_dim * self.model_config.activation_precision_bytes  # 2(k+v) * batch size * past sequence length * projection result tensor 
        # weights
        layernorm_weight_elements = 2 * self.model_config.hidden_size
        mlp_weight_elements = 3 * self.model_config.hidden_size * self.model_config.intermediate_size
        attn_kv_proj_weight_elements = 2 * self.model_config.num_key_value_heads * self.model_config.head_dim * self.model_config.hidden_size
        attn_qo_proj_weight_elements = 2 * self.model_config.hidden_size * self.model_config.hidden_size
        total_weight_bytes = (layernorm_weight_elements + mlp_weight_elements + attn_kv_proj_weight_elements + attn_qo_proj_weight_elements) * self.model_config.weight_precision_bytes
        # store kvcache into vram
        current_kvcache_bytes = 2 * batch_size * seq_len * self.model_config.num_key_value_heads * self.model_config.head_dim * self.model_config.activation_precision_bytes  # 2(k+v) * batch size * past sequence length * projection result tensor 
        return past_kvcache_bytes + total_weight_bytes + current_kvcache_bytes
        
    def _get_one_transformer_block_ops_for_decoding_phase_with_kvcache(self, seq_len: int, batch_size: int) -> float:
        """
        Calculate OPs for one transformer block during decoding with KV‐cache.
        Assumes we're processing one new token with a past context of length (seq_len - 1).
        GEMM/GEMV flops counted as 2*m*n*k (multiply + add).
        """
        config = self.model_config
        head_dim = config.head_dim
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        # Number of past tokens in KV cache
        past_len = max(seq_len - 1, 0)

        # 1. Projections for the new token
        # Q projection: hidden_size × hidden_size GEMV
        q_proj_ops = 2 * batch_size * hidden_size * hidden_size
        # K and V projections (separate heads): hidden_size × (num_kv_heads * head_dim)
        k_proj_ops = 2 * batch_size * hidden_size * (num_kv_heads * head_dim)
        v_proj_ops = 2 * batch_size * hidden_size * (num_kv_heads * head_dim)

        # 2. Attention
        # 2a) Attention scores: Q · Kᵀ over past tokens
        attn_scores_ops = 2 * batch_size * num_heads * past_len * head_dim
        # 2b) Softmax over each head’s score vector
        softmax_ops = 2 * batch_size * num_heads * past_len
        # 2c) Weighted sum: Attention weights · V
        attn_v_ops = 2 * batch_size * num_heads * past_len * head_dim
        # 2d) Output projection: concat heads back to hidden_size
        o_proj_ops = 2 * batch_size * hidden_size * hidden_size

        # 3. SwiGLU MLP
        # Gate & up projections: two hidden_size→intermediate_size GEMVs
        gate_proj_ops = 2 * batch_size * hidden_size * intermediate_size
        up_proj_ops   = 2 * batch_size * hidden_size * intermediate_size
        # Elementwise gate multiplication + activation (approx 1 flop each)
        gate_ops       = batch_size * intermediate_size
        activation_ops = batch_size * intermediate_size
        # Down projection: intermediate_size→hidden_size GEMV
        down_proj_ops = 2 * batch_size * intermediate_size * hidden_size
        mlp_ops = (gate_proj_ops + up_proj_ops + gate_ops + activation_ops + down_proj_ops)

        # 4. LayerNorms (two per block)
        # Approx ~5 flops per element (mean, var, scale, shift)
        ln_ops = 5 * 2 * batch_size * hidden_size

        total_ops = (
            q_proj_ops
        + k_proj_ops
        + v_proj_ops
        + attn_scores_ops
        + softmax_ops
        + attn_v_ops
        + o_proj_ops
        + mlp_ops
        + ln_ops
        )
        return total_ops

    def estimate_total_storage_for_client_and_servers(self, seq_len: int, batch_size: int):
        """
        Estimate the total storage requirements (in bytes) for the client and each server.

        1. The client must hold its own model weights (embedding + LM head) plus an inputs cache for all servers. 
        2. Each server must hold its transformer-block weights and a KV-cache for the current decoding batch.
        """
        
        client_storage = {
            "embedding_layer_weights": self.model_config.vocab_size * self.model_config.hidden_size * self.model_config.weight_precision_bytes,
            "lm_head_layer_weights": self.model_config.vocab_size * self.model_config.hidden_size * self.model_config.weight_precision_bytes,
            "inputs_cache": len(self.server_configs) * seq_len * (batch_size * 1 * self.model_config.hidden_size * self.model_config.activation_precision_bytes)
        }
        server_storage = []
        
        for server in self.server_configs:
            
            layernorm_weight_elements = 2 * self.model_config.hidden_size
            mlp_weight_elements = 3 * self.model_config.hidden_size * self.model_config.intermediate_size
            attn_kv_proj_weight_elements = 2 * self.model_config.num_key_value_heads * self.model_config.head_dim * self.model_config.hidden_size
            attn_qo_proj_weight_elements = 2 * self.model_config.hidden_size * self.model_config.hidden_size
            total_weight_bytes = (layernorm_weight_elements + mlp_weight_elements + attn_kv_proj_weight_elements + attn_qo_proj_weight_elements) * self.model_config.weight_precision_bytes
            total_weight_bytes *= server.num_blocks
            
            kvcache_bytes = 2 * batch_size * seq_len * self.model_config.num_key_value_heads * self.model_config.head_dim * self.model_config.activation_precision_bytes
            kvcache_bytes *= server.num_blocks
        
            server_storage.append({
                    "total_weights(all_layers)":  total_weight_bytes,
                    "kvcache": kvcache_bytes,
                })
            
        return client_storage, server_storage
        
    
    def estimate_total_latency(self, seq_len: int, batch_size: int) -> dict[str, float]:
        """
        Estimate the total latency for one forward pass during the decoding phase from the client's perspective.

        Client Perspective:
        1. inputs = embeddings(next_token)  # According to the Petals source code, client-side computations are performed on the CPU by default.
        2. WHILE iterating through all servers:
            a. outputs = server.send(inputs)
            b. cache[server].append(inputs)  # Client caches inputs for fault tolerance
            c. inputs = outputs
        3. logits = compute_logits(outputs, embeddings) # According to the Petals source code, client-side computations are performed on the CPU by default.

        Server Perspective (per server):
        1. inputs = stream.receive()    
        # This step overlaps with steps 3, and Client Perspective 2.a.
        # Communication occurs concurrently — outputs is sent to the client and next server simultaneously (refer to Section 3.2 in the paper).
        2. FOR each local layer:
            a. past_kv = cache[layer]
            b. inputs, new_kv = forward(layer, inputs, past_kv)
            c. cache[layer].append(new_kv)
        3. stream.send(inputs)
        """

        latency_components = defaultdict(int)
        
        embedding_lookup_latency = 0    # The embedding lookup latency is nearly zero because it only needs to retrieve the embedding vector for "one token" and doesn't require any computation.
        latency_components["(memory)embeddinig_lookup_latency"] += embedding_lookup_latency

        for i, server in enumerate(self.server_configs):
            inputs_bytes = batch_size * 1 * self.model_config.hidden_size * self.model_config.activation_precision_bytes # Due to the KV Cache, the inputs size is just one token's hidden vector
            if i == 0: # On the first server, communication time includes transmission from client to server
                comm_bw = min(self.client_config.upload_network_bandwidth, server.download_network_bandwidth)   # Use the lower bandwidth between client upload and server download
                inputs_comm_latency = inputs_bytes / comm_bw
                inputs_comm_latency += self._get_random_rtt()
                latency_components["(comm)first_server_inputs_comm_latency"] += inputs_comm_latency
            else:   # On the other server, the communication happen here is from previous server to next server and also previous server to client
                previous_server = self.server_configs[i - 1]
                previous_server_bw = previous_server.upload_network_bandwidth / 2 # This needs to be divided by two because the data is sent to both the client and the next server simultaneously
                comm_bw = min(server.download_network_bandwidth, previous_server_bw)
                inputs_comm_latency = inputs_bytes / comm_bw
                inputs_comm_latency += self._get_random_rtt()
                latency_components["(comm)total_server_to_server_inputs_comm_latency"] += inputs_comm_latency
                # The below is happen at the same time for Client but because it can happen async in the background and the major inference forward process is still happen because the inputs for next server is already send from previous server to next server
                # a. received "outputs" from previous server
                # b. cache the previous server inputs
            
            # after the server receive the inputs, it should transfer the inputs from cpu to gpu
            # TODO: need verified: if petals use GPU RDMA to directly transfer data into GPU VRAM or not?
            inputs_cpu_to_gpu_latency = inputs_bytes / server.pcie_bandwidth
            latency_components["(memory)pcie_overhead"] += inputs_cpu_to_gpu_latency
            
            for _ in range(server.num_blocks):  # iterate all local layers in each server     
                # to estimate the forward latency, we estimate from 2 perspective (memory and compute) and based on each theoratical latency to know it is comptue bound or memory bound for this server
                # first is the memory latency estimation, when the GPU doing computation, it needs to move the input from VRAM into register, and store output from register to input, if the operation is memory-bound, the total latency should bound at (total_datamovements/vram bandwidth)
                # next is the compute latency estimation, ....
                datamovement_bytes = self._get_one_transformer_block_datamovements_for_decoding_phase_with_kvcache(seq_len, batch_size)
                datamovement_latency = datamovement_bytes / server.vram_bandwidth
                compute_ops = self._get_one_transformer_block_ops_for_decoding_phase_with_kvcache(seq_len, batch_size)
                compute_latency = compute_ops / (server.ops * server.ops_utilization)
                
                if datamovement_latency >= compute_latency: # memory bound
                    forward_latency = datamovement_latency
                else:   # compute bound
                    forward_latency = compute_latency
                latency_components["(compute&memory)transformer_block_forward_latency"] += forward_latency
            
            outputs_bytes = batch_size * 1 * self.model_config.hidden_size * self.model_config.activation_precision_bytes
            # after the server finished computations, it should transfer the outputs from gpu to cpu 
            outpus_gpu_to_cpu_latency = outputs_bytes / server.pcie_bandwidth
            latency_components["(memory)pcie_overhead"] += outpus_gpu_to_cpu_latency
            
            # After finished all computation, send the outputs to client and next server, since "server to server" communication latency already be estimated above, here we only consider the last server scenario, which is the last server need to transfer the outputs back to client
            if i == len(self.server_configs) - 1:
                comm_bw = min(self.client_config.upload_network_bandwidth, server.download_network_bandwidth)   # Use the lower bandwidth between server upload and client download
                outputs_comm_latency = outputs_bytes / comm_bw
                outputs_comm_latency += self._get_random_rtt()
                latency_components["(comm)last_server_outputs_comm_latency"] += inputs_comm_latency

        compute_logits_latency = 0 
        latency_components["(compute&memory)compute_logits_latency"] += compute_logits_latency
        # logits = compute_logits(outputs, embeddings)
        # This part should also follow the memory-bound/compute-bound estimation types 
        # to estimate the forward pass that computes the lm_head for outputs, generates the logits, 
        # calculates the argmax, etc.
        # However, for now, i simplify the process and assume the latency is zero. 
        # While this may not be a bottleneck, i leave room for a more rigorous estimation in the future.
        
        return latency_components
    
    def run_and_print(self, seq_len: int, batch_size: int) -> EstimateResults:
        print("\nPetals Distributed Inference Latency Estimation\n" + "="*60)
        print(f"Model:  Hidden Size = {self.model_config.hidden_size}")
        print(f"Layers: {self.model_config.num_layers}  |  Heads: {self.model_config.num_attention_heads}\n")

        results = self.estimate_total_latency(seq_len, batch_size)
        total_latency = sum(results.values())
        current_token_per_seconds = 1 / total_latency
        print(f"Config: context={seq_len:<4}  batch={batch_size:<2} -> Total latency: {total_latency*1000:6.2f} ms | Throughput (token/s): {current_token_per_seconds}")
        print("Component-wise breakdown:")
        print(f"{'Component':50s} {'Latency (ms)':>12s}")
        print("-"*65)
        for comp, lat in sorted(results.items(), key=lambda item: item[1], reverse=True):
            print(f"{comp:50s} {lat*1000:12.2f}")
        print("Storage breakdown:")
        print("-"*65)
        client_storage, server_storage = self.estimate_total_storage_for_client_and_servers(seq_len, batch_size)
        pretty_print_storage(client_storage=client_storage, server_storage=server_storage)
        print()
        
        return EstimateResults(
            seq_len=seq_len,
            token_per_s=current_token_per_seconds
        )