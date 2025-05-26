from dataclasses import dataclass

@dataclass
class EstimateResults:
    seq_len: int
    token_per_s: float

def human_readable(n_bytes: float) -> str:
    """Convert a byte count into a human-readable string."""
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while n_bytes >= 1024 and i < len(suffixes)-1:
        n_bytes /= 1024.
        i += 1
    return f"{n_bytes:.2f} {suffixes[i]}"

def pretty_print_storage(client_storage: dict[str, float],
                         server_storage: list[dict[str, float]]) -> None:
    # Client
    print("Client Storage:")
    max_key_len = max(len(k) for k in client_storage)
    for name, val in client_storage.items():
        print(f"  {name.ljust(max_key_len)} : {human_readable(val)}")
    print()

    # Servers
    print("Server Storage:")
    # header
    headers = ["Server", "Total Weights (all layers)", "KV Cache"]
    print(f"  {headers[0]:<6}  {headers[1]:<25}  {headers[2]:<15}")
    print("  " + "-"*(6 + 2 + 25 + 2 + 15))
    total_weights_bytes = 0
    total_kvcache_bytes = 0
    for idx, srv in enumerate(server_storage, start=1):
        total_weights_bytes += srv["total_weights(all_layers)"]
        total_kvcache_bytes += srv["kvcache"] 
        w = human_readable(srv["total_weights(all_layers)"])
        kv = human_readable(srv["kvcache"])
        print(f"  {idx:<6}  {w:<25}  {kv:<15}")
    w = human_readable(total_weights_bytes)
    kv = human_readable(total_kvcache_bytes)
    print(f"  Total  {w:<25}  {kv:<15}")
    print()