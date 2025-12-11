import numpy as np
import time

# --------------------------
# CONFIGURATION
# --------------------------
RTT_MS = 25.0                 # WiFi round-trip time
LOCAL_COMPUTE_MS = 0.5        # PQ or local search time on each node
SERVER_COMPUTE_MS = 0.5       # server processing time per query
CONCURRENCY_LEVELS = [1, 5, 10, 20, 50, 100, 200, 400]

def simulate_distributed(concurrency):
    """
    Distributed system: each node handles its own queries.
    No network, no queueing. Time = local compute only.
    """
    return LOCAL_COMPUTE_MS

def simulate_centralized(concurrency):
    """
    Central server model:
    - Every query incurs network RTT
    - Server processes queries in a queue
    - Queueing delay = (concurrency - 1) * server_compute_time
    """
    network_delay = RTT_MS
    queue_delay = (concurrency - 1) * SERVER_COMPUTE_MS
    compute = SERVER_COMPUTE_MS
    return network_delay + queue_delay + compute

# --------------------------
# RUN EXPERIMENT
# --------------------------
print("\n===== Load Distribution vs Centralized Server =====\n")
print("Concurrency, Distributed(ms), Centralized(ms)")

distributed_results = []
central_results = []

for c in CONCURRENCY_LEVELS:
    lat_d = simulate_distributed(c)
    lat_c = simulate_centralized(c)

    distributed_results.append(lat_d)
    central_results.append(lat_c)

    print(f"{c}, {lat_d:.3f}, {lat_c:.3f}")

# --------------------------
# FINAL SUMMARY
# --------------------------
print("\n===== SUMMARY TABLE =====")
print("Concurrency,Dist(ms),Central(ms)")

for c, d, s in zip(CONCURRENCY_LEVELS, distributed_results, central_results):
    print(f"{c},{d:.3f},{s:.3f}")

