import numpy as np
import time
from pq_runtime import PQRuntime   # make sure this import matches your class name/path

# ============================================================
# CONFIG
# ============================================================
N_QUERIES = 200          # total random queries
TOP_K = 5                # K nearest neighbors
D = 128                  # vector dimensionality
SEED = 42

np.random.seed(SEED)

# ============================================================
# LOAD RUNTIME (PQ + LOCAL)
# ============================================================
runtime = PQRuntime(
    pq_centroids_file="pq_artifacts/pq_centroids.npy",
    pq_codebooks_file="pq_artifacts/pq_meta.npy",
    local_codes_file="node1_local.npy",              # local embeddings
    backup_codes_file="node1_pq_backup.npy",         # remote PQ backups (combined)
    D=D
)

print("=== BENCHMARK START ===")
print(f"Queries: {N_QUERIES}, Dim: {D}, k={TOP_K}")
print()

# ============================================================
# STORAGE FOR RESULTS
# ============================================================
local_times = []
pq_times = []

# ============================================================
# RUN EXPERIMENTS
# ============================================================
for i in range(N_QUERIES):
    q = np.random.randn(D).astype(np.float32)

    # ---------------- Local Search ----------------
    t0 = time.time()
    r_local = runtime.query(q, top_k=TOP_K)
    t1 = time.time()
    local_times.append((t1 - t0) * 1000)     # ms

    # ---------------- PQ Search ----------------
    t0 = time.time()
    r_pq = runtime.query_pq(q, top_k=TOP_K)
    t1 = time.time()
    pq_times.append((t1 - t0) * 1000)         # ms

# ============================================================
# PRINT SUMMARY (copy into Excel)
# ============================================================
print("=== RESULTS (ms) ===")
print()
print("local_ms,pq_ms")

for i in range(N_QUERIES):
    print(f"{local_times[i]:.4f},{pq_times[i]:.4f}")

print("\n=== Averages ===")
print(f"avg_local_ms = {np.mean(local_times):.4f}")
print(f"avg_pq_ms     = {np.mean(pq_times):.4f}")

print("\nDONE.")

