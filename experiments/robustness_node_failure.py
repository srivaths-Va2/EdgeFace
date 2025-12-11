import numpy as np
import time
import faiss

# -------------------------------------------------------
# Helper: compute exact neighbors (L2)
# -------------------------------------------------------
def exact_knn(db, queries, k):
    index = faiss.IndexFlatL2(db.shape[1])
    index.add(db)
    distances, idx = index.search(queries, k)
    return idx


# -------------------------------------------------------
# Helper: train PQ and encode full DB
# -------------------------------------------------------
def train_pq(db_vectors, M=8, ksub=256):
    D = db_vectors.shape[1]
    pq = faiss.ProductQuantizer(D, M, int(np.log2(ksub)))

    # train PQ
    pq.train(db_vectors)

    # encode
    codes = np.zeros((db_vectors.shape[0], M), dtype=np.uint8)
    for i in range(db_vectors.shape[0]):
        codes[i, :] = pq.compute_codes(db_vectors[i:i+1]).ravel()

    return pq, codes


def pq_amortized_lookup(pq, codes, queries, M, ksub):
    D = queries.shape[1]
    Nq = queries.shape[0]
    Ndb = codes.shape[0]

    # properly convert FAISS PQ centroids
    codebooks_np = faiss.vector_to_array(pq.centroids).reshape(M, ksub, D // M)
    codebooks = codebooks_np
    cb_sq = np.sum(codebooks ** 2, axis=2)  # (M, ksub)

    results = []
    start_time = time.time()

    for q in queries:
        lookup = np.zeros((M, ksub), dtype=np.float32)

        for m in range(M):
            qsub = q[m*(D//M):(m+1)*(D//M)]
            lookup[m] = np.sum((codebooks[m] - qsub) ** 2, axis=1)

        dist = np.zeros(Ndb, dtype=np.float32)
        for m in range(M):
            dist += lookup[m][codes[:, m]]

        idx = np.argsort(dist)[:5]
        results.append(idx)

    latency = (time.time() - start_time) / Nq * 1000
    return np.array(results), latency



# -------------------------------------------------------
# Experiment parameters
# -------------------------------------------------------
D = 128
N_per_node = 4000
N_queries = 200
Ks = 5

np.random.seed(0)

# synthetic data for 3 nodes (total = 12k embeddings)
node1 = np.random.randn(N_per_node, D).astype(np.float32)
node2 = np.random.randn(N_per_node, D).astype(np.float32)
node3 = np.random.randn(N_per_node, D).astype(np.float32)

# queries drawn from node2 to test robustness
queries = node2[np.random.choice(N_per_node, N_queries, replace=False)]

# full DB for baseline (before failure)
full_db = np.vstack([node1, node2, node3])


# -------------------------------------------------------
# Baseline: exact recall before failure
# -------------------------------------------------------
print("\n===== BASELINE (Before Node Failure) =====")

# exact nearest neighbors
gt_idx = exact_knn(full_db, queries, Ks)

# measure baseline latency
start = time.time()
_ = exact_knn(node1, queries, Ks)
local_latency_ms = (time.time() - start) / N_queries * 1000

print(f"Local exact latency (no failure): {local_latency_ms:.3f} ms/query")


# -------------------------------------------------------
# After Failure: Node1 must use PQ for Node2 + Node3
# -------------------------------------------------------
print("\n===== AFTER NODE2 FAILURE — PQ BACKUPS =====")

# train PQ on node2 + node3 only (simulate PQ backups)
remote_db = np.vstack([node2, node3])

M = 8
ksub = 256

pq, remote_codes = train_pq(remote_db, M=M, ksub=ksub)

# PQ search
pq_results, pq_latency_ms = pq_amortized_lookup(pq, remote_codes, queries, M, ksub)

print(f"PQ latency (after failure): {pq_latency_ms:.3f} ms/query")


# -------------------------------------------------------
# Compute Recall@5 after failure
# -------------------------------------------------------
# convert indices to ids relative to full_db
offset_node2 = N_per_node        # node2 starts at 4000
offset_node3 = 2 * N_per_node    # node3 starts at 8000

# Build a mapping: remote_db index → full_db index
remote_to_full = np.arange(len(remote_db))
remote_to_full[:N_per_node] += offset_node2       # node2 block
remote_to_full[N_per_node:] += offset_node3 - N_per_node

pq_full_indices = remote_to_full[pq_results]

# Compute recall
recall = 0
for i in range(N_queries):
    gt_set = set(gt_idx[i])
    pq_set = set(pq_full_indices[i])
    recall += len(gt_set.intersection(pq_set)) / Ks

recall /= N_queries

print(f"Recall@5 after failure: {recall:.4f}")


# -------------------------------------------------------
# Print Summary
# -------------------------------------------------------
print("\n===== FINAL SUMMARY =====")
print(f"Local exact latency (before failure): {local_latency_ms:.3f} ms/query")
print(f"PQ lookup latency (after failure):    {pq_latency_ms:.3f} ms/query")
print(f"Recall@5 after failure:               {recall:.4f}")

