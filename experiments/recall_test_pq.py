import numpy as np
import faiss
import time

# ============================================================
# Parameters
# ============================================================
D = 128
N = 50000       # database size
Q = 200        # number of queries
ksub = 256     # 8-bit PQ
M_values = [2, 4, 8, 16, 32, 64]
K_values = [1, 5, 10, 20, 50, 100]

# ============================================================
# Generate synthetic data
# ============================================================
np.random.seed(42)
db_vectors = np.random.randn(N, D).astype(np.float32)
query_vectors = np.random.randn(Q, D).astype(np.float32)

# Normalize for cosine-sim-like behavior
db_vectors /= np.linalg.norm(db_vectors, axis=1, keepdims=True) + 1e-9
query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-9

# ============================================================
# Compute EXACT top-k (ground truth)
# ============================================================
print("Computing exact L2 neighbors for ground truth...")

index_exact = faiss.IndexFlatL2(D)
index_exact.add(db_vectors)

ground_truth = {}
for k in K_values:
    _, gt_idx = index_exact.search(query_vectors, k)
    ground_truth[k] = gt_idx


# ============================================================
# PQ Recall Testing
# ============================================================
def train_and_encode_PQ(x, M, ksub):
    """Train PQ with M subquantizers and encode x into PQ codes."""
    D = x.shape[1]
    pq = faiss.ProductQuantizer(D, M, int(np.log2(ksub)))
    pq.train(x)     # train codebooks
    codes = pq.compute_codes(x)  # encode database
    return pq, codes


def pq_search(pq, codes, queries, M, k):
    """Perform ADC PQ search for k neighbors."""
    D = queries.shape[1]
    N = codes.shape[0]

    # Convert FAISS centroid storage into numpy array
    centroids = faiss.vector_to_array(pq.centroids)
    centroids = centroids.reshape(pq.M, pq.ksub, pq.dsub)  # correct shape

    codebooks = centroids
    cb_norms = np.sum(codebooks ** 2, axis=2)

    distances = np.zeros((queries.shape[0], N), dtype=np.float32)

    for qi, q in enumerate(queries):
        lookup = np.zeros((M, pq.ksub), dtype=np.float32)

        for m in range(M):
            qsub = q[m * pq.dsub : (m + 1) * pq.dsub]
            dot = codebooks[m] @ qsub
            qsq = np.sum(qsub * qsub)
            lookup[m] = qsq - 2 * dot + cb_norms[m]

        # accumulate distances
        dist = np.zeros(N, dtype=np.float32)
        for m in range(M):
            dist += lookup[m][codes[:, m]]

        # top-k
        idx = np.argpartition(dist, k)[:k]
        sorted_idx = idx[np.argsort(dist[idx])]
        distances[qi] = dist

    return distances


# ============================================================
# Main experiment loop
# ============================================================
print("\n==== PQ Recall Benchmark ====\n")

results = {}  # store recall for each M

for M in M_values:
    print(f"\n======================")
    print(f"Testing M = {M}")
    print(f"======================")

    start = time.time()
    pq, codes = train_and_encode_PQ(db_vectors, M, ksub)
    train_time = time.time() - start

    print(f"PQ trained and encoded in {train_time:.3f}s")

    # Run PQ search
    distances = pq_search(pq, codes, query_vectors, M, ksub)

    # Compute recall for all k values
    recall_per_k = {}

    for k in K_values:
        matches = 0

        for qi in range(Q):
            # PQ predicted top-k
            pq_idx = np.argpartition(distances[qi], k)[:k]
            pq_idx = pq_idx[np.argsort(distances[qi][pq_idx])]

            # ground truth exact top-k
            gt_idx = ground_truth[k][qi]

            # overlap count
            matches += len(set(pq_idx).intersection(set(gt_idx)))

        recall = matches / (Q * k)
        recall_per_k[k] = recall

        print(f"Recall@{k}: {recall:.4f}")

    results[M] = recall_per_k

# ============================================================
# Final summary (copy this into Excel)
# ============================================================
print("\n\n===== FINAL SUMMARY TABLE =====")
print("M,", ",".join([f"Recall@{k}" for k in K_values]))

for M in M_values:
    row = [f"{results[M][k]:.4f}" for k in K_values]
    print(f"{M}, " + ", ".join(row))

