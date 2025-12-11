import numpy as np
import time
import faiss

# ================================================================
# Utility: cosine similarity for local search
# ================================================================
def cosine_similarity_search(query, db_vectors):
    qn = query / (np.linalg.norm(query) + 1e-9)
    dn = db_vectors / (np.linalg.norm(db_vectors, axis=1, keepdims=True) + 1e-9)
    sims = dn @ qn
    best_idx = np.argmax(sims)
    return best_idx, sims[best_idx]

# ================================================================
# PQ search (ADC)
# ================================================================
def pq_adc_search(pq, codes, query, M, ksub):
    # lookup table creation
    D = len(query)
    subdim = D // M
    query = query.astype(np.float32)
    
    # get centroids from FAISS PQ object
    centroids = faiss.vector_to_array(pq.centroids).reshape(M, ksub, subdim)

    lookup = np.zeros((M, ksub), dtype=np.float32)
    for m in range(M):
        qsub = query[m*subdim:(m+1)*subdim]
        c = centroids[m]
        lookup[m] = np.sum((c - qsub)**2, axis=1)

    # compute PQ distance
    dists = np.zeros(codes.shape[0], dtype=np.float32)
    for m in range(M):
        dists += lookup[m][codes[:, m]]

    idx = np.argmin(dists)
    return idx, dists[idx]

# ================================================================
# Benchmark helper
# ================================================================
def measure_time(func, repeat=50):
    times = []
    for _ in range(repeat):
        t0 = time.time()
        func()
        times.append((time.time() - t0) * 1000)   # ms
    return {
        "mean": np.mean(times),
        "median": np.median(times),
        "min": np.min(times),
        "max": np.max(times)
    }

# ================================================================
# Main Experiment
# ================================================================
if __name__ == "__main__":
    D = 128
    M = 8
    ksub = 256
    local_size = 4000
    remote_size = 8000
    total = local_size + remote_size

    print("\n=== Distributed Query Time Benchmark ===")

    # --------------------------------------------
    # Step 1 - create synthetic embeddings
    # --------------------------------------------
    np.random.seed(42)
    local_db = np.random.randn(local_size, D).astype(np.float32)
    remote_db = np.random.randn(remote_size, D).astype(np.float32)

    # --------------------------------------------
    # Step 2 - train PQ for backups
    # --------------------------------------------
    print("Training PQ...")
    pq = faiss.ProductQuantizer(D, M, int(np.log2(ksub)))
    pq.train(remote_db)

    codes = pq.compute_codes(remote_db)

    print("PQ training complete. Remote codes shape:", codes.shape)

    # --------------------------------------------
    # Benchmarking Functions
    # --------------------------------------------
    test_queries = np.random.randn(100, D).astype(np.float32)

    # 1) Local search time
    def local_test():
        q = test_queries[np.random.randint(0, len(test_queries))]
        cosine_similarity_search(q, local_db)

    # 2) PQ search time
    def pq_test():
        q = test_queries[np.random.randint(0, len(test_queries))]
        pq_adc_search(pq, codes, q, M, ksub)

    # 3) Hybrid (local → PQ)
    def hybrid_test():
        q = test_queries[np.random.randint(0, len(test_queries))]
        best_idx, score = cosine_similarity_search(q, local_db)
        # assume threshold = 0.80
        if score < 0.80:
            pq_adc_search(pq, codes, q, M, ksub)

    # 4) Central server (12k exact search + network RTT)
    RTT_ms = 40  # assume realistic WiFi/campus RTT
    full_db = np.vstack([local_db, remote_db])

    def central_server_test():
        q = test_queries[np.random.randint(0, len(test_queries))]
        cosine_similarity_search(q, full_db)
        time.sleep(RTT_ms / 1000.0)

    # --------------------------------------------
    # Run benchmarks
    # --------------------------------------------
    print("\nRunning timing benchmarks...\n")

    t_local = measure_time(local_test)
    t_pq = measure_time(pq_test)
    t_hybrid = measure_time(hybrid_test)
    t_central = measure_time(central_server_test)

    # --------------------------------------------
    # Print results
    # --------------------------------------------
    def show(name, t):
        print(f"{name:20s} | mean={t['mean']:.3f} ms | median={t['median']:.3f} | min={t['min']:.3f} | max={t['max']:.3f}")

    print("===== RESULTS =====")
    show("Local Search", t_local)
    show("PQ Search", t_pq)
    show("Hybrid Search", t_hybrid)
    show("Central Server", t_central)

    print("\nDone.")

