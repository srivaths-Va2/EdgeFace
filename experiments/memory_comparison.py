import numpy as np

# ======================================================
# Memory Comparison Experiment: Local FP32 vs PQ Codes
# ======================================================

if __name__ == "__main__":

    N = 4000         # number of embeddings
    D = 128          # dimension
    bytes_fp32 = 4   # size of float32
    
    # Raw local storage
    local_storage_bytes = N * D * bytes_fp32
    local_storage_kb = local_storage_bytes / 1024
    local_storage_mb = local_storage_kb / 1024
    
    print("\n=== MEMORY COMPARISON: LOCAL FP32 vs PQ ===\n")
    print(f"Local FP32 embeddings = {local_storage_mb:.2f} MB ({local_storage_kb:.1f} KB)")
    
    # PQ parameters to test
    M_values = [4, 8, 16, 32, 64]
    
    print("\nM, PQ Storage (KB), Compression Ratio\n")
    for M in M_values:
        pq_bytes = N * M * 1          # 1 byte per code index
        pq_kb = pq_bytes / 1024
        compression = local_storage_bytes / pq_bytes
        
        print(f"{M:2d}, {pq_kb:10.2f} KB, {compression:8.1f}× smaller")

    print("\nDone.\n")

