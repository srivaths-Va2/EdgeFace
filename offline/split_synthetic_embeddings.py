import numpy as np

# ---- CONFIG ----
MASTER_FILE = "face_embeddings.npy"    # your file with shape (12000, 128)
N_NODES = 3
EMB_PER_NODE = 4000                   # since 12000 / 3 = 4000
OUT_PREFIX = "node"

# ---- LOAD MASTER EMBEDDINGS ----
embeddings = np.load(MASTER_FILE)
print("Loaded embeddings:", embeddings.shape)

# ---- SAFETY CHECK ----
assert embeddings.shape[0] == N_NODES * EMB_PER_NODE, \
    f"Expected {N_NODES * EMB_PER_NODE} embeddings but found {embeddings.shape[0]}"

# ---- SPLIT INTO THREE FILES ----
for node_id in range(1, N_NODES + 1):
    start = (node_id - 1) * EMB_PER_NODE
    end   = node_id * EMB_PER_NODE
    chunk = embeddings[start:end]

    out_file = f"{OUT_PREFIX}{node_id}_local.npy"
    np.save(out_file, chunk)
    print(f"Saved {chunk.shape} → {out_file}")

