# train_pq_faiss.py
# Offline: train PQ on a large set of embeddings using FAISS and export codebooks.
# Usage: python train_pq_faiss.py --embeddings embeddings.npy --m 8 --ksub 256 --out_dir ./pq_artifact

import argparse
import numpy as np
import faiss
import os

def train_and_export(emb_path, M=8, ksub=256, out_dir="pq_artifact", sample_limit=None):
    os.makedirs(out_dir, exist_ok=True)
    X = np.load(emb_path)  # shape (N, D), dtype=float32
    if sample_limit is not None and sample_limit < X.shape[0]:
        # sample for faster training
        idx = np.random.choice(X.shape[0], sample_limit, replace=False)
        train_X = X[idx]
    else:
        train_X = X

    N, D = train_X.shape
    print(f"Training PQ on {N} samples, D={D}, M={M}, ksub={ksub}")

    # FAISS IndexPQ: dimension D, M subquantizers, 8 bits => 256 centroids
    pq_index = faiss.IndexPQ(D, M, int(np.log2(ksub)))
    pq_index.train(train_X.astype(np.float32))
    print("FAISS PQ trained.")

    # save the full index (it contains codebooks)
    index_file = os.path.join(out_dir, "pq_index.faiss")
    faiss.write_index(pq_index, index_file)
    print("Saved FAISS index to", index_file)

    # extract codebooks as numpy arrays: pq_index.pq.centroids is (M, ksub, D/M)
    # In some FAISS versions, the layout is flat; use reconstruct_centroids
    codebooks = []
    subdim = D // M
    #centroids = pq_index.pq.centroids  # shape: (M*ksub*subdim) flat
    #centroids = centroids.reshape(M, ksub, subdim).copy()
    centroids = faiss.vector_to_array(pq_index.pq.centroids)  # convert to numpy array
    centroids = centroids.reshape(M, ksub, subdim).copy()
    
    for m in range(M):
        cb = centroids[m].astype(np.float32)
        codebooks.append(cb)
        np.save(os.path.join(out_dir, f"codebook_m{m}.npy"), cb)
        print(f"Saved codebook_m{m}.npy shape={cb.shape}")

    # metadata
    meta = {
        "D": int(D),
        "M": int(M),
        "ksub": int(ksub),
        "subdim": int(subdim)
    }
    np.save(os.path.join(out_dir, "pq_meta.npy"), meta)
    print("Saved metadata and codebooks.")

    return out_dir

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True, help="np .npy file with shape (N,D) float32")
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--ksub", type=int, default=256)
    p.add_argument("--out_dir", default="pq_artifact")
    p.add_argument("--sample_limit", type=int, default=None)
    args = p.parse_args()
    train_and_export(args.embeddings, M=args.m, ksub=args.ksub, out_dir=args.out_dir, sample_limit=args.sample_limit)

