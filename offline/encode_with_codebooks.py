# encode_with_codebooks.py
# Pure-NumPy PQ encoder using exported codebooks.
# Usage: python encode_with_codebooks.py --in emb.npy --codebook_dir pq_artifact --out codes.npy

import argparse
import numpy as np
import os

def load_codebooks(codebook_dir):
    meta = np.load(os.path.join(codebook_dir, "pq_meta.npy"), allow_pickle=True).item()
    M = meta["M"]; ksub = meta["ksub"]; subdim = meta["subdim"]; D = meta["D"]
    codebooks = []
    for m in range(M):
        cb = np.load(os.path.join(codebook_dir, f"codebook_m{m}.npy"))
        assert cb.shape == (ksub, subdim)
        codebooks.append(cb)
    return meta, codebooks

def encode_embeddings(emb_path, codebook_dir, out_codes_path, batch=4096):
    meta, codebooks = load_codebooks(codebook_dir)
    M = meta["M"]; subdim = meta["subdim"]; D = meta["D"]; ksub = meta["ksub"]
    X = np.load(emb_path).astype(np.float32)
    N = X.shape[0]
    assert X.shape[1] == D
    codes = np.empty((N, M), dtype=np.uint8)
    for i in range(0, N, batch):
        b = X[i:i+batch]
        # split into M chunks
        for m in range(M):
            sub = b[:, m*subdim:(m+1)*subdim]  # (B, subdim)
            cb = codebooks[m]  # (ksub, subdim)
            # compute squared distances (B, ksub)
            # (a-b)^2 = a^2 -2ab + b^2
            # use (B,1)*(1,ksub) trick
            # compute  -2 * a @ cb.T + ||cb||^2 + ||a||^2 (||a||^2 same for all centroids)
            prod = np.dot(sub, cb.T)  # (B, ksub)
            a_sq = np.sum(sub*sub, axis=1, keepdims=True)  # (B,1)
            cb_sq = np.sum(cb*cb, axis=1)  # (ksub,)
            # dist = a_sq - 2*prod + cb_sq
            dists = a_sq - 2.0 * prod + cb_sq[None, :]
            codes[i:i+batch, m] = np.argmin(dists, axis=1).astype(np.uint8)
    np.save(out_codes_path, codes)
    print(f"Saved PQ codes to {out_codes_path} shape {codes.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="emb", required=True)
    parser.add_argument("--codebook_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    encode_embeddings(args.emb, args.codebook_dir, args.out)

