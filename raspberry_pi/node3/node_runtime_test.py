# node_runtime.py
# RPi runtime: load local full embeddings, load PQ codebooks and PQ codes for backups, run queries (exact local + ADC PQ).
# Also supports re-encoding changed local embeddings to PQ codes for delta packaging.

import numpy as np
import os
import heapq
import time

class PQRuntime:
    def __init__(self, codebook_dir, backup_codes_file=None, backup_meta=None, local_db=None):
        # load codebooks
        meta = np.load(os.path.join(codebook_dir, "pq_meta.npy"), allow_pickle=True).item()
        self.meta = meta
        self.M = meta["M"]
        self.subdim = meta["subdim"]
        self.D = meta["D"]
        self.ksub = meta["ksub"]
        self.codebooks = [np.load(os.path.join(codebook_dir, f"codebook_m{m}.npy")) for m in range(self.M)]
        # precompute centroid norms
        self.cb_sq = [np.sum(cb*cb, axis=1) for cb in self.codebooks]  # list of (ksub,)
        # load backup codes (np.uint8 array shape (N_backup, M))
        # load backup codes (npz: dict of arrays, npy: raw array)
        self.backup_codes = None
        self.backup_ids = None
        if backup_codes_file:
            data = np.load(backup_codes_file, allow_pickle=True)

            # Case 1: .npz file with keys
            if isinstance(data, np.lib.npyio.NpzFile):
                self.backup_ids = data["ids"]
                self.backup_codes = data["codes"]

            # Case 2: .npy file containing dict {ids, codes}
            elif hasattr(data, "item"):
                payload = data.item()
                self.backup_ids = payload["ids"]
                self.backup_codes = payload["codes"]

            # Case 3: raw array of codes
            else:
                self.backup_codes = data

            print(f"[PQRuntime] Loaded backup_codes shape {self.backup_codes.shape}")
        # local_db: dict {person_id: np.ndarray (D,)} OR path to npz
        self.local_db = {}
        if local_db:
            if isinstance(local_db, str) and os.path.exists(local_db):
                # expect npz with arrays 'ids' and 'embs' or a dict npy
                try:
                    data = np.load(local_db, allow_pickle=True)
                    if 'ids' in data and 'embs' in data:
                        ids = data['ids']
                        embs = data['embs']
                        for pid, emb in zip(ids, embs):
                            self.local_db[str(pid)] = emb.astype(np.float32)
                    else:
                        obj = data.item()
                        self.local_db = {str(k): np.array(v, dtype=np.float32) for k, v in obj.items()}
                    print(f"[PQRuntime] Loaded local_db with {len(self.local_db)} entries")
                except Exception as e:
                    print("Failed load local_db:", e)
            elif isinstance(local_db, dict):
                self.local_db = local_db
            else:
                print("[PQRuntime] no local db loaded")

    def add_local(self, person_id, emb):
        self.local_db[str(person_id)] = emb.astype(np.float32)

    def local_exact_search(self, q, threshold=0.8):
        # cosine similarity search over local_db
        best_id, best_score = None, -1.0
        qn = q / (np.linalg.norm(q) + 1e-12)
        for pid, emb in self.local_db.items():
            embn = emb / (np.linalg.norm(emb) + 1e-12)
            sim = float(np.dot(qn, embn))
            if sim > best_score:
                best_score = sim; best_id = pid
        if best_score >= threshold:
            return best_id, best_score
        return None, best_score

    def adc_search(self, q, topk=5):
        """
        Asymmetric distance computation: compute distances from query q (FP32) to PQ-coded database.
        Returns topk (id, approx_dist)
        Lower dist = closer (L2). We can convert to similarity if desired.
        """
        if self.backup_codes is None:
            return []
        # build lookup tables per subvector: shape (M, ksub)
        q = q.astype(np.float32)
        M = self.M; sd = self.subdim
        lookup = np.empty((M, self.ksub), dtype=np.float32)
        for m in range(M):
            qsub = q[m*sd:(m+1)*sd]  # (sd,)
            # dist to all centroids: ||qsub - c||^2 = ||qsub||^2 - 2 qsub·c + ||c||^2
            prod = np.dot(self.codebooks[m], qsub)  # (ksub,)
            qsq = np.sum(qsub*qsub)
            lookup[m, :] = qsq - 2.0 * prod + self.cb_sq[m]  # (ksub,)
        # vectorized sum of lookup over codes: for each DB vector sum lookup[m, codes[:,m]]
        codes = self.backup_codes  # (N, M) uint8
        # gather distances per subspace and sum across M
        # efficient gather: for m in M, distances_m = lookup[m, codes[:, m]]  -> shape (N,)
        N = codes.shape[0]
        dist = np.zeros(N, dtype=np.float32)
        for m in range(M):
            dist += lookup[m][codes[:, m]]
        # get topk smallest distances
        idx = np.argpartition(dist, min(topk, N)-1)[:topk]
        top_idx = idx[np.argsort(dist[idx])]
        results = []
        for i in top_idx:
            pid = self.backup_ids[i] if self.backup_ids is not None else int(i)
            results.append((pid, float(dist[i])))
        return results

    def query(self, q, local_thresh=0.8, topk=5):
        # 1) try local exact
        lid, lscore = self.local_exact_search(q, threshold=local_thresh)
        if lid is not None:
            return {"mode": "local", "id": lid, "score": lscore}
        # 2) PQ ADC search
        pq_res = self.adc_search(q, topk=topk)
        if len(pq_res) > 0:
            # if you want final verification, you can contact owner node for exact FP32 comparison
            return {"mode": "pq", "candidates": pq_res}
        return {"mode": "miss", "candidates": []}

    def reencode_local_changes(self, changed_ids, out_codes_path):
        """
        Re-encode changed local embeddings (using current codebooks), produce (ids, codes) for delta.
        changed_ids: iterable of person_id strings
        """
        ids = []
        codes_list = []
        for pid in changed_ids:
            emb = self.local_db.get(pid)
            if emb is None:
                continue
            # encode using brute-force per-subvector
            code = np.zeros(self.M, dtype=np.uint8)
            for m in range(self.M):
                sub = emb[m*self.subdim:(m+1)*self.subdim]
                cb = self.codebooks[m]  # (ksub, subdim)
                prod = np.dot(cb, sub)  # (ksub,)
                cb_sq = self.cb_sq[m]
                sub_sq = np.sum(sub*sub)
                dists = sub_sq - 2.0 * prod + cb_sq
                code[m] = np.argmin(dists)
            ids.append(pid)
            codes_list.append(code)
        ids = np.array(ids, dtype=object)
        codes = np.vstack(codes_list).astype(np.uint8) if codes_list else np.empty((0, self.M), dtype=np.uint8)
        # save as a dict .npz for atomic transfer
        np.savez_compressed(out_codes_path, ids=ids, codes=codes)
        print(f"[reencode_local_changes] saved delta to {out_codes_path} (count={len(ids)})")
        return out_codes_path

if __name__ == "__main__":
    print("=== Local Test: PQRuntime on PC ===")

    # ------------------------------
    # 1. Load codebooks (already created offline)
    # ------------------------------
    codebook_dir = "./pq_artifact"
    if not os.path.exists(codebook_dir):
        raise FileNotFoundError("pq_artifact folder not found. Run training first.")

    # ------------------------------
    # 2. Create synthetic local DB
    # ------------------------------
    D = 128
    M = 8
    N_local = 50  # number of people stored locally

    np.random.seed(0)
    local_vectors = np.random.randn(N_local, D).astype(np.float32)
    local_ids = [f"local_{i}" for i in range(N_local)]
    local_db = {pid: emb for pid, emb in zip(local_ids, local_vectors)}

    # ------------------------------
    # 3. Create synthetic PQ backups (200 remote users)
    # ------------------------------
    N_remote = 200
    remote_vectors = np.random.randn(N_remote, D).astype(np.float32)

    # Load meta + codebooks to encode these PQ backups
    meta = np.load(os.path.join(codebook_dir, "pq_meta.npy"), allow_pickle=True).item()
    subdim = meta["subdim"]
    ksub = meta["ksub"]

    # encode PQ backups
    print("Encoding synthetic PQ backups...")
    codebooks = [np.load(os.path.join(codebook_dir, f"codebook_m{m}.npy")) for m in range(M)]
    cb_sq = [np.sum(cb * cb, axis=1) for cb in codebooks]

    pq_codes = np.zeros((N_remote, M), dtype=np.uint8)
    for i, vec in enumerate(remote_vectors):
        for m in range(M):
            sub = vec[m*subdim:(m+1)*subdim]
            cb = codebooks[m]
            dist = np.sum(sub**2) - 2 * (cb @ sub) + cb_sq[m]
            pq_codes[i, m] = np.argmin(dist)

    remote_ids = np.array([f"remote_{i}" for i in range(N_remote)], dtype=object)
    backup_path = "synthetic_backup.npz"
    np.savez_compressed(backup_path, ids=remote_ids, codes=pq_codes)

    # ------------------------------
    # 4. Create PQRuntime instance
    # ------------------------------
    runtime = PQRuntime(
        codebook_dir=codebook_dir,
        backup_codes_file=backup_path,
        local_db=local_db
    )

    print("Runtime initialized.")
    print("Local people:", len(runtime.local_db))
    print("Backup PQ entries:", runtime.backup_codes.shape)

    # ------------------------------
    # 5. Test a query
    # ------------------------------
    print("\n=== Running Test Query ===")
    q = np.random.randn(D).astype(np.float32)  # random query
    result = runtime.query(q)

    print("Query result:")
    print(result)

