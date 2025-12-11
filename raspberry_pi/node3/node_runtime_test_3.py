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
        # load backup codes (np.uint8 array OR dict with ids/codes)
        self.backup_codes = None
        self.backup_ids = None

        if backup_codes_file:
            data = np.load(backup_codes_file, allow_pickle=True)

            # -------------------------
            # Case 1 — npz dict format
            # -------------------------
            if isinstance(data, np.lib.npyio.NpzFile) and "codes" in data:
                self.backup_codes = data["codes"]
                if "ids" in data:
                    self.backup_ids = data["ids"]
                else:
                    N = self.backup_codes.shape[0]
                    self.backup_ids = np.array([f"remote_{i}" for i in range(N)], dtype=object)

            # -------------------------
            # Case 2 — Python dict inside an object array
            # -------------------------
            elif isinstance(data, np.ndarray) and data.dtype == object:
                obj = data.item()  # unpack the dict

                if isinstance(obj, dict) and "codes" in obj:
                    self.backup_codes = obj["codes"]

                    if "ids" in obj:
                        self.backup_ids = obj["ids"]
                    else:
                        N = self.backup_codes.shape[0]
                        self.backup_ids = np.array([f"remote_{i}" for i in range(N)], dtype=object)
                else:
                    raise ValueError(f"Unsupported object-format backup file: {backup_codes_file}")

            # -------------------------
            # Case 3 — raw ndarray (your case)
            # -------------------------
            elif isinstance(data, np.ndarray):
                # raw PQ codes only: shape (N, M) uint8
                self.backup_codes = data
                N = data.shape[0]
                self.backup_ids = np.array([f"remote_{i}" for i in range(N)], dtype=object)

            # -------------------------
            # Unknown format
            # -------------------------
            else:
                raise ValueError(f"Unrecognized PQ backup format in {backup_codes_file}")

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

    print("=== Node 3 Test: PQRuntime on PC with full backups ===")

    # ------------------------------
    # 1. Paths (match your repo layout)
    # ------------------------------
    codebook_dir = "./pq_artifact"         # contains codebook_m0.npy ... codebook_m7.npy and pq_meta.npy
    local_file = "node3_local.npy"         # 4000 x 128 FP32
    backup_files = [
        "backups/node1_pq_backup.npy",     # 4000 x M (uint8)
        "backups/node2_pq_backup.npy"
    ]

    if not os.path.exists(codebook_dir):
        raise FileNotFoundError(f"Missing codebook_dir: {codebook_dir}")
    if not os.path.exists(local_file):
        raise FileNotFoundError(f"Missing local embeddings: {local_file}")

    # ------------------------------
    # 2. Load local embeddings into dict expected by PQRuntime
    # ------------------------------
    local_vectors = np.load(local_file).astype(np.float32)
    local_ids = [f"node3_{i}" for i in range(local_vectors.shape[0])]
    local_db = {pid: emb for pid, emb in zip(local_ids, local_vectors)}
    print(f"[Node3] Loaded local embeddings: {len(local_db)}")

    # ------------------------------
    # 3. Load and combine neighbor PQ backups (supports raw npy arrays)
    # ------------------------------
    backup_codes_list = []
    backup_ids_list = []
    for bf in backup_files:
        if not os.path.exists(bf):
            raise FileNotFoundError(f"Missing backup file: {bf}")
        data = np.load(bf, allow_pickle=True)

        # if dict-like npz
        if isinstance(data, np.lib.npyio.NpzFile):
            codes = data["codes"]
            ids = data["ids"] if "ids" in data else np.array([f"remote_{i}" for i in range(codes.shape[0])], dtype=object)
        # if raw ndarray of codes (your case)
        elif isinstance(data, np.ndarray):
            codes = data
            ids = np.array([f"remote_{i}" for i in range(codes.shape[0])], dtype=object)
        else:
            raise ValueError(f"Unsupported backup format: {bf}")

        backup_codes_list.append(codes)
        backup_ids_list.append(ids)
        print(f"[Node3] Loaded backup file {bf} -> codes shape {codes.shape}")

    backup_codes = np.vstack(backup_codes_list)
    backup_ids = np.hstack(backup_ids_list)
    print(f"[Node3] Combined backup codes shape: {backup_codes.shape}")

    # Save combined backup as an npz (ids + codes) for PQRuntime to load
    combined_backup_path = "combined_backup_node3.npz"
    np.savez_compressed(combined_backup_path, ids=backup_ids, codes=backup_codes)
    print(f"[Node3] Wrote combined backup -> {combined_backup_path}")

    # ------------------------------
    # 4. Initialize PQRuntime
    # ------------------------------
    runtime = PQRuntime(
        codebook_dir=codebook_dir,
        backup_codes_file=combined_backup_path,
        local_db=local_db
    )
    print("Runtime initialized.")

    # ------------------------------
    # 5. Run some test queries
    # ------------------------------
    D = 128
    np.random.seed(42)
    for i in range(5):
        q = np.random.randn(D).astype(np.float32)
        t0 = time.time()
        result = runtime.query(q)
        t1 = time.time()
        print(f"\nQuery {i+1} result (t={1000*(t1-t0):.1f} ms):")
        print(result)


