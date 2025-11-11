import os
import sqlite3
import numpy as np
from collections import OrderedDict
from datetime import datetime
import random
import time

class FaceCache:
    def __init__(self, db_path="face_cache.db", embedding_dim=128, max_size=500):
        """
        Face embedding cache for Raspberry Pi edge nodes.
        :param db_path: path to SQLite database file
        :param embedding_dim: dimension of the face embedding vector
        :param max_size: maximum number of entries to keep in memory
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.max_size = max_size
        self.cache = OrderedDict()  # in-memory LRU
        self._init_db()
        self._load_from_db()

    # --------------------------- Database Setup ---------------------------
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                person_id TEXT PRIMARY KEY,
                embedding BLOB,
                last_updated TEXT
            );
        """)
        conn.commit()
        conn.close()

    # --------------------------- Core Operations ---------------------------
    def add_embedding(self, person_id, embedding: np.ndarray):
        """Add or update an embedding in cache + database."""
        if embedding.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {embedding.shape[-1]}")

        emb_bytes = embedding.astype(np.float32).tobytes()
        timestamp = datetime.utcnow().isoformat()

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO embeddings (person_id, embedding, last_updated)
            VALUES (?, ?, ?)
            ON CONFLICT(person_id) DO UPDATE SET
                embedding=excluded.embedding,
                last_updated=excluded.last_updated;
        """, (person_id, emb_bytes, timestamp))
        conn.commit()
        conn.close()

        self.cache[person_id] = embedding
        self.cache.move_to_end(person_id)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Evict least recently used

    def find_match(self, embedding, threshold=0.8):
        """Find the best matching person_id based on cosine similarity."""
        if not self.cache:
            return None, None

        best_id, best_score = None, -1
        for pid, vec in self.cache.items():
            sim = self._cosine_similarity(embedding, vec)
            if sim > best_score:
                best_id, best_score = pid, sim

        if best_score >= threshold:
            return best_id, best_score
        return None, best_score

    # --------------------------- Utilities ---------------------------
    def _load_from_db(self):
        """Load embeddings from database into memory (up to max_size)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT person_id, embedding FROM embeddings ORDER BY last_updated DESC LIMIT ?;", (self.max_size,))
        rows = cur.fetchall()
        conn.close()

        for pid, emb_bytes in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            if emb.shape[0] == self.embedding_dim:
                self.cache[pid] = emb
        print(f"[FaceCache] Loaded {len(self.cache)} embeddings into memory.")

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return np.dot(a, b) / (a_norm * b_norm)

    def sync_from_server(self, new_data: dict):
        """Bulk update cache from server-provided embeddings."""
        for pid, emb in new_data.items():
            self.add_embedding(pid, emb)
        print(f"[FaceCache] Synced {len(new_data)} embeddings from server.")

    def __len__(self):
        return len(self.cache)

# --------------------------- Simulation / Demo ---------------------------

if __name__ == "__main__":
    print("\n[Simulation] Initializing face embedding cache...")
    cache = FaceCache(embedding_dim=128, max_size=200)

    # --- Populate cache with random embeddings for 200 people ---
    num_people = 200
    print(f"[Simulation] Populating cache with {num_people} embeddings...")
    for i in range(num_people):
        pid = f"person_{i:04d}"
        emb = np.random.rand(cache.embedding_dim).astype(np.float32)
        cache.add_embedding(pid, emb)

    print(f"[Simulation] Cache now holds {len(cache)} embeddings.")

    # --- Query Simulation ---
    num_queries = 50
    hit_count = 0
    start_time = time.time()

    for _ in range(num_queries):
        # pick a random person
        target_pid = f"person_{random.randint(0, num_people - 1):04d}"
        target_vec = cache.cache[target_pid]
        # create slightly noisy version
        query_vec = target_vec + np.random.normal(0, 0.02, size=cache.embedding_dim)
        match_pid, score = cache.find_match(query_vec, threshold=0.75)
        if match_pid == target_pid:
            hit_count += 1

    duration = time.time() - start_time
    print(f"\n[Results]")
    print(f"Queries run: {num_queries}")
    print(f"Hit rate: {hit_count / num_queries * 100:.2f}%")
    print(f"Avg query time: {duration / num_queries * 1000:.2f} ms")

    print(f"\nCache test completed")
