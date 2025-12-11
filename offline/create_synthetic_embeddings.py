import numpy as np

# Parameters
num_embeddings = 12000
embedding_dim = 128
output_file = "face_embeddings.npy"

# Generate random embeddings
embeddings = np.random.randn(num_embeddings, embedding_dim)

# Normalize embeddings to unit length (common for face embeddings)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Save embeddings to a .npy file
np.save(output_file, embeddings)

print(f"Saved {num_embeddings} embeddings of dimension {embedding_dim} to '{output_file}'")

