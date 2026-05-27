import faiss
import numpy as np

# Suppose each embedding has 384 dimensions
dimension = 384

# Create sample document embeddings
embeddings = np.random.random((1000, dimension)).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(dimension)

# Add vectors to index
index.add(embeddings)

# Create query embedding
query = np.random.random((1, dimension)).astype("float32")

# Search top 5 nearest vectors
distances, ids = index.search(query, 5)

print(ids)
print(distances)
