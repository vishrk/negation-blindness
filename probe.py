"""The smallest demonstration of negation blindness.

Embed one statement and its direct negation, print the cosine similarity.
A model that understood "not" would score these far apart. It does not.
"""

from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

STATEMENT = "Paris is the capital of France."
NEGATION = "Paris is not the capital of France."

model = SentenceTransformer(MODEL)
embeddings = model.encode([STATEMENT, NEGATION])
similarity = model.similarity(embeddings[0], embeddings[1]).item()

print(f"model:     {MODEL}")
print(f"statement: {STATEMENT}")
print(f"negation:  {NEGATION}")
print(f"cosine similarity: {similarity:.4f}")
