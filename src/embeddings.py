# importing sentence transformer for embedding model(all-MiniLM-L6-v2) to convert the extracted text into vextor representation
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Load the embedding model.
        """
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of text chunks.
        """
        # the embedding from the same model will be of the same size.
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,# storing as numpy array
            normalize_embeddings=True # normalizing
        )

        return embeddings
