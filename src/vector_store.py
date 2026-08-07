# importing packages to connect with qdrant database
# QdrantClient(class) creates the connection
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct # A point is one record stored in Qdrant similar to arow in a table.


class VectorStore:
    # this unction automatically runs when we create a object for the class to initialize the required parameters
    # self refers to the current VectorStore object
    def __init__( self, collection_name="pdf_documents", host="localhost", port=6333):
        """
        Connect to the local Qdrant server.
        """
        '''This creates the connection object.

            Conceptually:

            Python
            │
            │ QdrantClient
            ▼
            localhost:6333
            │
            ▼
            Qdrant'''
        self.client = QdrantClient(
            host=host,
            port=port
        )

        self.collection_name = collection_name

    def create_collection(self):
        """
        Create the Qdrant collection if it doesn't already exist.
        """
        # get the existing colectiion
        collections = self.client.get_collections()
        # Extract collection names such as pdf_documments
        existing_collections = [
            collection.name for collection in collections.collections
        ]
        # check if the collection exists
        if self.collection_name not in existing_collections:
            # if it does not exists then create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )

    def add_documents(self, documents, embeddings):
        """
        Store document embeddings and metadata in Qdrant
        in smaller batches to avoid oversized requests.
        """

        batch_size = 100

        for start in range(0, len(documents), batch_size):

            end = start + batch_size

            batch_documents = documents[start:end]
            batch_embeddings = embeddings[start:end]

            points = []

            for index, (document, embedding) in enumerate(
                zip(batch_documents, batch_embeddings),
                start=start
            ):

                point = PointStruct(
                    id=index,
                    vector=embedding.tolist(),
                    payload={
                        "filename": document["filename"],
                        "page": document["page"],
                        "content": document["content"]
                    }
                )

                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            print(f"Uploaded {min(end, len(documents))}/{len(documents)} chunks")



    def search(self, query_embedding, top_k=5):
        """
        Search Qdrant for the most relevant document chunks.
        """
# self.client.query_points() is the Qdrant function that searches the vector database for the most relevant vectors.
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )

        return results.points
