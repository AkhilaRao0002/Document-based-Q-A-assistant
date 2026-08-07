from src.pdf_loader import loading_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks.
    """
    # split each words
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        # take first 500 words
        end = start + chunk_size
        # join them into a chunk
        chunk = " ".join(words[start:end])
        # store it
        if chunk:
            chunks.append(chunk)
        # increment start considering overlapping
        start += chunk_size - overlap

    return chunks


def prepare_documents(documents):
    """
    Split each PDF page into chunks while preserving
    filename and page number.
    """

    chunked_documents = []

    for document in documents:
        # calling chunk function for text in every document
        chunks = chunk_text(document["content"])

        for chunk in chunks:

            chunked_documents.append(
                {
                    "filename": document["filename"],
                    "page": document["page"],
                    "content": chunk
                }
            )

    return chunked_documents


def main():

    # 1. Load PDF pages
    documents = loading_documents(r"C:\Users\Akhil\OneDrive\Desktop\company_project\ConversAIlabs\cases_pdf")

    print(f"Loaded {len(documents)} PDF pages.")

    # 2. Split pages into chunks
    documents = prepare_documents(documents)

    print(f"Created {len(documents)} chunks.")

    # 3. Extract text for embedding
    texts = [
        document["content"]
        for document in documents
    ]

    # 4. Generate embeddings
    embedding_model = EmbeddingModel()

    embeddings = embedding_model.generate_embeddings(texts)

    print("Embeddings generated.")

    # 5. Connect to Qdrant
    vector_store = VectorStore()

    # 6. Create collection
    vector_store.create_collection()

    # 7. Store vectors and metadata
    vector_store.add_documents(
        documents,
        embeddings
    )

    print("Documents successfully stored in Qdrant.")


if __name__ == "__main__":
    main()

