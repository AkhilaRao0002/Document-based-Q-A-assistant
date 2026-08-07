import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


load_dotenv()


class RAGPipeline:

    def __init__(self):

        # Load the OpenRouter API key
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set in the .env file."
            )

        # OpenRouter provides an OpenAI-compatible API
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Load embedding model
        self.embedding_model = EmbeddingModel()

        # Connect to Qdrant
        self.vector_store = VectorStore()

    def retrieve(self, question, top_k=5):

        query_embedding = self.embedding_model.generate_embeddings(
            [question]
        )[0]

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        # Keep only sufficiently relevant results
        MIN_SCORE = 0.50

        relevant_results = [
            result
            for result in results
            if result.score >= MIN_SCORE
        ]

        return relevant_results

    def get_relevant_snippet(self, question, text, max_length=500):

        # Convert question into important words
        question_words = set(
            word.lower().strip(".,?!()[]{}:")
            for word in question.split()
            if len(word) > 2
        )

        # Split retrieved text into sentences
        sentences = text.replace("\n", " ").split(". ")

        scored_sentences = []

        for sentence in sentences:

            sentence_words = set(
                word.lower().strip(".,?!()[]{}:")
                for word in sentence.split()
                if len(word) > 2
            )

            # Count overlapping words
            score = len(question_words.intersection(sentence_words))

            scored_sentences.append(
                (score, sentence.strip())
            )

        # Sort sentences by relevance
        scored_sentences.sort(
            key=lambda x: x[0],
            reverse=True
        )

        # Select the best sentences
        selected_sentences = []

        current_length = 0

        for score, sentence in scored_sentences:

            if score == 0:
                continue

            if current_length + len(sentence) > max_length:
                continue

            selected_sentences.append(sentence)

            current_length += len(sentence)

            if current_length >= 300:
                break

        # If nothing useful was found
        if not selected_sentences:

            return text[:max_length] + "..."

        return ". ".join(selected_sentences) + "."

    def generate_answer(self, question, retrieved_documents):

        if not retrieved_documents:

            return {
                "answer": (
                    "The information is not available "
                    "in the supplied documents."
                ),
                "sources": []
            }

        # ---------------------------------------------------------
        # Create numbered context
        # ---------------------------------------------------------

        context_parts = []

        for index, result in enumerate(retrieved_documents, start=1):

            payload = result.payload

            context_parts.append(
                f"""
SOURCE_ID: {index}
Document: {payload['filename']}
Page: {payload['page']}
Text:
{payload['content']}
"""
            )

        context = "\n".join(context_parts)

        # ---------------------------------------------------------
        # Prompt
        # ---------------------------------------------------------

        prompt = f"""
You are a document question-answering assistant.

Your task is to answer the question using ONLY the supplied
document context.

Do NOT use outside knowledge.

If the answer cannot be found in the supplied context,
the answer must be:

The information is not available in the supplied documents.

You must also identify which SOURCE_ID numbers directly support
your answer.

Question:
{question}

Document Context:
{context}

Return your response ONLY as valid JSON in this format:

{{
    "answer": "your answer here",
    "source_ids": [1, 3]
}}

Rules:

1. Answer only using the supplied context.
2. Do not make up facts.
3. source_ids must contain only SOURCE_ID numbers from the context.
4. Select only sources that directly support the answer.
5. If the answer is unavailable, return:
   {{
       "answer": "The information is not available in the supplied documents.",
       "source_ids": []
   }}
"""

        # ---------------------------------------------------------
        # Ask OpenRouter model
        # ---------------------------------------------------------

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the provided document context. "
                        "Return valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        raw_answer = response.choices[0].message.content.strip()

        # ---------------------------------------------------------
        # Parse JSON returned by model
        # ---------------------------------------------------------

        try:

            result_data = json.loads(raw_answer)

            answer = result_data.get("answer", "").strip()

            source_ids = result_data.get("source_ids", [])

        except (json.JSONDecodeError, AttributeError):

            # Fallback if model does not return valid JSON

            answer = raw_answer
            source_ids = []

        # ---------------------------------------------------------
        # If model says information is unavailable
        # ---------------------------------------------------------

        unavailable_message = (
            "The information is not available "
            "in the supplied documents."
        )

        if unavailable_message.lower() in answer.lower():

            return {
                "answer": unavailable_message,
                "sources": []
            }

        # ---------------------------------------------------------
        # Convert source IDs into actual citations
        # ---------------------------------------------------------

        sources = []

        for source_id in source_ids:

            try:
                source_id = int(source_id)
            except (ValueError, TypeError):
                continue

            # SOURCE_ID starts from 1
            index = source_id - 1

            if index < 0 or index >= len(retrieved_documents):
                continue

            result = retrieved_documents[index]

            payload = result.payload

            text = self.get_relevant_snippet(
                question,
                payload["content"]
            )

            sources.append(
                {
                    "filename": payload["filename"],
                    "page": payload["page"],
                    "text": text,
                    "score": result.score
                }
            )

        return {
            "answer": answer,
            "sources": sources
        }

    def ask(self, question):

        # Retrieve relevant document chunks
        retrieved_documents = self.retrieve(question)

        # Generate answer and citations
        result = self.generate_answer(
            question,
            retrieved_documents
        )

        return result