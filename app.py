import streamlit as st

from src.rag import RAGPipeline


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("📚 Document Q&A Assistant")

st.write(
    "Ask questions about the information contained "
    "in the supplied PDF documents."
)


# ---------------------------------------------------------
# Load RAG pipeline
# ---------------------------------------------------------

@st.cache_resource
def load_rag():

    return RAGPipeline()


rag = load_rag()


# ---------------------------------------------------------
# Question input
# ---------------------------------------------------------

question = st.text_input(
    "Ask a question:",
    placeholder="Example: What does Article 21 say?"
)


# ---------------------------------------------------------
# Ask button
# ---------------------------------------------------------

if st.button("Ask Question"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching documents..."):

            result = rag.ask(question)


        # -------------------------------------------------
        # Display answer
        # -------------------------------------------------

        st.subheader("Answer")

        st.write(result["answer"])


        # -------------------------------------------------
        # Display sources
        # -------------------------------------------------

        if result["sources"]:

            st.subheader("Sources")

            for i, source in enumerate(
                result["sources"],
                start=1
            ):

                st.markdown(
                    f"### Source {i}"
                )

                st.markdown(
                    f"**Document:** {source['filename']}"
                )

                st.markdown(
                    f"**Page:** {source['page']}"
                )

                st.markdown(
                    "**Retrieved Text:**"
                )

                st.info(source["text"])

        else:

            st.info(
                "No relevant sources found in the supplied documents."
            )