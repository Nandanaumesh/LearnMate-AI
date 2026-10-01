import tempfile

import streamlit as st

from rag.pdf_processor import extract_text_from_pdf
from rag.chunker import create_chunks
from rag.embeddings import create_embeddings
from rag.vector_store import VectorStore
from rag.rag_pipeline import generate_answer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="LearnMate AI",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📚 LearnMate AI")

    st.markdown("### 📄 Study Material")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    process_button = st.button(
        "⚙️ Process Document",
        use_container_width=True
    )
    summary_button = st.button(
        "📋 Summarize Document",
        use_container_width=True
    )

    st.divider()

    if st.session_state.processed:

        st.success(
            f"📘 {st.session_state.document_name}"
        )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if uploaded_file is not None and process_button:

    with st.spinner("Processing your document..."):

        try:

            # ---------------------------------------------
            # Save PDF temporarily
            # ---------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                pdf_path = temp_file.name


            # ---------------------------------------------
            # Extract PDF text
            # ---------------------------------------------

            pages = extract_text_from_pdf(
                pdf_path
            )

            if not pages:

                st.error(
                    "❌ No readable text found in this PDF."
                )

                st.stop()


            # ---------------------------------------------
            # Create chunks
            # ---------------------------------------------

            chunks = create_chunks(
                pages,
                chunk_size=100,
                overlap=20
            )

            if not chunks:

                st.error(
                    "❌ No chunks were created."
                )

                st.stop()


            # ---------------------------------------------
            # Create embeddings
            # ---------------------------------------------

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            embeddings = create_embeddings(
                texts
            )


            # ---------------------------------------------
            # Create FAISS vector store
            # ---------------------------------------------

            dimension = embeddings.shape[1]

            vector_store = VectorStore(
                dimension
            )

            vector_store.add_embeddings(
                embeddings,
                chunks
            )


            # ---------------------------------------------
            # Save state
            # ---------------------------------------------

            st.session_state.vector_store = (
                vector_store
            )

            st.session_state.document_name = (
                uploaded_file.name
            )

            st.session_state.processed = True

            st.session_state.chat_history = []


            st.success(
                f"✅ {uploaded_file.name} is ready!"
            )

            st.rerun()


        except Exception as e:

            st.error(
                f"❌ Error while processing document:\n\n{e}"
            )
        # =========================================================
# DOCUMENT SUMMARY
# =========================================================

if summary_button:

    if not st.session_state.processed:

        st.warning(
            "📄 Please upload and process a PDF first."
        )

    else:

        with st.spinner("📖 Creating document summary..."):

            try:

                result = generate_answer(
                    "Summarize the main concepts and important points from the uploaded document.",
                    st.session_state.vector_store,
                    top_k=8
                )

                st.session_state.chat_history.append(
                    {
                        "question": "📋 Summarize this document",
                        "answer": result["answer"],
                        "sources": result["sources"]
                    }
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"❌ Error creating summary:\n\n{e}"
                )


# =========================================================
# MAIN HEADER
# =========================================================

st.title("💬 LearnMate AI")

if st.session_state.processed:

    st.caption(
        f"Ask questions about **{st.session_state.document_name}**"
    )

else:

    st.caption(
        "Upload a study document and start learning."
    )


# =========================================================
# CHAT AREA
# =========================================================

if not st.session_state.chat_history:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding-top:120px;
            padding-bottom:100px;
        ">

        <div style="font-size:60px;">📚</div>

        <h2>How can I help you learn?</h2>

        <p style="color:gray;">
        Upload your study material and ask questions about it.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for chat in st.session_state.chat_history:

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.write(
            chat["question"]
        )


    # -----------------------------------------------------
    # AI MESSAGE
    # -----------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="📚"
    ):

        st.write(
            chat["answer"]
        )


        # ---------------------------------------------
        # SOURCES
        # ---------------------------------------------

        sources = chat.get(
            "sources",
            []
        )

        if sources:

            pages = sorted(
                set(
                    source["page_number"]
                    for source in sources
                )
            )

            st.caption(
                "📌 Sources: "
                + ", ".join(
                    f"Page {page}"
                    for page in pages
                )
            )


# =========================================================
# CHAT INPUT — BOTTOM
# =========================================================

question = st.chat_input(
    "Ask anything about your document..."
)


# =========================================================
# HANDLE QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # Check document
    # -----------------------------------------------------

    if not st.session_state.processed:

        st.warning(
            "📄 Please upload and process a PDF first."
        )

        st.stop()


    # -----------------------------------------------------
    # Show user message immediately
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.write(
            question
        )


    # -----------------------------------------------------
    # Generate AI answer
    # -----------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="📚"
    ):

        with st.spinner("Thinking..."):

            try:

                result = generate_answer(
                    question,
                    st.session_state.vector_store,
                    top_k=3
                )


                answer = result["answer"]

                sources = result["sources"]


                # -----------------------------------------
                # Display answer
                # -----------------------------------------

                st.write(
                    answer
                )


                # -----------------------------------------
                # Display sources
                # -----------------------------------------

                if sources:

                    pages = sorted(
                        set(
                            source["page_number"]
                            for source in sources
                        )
                    )

                    st.caption(
                        "📌 Sources: "
                        + ", ".join(
                            f"Page {page}"
                            for page in pages
                        )
                    )


                # -----------------------------------------
                # Save chat
                # -----------------------------------------

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    }
                )


            except Exception as e:

                st.error(
                    f"❌ Error generating answer:\n\n{e}"
                )