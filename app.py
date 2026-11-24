import streamlit as st
import tempfile
import hashlib
import time
from pathlib import Path

from rag import (
    extract_pdf_pages,
    chunk_pages,
    generate_embeddings,
    create_faiss_index,
    search_similar_chunks,
    generate_answer,
    summarize_document
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)


# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================

DEFAULTS = {
    "file_hash": None,
    "chunks": None,
    "index": None,
    "page_count": 0,
    "last_answer": None,
    "last_results": None,
    "last_question": None,
    "performance": None,
    "processing_time": None,
    "document_summary": None,
    "summary_time": None
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.title("📄 AI Document Assistant")

st.write(
    "Upload a PDF document and ask questions "
    "about its content using AI."
)

st.caption(
    "Powered by Python, Streamlit, FAISS, "
    "Sentence Transformers and Google Gemini"
)

st.divider()


# ==========================================
# 4. PDF FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"],
    help="Upload a text-based PDF document."
)


if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()


    # ======================================
    # 5. PROCESS NEW PDF DOCUMENT
    # ======================================

    if st.session_state["file_hash"] != file_hash:

        # Clear previous document information
        st.session_state["file_hash"] = None
        st.session_state["chunks"] = None
        st.session_state["index"] = None
        st.session_state["page_count"] = 0

        # Clear previous answers
        st.session_state["last_answer"] = None
        st.session_state["last_results"] = None
        st.session_state["last_question"] = None
        st.session_state["performance"] = None
        st.session_state["processing_time"] = None

        processing_start = time.perf_counter()

        try:

            with st.spinner(
                "📄 Processing your PDF document..."
            ):

                with tempfile.TemporaryDirectory() as temp_dir:

                    pdf_path = Path(temp_dir) / "uploaded.pdf"

                    # Save uploaded PDF temporarily
                    pdf_path.write_bytes(file_bytes)


                    # STEP 1: EXTRACT PDF TEXT

                    pages = extract_pdf_pages(pdf_path)

                    if not pages:
                        raise ValueError(
                            "No readable text found in this PDF. "
                            "Please upload a text-based PDF."
                        )


                    # STEP 2: CREATE TEXT CHUNKS

                    chunks = chunk_pages(pages)

                    if not chunks:
                        raise ValueError(
                            "No text chunks could be generated."
                        )


                    # STEP 3: GENERATE EMBEDDINGS

                    embeddings = generate_embeddings(chunks)

                    if embeddings.shape[0] == 0:
                        raise ValueError(
                            "Embedding generation returned no vectors."
                        )


                    # STEP 4: CREATE FAISS INDEX

                    index = create_faiss_index(embeddings)


                # Store processed document
                st.session_state["chunks"] = chunks
                st.session_state["index"] = index
                st.session_state["page_count"] = len(pages)
                st.session_state["file_hash"] = file_hash

                processing_time = (
                    time.perf_counter() - processing_start
                )

                st.session_state["processing_time"] = processing_time

                st.success(
                    "PDF processed successfully! "
                    f"{len(pages)} pages and "
                    f"{len(chunks)} chunks created."
                )

                print(
                    f"PDF processing time: "
                    f"{processing_time:.2f} seconds"
                )

        except Exception as e:

            # Keep the document marked as unprocessed
            st.session_state["file_hash"] = None

            st.error(
                f"PDF processing failed: {str(e)}"
            )

            st.stop()

    else:

        st.success("PDF uploaded and already processed!")


    # ======================================
    # 6. DOCUMENT INFORMATION
    # ======================================

    if (
        st.session_state["chunks"] is not None
        and st.session_state["index"] is not None
    ):

        st.info("📚 Your document is ready!")

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Pages",
            st.session_state["page_count"]
        )

        col2.metric(
            "Total Chunks",
            len(st.session_state["chunks"])
        )

        if st.session_state["processing_time"] is not None:

            st.caption(
                "Document processing time: "
                f"{st.session_state['processing_time']:.2f} seconds"
            )

# ==========================================
# DOCUMENT SUMMARISATION
# ==========================================

st.divider()

st.subheader("📖 Document Summarisation")

st.write(
    "Generate a summary using information from "
    "the entire uploaded PDF."
)

if st.button("📄 Summarise Entire Document"):

    summary_start = time.perf_counter()

    try:

        with st.spinner(
            "🤖 Analysing all document sections..."
        ):

            summary = summarize_document(
                st.session_state["chunks"],
                batch_size=5
            )

        st.session_state["document_summary"] = summary

        st.session_state["summary_time"] = (
            time.perf_counter() - summary_start
        )

    except Exception as e:

        st.error(
            f"Document summarisation failed: {str(e)}"
        )


# Display saved summary

if st.session_state["document_summary"]:

    st.subheader("📚 Full Document Summary")

    st.markdown(
        st.session_state["document_summary"]
    )

    if st.session_state["summary_time"] is not None:

        st.caption(
            "Summarisation time: "
            f"{st.session_state['summary_time']:.2f} seconds"
        )


# ======================================
# 7. QUESTION INPUT
# ======================================

st.divider()

st.subheader("🔍 Ask Your Question")

with st.form("question_form"):
    question = st.text_input(
        "Enter your question about the document",
        placeholder=(
            "Example: What was the First "
            "Constituent Assembly of India?"
        )
    )

    submitted = st.form_submit_button(
        "🔍 Get Answer"
    )

    # ======================================
    # 8. RETRIEVAL AND GEMINI GENERATION
    # ======================================

    if submitted:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        elif (
            st.session_state["index"] is None
            or st.session_state["chunks"] is None
        ):

            st.warning(
                "Please upload and process a PDF first."
            )

        else:

            # Clear the previous question result
            st.session_state["last_answer"] = None
            st.session_state["last_results"] = None
            st.session_state["last_question"] = None
            st.session_state["performance"] = None
            st.session_state["document_summary"] = None
            st.session_state["summary_time"] = None 

            total_start = time.perf_counter()

            retrieval_time = 0.0
            generation_time = 0.0

            try:

                # ----------------------------------
                # STEP 1: RETRIEVE RELEVANT CHUNKS
                # ----------------------------------

                retrieval_start = time.perf_counter()

                with st.spinner(
                    "🔎 Searching your document..."
                ):

                    results = search_similar_chunks(
                        question,
                        st.session_state["index"],
                        st.session_state["chunks"],
                        top_k=3
                    )

                retrieval_time = (
                    time.perf_counter() - retrieval_start
                )


                # Save retrieved information
                st.session_state["last_results"] = results


                # ----------------------------------
                # STEP 2: GENERATE GEMINI ANSWER
                # ----------------------------------

                if results:

                    generation_start = time.perf_counter()

                    with st.spinner(
                        "🤖 Gemini is generating your answer..."
                    ):

                        answer = generate_answer(
                            question,
                            results
                        )

                    generation_time = (
                        time.perf_counter() - generation_start
                    )

                    st.session_state["last_answer"] = answer

                else:

                    st.session_state["last_answer"] = (
                        "No relevant information was found "
                        "in the uploaded document."
                    )


                # ----------------------------------
                # STEP 3: CALCULATE TOTAL TIME
                # ----------------------------------

                total_time = (
                    time.perf_counter() - total_start
                )

                st.session_state["last_question"] = question

                st.session_state["performance"] = {
                    "retrieval": retrieval_time,
                    "generation": generation_time,
                    "total": total_time
                }

                print(
                    f"Retrieval time: {retrieval_time:.2f}s"
                )

                print(
                    f"Gemini generation time: "
                    f"{generation_time:.2f}s"
                )

                print(
                    f"Total response time: {total_time:.2f}s"
                )


            except Exception as e:

                total_time = (
                    time.perf_counter() - total_start
                )

                st.session_state["performance"] = {
                    "retrieval": retrieval_time,
                    "generation": generation_time,
                    "total": total_time
                }

                st.error(
                    "An error occurred while processing "
                    f"your question: {str(e)}"
                )


    # ======================================
    # 9. DISPLAY GENERATED ANSWER
    # ======================================

    if st.session_state["last_answer"] is not None:

        st.divider()

        st.subheader("🤖 AI Generated Answer")

        answer = st.session_state["last_answer"]

        # Handle API errors separately
        if answer.startswith("Gemini API error:"):

            st.error(answer)

        elif "unable to generate" in answer.lower():

            st.warning(answer)

        elif "returned an empty response" in answer.lower():

            st.warning(answer)

        else:

            st.markdown(answer)


        # ==================================
        # 10. SOURCE PAGE REFERENCES
        # ==================================

        results = st.session_state["last_results"]

        if results:

            source_pages = sorted({
                result["chunk"]["page"]
                for result in results
            })

            st.caption(
                "📄 Source Pages: "
                + ", ".join(map(str, source_pages))
            )


        # ==================================
        # 11. RETRIEVED INFORMATION
        # ==================================

        if results:

            with st.expander(
                "📚 View Retrieved Information"
            ):

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    st.markdown(
                        f"### Result {i}"
                    )

                    st.write(
                        f"**Page:** "
                        f"{result['chunk']['page']}"
                    )

                    st.write(
                        result["chunk"]["text"]
                    )

                    st.caption(
                        f"Distance: "
                        f"{result['distance']:.4f}"
                    )

                    st.divider()


        # ==================================
        # 12. PERFORMANCE STATISTICS
        # ==================================

        performance = st.session_state["performance"]

        if performance:

            st.divider()

            st.subheader("⚡ Performance Statistics")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Retrieval Time",
                f"{performance['retrieval']:.2f}s"
            )

            col2.metric(
                "Gemini Generation Time",
                f"{performance['generation']:.2f}s"
            )

            col3.metric(
                "Total Response Time",
                f"{performance['total']:.2f}s"
            )

            st.caption(
                "Total response time includes retrieval "
                "and answer generation."
            )


# ==========================================
# 13. APPLICATION FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Document Assistant | Built with "
    "Python, Streamlit, FAISS, Sentence Transformers "
    "and Google Gemini"
)
