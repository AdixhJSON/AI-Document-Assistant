from pathlib import Path
from functools import lru_cache
import os
import re
import time
import unicodedata

import pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai


# ==========================================
# 1. GEMINI API CONFIGURATION
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env file")

client = genai.Client(api_key=api_key)

GEMINI_MODEL = "gemini-3.5-flash-lite"


# ==========================================
# 2. CLEAN PDF TEXT
# ==========================================

def clean_pdf_text(text):
    """Normalize extracted PDF text."""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufffd", "")

    text = "".join(
        char for char in text
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )

    # Join words split by hyphens at line endings
    text = re.sub(r"-\s*\n\s*", "", text)

    # Normalize spaces and preserve paragraphs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ==========================================
# 3. EXTRACT TEXT FROM PDF
# ==========================================

def extract_pdf_pages(pdf_path):
    """Extract text page by page from a PDF."""

    pages = []

    with pymupdf.open(pdf_path) as document:

        for page_number, page in enumerate(document, start=1):

            raw_text = page.get_text("text", sort=True)
            text = clean_pdf_text(raw_text)

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages


# ==========================================
# 4. SENTENCE-AWARE TEXT CHUNKING
# ==========================================

def split_long_sentence(sentence, chunk_size):
    """Split long sentences without cutting words."""

    words = sentence.split()
    pieces = []
    current = ""

    for word in words:

        if len(word) > chunk_size:
            if current:
                pieces.append(current)
                current = ""

            # Handle an individual word longer than chunk_size
            for start in range(0, len(word), chunk_size):
                pieces.append(word[start:start + chunk_size])

        elif not current:
            current = word

        elif len(current) + len(word) + 1 <= chunk_size:
            current += " " + word

        else:
            pieces.append(current)
            current = word

    if current:
        pieces.append(current)

    return pieces


def chunk_pages(pages, chunk_size=1000, overlap=150):
    """
    Create sentence-aware chunks while preserving page numbers.
    Overlap is measured in characters.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    for page in pages:

        text = page.get("text", "").strip()

        if not text:
            continue

        # Split into sentences and paragraphs
        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            text
        )

        processed_sentences = []

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) > chunk_size:
                processed_sentences.extend(
                    split_long_sentence(sentence, chunk_size)
                )
            else:
                processed_sentences.append(sentence)

        current_sentences = []
        current_length = 0

        for sentence in processed_sentences:

            separator_length = 1 if current_sentences else 0
            required_length = len(sentence) + separator_length

            if current_length + required_length <= chunk_size:

                current_sentences.append(sentence)
                current_length += required_length

            else:

                # Save the completed chunk
                chunk_text = " ".join(current_sentences).strip()

                if chunk_text:
                    chunks.append({
                        "chunk_id": len(chunks) + 1,
                        "page": page["page"],
                        "text": chunk_text
                    })

                # Build overlap from the end of the previous chunk
                overlap_sentences = []
                overlap_length = 0

                for previous_sentence in reversed(current_sentences):

                    added_length = len(previous_sentence)

                    if overlap_sentences:
                        added_length += 1

                    if overlap_length + added_length > overlap:
                        break

                    overlap_sentences.insert(0, previous_sentence)
                    overlap_length += added_length

                # Avoid overlap preventing the next sentence from fitting
                while (
                    overlap_sentences
                    and overlap_length + 1 + len(sentence) > chunk_size
                ):
                    removed = overlap_sentences.pop(0)
                    overlap_length -= len(removed)

                    if overlap_sentences:
                        overlap_length -= 1

                current_sentences = overlap_sentences
                current_length = overlap_length

                # Add the new sentence
                separator_length = 1 if current_sentences else 0
                current_sentences.append(sentence)
                current_length += len(sentence) + separator_length

        # Save the final chunk of the page
        if current_sentences:

            chunk_text = " ".join(current_sentences).strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": len(chunks) + 1,
                    "page": page["page"],
                    "text": chunk_text
                })

    return chunks


# ==========================================
# 5. LOAD EMBEDDING MODEL
# ==========================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the embedding model only once."""

    return SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# 6. GENERATE EMBEDDINGS
# ==========================================

def generate_embeddings(chunks):
    """Convert text chunks into normalized embeddings."""

    model = get_embedding_model()

    if not chunks:
        return np.empty(
            (0, model.get_sentence_embedding_dimension()),
            dtype=np.float32
        )

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.asarray(embeddings, dtype=np.float32)


# ==========================================
# 7. CREATE FAISS INDEX
# ==========================================

def create_faiss_index(embeddings):
    """Create a FAISS index from embeddings."""

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            "Cannot create FAISS index from empty embeddings."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numeric values.")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


# ==========================================
# 8. SEARCH SIMILAR CHUNKS
# ==========================================

def search_similar_chunks(
    query,
    index,
    chunks,
    top_k=3,
    distance_threshold=1.0
):
    """
    Retrieve relevant chunks using FAISS.

    Embeddings are normalized, so lower squared L2
    distance indicates greater similarity.

    The threshold is configurable and should be tested
    against different documents and questions.
    """

    if index is None or index.ntotal == 0:
        return []

    if not query or not query.strip():
        return []

    if top_k <= 0:
        return []

    if distance_threshold < 0:
        raise ValueError("distance_threshold cannot be negative.")

    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index size does not match the number of chunks."
        )

    model = get_embedding_model()

    query_embedding = model.encode(
        [query.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx < 0:
            continue

        distance = float(distance)

        if distance > distance_threshold:
            continue

        results.append({
            "chunk": chunks[int(idx)],
            "distance": distance
        })

    return results


# ==========================================
# 9. GEMINI ANSWER GENERATION
# ==========================================

def generate_answer(question, results):
    """Generate an answer using retrieved PDF context."""

    if not results:
        return "I couldn't find relevant information in the PDF."

    if not question or not question.strip():
        return "Please enter a question."

    context = "\n\n".join(
        f"Page {result['chunk']['page']}:\n"
        f"{result['chunk']['text']}"
        for result in results
    )

    prompt = f"""
You are an AI Document Assistant.

Answer the user's question using ONLY
the information in the document context.

Instructions:
1. Give clear and accurate answers.
2. Do not invent information.
3. If information is unavailable, say so.
4. Mention relevant page numbers.
5. Use headings or bullet points when appropriate.
6. Do not use outside knowledge.
7. Do not repeat the question unnecessarily.
8. Distinguish clearly between facts stated in the document
   and information that the document does not provide.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    max_retries = 2

    for attempt in range(max_retries):

        try:

            start_time = time.perf_counter()

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            elapsed = time.perf_counter() - start_time

            print(
                f"Gemini generation time: {elapsed:.2f} seconds"
            )

            if response.text and response.text.strip():
                return response.text.strip()

            return "Gemini returned an empty response."

        except Exception as e:

            error_message = str(e)
            error_upper = error_message.upper()

            print(f"Gemini API error: {error_message}")

            temporary_errors = [
                "429",
                "500",
                "502",
                "503",
                "504",
                "UNAVAILABLE",
                "RESOURCE_EXHAUSTED",
                "INTERNAL",
                "DEADLINE_EXCEEDED"
            ]

            should_retry = any(
                error in error_upper
                for error in temporary_errors
            )

            if should_retry and attempt < max_retries - 1:

                wait_time = 2 ** (attempt + 1)

                print(
                    f"Retrying Gemini API in {wait_time} seconds..."
                )

                time.sleep(wait_time)
                continue

            return f"Gemini API error: {error_message}"

    return "Unable to generate an answer. Please try again."


# ==========================================
# 10. GEMINI HELPER FOR SUMMARISATION
# ==========================================

def call_gemini(prompt):
    """Send a prompt to Gemini with retry handling."""

    max_retries = 2

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            if response.text and response.text.strip():
                return response.text.strip()

            raise ValueError("Gemini returned an empty response.")

        except Exception as e:

            error_message = str(e)
            error_upper = error_message.upper()

            temporary_errors = [
                "429",
                "500",
                "502",
                "503",
                "504",
                "UNAVAILABLE",
                "RESOURCE_EXHAUSTED",
                "INTERNAL",
                "DEADLINE_EXCEEDED"
            ]

            should_retry = any(
                error in error_upper
                for error in temporary_errors
            )

            if should_retry and attempt < max_retries - 1:

                wait_time = 2 ** (attempt + 1)
                time.sleep(wait_time)
                continue

            raise RuntimeError(
                f"Gemini API error: {error_message}"
            ) from e

    raise RuntimeError("Unable to generate the requested content.")


# ==========================================
# 11. FULL DOCUMENT SUMMARISATION
# ==========================================

def summarize_document(chunks, batch_size=5):
    """
    Summarize the entire PDF in batches.

    1. Summarize chunks in batches.
    2. Combine batch summaries if necessary.
    3. Generate the final document summary.
    """

    if not chunks:
        return "No document content is available to summarize."

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    batch_summaries = []

    # Summarize each batch
    for start in range(0, len(chunks), batch_size):

        batch = chunks[start:start + batch_size]

        context = "\n\n".join(
            f"Page {chunk['page']}:\n{chunk['text']}"
            for chunk in batch
        )

        prompt = f"""
You are an AI Document Summarization Assistant.

Summarize the following section of an uploaded PDF.

Instructions:
1. Use only information present in the supplied text.
2. Identify the main topics and important points.
3. Preserve important names, dates, figures and concepts.
4. Do not invent facts or add outside information.
5. Include relevant page numbers.
6. Avoid unnecessary personal identifiers such as student IDs.
7. If the section contains assignment instructions, identify them clearly.
8. Do not assume that this section represents the entire document.

DOCUMENT SECTION:

{context}

Write a clear and structured section summary.
"""

        summary = call_gemini(prompt)
        batch_summaries.append(summary)

    # Combine summaries in groups if there are many
    while len(batch_summaries) > 8:

        combined_summaries = []

        for start in range(0, len(batch_summaries), 8):

            group = batch_summaries[start:start + 8]
            group_text = "\n\n".join(group)

            prompt = f"""
Combine the following section summaries into one concise
intermediate summary.

Instructions:
1. Preserve all important topics and facts.
2. Remove repeated information.
3. Do not add facts that are not present.
4. Retain page references.
5. Do not invent a conclusion.

SECTION SUMMARIES:

{group_text}

Provide a structured combined summary.
"""

            combined_summaries.append(call_gemini(prompt))

        batch_summaries = combined_summaries

    # Generate final summary
    final_context = "\n\n".join(batch_summaries)

    final_prompt = f"""
You are an AI Document Assistant.

Create a comprehensive summary of the uploaded PDF
using the following summaries of its sections.

Instructions:
1. Cover all major topics represented in the summaries.
2. Organize the response using headings and bullet points.
3. Include important names, dates, figures and concepts.
4. Include relevant page numbers wherever available.
5. Do not repeat the same information unnecessarily.
6. Do not include unnecessary personal identifiers.
7. Do not add external information.
8. Do not invent a conclusion.
9. If no conclusion is present in the supplied summaries,
   state that no conclusion was identified in the available content.
10. Make clear that the summary is based on the extracted
    document content.

SECTION SUMMARIES:

{final_context}

Generate the final document summary.
"""

    return call_gemini(final_prompt)


# ==========================================
# 12. BASIC RAG MODULE TEST
# ==========================================

if __name__ == "__main__":

    pdf_path = Path(__file__).parent / "data" / "sample.pdf"

    if not pdf_path.exists():

        print("sample.pdf was not found in the data folder.")

    else:

        pages = extract_pdf_pages(pdf_path)

        print("Total extracted pages:", len(pages))

        chunks = chunk_pages(pages)

        print("Total chunks created:", len(chunks))

        for chunk in chunks[:5]:

            print("\n--------------------")
            print("Chunk ID:", chunk["chunk_id"])
            print("Page:", chunk["page"])
            print("Text:", chunk["text"][:200])

        embeddings = generate_embeddings(chunks)

        print("Embedding shape:", embeddings.shape)

        if len(embeddings) > 0:

            index = create_faiss_index(embeddings)

            print("FAISS vectors stored:", index.ntotal)

            query = "What was the First Constituent Assembly of India?"

            results = search_similar_chunks(
                query,
                index,
                chunks,
                top_k=3
            )

            print("\nSIMILARITY SEARCH RESULTS:")

            for result in results:

                print("\n--------------------")
                print("Distance:", result["distance"])
                print("Page:", result["chunk"]["page"])
                print("Text:", result["chunk"]["text"])

        else:
            print("No embeddings were generated.")
