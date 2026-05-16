

import os
import tempfile
from typing import List
import numpy as np
import requests
import time
from dotenv import load_dotenv
from flask import Flask, jsonify, request

from llm_processor import LLMProcessor
from vector_db import NutritionHealthVectorDB


def compute_cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    a = np.asarray(vector_a, dtype=np.float32)
    b = np.asarray(vector_b, dtype=np.float32)
    # Since embeddings are already normalized, cosine similarity = dot product
    return float(np.dot(a, b))


def rank_top_k_indices(embeddings: List[List[float]], query_embedding: List[float], top_k: int = 6) -> List[int]:
    if not embeddings:
        return []
    
    # Convert to numpy arrays for vectorized operations
    embeddings_array = np.asarray(embeddings, dtype=np.float32)
    query_array = np.asarray(query_embedding, dtype=np.float32)
    
    # Compute dot products (cosine similarity since vectors are normalized)
    similarities = np.dot(embeddings_array, query_array)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return top_indices.tolist()


load_dotenv()
app = Flask(__name__)

EXPECTED_TOKEN = os.getenv(
    "HACKRX_TOKEN",
    "fbc6d34d08858901d26a10f1b8796c2e77577b24bf91053390f858a35af05df7",
)

VECTOR_DB_SINGLETON: NutritionHealthVectorDB | None = None
LLM_SINGLETON: LLMProcessor | None = None


def get_vector_db() -> NutritionHealthVectorDB:
    global VECTOR_DB_SINGLETON
    if VECTOR_DB_SINGLETON is None:
        VECTOR_DB_SINGLETON = NutritionHealthVectorDB(db_path="./diagnosis_vectordb")
    return VECTOR_DB_SINGLETON


def get_llm() -> LLMProcessor:
    global LLM_SINGLETON
    if LLM_SINGLETON is None:
        LLM_SINGLETON = LLMProcessor()
    return LLM_SINGLETON

try:
    _ = get_vector_db()
    _ = get_llm()
except Exception:
    pass


def download_document_to_temp(url: str, connect_timeout: int, read_timeout: int, max_retries: int = 3) -> str:
    # Determine file extension from URL
    file_ext = os.path.splitext(url.split('?')[0])[1].lower()
    if not file_ext:
        file_ext = '.pdf'  # Default to PDF if no extension
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    tmp_path = tmp.name
    tmp.close()
    headers = {"User-Agent": "MedRAG/1.0"}
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            # ULTRA FAST: Download entire file at once, no streaming
            with requests.get(url, timeout=(connect_timeout, read_timeout), headers=headers) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as fh:
                    fh.write(response.content)  # Write entire content at once
            return tmp_path
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(0.1)  # ULTRA FAST: Minimal backoff
            else:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                raise e


@app.route("/")
def health_check():
    return jsonify({"status": "healthy", "service": "MedRAG API"}), 200


@app.post("/hackrx/run")
def hackrx_run():
    # Authorization (optional public mode)
    allow_public = os.getenv("ALLOW_PUBLIC", "false").lower() == "true"
    if not allow_public:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        token = auth_header.split(" ", 1)[1]
        if EXPECTED_TOKEN and token != EXPECTED_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401

    # Content-Type and payload
    if not request.is_json:
        return jsonify({"error": "Unsupported Media Type. Expected application/json."}), 415

    payload = request.get_json(silent=True) or {}
    document_url = payload.get("documents")
    # Force a single fixed API user; do not accept user_id from clients
    user_id = os.getenv("DEFAULT_USER_ID", "hackrxadmin")
    questions = payload.get("questions")

    if not isinstance(document_url, str) or not document_url:
        return jsonify({"error": "'documents' must be a non-empty URL string"}), 400
    # user_id is fixed; no validation needed from client input
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        return jsonify({"error": "'questions' must be a list of strings"}), 400

    vector_db = get_vector_db()

    # Fast-path: if this exact URL is already indexed for this user, skip download/extraction
    try:
        # Use embedding-dimension aware table name to avoid dim mismatch across model changes
        table_name = vector_db.get_default_table_name()
        table = vector_db.create_table(table_name)
        try:
            existing_by_url = table.to_pandas(where=f"user_id = '{user_id}' AND source = '{document_url}'")
        except Exception:
            tmp_all = table.to_pandas()
            existing_by_url = tmp_all[(tmp_all["user_id"] == user_id) & (tmp_all["source"] == document_url)]
    except Exception:
        existing_by_url = []

    chunks = None
    chunk_embeddings = None
    cached_doc_available = False
    deterministic_doc_id = None

    if not (isinstance(existing_by_url, list) or existing_by_url is None or (hasattr(existing_by_url, "empty") and existing_by_url.empty)):
        # Cached by URL: reuse chunks and embeddings directly
        try:
            doc_rows = existing_by_url.sort_values("chunk_index").to_dict("records")
            chunks = [row.get("content", "") for row in doc_rows]
            chunk_embeddings = [row.get("embedding", []) for row in doc_rows]
            deterministic_doc_id = doc_rows[0].get("document_id") if doc_rows else None
            cached_doc_available = True
        except Exception:
            # Fall through to reprocess if for some reason cached rows are unreadable
            cached_doc_available = False

    if not cached_doc_available:
        # Download the document to a temporary file with retries
        try:
            temp_doc_path = download_document_to_temp(
                document_url,
                connect_timeout=int(os.getenv("DOWNLOAD_CONNECT_TIMEOUT", "10")),
                read_timeout=int(os.getenv("DOWNLOAD_READ_TIMEOUT", "120")),
                max_retries=int(os.getenv("DOWNLOAD_RETRIES", "3")),
            )
        except Exception as download_error:
            return jsonify({"error": f"Failed to download document: {str(download_error)}"}), 400

        # Per-user deduplication by content hash of the actual document bytes; fallback to URL hash
        import hashlib
        try:
            with open(temp_doc_path, "rb") as _fh:
                doc_bytes = _fh.read()
            deterministic_doc_id = hashlib.sha256(doc_bytes).hexdigest()
        except Exception:
            deterministic_doc_id = hashlib.sha256(document_url.encode("utf-8")).hexdigest()

        # Check if already in LanceDB for this user by document_id (content-identical) or URL
        try:
            table_name = vector_db.get_default_table_name()
            table = vector_db.create_table(table_name)
            try:
                existing = table.to_pandas(where=f"user_id = '{user_id}' AND (document_id = '{deterministic_doc_id}' OR source = '{document_url}')")
            except Exception:
                existing = table.to_pandas()
                existing = existing[(existing["user_id"] == user_id) & ((existing["document_id"] == deterministic_doc_id) | (existing["source"] == document_url))]
            if existing is None:
                existing = []
        except Exception:
            existing = []

    if not cached_doc_available and (isinstance(existing, list) or existing is None or (hasattr(existing, "empty") and existing.empty)):
        # Not cached: extract, chunk and embed, then insert once
        try:
            markdown_content = vector_db.extract_document_content(temp_doc_path)
        except Exception as extraction_error:
            try:
                os.unlink(temp_doc_path)
            except Exception:
                pass
            return jsonify({"error": f"Failed to extract document content: {str(extraction_error)}"}), 500
        finally:
            try:
                os.unlink(temp_doc_path)
            except Exception:
                pass

        if not markdown_content:
            return jsonify({"error": "No content could be extracted from the document"}), 400

        chunks = vector_db.chunk_text(markdown_content)
        if not chunks:
            return jsonify({"error": "Document produced no searchable chunks"}), 400

        chunk_embeddings = vector_db.create_embeddings(chunks)
        if not chunk_embeddings:
            return jsonify({"error": "Failed to compute embeddings for document"}), 500

        # Insert into LanceDB for this user once using precomputed chunks/embeddings
        try:
            vector_db.add_document_from_chunks(
                chunks=chunks,
                embeddings=chunk_embeddings,
                user_id=user_id,
                access_level="private",
                doc_type="clinical",
                 table_name=table_name,
                document_id=deterministic_doc_id,
                source=document_url,
            )
            cached_doc_available = True
        except Exception:
            # Proceed even if insert fails; answers can be generated from current process
            pass
    elif not cached_doc_available:
        # Cached: fetch chunks from LanceDB for this doc to build context quickly
        try:
            try:
                os.unlink(temp_doc_path)
            except Exception:
                pass
            doc_rows = existing.sort_values("chunk_index").to_dict("records")
            chunks = [row["content"] for row in doc_rows]
            chunk_embeddings = [row["embedding"] for row in doc_rows]
            cached_doc_available = True
        except Exception:
            # Fallback to reprocess if reading cached rows fails
            try:
                markdown_content = vector_db.extract_document_content(temp_doc_path)
            finally:
                try:
                    os.unlink(temp_doc_path)
                except Exception:
                    pass
            if not markdown_content:
                return jsonify({"error": "No content could be extracted from the document"}), 400
            chunks = vector_db.chunk_text(markdown_content)
            chunk_embeddings = vector_db.create_embeddings(chunks)

    llm_processor = get_llm()
    # Combine all questions into a single prompt for one-shot LLM response
    combined_questions = "\n".join(f"Q{i+1}: {q}" for i, q in enumerate(questions))
    # Use all top chunks for all questions as context (for brevity, use top 2 for each question)
    try:
        question_embeddings = vector_db.embedding_model.encode(
            questions,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except Exception:
        question_embeddings = [vector_db.embedding_model.encode([q], show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)[0] for q in questions]

    context_chunks = []
    for question_embedding in question_embeddings:
        top_indices = rank_top_k_indices(chunk_embeddings, question_embedding, top_k=2)
        context_chunks.extend([chunks[i] for i in top_indices if i < len(chunks)])
    # Deduplicate context chunks
    context = "\n\n".join(dict.fromkeys(context_chunks))

    if llm_processor.is_available():
        prompt = (
            "You are a medical health education assistant. Answer each question using ONLY the retrieved medical context. "
            "Return a numbered list of answers in the same order as the questions. "
            "Be specific about test values, ranges, diet guidance, or clinical facts when present in the context. "
            "Do not diagnose. If information is not in the context, say 'Information not found in provided documents.' "
            "Keep each answer concise.\n\n"
            f"Questions:\n{combined_questions}\n\nMedical context:\n{context}\nAnswers:"
        )
        response = llm_processor.llm.invoke(prompt)
        answer_text = (response.content or "").strip()
        # Split numbered answers into a list
        import re
        answers = re.findall(r"\d+\.\s*(.*?)(?=\n\d+\.|$)", answer_text, re.DOTALL)
        answers = [a.strip().replace('\n', ' ') for a in answers if a.strip()]
        # If not enough answers, fallback to splitting by newlines
        if len(answers) < len(questions):
            alt = [a.strip() for a in answer_text.split('\n') if a.strip()]
            if len(alt) == len(questions):
                answers = alt
        # If still not enough, pad with 'Information not found.'
        while len(answers) < len(questions):
            answers.append('Information not found.')
    else:
        answers = [(context[:200] + "...") if len(context) > 200 else context] * len(questions)

    return jsonify({"answers": answers})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
