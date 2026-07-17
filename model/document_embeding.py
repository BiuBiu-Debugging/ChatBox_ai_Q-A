import os
import re
import shutil
import tempfile

from model.model import *


import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from model.register_file import *

from pathlib import Path

from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def generate_doc_id(file_name) -> str:
    return hash_filename(file_name)


def load_document(file_path: str) -> str:

    ext = Path(file_path).suffix.lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        reader = PdfReader(file_path)

        pages = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

        return "\n".join(pages)
    elif ext == ".docx":
        doc = Document(file_path)

        paragraphs = []

        for p in doc.paragraphs:
            if p.text.strip():
                paragraphs.append(p.text)

        return "\n".join(paragraphs)

    else:
        raise ValueError(f"Không hỗ trợ định dạng: {ext}")


def chunk_document(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)


def load_and_chunk_document(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
):
    text = load_document(file_path)

    chunks = chunk_document(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    doc_id = generate_doc_id(Path(file_path).name)

    return doc_id, chunks



def get_data_path():
    data_path=[]
    for root, dirs, files in os.walk(UPLOADS_DIR):
        for file in files:
            data_path.append(os.path.join(root, file))
    return data_path
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())
class embedding:
    def __init__(self):
        self.model=get_embeding_model()
        self.chunksize=CHUNK_SIZE
        self.chunk: list[str]=[]
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: list[str] = []
        self.doc_ids: list[str] = []
        self.dimension: int | None = None
        self._vectors: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None
        self._load_if_exists()
    def _index_path(self) -> str:
        return os.path.join(VECTOR_DB_DIR, "faiss.index")

    def _chunks_path(self) -> str:
        return os.path.join(VECTOR_DB_DIR, "chunks.npy")

    def _doc_ids_path(self) -> str:
        return os.path.join(VECTOR_DB_DIR, "doc_ids.npy")

    def _vectors_path(self) -> str:
        return os.path.join(VECTOR_DB_DIR, "vectors.npy")

    @property
    def re_ranker(self) -> CrossEncoder:
        if not hasattr(self, "_re_ranker"):
            # Load a lightweight, highly accurate cross-encoder model locally
            self._re_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return self._re_ranker

    def _load_if_exists(self):
        index_path = self._index_path()
        chunks_path = self._chunks_path()
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_index = os.path.join(tmp, "faiss.index")
                shutil.copy2(index_path, tmp_index)
                self.index = faiss.read_index(tmp_index)
            self.dimension = self.index.d
            self.chunks = np.load(chunks_path, allow_pickle=True).tolist()

            doc_ids_path = self._doc_ids_path()
            if os.path.exists(doc_ids_path):
                self.doc_ids = np.load(doc_ids_path, allow_pickle=True).tolist()
            else:
                self.doc_ids = ["unknown"] * len(self.chunks)

            vectors_path = self._vectors_path()
            if os.path.exists(vectors_path):
                self._vectors = np.load(vectors_path)
            else:
                if self.index is not None and self.index.ntotal > 0:
                    self._vectors = faiss.rev_swig_ptr(
                        self.index.get_xb(), self.index.ntotal * self.index.d
                    ).reshape(self.index.ntotal, self.index.d).copy()
                else:
                    self._vectors = None

            self._rebuild_bm25()
    def _save(self):
        if self.index is not None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_index = os.path.join(tmp, "faiss.index")
                faiss.write_index(self.index, tmp_index)
                shutil.copy2(tmp_index, self._index_path())
            np.save(self._chunks_path(), np.array(self.chunks, dtype=object))
            np.save(self._doc_ids_path(), np.array(self.doc_ids, dtype=object))
            if self._vectors is not None:
                np.save(self._vectors_path(), self._vectors)

    def _rebuild_bm25(self):
        if self.chunks:
            tokenized = [_tokenize(chunk) for chunk in self.chunks]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def add_document(self, file_path: str):

        doc_id, chunks = load_and_chunk_document(
            file_path=file_path,
            chunk_size=self.chunksize,
            chunk_overlap=CHUNK_OVERLAP,
        )

        self.add_chunks(chunks, doc_id)

        return {
            "doc_id": doc_id,
            "filename": Path(file_path).name,
            "chunk_count": len(chunks),
        }

    def add_chunks(self, chunks: list[str], doc_id: str = "unknown"):
        if not chunks:
            return

        vectors = self.model.embed_documents(chunks)
        vectors_np = np.array(vectors, dtype="float32")

        if self.index is None:
            self.dimension = vectors_np.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
            self._vectors = vectors_np
        else:
            self._vectors = np.vstack([self._vectors, vectors_np])

        self.index.add(vectors_np)
        self.chunks.extend(chunks)
        self.doc_ids.extend([doc_id] * len(chunks))
        self._rebuild_bm25()
        self._save()

    def delete_by_doc_id(self, doc_id: str) -> int:
        if self.index is None:
            return 0

        keep_mask = [did != doc_id for did in self.doc_ids]
        removed_count = keep_mask.count(False)

        if removed_count == 0:
            return 0

        self.chunks = [c for c, keep in zip(self.chunks, keep_mask) if keep]
        self.doc_ids = [d for d, keep in zip(self.doc_ids, keep_mask) if keep]

        if self._vectors is not None:
            keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
            if keep_indices:
                self._vectors = self._vectors[keep_indices]
            else:
                self._vectors = None

        # rebuild both indexes from remaining data
        if self.chunks and self._vectors is not None and len(self._vectors) > 0:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self._vectors)
        else:
            self.index = None
            self.dimension = None
            self._vectors = None

        self._rebuild_bm25()
        self._save()
        return removed_count

    def clear(self):
        self.index = None
        self.chunks = []
        self.doc_ids = []
        self.dimension = None
        self._vectors = None
        self._bm25 = None
        for path in [self._index_path(), self._chunks_path(),
                     self._doc_ids_path(), self._vectors_path()]:
            if os.path.exists(path):
                os.remove(path)

    def _search_faiss(self, query: str, fetch_k: int, doc_ids: list[str] | None = None) -> list[tuple[int, float]]:

        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = self.model.embed_query(query)
        query_np = np.array([query_vector], dtype="float32")

        k = min(fetch_k, self.index.ntotal)
        distances, indices = self.index.search(query_np, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.chunks):
                if doc_ids and self.doc_ids[idx] not in doc_ids:
                    continue
                results.append((int(idx), float(dist)))
        return results

    def _search_bm25(self, query: str, fetch_k: int, doc_ids: list[str] | None = None) -> list[tuple[int, float]]:
        if self._bm25 is None or not self.chunks:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        scored = [(i, float(s)) for i, s in enumerate(scores) if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        # apply doc_ids filter
        if doc_ids:
            scored = [(i, s) for i, s in scored if self.doc_ids[i] in doc_ids]

        return scored[:fetch_k]

    def search(self, query: str, top_k: int = TOP_K, doc_ids: list[str] | None = None) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []

        fetch_k = min(top_k * 3, self.index.ntotal)

        faiss_results = self._search_faiss(query, fetch_k, doc_ids)
        bm25_results = self._search_bm25(query, fetch_k, doc_ids)

        RRF_K = 60
        rrf_scores: dict[int, float] = {}

        for rank, (idx, _) in enumerate(faiss_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        top_candidates = sorted_indices[:fetch_k]
        if not top_candidates:
            return []

        cross_inp = [[query, self.chunks[idx]] for idx in top_candidates]
        cross_scores = self.re_ranker.predict(cross_inp)

        reranked = [(idx, score) for idx, score in zip(top_candidates, cross_scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in reranked[:top_k]:
            results.append({
                "text": self.chunks[idx],
                "doc_id": self.doc_ids[idx],
                "chunk_idx": idx,
                "score": float(score)
            })
        return results

    def get_chunks_by_doc_id(self, doc_id: str) -> int:
        return sum(1 for did in self.doc_ids if did == doc_id)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)
