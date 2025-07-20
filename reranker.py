from typing import List
from langchain_core.documents import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from sentence_transformers import CrossEncoder
from pydantic import Field, PrivateAttr

class SentenceTransformersReranker(BaseDocumentCompressor):
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    _model: CrossEncoder = PrivateAttr()

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__(model_name=model_name)
        self._model = CrossEncoder(model_name)

    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        **kwargs  # ✅ THÊM để fix lỗi callbacks
    ) -> List[Document]:
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc.metadata["score"] = float(score)

        reranked_docs = [
            doc for score, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        ]
        return reranked_docs
