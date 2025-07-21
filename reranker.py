from typing import List
from langchain_core.documents import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from sentence_transformers import CrossEncoder
from pydantic import PrivateAttr

class SentenceTransformersReranker(BaseDocumentCompressor):
    _model: CrossEncoder = PrivateAttr()  

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__()
        object.__setattr__(self, "_model", CrossEncoder(model_name))

    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        **kwargs
    ) -> List[Document]:
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc.metadata["score"] = float(score)

        reranked_docs = sorted(
            documents,
            key=lambda doc: doc.metadata["score"],
            reverse=True
        )
        return reranked_docs
