from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25 = BM25Retriever.from_documents(chunks); bm25.k = 10
dense = vectordb.as_retriever(search_kwargs={"k": 10})
hybrid = EnsembleRetriever(
    retrievers=[bm25, dense],
    weights=[0.4, 0.6])  # tune by domain (heavier BM25 if lots of code/identifiers)