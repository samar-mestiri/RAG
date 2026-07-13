from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

compressor = LLMChainExtractor.from_llm(llm)  # extract only what's needed to answer
compressed = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid)