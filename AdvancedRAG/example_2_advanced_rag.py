from __future__ import annotations
from typing import List
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# ════════════════════════════════════════════════════════════════
# 0. Data: six engineering policy documents
# ════════════════════════════════════════════════════════════════
KB = [
    {"source": "ENG/Coding_Standards.md",
     "content": "Python follows PEP 8 and the black formatter. Line length 100. Type hints on every "
                "public function. Function names snake_case, classes PascalCase, constants UPPER_SNAKE_CASE."},
    {"source": "ENG/Code_Review_Policy.md",
     "content": "PR merges require at least 2 approvals. One must be a senior. Security changes need "
                "additional approval from the security team. Recommended PR size 400 lines, ask for a "
                "split if it exceeds 1000. Reviews within 2 business days."},
    {"source": "ENG/Deploy_Process.md",
     "content": "Production deploys are Tue/Wed/Thu, 10:00–16:00. Forbidden on Fridays and the day "
                "before holidays. Validate on staging for 24 hours before deploying. Hotfixes can "
                "bypass the time restriction with CTO approval. Wait 30 minutes monitoring after deploy."},
    {"source": "ENG/On_Call_Policy.md",
     "content": "On-call rotates weekly. Target response: P1 within 15 minutes, P2 within 1 hour. "
                "Night (22:00–08:00) and weekend on-call earns extra hourly compensation. Vacations "
                "require a swap arranged in advance."},
    {"source": "ENG/Tech_Stack.md",
     "content": "Backend standard is Python 3.12 + FastAPI. DB: PostgreSQL 16, cache: Redis 7, "
                "MQ: RabbitMQ. Frontend TypeScript + React 18. AWS (ECS/RDS/S3) + Terraform."},
    {"source": "HR/Remote_Work.md",
     "content": "Full-timers can work from home twice a week. Manager approval required. Tue/Thu "
                "remote is discouraged. New hires come in every day for the first 3 months. Working "
                "abroad needs separate approval and tax review."},
]

# ════════════════════════════════════════════════════════════════
# 1. Indexing: for hybrid retrieval, build both Dense and BM25
# ════════════════════════════════════════════════════════════════
def build_retrievers():
    docs = [Document(page_content=d["content"], metadata={"source": d["source"]}) for d in KB]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350, chunk_overlap=70,
        separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_documents(docs)

    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})
    vectordb = Chroma.from_documents(chunks, emb, collection_name="adv_rag")

    dense = vectordb.as_retriever(search_kwargs={"k": 8})
    bm25 = BM25Retriever.from_documents(chunks); bm25.k = 8

    hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])
    return hybrid

# ════════════════════════════════════════════════════════════════
# 2. Query transformation: diversify with Multi-Query generation
# ════════════════════════════════════════════════════════════════
multi_query_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's question into 3 retrieval queries that preserve the meaning "
               "but vary the wording and angle. One per line, no numbering."),
    ("human", "{question}")
])

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", 
    temperature=0.7
)

def multi_queries(question: str) -> List[str]:
    raw = (multi_query_prompt | llm).invoke({"question": question}).content
    qs = [q.strip("-•123456789. ").strip() for q in raw.split("\n") if q.strip()]
    return [question] + qs[:3]  # original + 3 variants

# ════════════════════════════════════════════════════════════════
# 3. Reranking: sort candidates with a cross-encoder
# ════════════════════════════════════════════════════════════════
class Reranker:
    def __init__(self, name="BAAI/bge-reranker-v2-m3"):
        self.m = CrossEncoder(name, max_length=512)

    def __call__(self, query: str, docs: List[Document], top_n=4) -> List[Document]:
        if not docs: return []
        scores = self.m.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        # Deduplicate by page_content
        seen, out = set(), []
        for s, d in ranked:
            if d.page_content in seen: continue
            seen.add(d.page_content)
            d.metadata["rerank_score"] = float(s)
            out.append(d)
            if len(out) == top_n: break
        return out

# ════════════════════════════════════════════════════════════════
# 4. Answer generation: forced citations
# ════════════════════════════════════════════════════════════════
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "ACME engineering assistant. Answer based only on the [reference documents] below.\n"
     "Rules:\n"
     "1) End every fact with a [number] citation.\n"
     "2) If multiple documents support a claim, cite all of them like [1][3].\n"
     "3) If the docs don't say, answer 'Not specified in the documents.'\n"
     "4) Keep it concise and to the point."),
    ("human", "[reference documents]\n{context}\n\n[question]\n{question}")
])

@dataclass
class Result:
    answer: str
    sources: List[Document]
    queries_used: List[str]

def make_ctx(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {d.metadata['source']})\n{d.page_content}"
        for i, d in enumerate(docs, 1))

def advanced_rag(question: str, hybrid, reranker) -> Result:
    # ① Multi-Query transformation
    queries = multi_queries(question)

    # ② Hybrid search (per query variant)
    candidates: List[Document] = []
    seen = set()
    for q in queries:
        for d in hybrid.invoke(q):
            key = d.page_content
            if key not in seen:
                seen.add(key); candidates.append(d)

    # ③ Reranking (precise sort against the original question)
    top = reranker(question, candidates, top_n=4)

    # ④ Generate answer (forced citations)
    msg = answer_prompt.invoke({"context": make_ctx(top), "question": question})
    ans = llm.invoke(msg).content
    return Result(answer=ans, sources=top, queries_used=queries)

# ════════════════════════════════════════════════════════════════
# 5. Self-Evaluation: automatic faithfulness check
# ════════════════════════════════════════════════════════════════
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", "Judge the faithfulness of a RAG answer. If every fact in the [answer] is supported "
               "by the [reference documents], return PASS; if any fact lacks support, FAIL. "
               "First line PASS/FAIL, the rest the reasoning."),
    ("human", "[reference documents]\n{context}\n\n[answer]\n{answer}\n\nVerdict:")
])

def judge(res: Result) -> str:
    msg = judge_prompt.invoke({"context": make_ctx(res.sources), "answer": res.answer})
    return llm.invoke(msg).content

# ════════════════════════════════════════════════════════════════
# 6. Run
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    hybrid = build_retrievers()
    rerank = Reranker()

    for q in [
        "Who needs to approve a security-related PR merge?",
        "Can I push a hotfix on Friday afternoon?",
        "Can a new hire apply for remote work?",
        "What do we use for DB and cache?",
        "What's on the company lunch menu?",  # not in docs
    ]:
        print(f"\n{'='*72}\nQ: {q}")
        r = advanced_rag(q, hybrid, rerank)
        print(f"\nQuery variants: {r.queries_used}")
        print(f"\nRetrieved + reranked top-{len(r.sources)}:")
        for i, d in enumerate(r.sources, 1):
            print(f"  [{i}] {d.metadata['source']:28s}  "
                  f"score={d.metadata.get('rerank_score',0):+.2f}")
        print(f"\n> Answer:\n{r.answer}")
        print(f"\nSelf-Eval:\n{judge(r)}")