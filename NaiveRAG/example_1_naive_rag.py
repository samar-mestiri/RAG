from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── 1) Documents (in production, load from PDFs/Notion/DB) ─
raw = [
    Document(page_content=(
        "ACME annual leave policy: full-time employees receive 15 days of annual leave "
        "after one year of employment. 16 days after 3 years, 18 days after 5 years, "
        "20 days after 10 years. Unused leave can be carried over to June 30 of the "
        "following year, after which it expires."),
        metadata={"source": "HR/leave_policy_v3.md"}),
    Document(page_content=(
        "Special leave: 5 days for own marriage, 1 day for child's marriage, "
        "10 days for spouse's childbirth, 5 days for death of own/spouse's parent, "
        "3 days for death of grandparent. Family events such as a parent's 60th or 70th "
        "birthday do not qualify for special leave and must be taken as annual leave."),
        metadata={"source": "HR/special_leave.md"}),
    Document(page_content=(
        "Remote work: full-time employees can work from home twice a week. Manager "
        "approval required in advance. Tue/Thu remote is discouraged (company-wide "
        "meetings). New hires must come in every day for the first 3 months."),
        metadata={"source": "HR/remote_work_policy.md"}),
]

# ── 2) Chunking ──────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""])
chunks = splitter.split_documents(raw)

# ── 3) Embed + vector store ─────────────────────────────
emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True})
vectordb = Chroma.from_documents(chunks, emb, collection_name="acme_hr")

# ── 4) Retriever ────────────────────────────────────────
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# ── 5) Prompt + LLM + chain ─────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "ACME HR assistant. Answer based only on the [reference documents] below. "
     "If you can't find an answer, reply 'I cannot find the answer in the provided documents.' "
     "End each claim with [filename] as the citation."),
    ("human", "[reference documents]\n{context}\n\n[question]\n{question}")])

#llm = ChatAnthropic(model="claude-opus-4-7", temperature=0)
llm = ChatGoogleGenAI(
    model="gemini-2.5-flash", 
    temperature=0.7
)


def fmt(docs):
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)

# LCEL: LangChain Expression Language. Components are chained with `|`.
# Same idea as `cat file | grep ... | wc -l` in a Unix shell.
chain = ({"context": retriever | fmt, "question": RunnablePassthrough()}
         | prompt | llm | StrOutputParser())

# ── Run ────────────────────────────────────────────────
for q in [
    "If my parent's 60th birthday falls in my first year, how many days of leave do I get?",
    "How many days of annual leave does someone with 7 years of tenure get?",
    "Can a new hire work from home?",
    "Does the company cover lunch?",  # not in the docs → should refuse
]:
    print(f"\n━━━ Q: {q}\n> {chain.invoke(q)}")