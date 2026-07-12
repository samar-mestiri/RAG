from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", 
    temperature=0.7
)
hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a single plausible paragraph answering the question, regardless of factual accuracy."),
    ("human", "{question}")
])

def hyde_search(question: str, retriever):
    hypothetical = (hyde_prompt | llm).invoke({"question": question}).content
    # Embed the hypothetical answer for retrieval (usually more accurate than searching with the question itself)
    return retriever.invoke(hypothetical)