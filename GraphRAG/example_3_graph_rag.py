from __future__ import annotations
import json
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field

import networkx as nx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ════════════════════════════════════════════════════════════════
# 0. Data: fictional company wiki rich in person/project/org relations
# ════════════════════════════════════════════════════════════════
DOCS = [
    "John Kim is the CTO of ACME and joined in 2019. Before that he was a senior engineer at "
    "BlueTech. He currently leads Project Alpha and concurrently serves as director of the "
    "Machine Learning Infrastructure team.",

    "Jane Park is the security lead at ACME and serves as the security owner for Project Alpha. "
    "She previously spent 10 years at SecureCorp, and was a colleague of John Kim back at BlueTech.",

    "Minsoo Lee is a senior engineer on ACME's Data Platform team. He owns the data pipeline for "
    "Project Alpha and reports directly to John Kim. He's also collaborating with Jane Park on a "
    "security audit.",

    "Jihoon Choi is the PM for Project Beta. Beta aims to build a new payments system, and "
    "Minsoo Lee is partially involved in Beta as well, supporting the data migration.",

    "Project Alpha is ACME's next-generation recommendation system, started in January 2024. "
    "Project Beta is the payments system project, started in June 2024. "
    "Both projects are supported by the Machine Learning Infrastructure team.",
]

# ════════════════════════════════════════════════════════════════
# 1. Extract entity/relation triples with the LLM
# ════════════════════════════════════════════════════════════════
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", 
    temperature=0.7
)

extract_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extract entities and relations from the following document and output them as a JSON list of triples.\n"
     "Each triple is in the form {\"s\": subject, \"r\": relation, \"o\": object}.\n"
     "Entities should be clear concrete things only, people, organizations, projects, roles.\n"
     "Relations should be short verb phrases (e.g., WORKS_AT, LEADS, IS_CTO_OF, REPORTS_TO, COLLABORATES_WITH).\n"
     "Normalize different mentions of the same person to one name.\n"
     "Output pure JSON array only, no other text."),
    ("human", "{document}")
])

def _safe_json_parse(text: str, default):
    """Extract just the JSON portion from the LLM response (strip markdown fences, etc.)"""
    import re
    m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if not m:
        return default
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return default

def extract_triples(doc: str) -> List[Dict]:
    raw = (extract_prompt | llm).invoke({"document": doc}).content
    return _safe_json_parse(raw, default=[])

# ════════════════════════════════════════════════════════════════
# 2. Build a NetworkX graph (+ index of source documents)
# ════════════════════════════════════════════════════════════════
@dataclass
class KnowledgeGraph:
    G: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    # entity → set of source document indices it appears in
    ent2docs: Dict[str, Set[int]] = field(default_factory=dict)
    docs: List[str] = field(default_factory=list)

def build_kg(docs: List[str]) -> KnowledgeGraph:
    kg = KnowledgeGraph(docs=docs)
    for i, d in enumerate(docs):
        triples = extract_triples(d)
        for t in triples:
            s, r, o = t.get("s"), t.get("r"), t.get("o")
            if not (s and r and o): continue
            kg.G.add_edge(s, o, relation=r, doc_idx=i)
            kg.ent2docs.setdefault(s, set()).add(i)
            kg.ent2docs.setdefault(o, set()).add(i)
    return kg

# ════════════════════════════════════════════════════════════════
# 3. Extract entities from the query
# ════════════════════════════════════════════════════════════════
query_ent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract only the entities (people, organizations, projects, roles) mentioned in the question, "
               "as a JSON array. e.g., [\"John Kim\", \"Project Alpha\"]. No other text."),
    ("human", "{question}")
])

def extract_query_entities(q: str) -> List[str]:
    raw = (query_ent_prompt | llm).invoke({"question": q}).content
    return _safe_json_parse(raw, default=[])

# ════════════════════════════════════════════════════════════════
# 4. Graph traversal: N-hop subgraph around the query entities
# ════════════════════════════════════════════════════════════════
def find_node(kg: KnowledgeGraph, name: str) -> str | None:
    """Try exact match first, then fall back to substring matching"""
    if name in kg.G: return name
    for n in kg.G.nodes:
        if name in n or n in name:
            return n
    return None

def subgraph_around(kg: KnowledgeGraph, entities: List[str], hops: int = 2) -> Tuple[nx.MultiDiGraph, Set[int]]:
    """Subgraph collected from the N-hop neighborhood of the seed query entities + related document indices"""
    seed_nodes = {n for e in entities if (n := find_node(kg, e))}
    if not seed_nodes:
        return nx.MultiDiGraph(), set()

    # Convert to undirected for bidirectional BFS
    undirected = kg.G.to_undirected()
    visited = set(seed_nodes)
    frontier = set(seed_nodes)
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            if n not in undirected: continue
            next_frontier.update(undirected.neighbors(n))
        frontier = next_frontier - visited
        visited |= frontier

    sub = kg.G.subgraph(visited).copy()
    # Collect related document indices
    doc_ids = set()
    for n in visited:
        doc_ids.update(kg.ent2docs.get(n, set()))
    return sub, doc_ids

def serialize_subgraph(sub: nx.MultiDiGraph) -> str:
    """Convert the subgraph into text to pass to the LLM"""
    if sub.number_of_edges() == 0:
        return "(no related graph)"
    lines = []
    for u, v, data in sub.edges(data=True):
        lines.append(f"({u}) -[{data['relation']}]-> ({v})")
    return "\n".join(sorted(set(lines)))

# ════════════════════════════════════════════════════════════════
# 5. Answer generation: feed both the graph and source docs as context
# ════════════════════════════════════════════════════════════════
graph_answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an internal knowledge assistant. Answer based only on the [knowledge graph] and [source documents] below.\n"
     "- Walk the graph relationships to perform multi-hop reasoning.\n"
     "- Cite the relations you used in the form (A) -[relation]-> (B).\n"
     "- If evidence is insufficient, answer 'Cannot be answered with the provided information.'"),
    ("human",
     "[knowledge graph]\n{graph}\n\n[source documents]\n{docs}\n\n[question]\n{question}")
])

def graph_rag(question: str, kg: KnowledgeGraph) -> str:
    ents = extract_query_entities(question)
    sub, doc_ids = subgraph_around(kg, ents, hops=2)

    graph_text = serialize_subgraph(sub)
    doc_text = "\n\n".join(f"[doc{i}] {kg.docs[i]}" for i in sorted(doc_ids)) or "(no related documents)"

    print(f"  · Query entities: {ents}")
    print(f"  · Subgraph nodes {sub.number_of_nodes()}, edges {sub.number_of_edges()}")
    print(f"  · Related source docs: {sorted(doc_ids)}")

    msg = graph_answer_prompt.invoke({
        "graph": graph_text, "docs": doc_text, "question": question})
    return llm.invoke(msg).content

# ════════════════════════════════════════════════════════════════
# 6. Run
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[1/3] Building the graph...")
    kg = build_kg(DOCS)
    print(f"     done. nodes {kg.G.number_of_nodes()}, edges {kg.G.number_of_edges()}\n")

    # Preview the graph
    print("[2/3] Extracted triples (full):")
    for u, v, data in kg.G.edges(data=True):
        print(f"     ({u}) -[{data['relation']}]-> ({v})  (doc{data['doc_idx']})")

    # Questions that genuinely need multi-hop
    print("\n[3/3] Graph RAG Q&A:")
    for q in [
        # 1-hop: simple fact
        "Which company is John Kim CTO of?",
        # 2-hop: multi-hop, who from Kim's previous workplace are colleagues with him?
        "How do Jane Park and John Kim know each other?",
        # Relational intersection: a project both work on
        "Where do Minsoo Lee and Jane Park work together?",
        # Multi-hop + aggregation: one person across multiple projects
        "Which projects is Minsoo Lee involved in, and who are the other key members of those projects?",
        # Information not in the graph
        "What is John Kim's salary?",
    ]:
        print(f"\n━━━ Q: {q}")
        print(f"> {graph_rag(q, kg)}")

# Migrating to LangChain takes essentially two lines.

from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

graph = Neo4jGraph(url=..., username=..., password=...)
transformer = LLMGraphTransformer(llm=llm)
graph_documents = transformer.convert_to_graph_documents(docs)
graph.add_graph_documents(graph_documents)