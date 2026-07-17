"""example_4_llm_wiki.py, minimal LLM Wiki implementation
==================================================
A single file demonstrates:
  1) Sequentially ingesting 3 time-ordered sources
  2) On each ingest, the LLM creates/updates entities/projects/concepts pages
  3) Auto-maintaining index.md / log.md
  4) Time-evolution queries against the wiki itself as context
"""
from __future__ import annotations
import json, re, datetime, shutil
from pathlib import Path
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ════════════════════════════════════════════════════════════════
# 0. Sample raw sources: 3 time-ordered docs from a fictional company
# ════════════════════════════════════════════════════════════════
SAMPLE_SOURCES = {
    "2024Q3_strategy.md": """# Q3 2024 Strategy Meeting Summary

Q3 priorities announced by CTO John Kim:
1. Project Alpha (recommendation system overhaul) targeted for November launch
2. Data infrastructure team headcount up 50%
3. Stronger collaboration with the security team

Jane Park joins Project Alpha as security owner.
Minsoo Lee owns the data pipeline.""",

    "2024Q4_alpha_launch.md": """# Project Alpha Launch Retrospective (2024-11-30)

Launched Nov 15. Traffic up 30%, click-through up 12%.

Key contributors:
- John Kim (overall lead)
- Jane Park (security review)
- Minsoo Lee (data pipeline)
- Hyunwoo Jung (UI/UX, new joiner)

Failure mode: an early cache miss spike → resolved by scaling out the Redis cluster
Next: kick off Project Beta (payments system).""",

    "2025Q1_orgchange.md": """# Org Changes, January 2025

- John Kim: stays as CTO. Concurrently director of the Machine Learning Infrastructure team
- Jane Park: promoted to security team lead
- Minsoo Lee: promoted to Data Platform team lead
- Hyunwoo Jung: moves from the Alpha team to the Beta team
- Jihoon Choi: joins as PM of the Beta team (external hire)

Project Alpha shifts to maintenance mode. Project Beta becomes the new top priority."""
}

# ════════════════════════════════════════════════════════════════
# 1. Prompts: Ingest (action plan) / Query
# ════════════════════════════════════════════════════════════════
INGEST_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a wiki editor. Looking at a new source document, you decide how to update wiki pages.\n"
     "You'll receive every existing wiki page along with its content.\n\n"
     "Output must be a pure JSON array (no other explanation):\n"
     '  [{"op":"create","path":"entities/john_kim.md","content":"full markdown"},\n'
     '   {"op":"append","path":"projects/alpha.md","content":"markdown to append"}]\n\n'
     "Rules:\n"
     "- People: entities/<name>.md   Projects: projects/<name>.md   Concepts: concepts/<name>.md\n"
     "- Only act when there's new info. If purely duplicate, return an empty array.\n"
     "- In page bodies, use [[Other_Page_Name]] wikilinks generously.\n"
     "- End each page with '> source: [source_filename]' (also when appending).\n"
     "- If you find contradictions, add a 'TODO: review contradiction, ...' note."),
    ("human",
     "[new source: {source_name}]\n{source_text}\n\n"
     "[current wiki]\n{existing_pages}\n\n"
     "Action JSON array to reflect this source:")
])

QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a wiki assistant. Answer questions based only on the [wiki pages] below.\n"
     "Cite the source page after each fact in [[Page Name]] form.\n"
     "If evidence is insufficient, answer 'The wiki doesn't have enough information.'"),
    ("human", "[wiki pages]\n{pages}\n\n[question]\n{question}")
])


# ════════════════════════════════════════════════════════════════
# 2. WikiAgent: core logic
# ════════════════════════════════════════════════════════════════
class WikiAgent:
    def __init__(self, root: str):
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.wiki_dir = self.root / "wiki"
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite", 
            temperature=0.7
        )

    # ── Setup: directories + sample sources + empty index/log ──
    def setup(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.raw_dir.mkdir(parents=True)
        self.wiki_dir.mkdir(parents=True)
        for name, content in SAMPLE_SOURCES.items():
            (self.raw_dir / name).write_text(content, encoding="utf-8")
        (self.wiki_dir / "index.md").write_text("# Wiki Index\n\n", encoding="utf-8")
        (self.wiki_dir / "log.md").write_text("# Operation Log\n\n", encoding="utf-8")

    # ── All current wiki pages (path → content). Excludes index/log ──
    def _list_pages(self) -> Dict[str, str]:
        out = {}
        for p in self.wiki_dir.rglob("*.md"):
            rel = p.relative_to(self.wiki_dir).as_posix()
            if rel in ("index.md", "log.md"):
                continue
            out[rel] = p.read_text(encoding="utf-8")
        return out

    @staticmethod
    def _parse_json_array(text: str):
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m: return []
        try: return json.loads(m.group(0))
        except json.JSONDecodeError: return []

    # ── Ingest: absorb one source into the wiki ──
    def ingest(self, source_name: str):
        print(f"\nIngest: {source_name}")
        source_text = (self.raw_dir / source_name).read_text(encoding="utf-8")

        pages = self._list_pages()
        existing = "(no pages yet)" if not pages else "\n\n".join(
            f"### {path}\n{content}" for path, content in pages.items())

        msg = INGEST_PROMPT.invoke({
            "source_name": source_name,
            "source_text": source_text,
            "existing_pages": existing,
        })
        actions = self._parse_json_array(self.llm.invoke(msg).content)

        # Execute actions
        for a in actions:
            target = self.wiki_dir / a["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if a["op"] == "create":
                target.write_text(a["content"].rstrip() + "\n", encoding="utf-8")
                print(f"    CREATE  {a['path']}")
            elif a["op"] == "append":
                cur = target.read_text(encoding="utf-8") if target.exists() else ""
                target.write_text(cur.rstrip() + "\n\n" + a["content"].rstrip() + "\n",
                                  encoding="utf-8")
                print(f"    APPEND  {a['path']}")

        self._update_index()
        self._append_log(f"ingest | {source_name} | actions={len(actions)}")

    # ── Rebuild index: grouped by category + one-line summary ──
    def _update_index(self):
        pages = self._list_pages()
        groups: Dict[str, list] = {}
        for path in sorted(pages):
            cat = path.split("/")[0] if "/" in path else "root"
            groups.setdefault(cat, []).append(path)
        lines = ["# Wiki Index",
                 f"\n_updated: {datetime.date.today()}_  / {len(pages)} pages\n"]
        for cat, paths in groups.items():
            lines.append(f"\n## {cat}")
            for p in paths:
                first = pages[p].splitlines()[0].lstrip("# ").strip()
                lines.append(f"- [[{p[:-3]}]], {first}")
        (self.wiki_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── Append to log ──
    def _append_log(self, msg: str):
        line = f"## [{datetime.date.today()}] {msg}\n"
        with (self.wiki_dir / "log.md").open("a", encoding="utf-8") as f:
            f.write(line)

    # ── Query: answer using the whole wiki as context (small-scale demo) ──
    def query(self, question: str) -> str:
        # In production: read the index first, then have the LLM open relevant pages as a tool
        # The demo is small, so just put every page into the context at once
        pages = self._list_pages()
        joined = "\n\n".join(f"### [[{p[:-3]}]]\n{c}" for p, c in pages.items())
        msg = QUERY_PROMPT.invoke({"pages": joined, "question": question})
        return self.llm.invoke(msg).content


# ════════════════════════════════════════════════════════════════
# 3. Main: time-ordered ingest, then evolution queries
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    agent = WikiAgent("./demo_wiki")
    agent.setup()

    # Ingest 3 sources in time order. Watch the wiki grow richer.
    for name in ["2024Q3_strategy.md", "2024Q4_alpha_launch.md", "2025Q1_orgchange.md"]:
        agent.ingest(name)

    # Final wiki tree
    print("\nFinal wiki structure:")
    for p in sorted(Path("./demo_wiki/wiki").rglob("*.md")):
        rel = p.relative_to("./demo_wiki/wiki")
        print(f"   {rel}  ({p.stat().st_size}B)")

    # Time-evolution queries, very hard for RAG
    # (need to integrate role changes for the same person across multiple sources)
    print("\nWiki queries:")
    for q in [
        "How have the core members of Project Alpha changed over time?",
        "How did Jane Park's role evolve?",
        "Is Minsoo Lee involved in both Alpha and Beta? How?",
    ]:
        print(f"\n━━━ Q: {q}")
        print(f"> {agent.query(q)}")