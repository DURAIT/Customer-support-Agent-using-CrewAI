"""
customer_support.py
===================
Multi-agent customer support system (CrewAI + Streamlit).

Agents (process = sequential):
  1. Assistant           -> answers ONLY from the PDF knowledge base (FAISS RAG tool)
  2. Web Search Assistant-> searches the web and writes a referenced answer
  3. Entry Agent         -> appends question, both answers and a timestamp to Google Sheets

Run:
    streamlit run customer_support.py

.env file (same folder):
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini                      # optional
    EMBEDDING_MODEL=text-embedding-3-small        # optional
    GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json
    GOOGLE_SHEET_ID=<id from the sheet URL>       # share the sheet with the service-account email
"""

# ----------------------------------------------------------------------------
# Imports & environment
# ----------------------------------------------------------------------------
import io
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()  # load OPENAI_API_KEY etc. from .env

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
STORAGE_DIR = BASE_DIR / "storage"          # CrewAI long-term memory lives here
FAISS_DIR = BASE_DIR / "faiss_store"        # local FAISS vector store
LOG_DIR = BASE_DIR / "logs"                 # local log store
HISTORY_DB = BASE_DIR / "storage" / "chat_history.db"
DEFAULT_PDF = BASE_DIR / "sample_customer_support_data.pdf"

for _d in (STORAGE_DIR, FAISS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Must be set before CrewAI creates its memory storage
os.environ.setdefault("CREWAI_STORAGE_DIR", str(STORAGE_DIR))

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402
from openai import OpenAI  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402
from pypdf import PdfReader  # noqa: E402

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# File logger (verbose crew output is also written to logs/crew_log.txt)
logging.basicConfig(
    filename=str(LOG_DIR / "app.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("customer_support")


# ----------------------------------------------------------------------------
# Pydantic models (data validation)
# ----------------------------------------------------------------------------
class UserQuery(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Question is too short.")
        return v


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)


class RAGAnswer(BaseModel):
    answer: str = Field(..., description="Answer based only on the knowledge base")
    sources: List[str] = Field(default_factory=list, description="e.g. 'file.pdf p.2'")
    found_in_kb: bool = Field(..., description="True if the KB fully answers the question")


class WebAnswer(BaseModel):
    answer: str = Field(..., description="Answer based on web search results")
    references: List[str] = Field(default_factory=list, description="Source URLs")


class SheetEntry(BaseModel):
    timestamp: str
    question: str = Field(..., min_length=1)
    assistant_answer: str
    web_answer: str


class SheetResult(BaseModel):
    status: str = Field(..., description="'success' or 'error'")
    message: str


# ----------------------------------------------------------------------------
# Long-term chat history (local SQLite, shown in the UI)
# ----------------------------------------------------------------------------
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(HISTORY_DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chat (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               ts TEXT, question TEXT, assistant TEXT, web TEXT, sheet TEXT)"""
    )
    return conn


def save_history(question: str, assistant: str, web: str, sheet: str) -> None:
    with _db() as conn:
        conn.execute(
            "INSERT INTO chat (ts, question, assistant, web, sheet) VALUES (?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question, assistant, web, sheet),
        )


def load_history(limit: int = 50) -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT ts, question, assistant, web, sheet FROM chat ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    rows.reverse()
    return [dict(zip(("ts", "question", "assistant", "web", "sheet"), r)) for r in rows]


def history_as_text(n: int = 4) -> str:
    items = load_history(n)
    if not items:
        return "No previous conversation."
    return "\n".join(
        f"[{h['ts']}] Customer: {h['question']}\nAssistant: {h['assistant'][:400]}" for h in items
    )


# ----------------------------------------------------------------------------
# RAG: PDF -> chunks -> OpenAI embeddings -> local FAISS index
# ----------------------------------------------------------------------------
INDEX_FILE = FAISS_DIR / "index.faiss"
META_FILE = FAISS_DIR / "chunks.json"


def _embed(texts: List[str]) -> np.ndarray:
    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 64):
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[i : i + 64])
        vectors.extend(d.embedding for d in resp.data)
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)  # cosine similarity via inner product
    return arr


def _chunk(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 30]


def build_index(pdfs: list[tuple[str, bytes]]) -> int:
    """Build and persist the FAISS index from (filename, bytes) PDFs. Returns chunk count."""
    records: list[dict] = []
    for name, data in pdfs:
        reader = PdfReader(io.BytesIO(data))
        for page_no, page in enumerate(reader.pages, start=1):
            for piece in _chunk(page.extract_text() or ""):
                records.append({"source": name, "page": page_no, "text": piece})
    if not records:
        raise ValueError("No extractable text found in the PDF(s).")

    vectors = _embed([r["text"] for r in records])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    logger.info("Indexed %d chunks from %d PDF(s)", len(records), len(pdfs))
    return len(records)


def index_ready() -> bool:
    return INDEX_FILE.exists() and META_FILE.exists()


# ----------------------------------------------------------------------------
# Tools (@tool decorator)
# ----------------------------------------------------------------------------
@tool("Search Knowledge Base")
def search_knowledge_base(query: str) -> str:
    """Search the uploaded customer-support PDF knowledge base (FAISS RAG).
    Input: the customer's question as plain text.
    Returns the most relevant passages with source file and page number."""
    try:
        q = SearchQuery(query=query).query
    except Exception as exc:
        return f"Invalid query: {exc}"
    if not index_ready():
        return "KNOWLEDGE_BASE_EMPTY: no PDF has been indexed yet."

    index = faiss.read_index(str(INDEX_FILE))
    records = json.loads(META_FILE.read_text(encoding="utf-8"))
    scores, ids = index.search(_embed([q]), 4)

    out = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        r = records[idx]
        out.append(f"[{r['source']} p.{r['page']} | relevance {score:.2f}]\n{r['text']}")
    logger.info("RAG search: %s", q)
    return "\n\n---\n\n".join(out) or "NO_RELEVANT_PASSAGES_FOUND"


@tool("Web Search")
def web_search(query: str) -> str:
    """Search the public web (DuckDuckGo) for up-to-date information.
    Input: a focused search query. Returns titles, URLs and snippets."""
    try:
        q = SearchQuery(query=query).query
    except Exception as exc:
        return f"Invalid query: {exc}"
    try:
        try:
            from ddgs import DDGS  # newer package name
        except ImportError:
            from duckduckgo_search import DDGS  # older package name
        results = list(DDGS().text(q, max_results=6))
    except Exception as exc:
        logger.exception("Web search failed")
        return f"WEB_SEARCH_ERROR: {exc}"
    if not results:
        return "NO_WEB_RESULTS"
    logger.info("Web search: %s", q)
    return "\n\n".join(
        f"[{i}] {r.get('title')}\nURL: {r.get('href')}\n{r.get('body')}"
        for i, r in enumerate(results, start=1)
    )


@tool("Save To Google Sheet")
def save_to_google_sheet(question: str, assistant_answer: str, web_answer: str) -> str:
    """Append one row to the Google Sheet: timestamp, customer question,
    Assistant (RAG) answer, Web Search answer. Returns a success/error message."""
    try:
        entry = SheetEntry(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question=question,
            assistant_answer=assistant_answer,
            web_answer=web_answer,
        )
        import gspread

        cred_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
        sheet_id = os.getenv("GOOGLE_SHEET_ID")
        if not sheet_id:
            return "ERROR: GOOGLE_SHEET_ID is not set in .env"

        ws = gspread.service_account(filename=cred_file).open_by_key(sheet_id).sheet1
        if not ws.get_all_values():  # empty sheet -> add header
            ws.append_row(["User Query", "Answer from Agent1", "Answer from Agent2", "Date and Time"])
        # Column order matches the sheet: A=User Query, B=Agent1, C=Agent2, D=Date and Time
        ws.append_row(
            [entry.question, entry.assistant_answer, entry.web_answer, entry.timestamp],
            value_input_option="RAW",
        )
        logger.info("Sheet row saved")
        return f"SUCCESS: row saved at {entry.timestamp}"
    except Exception as exc:
        logger.exception("Google Sheet save failed")
        return f"ERROR: {exc}"


# ----------------------------------------------------------------------------
# CrewAI: agents, tasks, crew
# ----------------------------------------------------------------------------
def run_crew(question: str) -> dict:
    llm = LLM(model=f"openai/{OPENAI_MODEL}", temperature=0.1)

    assistant = Agent(
        role="Assistant",
        goal="Answer the customer's question using ONLY the company knowledge base (PDF RAG).",
        backstory=(
            "You are a senior AcmeCloud support specialist. You never use outside knowledge; "
            "if the knowledge base does not contain the answer you say so clearly."
        ),
        tools=[search_knowledge_base],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    web_agent = Agent(
        role="Web Search Assistant",
        goal="Search the web and write an accurate answer with references (URLs).",
        backstory=(
            "You are a research-minded support engineer who verifies facts on the public web "
            "and always cites the URLs you used."
        ),
        tools=[web_search],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    entry_agent = Agent(
        role="Entry Agent",
        goal="Record the customer question, both answers and the time in the Google Sheet.",
        backstory="You are a meticulous operations clerk who logs every support interaction.",
        tools=[save_to_google_sheet],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    t1 = Task(
        description=(
            "Customer question: {question}\n\n"
            "Recent conversation (for context only):\n{history}\n\n"
            "Use the 'Search Knowledge Base' tool, then answer using ONLY the retrieved passages. "
            "Set found_in_kb=false if the passages do not fully answer the question, and say what is missing. "
            "List sources as 'file p.N'."
        ),
        expected_output="A RAGAnswer JSON object (answer, sources, found_in_kb).",
        agent=assistant,
        output_pydantic=RAGAnswer,
    )
    t2 = Task(
        description=(
            "Customer question: {question}\n\n"
            "Review the Assistant's knowledge-base answer (provided as context). Use the 'Web Search' tool "
            "to find an answer. If the knowledge base was insufficient, give a complete web-based answer; "
            "if it was sufficient, verify and supplement it with current public information. "
            "Cite every source URL in 'references'."
        ),
        expected_output="A WebAnswer JSON object (answer, references).",
        agent=web_agent,
        context=[t1],
        output_pydantic=WebAnswer,
    )
    t3 = Task(
        description=(
            "Call the 'Save To Google Sheet' tool exactly once with:\n"
            "- question: {question}\n"
            "- assistant_answer: the Assistant's answer (from context)\n"
            "- web_answer: the Web Search Assistant's answer (from context)\n"
            "The tool adds the timestamp itself. Report whether it succeeded."
        ),
        expected_output="A SheetResult JSON object (status 'success' or 'error', message).",
        agent=entry_agent,
        context=[t1, t2],
        output_pydantic=SheetResult,
    )

    crew_kwargs = dict(
        agents=[assistant, web_agent, entry_agent],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=True,
        output_log_file=str(LOG_DIR / "crew_log.txt"),  # local log store
        memory=True,
    )
    # Explicit long-term memory (local SQLite) when this CrewAI version exposes it
    try:
        from crewai.memory import LongTermMemory
        from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

        crew_kwargs["long_term_memory"] = LongTermMemory(
            storage=LTMSQLiteStorage(db_path=str(STORAGE_DIR / "long_term_memory.db"))
        )
    except Exception as exc:  # newer/older API layouts fall back to memory=True
        logger.info("Using default CrewAI memory (%s)", exc)

    crew = Crew(**crew_kwargs)
    result = crew.kickoff(inputs={"question": question, "history": history_as_text()})

    outs = result.tasks_output
    rag = outs[0].pydantic or RAGAnswer(answer=outs[0].raw, found_in_kb=False)
    web = outs[1].pydantic or WebAnswer(answer=outs[1].raw)
    sheet = outs[2].pydantic or SheetResult(status="unknown", message=outs[2].raw)
    return {"rag": rag, "web": web, "sheet": sheet}


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
def render_answers(rag: dict, web: dict, sheet: str) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🤖 Assistant (PDF RAG)")
        st.markdown(rag["answer"])
        if rag.get("sources"):
            st.caption("Sources: " + ", ".join(rag["sources"]))
        if rag.get("found_in_kb") is False:
            st.warning("Knowledge base did not fully answer this question.")
    with c2:
        st.subheader("🌐 Web Search Assistant")
        st.markdown(web["answer"])
        if web.get("references"):
            st.markdown("**References**")
            for ref in web["references"]:
                st.markdown(f"- {ref}")
    st.caption(f"📝 Google Sheet: {sheet}")


def main() -> None:
    st.set_page_config(page_title="Customer Support Knowledge Base", page_icon="💬", layout="wide")
    st.title("💬 Customer Support Knowledge Base")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Add it to your .env file and restart.")
        st.stop()

    # ---- Sidebar: knowledge base ----
    with st.sidebar:
        st.header("Knowledge Base (PDF)")
        uploads = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
        if st.button("Build / Rebuild index", use_container_width=True):
            files = [(u.name, u.getvalue()) for u in uploads or []]
            if not files and DEFAULT_PDF.exists():
                files = [(DEFAULT_PDF.name, DEFAULT_PDF.read_bytes())]
            if not files:
                st.warning("Upload a PDF first.")
            else:
                with st.spinner("Embedding & indexing..."):
                    n = build_index(files)
                st.success(f"Indexed {n} chunks.")
        st.write("Index status: " + ("✅ ready" if index_ready() else "❌ not built"))
        st.caption(f"Logs: {LOG_DIR}")
        if st.button("Clear displayed history", use_container_width=True):
            st.session_state.pop("session_chat", None)

    # ---- Past conversations (long-term chat history) ----
    with st.expander("Previous conversations (long-term history)"):
        past = load_history(20)
        if not past:
            st.write("Nothing yet.")
        for h in reversed(past):
            st.markdown(f"**{h['ts']}** — {h['question']}")

    st.session_state.setdefault("session_chat", [])
    for item in st.session_state["session_chat"]:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            render_answers(item["rag"], item["web"], item["sheet"])

    # ---- Chat input ----
    raw = st.chat_input("Ask a customer support question...")
    if not raw:
        return
    try:
        question = UserQuery(question=raw).question
    except Exception as exc:
        st.error(f"Invalid question: {exc}")
        return
    if not index_ready():
        st.error("Build the knowledge-base index from the sidebar first.")
        return

    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Agents are working (Assistant → Web Search → Entry)..."):
            try:
                res = run_crew(question)
            except Exception as exc:
                logger.exception("Crew failed")
                st.error(f"Something went wrong: {exc}")
                return
        rag, web, sheet = res["rag"].model_dump(), res["web"].model_dump(), res["sheet"]
        sheet_msg = f"{sheet.status} – {sheet.message}"
        render_answers(rag, web, sheet_msg)

    save_history(question, rag["answer"], web["answer"], sheet_msg)
    st.session_state["session_chat"].append(
        {"question": question, "rag": rag, "web": web, "sheet": sheet_msg}
    )


if __name__ == "__main__":
    main()
