# AcmeCloud Customer Support Agent (CrewAI + Streamlit)

A multi-agent customer support system built with **CrewAI**. A customer asks a question in a **Streamlit** chat UI, and three agents work one after another:

| # | Agent | Job |
|---|-------|-----|
| 1 | **Assistant** | Answers using **only** the uploaded PDF knowledge base (RAG with FAISS) |
| 2 | **Web Search Assistant** | Searches the web and writes an answer with reference URLs |
| 3 | **Entry Agent** | Saves the question, both answers and the date/time to a Google Sheet |

The UI shows the Assistant answer and the Web Search answer side by side.

Everything lives in a single file: `customer_support.py`.

---

## Features

- **RAG from PDFs**: the PDF is split into chunks, embedded with OpenAI embeddings and stored in a local FAISS index.
- **Sequential process**: `Process.sequential` (Assistant, then Web Search, then Entry).
- **Tools via `@tool`**: `search_knowledge_base`, `web_search` and `save_to_google_sheet` are plain decorated functions, not classes.
- **Pydantic validation**: for the user question, tool inputs, each agent's structured output and the sheet row.
- **Verbose logging** with a local log store (`logs/`).
- **Long-term memory**: CrewAI long-term memory plus a local SQLite chat history that feeds recent turns back into the prompt.
- **Secrets in `.env`**, loaded with `load_dotenv()`.

---

## Architecture

```
Customer question (Streamlit)
        |
        v
  Pydantic validation (UserQuery)
        |
        v
+--------------------------------------------------------------+
| CrewAI Crew  (process = sequential, verbose = True)          |
|                                                              |
|  Task 1: Assistant ---- search_knowledge_base (FAISS RAG)    |
|        | RAGAnswer(answer, sources, found_in_kb)             |
|        v                                                     |
|  Task 2: Web Search Assistant ---- web_search (DuckDuckGo)   |
|        | WebAnswer(answer, references)                       |
|        v                                                     |
|  Task 3: Entry Agent ---- save_to_google_sheet (gspread)     |
|          SheetResult(status, message)                        |
+--------------------------------------------------------------+
        |
        v
Both answers shown in the UI + chat history saved locally
```

Notes on behaviour:

- Agent 2 runs for **every** question so you always get both answers. If the knowledge base fully answered the question, it verifies and supplements that answer with public information. If the knowledge base was not enough, it gives a complete web-based answer.
- Agent 1 sets `found_in_kb = false` and explains what is missing when the PDF does not contain the answer, and the UI then shows a warning.

---

## Project Structure

```
Customer-support-Agent-using-CrewAI/
├── customer_support.py              # The whole application
├── sample_customer_support_data.pdf # Optional default PDF for the knowledge base
├── .env                             # Your secrets (never commit)
├── service_account.json             # Google service account key (never commit)
├── README.md
│
├── faiss_store/                     # Created automatically: local vector store
│   ├── index.faiss
│   └── chunks.json
├── logs/                            # Created automatically: local log store
│   ├── app.log                      # Application log
│   └── crew_log.txt                 # Verbose CrewAI task/agent log
└── storage/                         # Created automatically: memory
    ├── chat_history.db              # Chat history shown in the UI
    └── long_term_memory.db          # CrewAI long-term memory
```

---

## Prerequisites

- **Python 3.12** (CrewAI supports Python 3.10 to 3.13)
- An **OpenAI API key**
- A **Google account** (for Google Sheets and a Google Cloud project)

---

## Setup

### 1. Get the code

Place `customer_support.py` (and the PDF, if you want it as the default knowledge base) in a project folder, then open a terminal in that folder.

### 2. Create a virtual environment

Windows:
```bash
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

macOS / Linux:
```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install dependencies

Install CrewAI first (it pins its own versions of openai, pydantic and others), then the rest:

```bash
pip install crewai
pip install streamlit faiss-cpu pypdf gspread ddgs
```

Or use a `requirements.txt`:

```
crewai
streamlit
faiss-cpu
pypdf
gspread
ddgs
python-dotenv
```

```bash
pip install -r requirements.txt
```

### 4. Prepare the Google Sheet

1. Create a Google Sheet (for example `Customer_Support_Tracker`).
2. In row 1, add these headers in this order:

   | A | B | C | D |
   |---|---|---|---|
   | User Query | Answer from Agent1 | Answer from Agent2 | Date and Time |

   (If the sheet is completely empty, the app adds these headers itself.)
3. Copy the **Sheet ID** from the URL. It is the part between `/d/` and `/edit`:
   ```
   https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit?gid=0#gid=0
   ```

### 5. Create a Google service account

A service account lets the app write to your sheet without a login pop-up.

1. Open the [Google Cloud Console](https://console.cloud.google.com) and create (or select) a project.
2. Go to **APIs & Services → Library**, then enable the **Google Sheets API** and the **Google Drive API**.
3. Go to **APIs & Services → Credentials → Create Credentials → Service account**. Give it a name (for example `support-sheet-bot`) and finish the wizard.
4. Open the service account, go to the **Keys** tab, choose **Add key → Create new key → JSON**, and download the file.
5. Put the file in your project folder and rename it to `service_account.json` (or keep its name and reference it in `.env`).
6. Open the JSON file, copy the `client_email` value, and **share your Google Sheet** with that email as **Editor**.

### 6. Create the `.env` file

In the project folder, create a file named `.env`:

```
OPENAI_API_KEY=sk-your-openai-key
GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json
GOOGLE_SHEET_ID=your-sheet-id-here
```

Optional settings:

```
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Used for the LLM and for embeddings |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | Yes | Path to the service account JSON key (relative paths are resolved next to `customer_support.py`) |
| `GOOGLE_SHEET_ID` | Yes | ID of the Google Sheet to write to |
| `OPENAI_MODEL` | No | Chat model for the agents (default `gpt-4o-mini`) |
| `EMBEDDING_MODEL` | No | Embedding model (default `text-embedding-3-small`) |

Use no quotes and no spaces around `=`.

---

## Run the App

```bash
streamlit run customer_support.py
```

Streamlit opens the app at `http://localhost:8501`.

---

## How to Use

1. **Upload the PDF** in the sidebar under *Knowledge Base (PDF)*. You can upload more than one. If you upload nothing and `sample_customer_support_data.pdf` is in the project folder, that file is used.
2. Click **Build / Rebuild index**. The app reads the PDF, splits it into chunks, creates embeddings and saves the FAISS index to `faiss_store/`. The sidebar then shows **Index status: ready**. You only need to do this again when the PDF changes. The index is reloaded from disk on later runs.
3. **Ask a question** in the chat box at the bottom, for example:
   - *What is the refund policy for annual plans?*
   - *Tell me about Support SLA*
   - *How do I fix a 504 Gateway Timeout?*
4. Wait while the three agents run (a spinner shows progress). You will see:
   - **Assistant (PDF RAG)**: the answer from your PDF, with sources like `sample_customer_support_data.pdf p.2`.
   - **Web Search Assistant**: the web-based answer with reference URLs.
   - **Google Sheet status**: whether the row was saved.
5. Check your Google Sheet: a new row appears with the question, both answers and the date/time.
6. Open **Previous conversations (long-term history)** at the top of the page to see earlier questions, even after restarting the app.

---

## How It Works

### Knowledge base (RAG)

- `pypdf` extracts text page by page.
- Text is split into chunks of about 900 characters with 150 characters of overlap.
- Chunks are embedded with the OpenAI embedding model and normalised, then stored in a `faiss.IndexFlatIP` index (cosine similarity).
- The index (`index.faiss`) and chunk metadata (`chunks.json`, with source file and page) are saved locally in `faiss_store/`.
- At question time, the `search_knowledge_base` tool embeds the question and returns the top 4 chunks with their source and page.

### Tools (`@tool`)

| Tool | Used by | What it does |
|------|---------|--------------|
| `search_knowledge_base` | Assistant | Semantic search over the FAISS index |
| `web_search` | Web Search Assistant | DuckDuckGo search (no API key needed), returns titles, URLs and snippets |
| `save_to_google_sheet` | Entry Agent | Appends `[question, assistant answer, web answer, timestamp]` as a new row via `gspread` |

### Pydantic models

| Model | Purpose |
|-------|---------|
| `UserQuery` | Validates the customer question (3 to 2000 characters, trimmed) |
| `SearchQuery` | Validates the input passed to the search tools |
| `RAGAnswer` | Structured output of the Assistant: `answer`, `sources`, `found_in_kb` |
| `WebAnswer` | Structured output of the Web Search Assistant: `answer`, `references` |
| `SheetEntry` | Validates the row before it is written to the sheet |
| `SheetResult` | Structured output of the Entry Agent: `status`, `message` |

### Memory and chat history

- CrewAI **long-term memory** is enabled and stored locally in `storage/long_term_memory.db`.
- Each question and answer is also saved to `storage/chat_history.db`. The last few turns are passed to the Assistant as context so follow-up questions work, and the history is shown in the UI.

### Logging

- `verbose=True` is set on all three agents and on the crew.
- `logs/crew_log.txt` records every task start and completion, with the agent and its output.
- `logs/app.log` records application events such as indexing, searches and sheet writes.

---

## Customisation

- **Change the LLM**: set `OPENAI_MODEL` in `.env` (for example `gpt-4o`).
- **Change the embedding model**: set `EMBEDDING_MODEL`, then rebuild the index.
- **Tune retrieval**: in `customer_support.py`, edit the chunk size and overlap in `_chunk()` and the number of results (`index.search(..., 4)`) in `search_knowledge_base`.
- **Change agent behaviour**: edit the `role`, `goal`, `backstory` and task `description` text inside `run_crew()`.
- **Use a different knowledge base**: upload other PDFs in the sidebar and click **Build / Rebuild index**. Rebuilding replaces the previous index.

---

## Security Notes

Never commit secrets to version control. Add this to `.gitignore`:

```
.env
service_account.json
venv/
storage/
faiss_store/
logs/
```

---

## Tech Stack

- [CrewAI](https://docs.crewai.com): multi-agent orchestration
- [Streamlit](https://streamlit.io): web UI
- [FAISS](https://github.com/facebookresearch/faiss): local vector store
- [OpenAI](https://platform.openai.com): LLM and embeddings
- [Pydantic](https://docs.pydantic.dev): data validation
- [gspread](https://docs.gspread.org): Google Sheets access
- [pypdf](https://pypdf.readthedocs.io): PDF text extraction
- [ddgs](https://pypi.org/project/ddgs/): DuckDuckGo web search
