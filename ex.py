
import base64
import io
import json
import os
import random
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

# =============================================================================
# OPTIONAL HEAVY DEPENDENCIES (the RAG / agent backend from Code 1)
# =============================================================================
_AGENT_IMPORT_ERROR = None
try:
    import fitz  # PyMuPDF
    from PIL import Image
    from groq import Groq
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_groq import ChatGroq
    from langchain_core.tools import StructuredTool
    from langchain.agents import create_agent

    AGENT_DEPS_AVAILABLE = True
except ImportError as _e:  # pragma: no cover - environment dependent
    AGENT_DEPS_AVAILABLE = False
    _AGENT_IMPORT_ERROR = str(_e)

# `ddgs` is only used for the manual "force search" button in the UI; the
# agent itself also has its own DuckDuckGo tool when AGENT_DEPS_AVAILABLE.
try:
    from ddgs import DDGS

    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

os.environ.setdefault("USER_AGENT", "BookWebAgent/1.0")  # silences WebBaseLoader warning


# =============================================================================
# PAGE CONFIG  (must be the first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="AI Agent Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CONSTANTS
# =============================================================================
APP_TITLE = "🤖 AI Agent Assistant"
APP_SUBTITLE = "Ask me anything — I can also search a book, a website, and the live web."

USER_AVATAR = "🧑"
AGENT_AVATAR = "🤖"

ROLE_USER = "user"
ROLE_AGENT = "assistant"
ROLE_SYSTEM = "system"

DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise AI assistant."

STORAGE_DIR = Path(__file__).parent / "conversations"
STORAGE_DIR.mkdir(exist_ok=True)

NEW_CHAT_TITLE = "New conversation"

# Groq/RAG defaults (from Code 1)
GROQ_KEY_ENV = os.getenv("GROQ_API_KEY", "")
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
LLM_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PDF_PATH = "imgread.pdf"
DEFAULT_WEB_URL = "https://en.wikipedia.org/wiki/Lahore"

SEARCH_TRIGGER_PATTERNS = [
    r"\blatest\b", r"\btoday\b", r"\bcurrent(ly)?\b", r"\bnews\b",
    r"\bright now\b", r"\bthis (week|month|year)\b", r"\bweather\b",
    r"\bstock price\b", r"\bwho (is|won)\b", r"\bwhen (is|did|will)\b",
    r"\bscore\b", r"\bprice of\b", r"\bupcoming\b",
]


# =============================================================================
# CONVERSATION STORAGE  (disk-persisted JSON, one file per conversation)
# =============================================================================
def _conversation_path(conversation_id: str) -> Path:
    return STORAGE_DIR / f"{conversation_id}.json"


def _new_conversation_id() -> str:
    return uuid.uuid4().hex[:12]


def save_conversation():
    conv_id = st.session_state.conversation_id
    data = {
        "id": conv_id,
        "title": st.session_state.conversation_title,
        "created_at": st.session_state.conversation_created_at,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "agent_name": st.session_state.agent_name,
        "system_prompt": st.session_state.system_prompt,
        "messages": st.session_state.messages,
    }
    try:
        _conversation_path(conv_id).write_text(json.dumps(data, indent=2))
    except OSError as exc:
        st.session_state.error_message = f"Couldn't save conversation: {exc}"


def load_conversation(conversation_id: str):
    path = _conversation_path(conversation_id)
    if not path.exists():
        st.session_state.error_message = "That conversation no longer exists."
        return
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        st.session_state.error_message = f"Couldn't load conversation: {exc}"
        return

    st.session_state.conversation_id = data["id"]
    st.session_state.conversation_title = data.get("title", NEW_CHAT_TITLE)
    st.session_state.conversation_created_at = data.get(
        "created_at", datetime.now().isoformat(timespec="seconds")
    )
    st.session_state.messages = data.get("messages", [])
    st.session_state.agent_name = data.get("agent_name", st.session_state.agent_name)
    st.session_state.system_prompt = data.get("system_prompt", st.session_state.system_prompt)
    st.session_state.is_processing = False
    st.session_state.error_message = None
    st.session_state.pending_user_input = None


def list_conversations() -> list:
    conversations = []
    for path in STORAGE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        conversations.append(
            {
                "id": data.get("id", path.stem),
                "title": data.get("title", NEW_CHAT_TITLE),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            }
        )
    conversations.sort(key=lambda c: c["updated_at"], reverse=True)
    return conversations


def delete_conversation(conversation_id: str):
    path = _conversation_path(conversation_id)
    if path.exists():
        path.unlink()


def _derive_title_from_message(message: str) -> str:
    cleaned = " ".join(message.strip().split())
    if len(cleaned) <= 48:
        return cleaned
    return cleaned[:45].rstrip() + "…"


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    defaults = {
        "messages": [],
        "is_processing": False,
        "conversation_id": _new_conversation_id(),
        "conversation_title": NEW_CHAT_TITLE,
        "conversation_created_at": datetime.now().isoformat(timespec="seconds"),
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "temperature": 0.7,
        "max_tokens": 512,
        "agent_name": "Assistant",
        "error_message": None,
        "pending_user_input": None,
        "web_search_enabled": True,
        "force_search_next": False,
        "rename_target_id": None,
        # --- new: RAG/agent backend config ---
        "groq_api_key": GROQ_KEY_ENV,
        "pdf_path": DEFAULT_PDF_PATH,
        "web_url": DEFAULT_WEB_URL,
        "book_diagram_images": [],
        "book_page_images": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation():
    st.session_state.messages = []
    st.session_state.conversation_id = _new_conversation_id()
    st.session_state.conversation_title = NEW_CHAT_TITLE
    st.session_state.conversation_created_at = datetime.now().isoformat(timespec="seconds")
    st.session_state.is_processing = False
    st.session_state.error_message = None
    st.session_state.pending_user_input = None
    st.session_state.force_search_next = False


# =============================================================================
# RAG / AGENT BACKEND  (adapted from Code 1)
# =============================================================================
def _to_b64(img, size=(512, 512), q=70) -> str:
    img = img.copy()
    img.thumbnail(size)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def _vision_describe(client, b64: str, prompt: str, tokens: int = 150) -> str:
    r = client.chat.completions.create(
        model=VISION_MODEL,
        max_tokens=tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return r.choices[0].message.content.strip()


def _page_pixmap(page, zoom: float = 2.0):
    p = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.frombytes("RGB", [p.width, p.height], p.samples)


def _load_pdf_chunks(client, pdf_path: str) -> list:
    """PDF -> list of text/caption chunks (mirrors Code 1's load_pdf)."""
    doc = fitz.open(pdf_path)
    chunks = []
    for pnum, page in enumerate(doc, 1):
        if t := page.get_text().strip():
            chunks.append(t)
        for img in page.get_images(full=True):
            pil = Image.open(io.BytesIO(doc.extract_image(img[0])["image"])).convert("RGB")
            try:
                cap = _vision_describe(client, _to_b64(pil), "Briefly describe this image for a RAG system.")
            except Exception:
                cap = "[Image: undescribed]"
            chunks.append(f"[Image p{pnum}]: {cap}")
    return chunks


def _load_website_chunks(url: str) -> list:
    docs = WebBaseLoader(url).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    return [d.page_content for d in split_docs]


def _build_vector_store(chunks: list):
    docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).create_documents(chunks)
    return FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name=EMBED_MODEL))


def _build_rag_chain(vs, groq_key: str):
    llm = ChatGroq(api_key=groq_key, model_name=LLM_MODEL, temperature=0)
    tpl = PromptTemplate.from_template("Context:\n{context}\n\nQ: {question}\nA:")
    return {
        "context": vs.as_retriever(search_kwargs={"k": 4}),
        "question": RunnablePassthrough(),
    } | tpl | llm | StrOutputParser()


def _build_agent(groq_key: str, book_rag, website_rag, web_url: str):
    llm = ChatGroq(api_key=groq_key, model_name=LLM_MODEL, temperature=0)
    duck_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
    duck = DuckDuckGoSearchRun(api_wrapper=duck_wrapper)

    def book_search_func(query: str) -> str:
        if book_rag is None:
            return "Book RAG system is not initialized (no PDF loaded)."
        try:
            return book_rag.invoke(query)
        except Exception as e:
            return f"Book search error: {e}"

    def web_search_func(query: str) -> str:
        try:
            result = duck.run(query)
            return result or "No search results found. Please try a different query."
        except Exception as e:
            return f"Web search temporarily unavailable: {e}."

    def website_search_func(query: str) -> str:
        if website_rag is None:
            return "Website RAG system is not initialized."
        try:
            return website_rag.invoke(query)
        except Exception as e:
            return f"Website search error: {e}"

    book_tool = StructuredTool.from_function(
        func=book_search_func, name="book_search",
        description="Search the indexed PDF book – use first for book-related questions.",
    )
    web_tool = StructuredTool.from_function(
        func=web_search_func, name="web_search",
        description="DuckDuckGo search – use for current events or topics not in the book or website.",
    )
    website_tool = StructuredTool.from_function(
        func=website_search_func, name="website_search",
        description=f"Search indexed content from the website ({web_url}) – use for questions about that page/topic.",
    )

    return create_agent(llm, [book_tool, web_tool, website_tool])


@st.cache_resource(show_spinner=False)
def _get_groq_client(api_key: str):
    return Groq(api_key=api_key, timeout=30.0)


@st.cache_resource(show_spinner="📚 Indexing the PDF book…")
def _get_book_rag(api_key: str, pdf_path: str):
    if not pdf_path or not os.path.exists(pdf_path):
        return None
    client = _get_groq_client(api_key)
    chunks = _load_pdf_chunks(client, pdf_path)
    vs = _build_vector_store(chunks)
    return _build_rag_chain(vs, api_key)


@st.cache_resource(show_spinner="🔗 Indexing the website…")
def _get_website_rag(api_key: str, url: str):
    if not url:
        return None
    chunks = _load_website_chunks(url)
    vs = _build_vector_store(chunks)
    return _build_rag_chain(vs, api_key)


@st.cache_resource(show_spinner="🧠 Booting the agent…")
def get_agent(api_key: str, pdf_path: str, url: str):
    book_rag = _get_book_rag(api_key, pdf_path)
    website_rag = _get_website_rag(api_key, url)
    return _build_agent(api_key, book_rag, website_rag, url)


def rebuild_agent_resources():
    """Clear all cached RAG/agent resources so the next call rebuilds them
    (e.g. after the user changes the PDF path, website URL, or API key)."""
    _get_book_rag.clear()
    _get_website_rag.clear()
    get_agent.clear()


def extract_book_pages(pdf_path: str, zoom: float = 1.5, limit: int = 20) -> list:
    """Streamlit-native replacement for Code 1's show_pages() (no Explorer)."""
    doc = fitz.open(pdf_path)
    return [_page_pixmap(p, zoom) for p in list(doc)[:limit]]


def extract_book_diagrams(pdf_path: str, limit_pages: int = 20) -> list:
    """Streamlit-native replacement for Code 1's show_diagrams() (no Explorer)."""
    client = _get_groq_client(st.session_state.groq_api_key)
    prompt = (
        "Find every photo/map/diagram (skip text). Output:\n"
        "DIAGRAM top% left% bottom% right%\nOne per line. If none: NONE"
    )
    doc = fitz.open(pdf_path)
    crops = []
    for page in list(doc)[:limit_pages]:
        full = _page_pixmap(page, zoom=3.0)
        W, H = full.size
        try:
            lines = _vision_describe(client, _to_b64(full.copy(), (768, 768), 85), prompt, 120).splitlines()
        except Exception:
            continue
        for ln in lines:
            if not ln.upper().startswith("DIAGRAM"):
                continue
            try:
                _, t, l, b, r = ln.split()
                t, l, b, r = max(0, float(t) - 2), max(0, float(l) - 2), min(100, float(b) + 2), min(100, float(r) + 2)
                crop = full.crop((int(l * W / 100), int(t * H / 100), int(r * W / 100), int(b * H / 100)))
                if min(crop.size) >= 50:
                    crops.append(crop)
            except Exception:
                pass
    return crops


# =============================================================================
# WEB SEARCH TOOL (manual "force search" button, independent of the agent)
# =============================================================================
def looks_like_it_needs_search(message: str) -> bool:
    text = message.lower()
    return any(re.search(pattern, text) for pattern in SEARCH_TRIGGER_PATTERNS)


def run_web_search(query: str, max_results: int = 5) -> list:
    if not WEB_SEARCH_AVAILABLE:
        raise RuntimeError("Web search isn't available because the `ddgs` package isn't installed. Run: pip install ddgs")
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
    return results or []


def format_search_results_for_prompt(results: list) -> str:
    if not results:
        return "No web search results were found."
    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "Untitled")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"{i}. {title} ({href})\n   {body}")
    return "\n".join(lines)


# =============================================================================
# BACKEND HOOK — now wired to the real LangGraph agent from Code 1
# =============================================================================
def call_agent_backend(
    user_message: str,
    history: list,
    settings: dict,
    search_results: list = None,
) -> str:
    """
    Send the user's message + conversation history (plus, optionally, extra
    web-search context from the manual "search the web" button) to the real
    book/website/web agent and return its text response.
    """
    if not AGENT_DEPS_AVAILABLE:
        raise RuntimeError(
            "Agent backend dependencies aren't installed. "
            f"Import error: {_AGENT_IMPORT_ERROR}. Install the RAG/agent "
            "requirements listed at the top of this file."
        )
    if not settings.get("groq_api_key"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Set it as an environment variable, or "
            "paste it in the sidebar Settings tab, then try again."
        )

    if user_message.strip().lower() == "trigger error":
        raise RuntimeError("Simulated backend failure (for testing error handling).")
    if not user_message.strip():
        raise ValueError("Empty message received by backend.")

    agent = get_agent(settings["groq_api_key"], settings["pdf_path"], settings["web_url"])

    # Fold in optional manual search results as extra grounding context.
    user_content = user_message
    if search_results is not None:
        if search_results:
            block = format_search_results_for_prompt(search_results)
            user_content = f"{user_message}\n\n[Manually retrieved web search results:]\n{block}"
        else:
            user_content = f"{user_message}\n\n[Note: a manual web search was attempted but returned no results.]"

    langchain_messages = [(m["role"], m["content"]) for m in history]
    langchain_messages.append(("user", user_content))

    response = agent.invoke({"messages": langchain_messages})
    return response["messages"][-1].content


# =============================================================================
# MESSAGE HELPERS
# =============================================================================
def add_message(role: str, content: str):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })


def get_history_for_backend() -> list:
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]


# =============================================================================
# SIDEBAR — recent chats
# =============================================================================
def render_recent_chats_tab():
    st.caption("Your saved conversations. Click one to switch to it.")

    if st.button("➕ New conversation", use_container_width=True, type="primary"):
        reset_conversation()
        st.rerun()

    conversations = list_conversations()
    if not conversations:
        st.info("No saved conversations yet. Send a message to start one.")
        return

    for conv in conversations:
        is_active = conv["id"] == st.session_state.conversation_id
        with st.container(border=True):
            if st.session_state.rename_target_id == conv["id"]:
                new_title = st.text_input(
                    "Rename", value=conv["title"],
                    key=f"rename_input_{conv['id']}", label_visibility="collapsed",
                )
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("Save", key=f"save_{conv['id']}", use_container_width=True):
                        if conv["id"] == st.session_state.conversation_id:
                            st.session_state.conversation_title = new_title
                            save_conversation()
                        else:
                            path = _conversation_path(conv["id"])
                            data = json.loads(path.read_text())
                            data["title"] = new_title
                            path.write_text(json.dumps(data, indent=2))
                        st.session_state.rename_target_id = None
                        st.rerun()
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_{conv['id']}", use_container_width=True):
                        st.session_state.rename_target_id = None
                        st.rerun()
                continue

            label = f"**{'🟢 ' if is_active else ''}{conv['title']}**"
            st.markdown(label)
            st.caption(f"{conv['message_count']} message(s) · updated {conv['updated_at'][:16].replace('T', ' ')}")

            col_open, col_rename, col_delete = st.columns([2, 1, 1])
            with col_open:
                if st.button(
                    "Open" if not is_active else "Active", key=f"open_{conv['id']}",
                    use_container_width=True, disabled=is_active,
                ):
                    if st.session_state.messages:
                        save_conversation()
                    load_conversation(conv["id"])
                    st.rerun()
            with col_rename:
                if st.button("✏️", key=f"rename_{conv['id']}", use_container_width=True, help="Rename"):
                    st.session_state.rename_target_id = conv["id"]
                    st.rerun()
            with col_delete:
                if st.button("🗑️", key=f"delete_{conv['id']}", use_container_width=True, help="Delete"):
                    delete_conversation(conv["id"])
                    if is_active:
                        reset_conversation()
                    st.rerun()


# =============================================================================
# SIDEBAR — settings / configuration
# =============================================================================
def render_settings_tab():
    st.session_state.agent_name = st.text_input(
        "Agent name", value=st.session_state.agent_name,
        help="Display name shown above agent responses.",
    )

    st.session_state.system_prompt = st.text_area(
        "System prompt", value=st.session_state.system_prompt, height=80,
        help="Currently informational only — the LangGraph agent from Code 1 "
             "doesn't take a system prompt in this version of create_agent(). "
             "Wire it in if your langchain version supports a `prompt=` kwarg.",
    )

    st.session_state.temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0,
        value=st.session_state.temperature, step=0.05,
        help="Currently informational — the agent's ChatGroq is built with "
             "temperature=0 for retrieval consistency. Change _build_agent() "
             "to use this value if you want it live.",
    )

    st.session_state.max_tokens = st.slider(
        "Max response length (tokens)", min_value=64, max_value=2048,
        value=st.session_state.max_tokens, step=64,
    )

    st.divider()
    st.markdown("##### 🧠 Agent backend (Code 1)")

    if not AGENT_DEPS_AVAILABLE:
        st.error(
            "Agent/RAG dependencies aren't installed, so the app is using no "
            f"real backend. Import error: `{_AGENT_IMPORT_ERROR}`",
            icon="🚨",
        )

    key_input = st.text_input(
        "GROQ_API_KEY", value=st.session_state.groq_api_key, type="password",
        help="Reads from the GROQ_API_KEY env var by default; you can override it here.",
    )
    pdf_input = st.text_input(
        "Book PDF path", value=st.session_state.pdf_path,
        help="Local path to the PDF indexed by the book_search tool.",
    )
    url_input = st.text_input(
        "Website URL", value=st.session_state.web_url,
        help="Webpage indexed by the website_search tool.",
    )

    config_changed = (
        key_input != st.session_state.groq_api_key
        or pdf_input != st.session_state.pdf_path
        or url_input != st.session_state.web_url
    )
    st.session_state.groq_api_key = key_input
    st.session_state.pdf_path = pdf_input
    st.session_state.web_url = url_input

    if st.button(
        "🔄 Rebuild index / agent", use_container_width=True,
        disabled=not AGENT_DEPS_AVAILABLE,
        help="Re-index the PDF/website and rebuild the agent with the settings above.",
    ):
        rebuild_agent_resources()
        st.success("Cleared cached agent/index — it will rebuild on the next message.")
    elif config_changed:
        st.caption("⚠️ Settings changed — click **Rebuild index / agent** to apply them.")

    with st.expander("📖 Book tools (from Code 1's show_pages / show_diagrams)"):
        colp, cold = st.columns(2)
        with colp:
            if st.button("Show pages", use_container_width=True, disabled=not AGENT_DEPS_AVAILABLE):
                if os.path.exists(st.session_state.pdf_path):
                    st.session_state.book_page_images = extract_book_pages(st.session_state.pdf_path)
                else:
                    st.session_state.error_message = f"PDF not found at '{st.session_state.pdf_path}'."
        with cold:
            if st.button("Show diagrams", use_container_width=True, disabled=not AGENT_DEPS_AVAILABLE):
                if os.path.exists(st.session_state.pdf_path):
                    st.session_state.book_diagram_images = extract_book_diagrams(st.session_state.pdf_path)
                else:
                    st.session_state.error_message = f"PDF not found at '{st.session_state.pdf_path}'."

    st.divider()
    st.markdown("##### 🔍 Manual web search")

    if not WEB_SEARCH_AVAILABLE:
        st.warning(
            "The `ddgs` package isn't installed, so the manual search button "
            "is disabled. Run `pip install ddgs` to enable it. (The agent's "
            "own web_search tool still works independently of this.)",
            icon="⚠️",
        )

    st.session_state.web_search_enabled = st.toggle(
        "Auto-hint the agent to search when a message looks time-sensitive",
        value=st.session_state.web_search_enabled, disabled=not WEB_SEARCH_AVAILABLE,
        help="When on, messages that look like they need current info "
             "(e.g. mention 'latest', 'today', 'news') run a DuckDuckGo "
             "search up front and pass the results to the agent as extra "
             "context, on top of whatever tools it decides to call itself.",
    )

    st.divider()
    st.markdown("##### 💬 Current conversation")
    st.caption(f"Title: **{st.session_state.conversation_title}**")
    st.caption(f"Session ID: `{st.session_state.conversation_id}`")
    st.caption(f"Messages: {len(st.session_state.messages)}")

    st.divider()
    st.caption("Tip: type **trigger error** as a message to test the app's error-handling path.")


def render_sidebar():
    with st.sidebar:
        st.markdown("### 🤖 AI Agent Chat")
        tab_chats, tab_settings = st.tabs(["💬 Recent Chats", "⚙️ Settings"])
        with tab_chats:
            render_recent_chats_tab()
        with tab_settings:
            render_settings_tab()


# =============================================================================
# MAIN HEADER
# =============================================================================
def render_header():
    st.markdown(f"## {APP_TITLE}")
    st.caption(APP_SUBTITLE)
    st.caption(f"📁 {st.session_state.conversation_title}")
    st.divider()


# =============================================================================
# BOOK PAGE / DIAGRAM GALLERY  (new, replaces Explorer popup from Code 1)
# =============================================================================
def render_book_gallery():
    if st.session_state.book_page_images:
        with st.expander(f"📄 Book pages ({len(st.session_state.book_page_images)})", expanded=False):
            for i, img in enumerate(st.session_state.book_page_images, 1):
                st.image(img, caption=f"Page {i}", use_container_width=True)
    if st.session_state.book_diagram_images:
        with st.expander(f"🖼️ Book diagrams ({len(st.session_state.book_diagram_images)})", expanded=False):
            for i, img in enumerate(st.session_state.book_diagram_images, 1):
                st.image(img, caption=f"Diagram {i}", use_container_width=True)


# =============================================================================
# CHAT HISTORY RENDERING
# =============================================================================
def render_chat_history():
    if not st.session_state.messages:
        st.info("👋 No messages yet. Start the conversation using the box below.", icon="💡")
        return

    for msg in st.session_state.messages:
        if msg["role"] == ROLE_USER:
            with st.chat_message(ROLE_USER, avatar=USER_AVATAR):
                st.markdown(msg["content"])
                st.caption(msg["timestamp"])
        else:
            with st.chat_message(ROLE_AGENT, avatar=AGENT_AVATAR):
                st.markdown(f"**{st.session_state.agent_name}**")
                st.markdown(msg["content"])
                st.caption(msg["timestamp"])


# =============================================================================
# ERROR DISPLAY
# =============================================================================
def render_error_if_any():
    if st.session_state.error_message:
        st.error(f"⚠️ {st.session_state.error_message}", icon="🚨")
        st.session_state.error_message = None


# =============================================================================
# AGENT CALL ORCHESTRATION
# =============================================================================
def process_pending_input():
    user_message = st.session_state.pending_user_input
    if user_message is None:
        return

    st.session_state.pending_user_input = None
    st.session_state.is_processing = True

    settings = {
        "system_prompt": st.session_state.system_prompt,
        "temperature": st.session_state.temperature,
        "max_tokens": st.session_state.max_tokens,
        "groq_api_key": st.session_state.groq_api_key,
        "pdf_path": st.session_state.pdf_path,
        "web_url": st.session_state.web_url,
    }

    should_search = WEB_SEARCH_AVAILABLE and (
        st.session_state.force_search_next
        or (st.session_state.web_search_enabled and looks_like_it_needs_search(user_message))
    )
    st.session_state.force_search_next = False

    search_results = None
    if should_search:
        with st.status(f"🔍 Searching the web for \"{user_message.strip()[:60]}\"...", expanded=False) as status:
            try:
                search_results = run_web_search(user_message)
                status.update(label=f"🔍 Found {len(search_results)} web result(s).", state="complete")
            except Exception as exc:
                status.update(label=f"🔍 Web search failed: {exc}", state="error")
                search_results = []

    with st.chat_message(ROLE_AGENT, avatar=AGENT_AVATAR):
        with st.spinner(f"{st.session_state.agent_name} is thinking (book + website + web agent)..."):
            try:
                history_before = get_history_for_backend()
                reply = call_agent_backend(user_message, history_before, settings, search_results=search_results)
                add_message(ROLE_AGENT, reply)
            except Exception as exc:
                error_text = f"The agent couldn't generate a response. Details: {exc}"
                st.session_state.error_message = error_text
                add_message(
                    ROLE_AGENT,
                    "⚠️ Sorry, something went wrong while generating a response. Please try again.",
                )
            finally:
                st.session_state.is_processing = False
                save_conversation()


# =============================================================================
# CHAT INPUT HANDLER
# =============================================================================
def render_chat_input():
    if WEB_SEARCH_AVAILABLE:
        col_hint, col_button = st.columns([4, 1.3])
        with col_hint:
            if st.session_state.force_search_next:
                st.caption("🔍 Web search will run before the next reply.")
        with col_button:
            if st.button(
                "🔍 Search the web", use_container_width=True,
                disabled=st.session_state.is_processing or st.session_state.force_search_next,
            ):
                st.session_state.force_search_next = True
                st.rerun()

    placeholder = "Message the agent..." if not st.session_state.is_processing else "Waiting for response..."
    user_input = st.chat_input(placeholder, disabled=st.session_state.is_processing)

    if user_input:
        cleaned = user_input.strip()
        if len(cleaned) == 0:
            st.session_state.error_message = "Please enter a non-empty message."
        elif len(cleaned) > 4000:
            st.session_state.error_message = "Message is too long (max 4000 characters). Please shorten it."
        else:
            if not st.session_state.messages:
                st.session_state.conversation_title = _derive_title_from_message(cleaned)
            add_message(ROLE_USER, cleaned)
            st.session_state.pending_user_input = cleaned
            save_conversation()
        st.rerun()


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    init_session_state()
    render_sidebar()
    render_header()
    render_error_if_any()
    render_book_gallery()
    render_chat_history()
    process_pending_input()
    render_chat_input()


if __name__ == "__main__":
    main()