
import streamlit as st
import hashlib
import html
from pathlib import Path
from datetime import datetime

from src.retriever import build_search_system, retrieve
from src.llm_generator import generate_answer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SALAAR AI | Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploaded_documents"
UPLOAD_DIR.mkdir(exist_ok=True)

TOOLS = [
    "PDF Chunks",
    "AI Assistant",
    "Questions & Answers",
    "Semantic Search"
]


# =========================================================
# CSS DESIGN
# =========================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background:
        radial-gradient(circle at 10% 5%, #35101d 0%, transparent 28%),
        radial-gradient(circle at 90% 90%, #172f20 0%, transparent 30%),
        #101014;
    color: #ffffff;
}

html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
    color: #ffffff;
}

h1, h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 800 !important;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #260c16,
        #171018 65%,
        #102018
    );
    border-right: 2px solid #a52b43;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

.brand-box {
    background: linear-gradient(
        135deg,
        #6b1428,
        #a52b43,
        #4b172d
    );
    border: 2px solid #ff647d;
    border-radius: 15px;
    padding: 20px 10px;
    text-align: center;
    margin-bottom: 20px;
}

.brand-title {
    font-size: 29px;
    font-weight: 1000;
    letter-spacing: 3px;
    color: #ffffff;
}

.brand-subtitle {
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 3px;
    color: #ffe1a6;
    margin-top: 5px;
}

.sidebar-heading {
    background: #4b1b28;
    border-left: 5px solid #ffad42;
    padding: 10px;
    border-radius: 7px;
    font-weight: 900;
    letter-spacing: 1px;
    margin-top: 17px;
    margin-bottom: 12px;
    color: #ffffff;
}

/* INPUTS */

.stTextInput input,
.stTextArea textarea {
    background: #211b22 !important;
    color: #ffffff !important;
    border: 2px solid #a64a57 !important;
    border-radius: 8px !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border: 2px solid #ffad42 !important;
}

[data-testid="stFileUploader"] {
    background: #21191f;
    border: 2px dashed #e45b65;
    border-radius: 10px;
    padding: 9px;
}

/* BUTTONS */

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(
        110deg,
        #8e1f35,
        #bd354b
    ) !important;
    color: #ffffff !important;
    border: 1px solid #f16a79 !important;
    border-radius: 9px !important;
    font-weight: 900 !important;
    transition: 0.2s ease-in-out;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(
        110deg,
        #c04437,
        #ed762e
    ) !important;
    border: 2px solid #ffd16b !important;
    color: #ffffff !important;
}

/* GREETING BOARD */

.greeting-board {
    background: linear-gradient(
        110deg,
        #6f1729 0%,
        #9e293d 42%,
        #8d3d24 72%,
        #37532b 100%
    );
    border: 2px solid #ffb34d;
    border-radius: 17px;
    padding: 23px;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px #00000070;
}

.greeting-label {
    color: #ffe1a6;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 3px;
}

.greeting-title {
    color: #ffffff;
    font-size: 29px;
    font-weight: 1000;
    margin-top: 8px;
}

.greeting-description {
    color: #ffffff;
    font-size: 13px;
    font-weight: 700;
    margin-top: 8px;
}

/* CONTENT CARDS */

.section-card {
    background: linear-gradient(
        145deg,
        #211a20,
        #19191c
    );
    border: 1px solid #6d3540;
    border-radius: 13px;
    padding: 18px;
    margin-bottom: 16px;
}

.section-heading {
    color: #ffd16b;
    font-size: 19px;
    font-weight: 1000;
    margin-bottom: 10px;
}

.stat-card {
    background: linear-gradient(
        135deg,
        #541b2b,
        #812d3b
    );
    border: 1px solid #f16a79;
    border-radius: 11px;
    padding: 15px 8px;
    text-align: center;
    min-height: 90px;
}

.stat-value {
    font-size: 25px;
    font-weight: 1000;
    color: #ffffff;
}

.stat-label {
    color: #ffe0a0;
    font-size: 11px;
    font-weight: 900;
    margin-top: 5px;
}

/* RIGHT TOOLS AREA */

.tools-heading {
    color: #ffd16b;
    font-size: 16px;
    font-weight: 1000;
    letter-spacing: 1.5px;
    text-align: center;
    padding: 12px 0;
    border-bottom: 2px solid #a64a57;
    margin-bottom: 12px;
}

.tools-subheading {
    color: #ffad69;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 1px;
    margin-top: 18px;
    margin-bottom: 8px;
}

/* ARROW BUTTONS */

.st-key-previous_arrow button,
.st-key-next_arrow button {
    width: 100% !important;
    min-height: 39px !important;
    padding: 0 !important;
    font-size: 23px !important;
    font-weight: 1000 !important;
    background: #6e1c30 !important;
    border: 2px solid #ffad42 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* TOOL BUTTONS */

.st-key-tool_pdf_chunks button,
.st-key-tool_ai_assistant button,
.st-key-tool_questions_answers button,
.st-key-tool_semantic_search button {
    width: 100% !important;
    min-height: 43px !important;
    height: auto !important;
    padding: 7px 3px !important;
    font-size: 11px !important;
    font-weight: 900 !important;
    line-height: 1.3 !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    text-align: center !important;
    border-radius: 8px !important;
}

/* WORKSPACE STATUS */

.workspace-status {
    background: linear-gradient(
        135deg,
        #24452c,
        #315b31
    );
    border: 1px solid #8dc56a;
    border-radius: 9px;
    padding: 11px 8px;
    margin-top: 8px;
    color: #ffffff;
    font-size: 11px;
    font-weight: 800;
}

/* CHAT */

.stChatMessage {
    background: #211b22;
    border: 1px solid #6d3540;
    border-radius: 10px;
}

/* EXPANDERS */

div[data-testid="stExpander"] {
    background: #1c1a1e;
    border: 1px solid #75404b;
    border-radius: 9px;
}

.streamlit-expanderHeader {
    background: #39202a !important;
    color: #ffffff !important;
    font-weight: 900 !important;
}

/* ALERTS */

div[data-testid="stAlert"] {
    background: #292126;
    border: 1px solid #d36b50;
    border-radius: 9px;
    color: #ffffff !important;
}

/* DIVIDERS */

hr {
    border-color: #75404b !important;
}

/* SCROLLBAR */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #171418;
}

::-webkit-scrollbar-thumb {
    background: #a52b43;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "current_tool": "PDF Chunks",
    "navigation_history": ["PDF Chunks"],
    "navigation_index": 0,
    "store": None,
    "metadata": [],
    "indexed_files": [],
    "messages": [],
    "qa_history": [],
    "last_search_results": [],
    "last_search_query": "",
    "user_name": "User",
    "upload_notice": "",
    "show_upload_celebration": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# NAVIGATION FUNCTIONS
# =========================================================

def navigate_to(tool):

    if tool not in TOOLS and tool != "Knowledge Base":
        return

    history = st.session_state.navigation_history
    index = st.session_state.navigation_index

    if history[index] != tool:

        history = history[:index + 1]
        history.append(tool)

        st.session_state.navigation_history = history
        st.session_state.navigation_index = len(history) - 1

    st.session_state.current_tool = tool
    st.rerun()


def go_previous():

    index = st.session_state.navigation_index

    if index > 0:

        index -= 1

        st.session_state.navigation_index = index
        st.session_state.current_tool = (
            st.session_state.navigation_history[index]
        )

        st.rerun()


def go_next():

    index = st.session_state.navigation_index
    history = st.session_state.navigation_history

    if index < len(history) - 1:

        index += 1

        st.session_state.navigation_index = index
        st.session_state.current_tool = history[index]

        st.rerun()


# =========================================================
# PDF FUNCTIONS
# =========================================================

def save_pdfs(uploaded_files):

    saved_paths = []

    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        safe_name = Path(uploaded_file.name).name

        destination = UPLOAD_DIR / f"{file_hash}_{safe_name}"

        if not destination.exists():
            destination.write_bytes(file_bytes)

        saved_paths.append(str(destination))

    return saved_paths


# =========================================================
# RETRIEVAL AND SOURCE FUNCTIONS
# =========================================================

def create_context(results):

    context_parts = []

    for result in results:

        filename = result.get("file", "Unknown PDF")
        page = result.get("page", "Unknown")
        content = result.get("text", "")

        context_parts.append(
            f"Source: {filename}\n"
            f"Page: {page}\n"
            f"Content:\n{content}"
        )

    return "\n\n".join(context_parts)


def show_sources(results):

    if not results:
        return

    with st.expander("VIEW SOURCE DOCUMENTS"):

        for index, result in enumerate(results, start=1):

            filename = result.get("file", "Unknown PDF")
            page = result.get("page", "Unknown")
            content = result.get("text", "")

            st.markdown(
                f"**SOURCE {index} — {filename} | PAGE {page}**"
            )

            st.write(content)
            st.divider()


def get_answer(question, top_k=5):

    if st.session_state.store is None:
        return "Please upload and index PDF documents first.", []

    results = retrieve(
        question,
        st.session_state.store,
        st.session_state.metadata,
        top_k=top_k
    )

    if not results:
        return "The document does not contain the answer.", []

    context = create_context(results)

    answer = generate_answer(question, context)

    return answer, results


# =========================================================
# GREETING
# =========================================================

def get_greeting():

    hour = datetime.now().hour

    if hour < 12:
        return "GOOD MORNING"

    elif hour < 17:
        return "GOOD AFTERNOON"

    return "GOOD EVENING"


def show_greeting_board():

    name = html.escape(
        st.session_state.user_name.strip() or "User"
    )

    greeting = get_greeting()

    # st.html renders the HTML directly.
    st.html(f"""
    <div class="greeting-board">

        <div class="greeting-label">
            SALAAR AI • DOCUMENT INTELLIGENCE
        </div>

        <div class="greeting-title">
            {greeting}, {name.upper()}!
        </div>

        <div class="greeting-description">
            Your intelligent document workspace is ready.
            Upload, explore and understand your PDF knowledge base.
        </div>

    </div>
    """)


# =========================================================
# DISPLAY HELPERS
# =========================================================

def show_stat(label, value):

    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))

    st.html(f"""
    <div class="stat-card">
        <div class="stat-value">{safe_value}</div>
        <div class="stat-label">{safe_label}</div>
    </div>
    """)


def show_page_heading(title, description):

    safe_title = html.escape(title)
    safe_description = html.escape(description)

    st.html(f"""
    <div class="section-card">
        <div class="section-heading">{safe_title}</div>

        <div style="color:#ffffff;font-weight:600;">
            {safe_description}
        </div>
    </div>
    """)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.html("""
    <div class="brand-box">
        <div class="brand-title">◈ SALAAR AI</div>
        <div class="brand-subtitle">DOCUMENT INTELLIGENCE</div>
    </div>
    """)

    # USER PROFILE

    st.html("""
    <div class="sidebar-heading">
        USER PROFILE
    </div>
    """)

    name_input = st.text_input(
        "Your Name",
        value=st.session_state.user_name,
        key="profile_name_input"
    )

    st.session_state.user_name = name_input.strip() or "User"

    # KNOWLEDGE BASE

    st.html("""
    <div class="sidebar-heading">
        KNOWLEDGE BASE
    </div>
    """)

    st.caption(
        "Upload PDF documents to build your AI knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if st.button(
        "BUILD KNOWLEDGE BASE",
        key="build_knowledge",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning("Please select at least one PDF.")

        else:

            try:

                with st.spinner("Processing PDF documents..."):

                    pdf_paths = save_pdfs(uploaded_files)

                    store, metadata = build_search_system(pdf_paths)

                    st.session_state.store = store
                    st.session_state.metadata = metadata

                    st.session_state.indexed_files = list(
                        dict.fromkeys(pdf_paths)
                    )

                    st.session_state.messages = []
                    st.session_state.qa_history = []
                    st.session_state.last_search_results = []
                    st.session_state.last_search_query = ""

                    st.session_state.upload_notice = (
                        "Knowledge base created successfully!"
                    )

                    st.session_state.show_upload_celebration = True

                st.rerun()

            except Exception as error:

                st.error(f"PDF processing failed: {error}")

    if st.session_state.upload_notice:
        st.success(st.session_state.upload_notice)

    # INDEX INFORMATION

    st.html("""
    <div class="sidebar-heading">
        INDEX INFORMATION
    </div>
    """)

    is_indexed = st.session_state.store is not None

    st.write(
        "Knowledge Base:",
        "🟢 Ready" if is_indexed else "🔴 Not Created"
    )

    st.write(
        "PDF Documents:",
        len(st.session_state.indexed_files)
    )

    st.write(
        "Total Chunks:",
        len(st.session_state.metadata)
    )

    if st.button(
        "OPEN KNOWLEDGE BASE",
        key="open_knowledge",
        use_container_width=True
    ):

        navigate_to("Knowledge Base")


# =========================================================
# MAIN 80% AND TOOLS 20%
# =========================================================

main_area, tool_area = st.columns(
    [4, 1],
    gap="medium"
)


# =========================================================
# RIGHT SIDE - TOOLS
# =========================================================

with tool_area:

    # PREVIOUS AND NEXT

    previous_col, next_col = st.columns(
        2,
        gap="small"
    )

    with previous_col:

        if st.button(
            "←",
            key="previous_arrow",
            help="Previous page",
            use_container_width=True
        ):
            go_previous()

    with next_col:

        if st.button(
            "→",
            key="next_arrow",
            help="Next page",
            use_container_width=True
        ):
            go_next()

    # TOOLS HEADING

    st.html("""
    <div class="tools-heading">
        YOUR TOOLS
    </div>
    """)

    # TOOL BUTTONS

    tool_keys = {
        "PDF Chunks": "tool_pdf_chunks",
        "AI Assistant": "tool_ai_assistant",
        "Questions & Answers": "tool_questions_answers",
        "Semantic Search": "tool_semantic_search"
    }

    for tool in TOOLS:

        is_active = (
            st.session_state.current_tool == tool
        )

        if st.button(
            tool,
            key=tool_keys[tool],
            type="primary" if is_active else "secondary",
            use_container_width=True
        ):
            navigate_to(tool)

    # WORKSPACE STATUS

    st.html("""
    <div class="tools-subheading">
        WORKSPACE STATUS
    </div>
    """)

    if st.session_state.store is not None:

        st.html("""
        <div class="workspace-status">
            🟢 SYSTEM READY<br><br>
            Knowledge Base Active
        </div>
        """)

    else:

        st.html("""
        <div class="workspace-status">
            🟡 WAITING FOR PDF<br><br>
            Upload documents to begin
        </div>
        """)


# =========================================================
# LEFT SIDE - MAIN WORKSPACE
# =========================================================

with main_area:

    # GREETING ALWAYS VISIBLE

    show_greeting_board()

    current_tool = st.session_state.current_tool


    # =====================================================
    # KNOWLEDGE BASE
    # =====================================================

    if current_tool == "Knowledge Base":

        show_page_heading(
            "KNOWLEDGE BASE",
            "Overview of your indexed PDF documents and extracted content."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            show_stat(
                "PDF DOCUMENTS",
                len(st.session_state.indexed_files)
            )

        with col2:
            show_stat(
                "TOTAL CHUNKS",
                len(st.session_state.metadata)
            )

        with col3:
            show_stat(
                "INDEX STATUS",
                "READY" if st.session_state.store is not None
                else "WAITING"
            )

        st.markdown("### INDEXED DOCUMENTS")

        if st.session_state.indexed_files:

            for path in st.session_state.indexed_files:

                safe_filename = html.escape(Path(path).name)

                st.html(f"""
                <div class="section-card">
                    📄 {safe_filename}
                </div>
                """)

        else:

            st.info("No PDF documents have been indexed yet.")


    # =====================================================
    # PDF CHUNKS
    # =====================================================

    elif current_tool == "PDF Chunks":

        show_page_heading(
            "PDF CHUNK EXPLORER",
            "Explore text extracted from your PDF documents."
        )

        if not st.session_state.metadata:

            st.warning(
                "No PDF chunks are available. Upload PDF documents "
                "from the sidebar and build your knowledge base."
            )

        else:

            col1, col2 = st.columns(2)

            with col1:
                show_stat(
                    "TOTAL CHUNKS",
                    len(st.session_state.metadata)
                )

            with col2:
                show_stat(
                    "PDF DOCUMENTS",
                    len(st.session_state.indexed_files)
                )

            filenames = sorted(set(
                item.get("file", "Unknown PDF")
                for item in st.session_state.metadata
            ))

            selected_file = st.selectbox(
                "FILTER BY PDF",
                ["All Documents"] + filenames
            )

            filtered_chunks = st.session_state.metadata

            if selected_file != "All Documents":

                filtered_chunks = [
                    item for item in filtered_chunks
                    if item.get("file") == selected_file
                ]

            st.write(f"Showing {len(filtered_chunks)} chunks")

            for index, chunk in enumerate(
                filtered_chunks,
                start=1
            ):

                filename = chunk.get("file", "Unknown PDF")
                page = chunk.get("page", "Unknown")
                content = chunk.get("text", "")

                with st.expander(
                    f"CHUNK {index} | {filename} | PAGE {page}"
                ):

                    st.write(content)


    # =====================================================
    # AI ASSISTANT
    # =====================================================

    elif current_tool == "AI Assistant":

        show_page_heading(
            "SALAAR AI ASSISTANT",
            "Ask questions about your uploaded PDF documents."
        )

        if st.session_state.store is None:

            st.warning(
                "Upload and index PDF documents first."
            )

        else:

            for message in st.session_state.messages:

                with st.chat_message(message["role"]):

                    st.markdown(message["content"])

                    if message.get("sources"):
                        show_sources(message["sources"])

            question = st.chat_input(
                "Ask something about your documents..."
            )

            if question:

                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })

                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):

                    with st.spinner(
                        "Searching documents and generating answer..."
                    ):

                        try:

                            answer, results = get_answer(question)

                            st.markdown(answer)
                            show_sources(results)

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": results
                            })

                        except Exception as error:

                            error_message = (
                                f"Unable to generate answer: {error}"
                            )

                            st.error(error_message)

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_message
                            })


    # =====================================================
    # QUESTIONS AND ANSWERS
    # =====================================================

    elif current_tool == "Questions & Answers":

        show_page_heading(
            "QUESTIONS & ANSWERS",
            "Ask questions and receive answers supported by your PDFs."
        )

        if st.session_state.store is None:

            st.warning(
                "Upload and index PDF documents first."
            )

        else:

            with st.form("qa_form"):

                question = st.text_area(
                    "ENTER YOUR QUESTION",
                    placeholder="Example: Explain while loops in Python.",
                    height=110
                )

                submitted = st.form_submit_button(
                    "GENERATE ANSWER"
                )

            if submitted:

                if not question.strip():

                    st.warning("Please enter a question.")

                else:

                    with st.spinner(
                        "Finding relevant information..."
                    ):

                        try:

                            answer, results = get_answer(question)

                            st.html("""
                            <div class="section-card">
                                <div class="section-heading">
                                    AI ANSWER
                                </div>
                            </div>
                            """)

                            st.markdown(answer)
                            show_sources(results)

                            st.session_state.qa_history.insert(
                                0,
                                {
                                    "question": question,
                                    "answer": answer,
                                    "sources": results
                                }
                            )

                        except Exception as error:

                            st.error(f"Error: {error}")

            if st.session_state.qa_history:

                st.markdown("### QUESTION HISTORY")

                for index, item in enumerate(
                    st.session_state.qa_history,
                    start=1
                ):

                    with st.expander(
                        f"QUESTION {index}: {item['question']}"
                    ):

                        st.markdown("**ANSWER**")
                        st.write(item["answer"])
                        show_sources(item["sources"])


    # =====================================================
    # SEMANTIC SEARCH
    # =====================================================

    elif current_tool == "Semantic Search":

        show_page_heading(
            "SEMANTIC SEARCH",
            "Find relevant PDF content using meaning-based search."
        )

        if st.session_state.store is None:

            st.warning(
                "Upload and index PDF documents first."
            )

        else:

            with st.form("semantic_search_form"):

                search_query = st.text_input(
                    "SEARCH YOUR DOCUMENTS",
                    placeholder="Example: Python looping statements"
                )

                top_k = st.slider(
                    "NUMBER OF RESULTS",
                    min_value=1,
                    max_value=10,
                    value=5
                )

                search_submitted = st.form_submit_button(
                    "SEARCH DOCUMENTS"
                )

            if search_submitted:

                if not search_query.strip():

                    st.warning("Enter a search query.")

                else:

                    with st.spinner(
                        "Searching for relevant chunks..."
                    ):

                        try:

                            results = retrieve(
                                search_query,
                                st.session_state.store,
                                st.session_state.metadata,
                                top_k=top_k
                            )

                            st.session_state.last_search_results = results
                            st.session_state.last_search_query = search_query

                        except Exception as error:

                            st.error(f"Search failed: {error}")

            if st.session_state.last_search_query:

                safe_query = html.escape(
                    st.session_state.last_search_query
                )

                st.html(f"""
                <div class="section-heading">
                    RESULTS FOR: {safe_query}
                </div>
                """)

                results = st.session_state.last_search_results

                if not results:

                    st.warning(
                        "No relevant content was found."
                    )

                else:

                    st.success(
                        f"{len(results)} relevant results found."
                    )

                    for index, result in enumerate(
                        results,
                        start=1
                    ):

                        filename = result.get("file", "Unknown PDF")
                        page = result.get("page", "Unknown")
                        content = result.get("text", "")

                        with st.expander(
                            f"RESULT {index} | {filename} | PAGE {page}",
                            expanded=(index == 1)
                        ):

                            st.write(content)


# =========================================================
# UPLOAD SUCCESS
# =========================================================

if st.session_state.show_upload_celebration:

    st.balloons()

    st.session_state.show_upload_celebration = False
    st.session_state.upload_notice = ""
