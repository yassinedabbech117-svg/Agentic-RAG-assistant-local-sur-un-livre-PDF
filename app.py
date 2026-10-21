
import streamlit as st
from rag_service import initialize_service, ask_question
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_UNIT, FINAL_TOP_K
@st.cache_resource
def load_rag():
    return initialize_service()
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: #0b1020;
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: #101729;
    border-right: 1px solid #263148;
}
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}
.hero {
    padding: 1.5rem;
    border: 1px solid #29334d;
    border-radius: 18px;
    background: linear-gradient(135deg, #151e35, #17152e);
    margin-bottom: 1.5rem;
}
.hero h1 {
    margin: 0;
    font-size: 2rem;
    color: #f8fafc;
}
.hero p { color: #a5b4fc; margin-bottom: 0; }
.status {
    color: #86efac;
    font-size: 0.85rem;
}
[data-testid="stChatMessage"] {
    border: 1px solid #27334b;
    border-radius: 14px;
    background: #111a2d;
    padding: 1rem;
}
div[data-testid="stChatInput"] textarea {
    background: #111a2d;
}
section[data-testid="stSidebar"] h2 {
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = {}

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🧠 Agentic RAG")
    st.caption("LOCAL KNOWLEDGE ASSISTANT")
    st.divider()

    st.markdown("### 📚 Knowledge base")
    st.markdown("**The Hitchhiker’s Guide to Agentic AI**")
    st.caption("PDF · Recherche sémantique + mots-clés · Ollama")
    st.caption(
        f"Chunks : {CHUNK_SIZE} {CHUNK_UNIT} · chevauchement : "
        f"{CHUNK_OVERLAP}"
    )

    st.divider()
    st.markdown("### ⚙️ Retrieval settings")

    top_k = st.slider(
        "Retrieved passages",
        min_value=2,
        max_value=10,
        value=FINAL_TOP_K,
    )

    st.caption("Nombre de passages fournis au modèle.")

    st.divider()
    st.markdown("### 🤖 Language model")
    model_name = st.selectbox(
        "Ollama model",
        ["qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b"],
        index=0,
    )

    st.caption("Le modèle doit être installé dans Ollama.")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.rerun()

    st.caption("Built with Streamlit · FAISS · BM25 · Ollama")

# ---------------- MAIN HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>Agentic RAG <span style="color:#a78bfa">✳</span></h1>
    <p>Explore your knowledge. Ask better questions.</p>
    <br>
    <span class="status">● Local AI workspace</span>
</div>
""", unsafe_allow_html=True)

# ---------------- STATUS CARDS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Retrieval", "Semantic + BM25")

with col2:
    st.metric("Vector index", "FAISS · Cosine")

with col3:
    st.metric("LLM", model_name)

st.markdown("### 💬 Ask your knowledge base")

if not st.session_state.messages:
    st.info(
        "Pose une question sur le livre. À la première question, le PDF et "
        "l’index sémantique sont préparés ; cette étape peut prendre un certain "
        "temps, surtout si Ollama utilise le processeur. Les questions suivantes "
        "réutilisent l’index. Ollama doit être démarré pour générer une réponse."
    )

    prompts = [
        "Explain the Transformer architecture",
        "What is the ReAct agent pattern?",
        "Explain RLHF, PPO and DPO",
    ]

    cols = st.columns(len(prompts))
    for i, prompt in enumerate(prompts):
        with cols[i]:
            if st.button(prompt, use_container_width=True):
                st.session_state.pending_question = prompt
                st.rerun()

# Render conversation history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = st.session_state.sources.get(i, [])
            if sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for source in sources:
                        st.markdown(
                            f"**Page {source.get('page_start', '?')}**"
                        )
                        st.caption(source.get("text", ""))
                        st.divider()

# Question input
question = st.chat_input("Ask anything about the book...")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")


if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    try:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.status(
                "Préparation des passages pertinents…",
                expanded=True,
            ) as thinking_status:
                thinking_status.write(
                    "Au premier lancement, le PDF et son index sémantique "
                    "sont préparés. Cette étape peut prendre un certain temps "
                    "si le processeur est utilisé. "
                    "La réponse s’affichera ensuite progressivement."
                )
                rag_state = load_rag()
                thinking_status.update(
                    label="Je réfléchis à votre question…",
                    state="running",
                )

                result = ask_question(
                    question,
                    rag_state,
                    top_k=top_k,
                    model_name=model_name,
                    stream=True,
                )
            # Keep generated text outside the status panel so collapsing that
            # panel never hides the answer along with the progress details.
            thinking_status.update(
                label="Génération de la réponse…",
                state="running",
                expanded=True,
            )
            answer = st.write_stream(result["answer"])
            thinking_status.update(
                label="Réponse prête",
                state="complete",
                expanded=False,
            )

            sources = result["sources"]

            if sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for source in sources:
                        start = source.get("page_start", "?")
                        end = source.get("page_end", start)

                        page_label = (
                            f"Page {start}"
                            if start == end
                            else f"Pages {start}-{end}"
                        )

                        st.markdown(f"**{page_label}**")
                        st.caption(source.get("text", ""))
                        st.divider()

        # Store the assistant message and its sources together.
        assistant_index = len(st.session_state.messages)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })

        st.session_state.sources[assistant_index] = sources

    except Exception as error:
        error_message = f"Erreur pendant le traitement : {error}"

        with st.chat_message("assistant"):
            st.error(error_message)

        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message,
        })

    st.rerun()
