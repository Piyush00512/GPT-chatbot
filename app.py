import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# Ollama's OpenAI-compatible endpoint
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

st.title("🤖 Local Llama Chatbot (RAG)")


# ---------------------------------------------------------------- RAG helpers
def extract_text(file) -> str:
    """Read text from an uploaded PDF / TXT / MD file."""
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping character chunks."""
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def embed(texts: list[str], embed_model: str) -> np.ndarray:
    """Embed texts with Ollama and return L2-normalised vectors."""
    vectors = []
    for i in range(0, len(texts), 32):  # batch to keep requests small
        resp = client.embeddings.create(model=embed_model, input=texts[i : i + 32])
        vectors.extend(d.embedding for d in resp.data)
    arr = np.array(vectors, dtype=np.float32)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10)


def build_index(files, embed_model, chunk_size, overlap):
    """Chunk + embed every uploaded file. Returns (chunks, vectors)."""
    chunks = []
    for f in files:
        for c in chunk_text(extract_text(f), chunk_size, overlap):
            chunks.append({"text": c, "source": f.name})
    vectors = embed([c["text"] for c in chunks], embed_model)
    return chunks, vectors


def retrieve(query, embed_model, k):
    """Return top-k (chunk, score) pairs for the query."""
    q = embed([query], embed_model)[0]
    scores = st.session_state.vectors @ q  # cosine similarity (vectors normalised)
    top = np.argsort(scores)[::-1][:k]
    return [(st.session_state.chunks[i], float(scores[i])) for i in top]


# -------------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Chat model", value="llama3.2")
    system_prompt = st.text_area("System prompt", "You are a helpful assistant.")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

    st.header("📚 RAG")
    use_rag = st.toggle("Use documents", value=True)
    embed_model = st.text_input("Embedding model", value="nomic-embed-text")
    uploaded = st.file_uploader(
        "Upload documents", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 4)
    chunk_size = st.slider("Chunk size (chars)", 300, 2000, 800, 100)
    overlap = st.slider("Chunk overlap (chars)", 0, 400, 150, 50)

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------- State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks, st.session_state.vectors = [], None
    st.session_state.index_key = None

# (Re)build the index only when files or chunk settings change
index_key = (
    tuple((f.name, f.size) for f in uploaded),
    embed_model,
    chunk_size,
    overlap,
)
if uploaded and index_key != st.session_state.index_key:
    with st.spinner("Indexing documents..."):
        try:
            st.session_state.chunks, st.session_state.vectors = build_index(
                uploaded, embed_model, chunk_size, overlap
            )
            st.session_state.index_key = index_key
        except Exception as e:
            st.sidebar.error(
                f"Indexing failed: {e}\n\nTry: `ollama pull {embed_model}`"
            )
elif not uploaded and st.session_state.index_key is not None:
    st.session_state.chunks, st.session_state.vectors = [], None
    st.session_state.index_key = None

if st.session_state.chunks:
    st.sidebar.success(
        f"Indexed {len(st.session_state.chunks)} chunks from {len(uploaded)} file(s)"
    )

# -------------------------------------------------------------------- History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📎 Sources"):
                for s in m["sources"]:
                    st.caption(f"**{s['source']}** (score {s['score']:.2f})")
                    st.write(s["text"])

# ------------------------------------------------------------------ New input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        sources = []
        sys_content = system_prompt
        try:
            # 1) Retrieve
            if use_rag and st.session_state.vectors is not None:
                hits = retrieve(prompt, embed_model, top_k)
                sources = [{**c, "score": s} for c, s in hits]
                context = "\n\n".join(
                    f"[{i + 1}] (from {c['source']})\n{c['text']}"
                    for i, (c, _) in enumerate(hits)
                )
                # 2) Augment: inject context into the system prompt
                sys_content = (
                    f"{system_prompt}\n\n"
                    "Answer using the context below. If the answer is not in the "
                    "context, say you couldn't find it in the documents. "
                    "Cite sources like [1], [2] where relevant.\n\n"
                    f"### Context\n{context}"
                )

            # 3) Generate
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            stream = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "system", "content": sys_content}] + history,
                stream=True,
            )
            reply = st.write_stream(
                chunk.choices[0].delta.content or "" for chunk in stream
            )

            if sources:
                with st.expander("📎 Sources"):
                    for s in sources:
                        st.caption(f"**{s['source']}** (score {s['score']:.2f})")
                        st.write(s["text"])
        except Exception as e:
            reply = f"Error: {e}"
            st.error(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )