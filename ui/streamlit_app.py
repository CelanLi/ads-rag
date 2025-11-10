# ui/streamlit_app.py
import streamlit as st
from rag.app.llm_interface import LLMInterface
from rag.embeddings.embedding_manager import EmbeddingManager
from rag.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL
from rag.retrieval.retriever import Retriever

# Page configuration
st.set_page_config(page_title="RAG System Chat", page_icon="💬", layout="wide")

# Initialize session state for chat history and model caching
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None


@st.cache_resource
def load_rag_system():
    """Load RAG system once and cache it"""
    embedding_manager = EmbeddingManager(embedding_model=DEFAULT_EMBEDDING_MODEL)
    llm_interface = LLMInterface(
        embedding_manager=embedding_manager, model_name=DEFAULT_LLM_MODEL
    )
    retriever = Retriever(embedding_manager)
    return llm_interface, retriever


# Initialize RAG system
if st.session_state.rag_system is None:
    with st.spinner("Loading RAG system..."):
        llm_interface, retriever = load_rag_system()
        st.session_state.rag_system = llm_interface
        st.session_state.retriever = retriever
else:
    llm_interface = st.session_state.rag_system
    retriever = st.session_state.retriever


# Sidebar for settings and information
with st.sidebar:
    st.title("⚙️ Settings")

    # Model information
    st.subheader("Model Information")
    st.info(
        f"**Embedding Model:** {DEFAULT_EMBEDDING_MODEL}\n\n**LLM Model:** {DEFAULT_LLM_MODEL}"
    )

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Show context toggle
    show_context = st.checkbox("Show Retrieved Context", value=False)

    st.divider()

    # Chat statistics
    st.subheader("📊 Chat Statistics")
    st.metric("Total Messages", len(st.session_state.messages))


# Main chat interface
st.title("💬 RAG System Chat Interface")
st.markdown(
    "Ask questions about your documents. The system will retrieve relevant context and generate answers."
)

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show retrieved context if enabled and available
            if show_context and message["role"] == "assistant" and "context" in message:
                with st.expander("📄 Retrieved Context", expanded=False):
                    for idx, ctx in enumerate(message["context"], 1):
                        st.markdown(f"**Context {idx}:**")
                        st.text_area(
                            label="",
                            value=ctx.get("text", ""),
                            height=100,
                            disabled=True,
                            key=f"context_{message.get('id', 0)}_{idx}",
                            label_visibility="collapsed",
                        )
                        if ctx.get("src_path"):
                            st.caption(f"Source: {', '.join(ctx['src_path'])}")
                        st.divider()


# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve context
            contexts = retriever.retrieve(prompt, top_k=3)

            # Generate response
            try:
                response = llm_interface.rag_qa(prompt)
            except Exception as e:
                st.error(f"Error generating response: {e}")
                response = "Sorry, I encountered an error while generating the response. Please try again."

            # Display response
            st.markdown(response)

            # Store assistant message with context
            message_id = len(st.session_state.messages)
            assistant_message = {
                "role": "assistant",
                "content": response,
                "context": contexts[0] if contexts else [],
                "id": message_id,
            }
            st.session_state.messages.append(assistant_message)

            # Show context if enabled
            if show_context and contexts:
                with st.expander("📄 Retrieved Context", expanded=False):
                    for idx, ctx in enumerate(contexts[0], 1):
                        st.markdown(f"**Context {idx}:**")
                        st.text_area(
                            label="",
                            value=ctx.get("text", ""),
                            height=100,
                            disabled=True,
                            key=f"context_new_{idx}",
                            label_visibility="collapsed",
                        )
                        if ctx.get("src_path"):
                            st.caption(f"Source: {', '.join(ctx['src_path'])}")
                        st.divider()

# Auto-scroll to bottom (optional enhancement)
# This would require custom JavaScript, but the chat interface should naturally scroll
