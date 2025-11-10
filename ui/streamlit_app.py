# ui/streamlit_app.py
import traceback
import streamlit as st
from rag.app.llm_interface import LLMInterface
from rag.app.memory import ConversationMemory
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

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationMemory(k=10)


@st.cache_resource
def load_rag_system():
    """Load RAG system once and cache it"""
    embedding_manager = EmbeddingManager(embedding_model=DEFAULT_EMBEDDING_MODEL)
    llm_interface = LLMInterface(
        embedding_manager=embedding_manager, model_name=DEFAULT_LLM_MODEL
    )
    retriever = Retriever(embedding_manager)
    return llm_interface, retriever


# Update LLMInterface initialization
if st.session_state.rag_system is None:
    with st.spinner("Loading RAG system..."):
        embedding_manager = EmbeddingManager(embedding_model=DEFAULT_EMBEDDING_MODEL)
        llm_interface = LLMInterface(
            embedding_manager=embedding_manager,
            model_name=DEFAULT_LLM_MODEL,
        )
        retriever = Retriever(embedding_manager)
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

    st.divider()

    # Show context toggle
    show_context = st.checkbox("Show Retrieved Context", value=False)

    st.divider()

    # Chat statistics
    st.subheader("📊 Chat Statistics")
    st.metric("Total Messages", len(st.session_state.messages))

    # Conversation history settings
    st.subheader("💬 Conversation Settings")
    enable_history = st.checkbox("Enable Conversation History", value=True)
    max_history = st.slider("Max History Messages", 0, 20, 10)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_memory.clear()
        st.rerun()


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

    # Add user message to conversation memory
    st.session_state.conversation_memory.add_message("user", prompt)

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # YOUR CUSTOM RETRIEVAL - UNCHANGED
            contexts = retriever.retrieve(prompt, top_k=3)

            # Get conversation history if enabled
            chat_history = None
            if enable_history:
                # FIX: Get history (will now include the user message we just added)
                chat_history = (
                    st.session_state.conversation_memory.get_history_for_llm()
                )
                # Limit to max_history (but exclude the current user message from limit)
                # Since we want to include current query + previous history
                if len(chat_history) > max_history + 1:  # +1 for current message
                    # Keep the last max_history messages (which includes current)
                    chat_history = chat_history[-(max_history + 1) :]
                # Remove the last message (current user message) from history
                # since it will be in the current query
                if len(chat_history) > 0 and chat_history[-1]["role"] == "user":
                    chat_history = chat_history[:-1]  # Remove current user message

                print("chat_history: ", chat_history)
                print(
                    "Number of messages in memory: ",
                    len(st.session_state.conversation_memory.get_history()),
                )

            try:
                # response = "hello world"
                response = llm_interface.rag_qa(prompt, chat_history=chat_history)
            except Exception as e:
                traceback.print_exc()  # Prints full stack trace to terminal/console
                st.error(f"Error generating response: {e}")
                response = "Sorry, I encountered an error."
            st.markdown(response)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "context": contexts[0] if contexts else [],
                }
            )

            # Add assistant message to conversation memory
            st.session_state.conversation_memory.add_message("assistant", response)

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
