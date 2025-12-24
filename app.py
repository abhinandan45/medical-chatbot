import streamlit as st
from backend import get_qa_chain
import time

# Page configuration for a professional look
st.set_page_config(page_title="MediBot Pro", page_icon="🩺", layout="wide")

# Custom CSS for better readability
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMarkdown h2 { color: #2e7d32; border-bottom: 2px solid #2e7d32; padding-bottom: 5px; }
    .stMarkdown h3 { color: #1565c0; margin-top: 20px; }
    .source-box { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #1565c0; }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Medical Intelligence System")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("📋 Menu")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.info("Directly grounded to: `medical_data.pdf`")

# Load Bot logic
@st.cache_resource
def load_bot():
    return get_qa_chain()

bot = load_bot()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Interaction
if prompt := st.chat_input("Enter your medical query here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing document structure..."):
            response_data = bot.ask(prompt)
            answer = response_data["answer"]
            sources = response_data["context"]

            # Streaming UI effect
            placeholder = st.empty()
            full_res = ""
            for word in answer.split(" "):
                full_res += word + " "
                time.sleep(0.01)
                placeholder.markdown(full_res + "▌")
            placeholder.markdown(full_res)

            # Sources Section with better design
            with st.expander("🔍 Verified Document Citations"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                    st.info(doc.page_content[:400] + "...")

    st.session_state.messages.append({"role": "assistant", "content": full_res})



