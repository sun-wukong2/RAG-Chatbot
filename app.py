import streamlit as st
from test_rag import query_and_validate
from query_data import query_rag

st.set_page_config(page_title="RAG Demo", layout="wide")

st.markdown("<h1>RAG Query Demo</h1>", unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.markdown("<h3>⚙️ Settings</h3>", unsafe_allow_html=True)
mode = st.sidebar.radio("Select Mode:", ("Query Only", "Query & Validate"))

# Main interface
col1, col2 = st.columns(2)

with col1:
    question = st.text_input("Enter your question:")

with col2:
    if mode == "Query & Validate":
        expected_response = st.text_input("Expected response:")
    else:
        expected_response = None

if st.button("Submit", type="primary"):
    if question:
        with st.spinner("Processing..."):
            response_text = query_rag(question)
            st.success("Query completed!")

        st.markdown("### Response")
        st.write(response_text)

        if mode == "Query & Validate" and expected_response:
            with st.spinner("Validating..."):
                result = query_and_validate(question, expected_response)
                if result:
                    st.success("✅ Response matches expected output")
                else:
                    st.error("❌ Response does not match expected output")
    else:
        st.warning("Please enter a question")
