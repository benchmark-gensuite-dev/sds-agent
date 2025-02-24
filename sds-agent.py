
import streamlit as st
import os
import logging
import pdfplumber
import requests
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic

from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
import tempfile
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
os.environ["SERPAPI_API_KEY"] = st.secrets["SERPAPI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

import streamlit as st
from sds_agent_backend import parse_sds_pdf, extract_and_assess_sds
from langchain_anthropic import ChatAnthropic
import streamlit as st
import tempfile
import os

# ----------------------------
# Custom Footer Function
# ----------------------------
def custom_footer():
    footer_html = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f5f5f5;
        color: #555;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #ddd;
    }
    </style>
    <div class="footer">
        <p>© Benchmark Gensuite 2025 | <a href="https://benchmarkgensuite.com/" target="_blank">benchmarkgensuite.com</a></p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# ----------------------------
# Main App Function
# ----------------------------
def main():
    # Configure the page
    st.set_page_config(
        page_title="Chemical Management Agent",
        page_icon="logo.png",
        layout="wide"
    )
    
    # ----------------------------
    # Sidebar Configuration
    # ----------------------------
    st.sidebar.image("logo.png", width=150)
    st.sidebar.title("Genny AI Agent Hub")
    st.sidebar.info("You're talking to the Chemical Management Agent.")
    
    # ----------------------------
    # Header with App Logo and Title
    # ----------------------------
    st.image("logo.png", width=200)
    st.title("Chemical Management Agent")
    st.text("I'm your Chemical Manaegement Agent. How can I help?")
    
    # ----------------------------
    # File Uploader and Progress Bar
    # ----------------------------
    uploaded_file = st.file_uploader("Upload SDS PDF", type=["pdf"])
    if uploaded_file is not None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Write uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            progress_bar.progress(25)
            status_text.text("Extracting text from PDF...")
            
            # Parse the PDF
            sds_text = parse_sds_pdf(tmp_file_path)
            os.unlink(tmp_file_path)  # Clean up temporary file
            
            progress_bar.progress(50)
            status_text.text("PDF text extracted.")
            
            # Show extracted text in a collapsible expander
            with st.expander("View Extracted Text"):
                st.text_area("PDF Content", sds_text, height=200)
            
            progress_bar.progress(75)
            status_text.text("Analyzing SDS content...")
            
            # Run the agent with a spinner to indicate activity
            with st.spinner("Running SDS analysis..."):
                final_answer = extract_and_assess_sds(sds_text)
            
            progress_bar.progress(100)
            status_text.text("Analysis complete.")
            
            st.subheader("Result")
            st.write(final_answer)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("Please upload a PDF file to begin.")
    
    # ----------------------------
    # Display Custom Footer
    # ----------------------------
    custom_footer()

if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
