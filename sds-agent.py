
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

st.cache_data.clear()

llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.2,
            max_tokens=1024,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize session state for counters
if 'tavily_search_call_count' not in st.session_state:
    st.session_state.tavily_search_call_count = 0
if 'serp_search_call_count' not in st.session_state:
    st.session_state.serp_search_call_count = 0

# Configure page
st.set_page_config(page_title="SDS Analyzer", layout="wide")
st.title("Safety Data Sheet (SDS) Analyzer")

# Sidebar for API keys
with st.sidebar:
    st.header("API Configuration")
    anthropic_api_key = st.text_input("Anthropic API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    serpapi_api_key = st.text_input("SerpAPI Key", type="password")
    
    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key
    if serpapi_api_key:
        os.environ["SERPAPI_API_KEY"] = serpapi_api_key

################################################################################
# PDF Parsing Function:
################################################################################
def parse_sds_pdf(pdf_file) -> str:
    text_content = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text_content.append(page.extract_text() or "")
    return "\n".join(text_content)

################################################################################
# Tool Definitions:
################################################################################
################################################################################
# Tool Definitions:
################################################################################
# Initialize Tavily tool:
tavily_tool_base = TavilySearchResults(
    include_answer=True, 
    include_raw_content=True,
    search_depth="advanced", 
    max_results=5
)
def tavily_extract(url: str) -> str:
    global tool_call_cache
    key = ("tavily_extract", url)
    if key in tool_call_cache:
        logging.info(f"[Tavily Extract] Cache hit for URL: {url}")
        return tool_call_cache[key]
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable not set."
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"urls": url, "include_images": False, "extract_depth": "basic"}
    try:
        response = requests.post("https://api.tavily.com/extract", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        if result.get("results") and len(result["results"]) > 0:
            final_result = result["results"][0].get("raw_content", "No raw content found.")
        else:
            final_result = "No results found."
    except Exception as e:
        final_result = f"Error during extraction: {str(e)}"
    tool_call_cache[key] = final_result
    return final_result


def tavily_search(query: str) -> str:
    global tavily_search_call_count
    tavily_search_call_count += 1
    logging.info(f"[Tavily Tool] Call #{tavily_search_call_count} with query: {query}")
    return tavily_tool_base.run(query)

def serp_search(query: str) -> str:
    global serp_search_call_count
    serp_search_call_count += 1
    logging.info(f"[SERPAPI] Call #{serp_search_call_count} with query: {query}")
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY environment variable not set."
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": "10"
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        logging.error(f"Error during SERPAPI call: {e}")
        return f"Error during SERPAPI call: {e}"
    
    organic_results = results.get("organic_results", [])
    if not organic_results:
        return "No organic search results found."
    
    # Build a summary of the search results with key metadata
    results_summary = ""
    for i, result in enumerate(organic_results):
        title = result.get("title", "No Title")
        snippet = result.get("snippet", "No Snippet")
        link = result.get("link", "No Link")
        results_summary += f"Result {i+1}:\nTitle: {title}\nSnippet: {snippet}\nLink: {link}\n\n"
    
    # Create a prompt for the LLM to identify the best SDS/MSDS URL.
    prompt = f"""
You are an expert in chemical safety data sheets (SDS/MSDS).
Below are search results from SERPAPI. Based solely on the titles, snippets, and links provided,
select the URL that most likely points to a valid SDS or MSDS document.
If none of the results appear to be relevant SDS documents, output only the word "None".

Search Results:
{results_summary}

Output only the chosen URL or "None".
"""
    # Invoke the LLM (using the existing llm instance)
    llm_response = llm.call_as_llm(prompt)
    best_link = llm_response.strip()

    if best_link.lower() == "none" or best_link == "":
        return "No SDS Document link found in the search results."
    else:
        return f"Found SDS Document: {best_link}"


# Wrap functions as LangChain Tools:
tavily_search_tool = Tool(
    name="tavily_search",
    func=tavily_search,
    description="Use Tavily to retrieve advanced textual data regarding SDS updates and chemical safety information."
)
serp_search_tool = Tool(
    name="serp_search",
    func=serp_search,
    description="Use SERP API to retrieve general web search results, e.g., for updated SDS versions."
)
tavily_extract_tool = Tool(
    name="tavily_extract",
    func=tavily_extract,
    description="Extract raw content from a URL using Tavily Extract."
)


################################################################################
# Main Streamlit Interface
################################################################################

# File uploader
st.write("Upload a Safety Data Sheet (PDF format)")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Update progress
        progress_bar.progress(25)
        status_text.text("Extracting text from PDF...")
        
        # Parse PDF
        sds_text = parse_sds_pdf(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Update progress
        progress_bar.progress(50)
        status_text.text("Analyzing SDS content...")
        
        # Display extracted text in expander
        with st.expander("View Extracted Text"):
            st.text_area("PDF Content", sds_text, height=200)
        
        # Initialize LLM with Claude
        
        
        # Prepare tools and prompt
        tools_for_agent = [tavily_search_tool, serp_search_tool, tavily_extract_tool]
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in tools_for_agent])
        tool_names = ", ".join([tool.name for tool in tools_for_agent])
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["pdf_text", "input", "tools", "tool_names", "agent_scratchpad"],
            template="""
            You are an expert in SDS (Safety Data Sheets).

            You have been provided a raw text from an SDS PDF:
            --------------------
            {pdf_text}
            --------------------

            You have access to the following tools:
            {tools}
            Tool Names: {tool_names}
            Current scratchpad: {agent_scratchpad}

            Process:
            1. Analyze the provided PDF text and identify the chemical and manufacturer.
            2. Search for the latest updated SDS for the chemical using serp_search and tavily_search - just call them both once each time.
            	Example tavily query: "Chemical Agent [insert name] from [manufacturer name] latest sds 2025 2024"
            3. If an SDS is found, select the best candidate.
            4. Use tavily_extract if needed to scrape the specific page.
            5. Extract hazards, chemical constituents, and version/update info.
            6. Compare with the provided PDF.
            7. Provide a preliminary assessment and recommended actions.
            8. Output your final answer starting with "Final Answer:".


            **Please actually call the tool and not just in your scratchpad**

             ## **INSTRUCTION FOR TOOL USE**
			    - When you need to perform a search, use the following **exact format**:
			      ```
			      {{ "action": "serp_search", "action_input": "Search query here" }}
			      ```
			      ```
			      {{ "action": "tavily_search", "action_input": "Search query here" }}
			      ```
			    - **DO NOT** generate Python function calls (`tavily_search("...")`)—ONLY use the JSON format.
			    - After retrieving search results, summarize the key updates.


			** DO NOT RUN MORE THAN 4 TIMES**



            ##Example output format:
            FINAL ANSWER:
			The latest available SDS for QUINTOLUBRIC 888-46 was found dated September 11, 2019, which is more recent than the provided version (1.01 from 04/14/2016).

			Latest SDS Details:
			- Product Name: QUINTOLUBRIC 888-46
			- Latest Version Date: September 11, 2019
			- Document Number: SDS 0061933
			- Available at: https://prolube.com.au/?wpfb_dl=710

			Key Changes/Updates from Original:
			1. Classification Status: Product remains non-hazardous
			2. Hazard Identification: No reportable hazardous substances
			3. Physical/Chemical Properties: Remain consistent with original SDS
			4. Environmental Information: Maintains 86.5% biodegradability rating

			Recommendations:
			1. Update to the 2019 version of the SDS for compliance purposes
			2. Note that while newer technical data sheets are available, the 2019 SDS appears to be the most recent full safety data sheet
			3. Continue monitoring for future updates as regulatory requirements evolve


            User Query: {input}
            Agent:"""
        )
        
        # Create agent
        agent = create_react_agent(
            llm=llm,
            tools=[tavily_search_tool, serp_search_tool, tavily_extract_tool],
            prompt=prompt_template,
            stop_sequence=True
        )
        
        # Create executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[tavily_search_tool, serp_search_tool, tavily_extract_tool],
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Update progress
        progress_bar.progress(75)
        status_text.text("Running analysis...")
        
        # Prepare input
        user_msg = "Please extract key SDS info, check for updates online using both tools, and provide a final summarized SDS assessment with references."
        inputs = {
            "pdf_text": sds_text,
            "input": user_msg,
            "tools": tools_str,
            "tool_names": tool_names,
            "agent_scratchpad": ""
        }
        
        # Run analysis
        with st.spinner('Analyzing SDS...'):
            result = agent_executor.invoke(inputs)
            
            # Extract final answer
            if isinstance(result, dict):
                output_text = result.get("output", "")
            else:
                output_text = result
                
            if "Final Answer:" in output_text:
                final_summary = output_text.split("Final Answer:")[-1].strip()
            else:
                final_summary = output_text.strip()
        
        # Update progress
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Display results
        st.header("Analysis Results")
        st.markdown(final_summary)
        
        # Display call counts in sidebar
        with st.sidebar:
            st.header("API Usage")
            st.write(f"Tavily Search Calls: {st.session_state.tavily_search_call_count}")
            st.write(f"SERP API Calls: {st.session_state.serp_search_call_count}")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error processing file: {e}")
    
    finally:
        # Clear progress bar and status
        progress_bar.empty()
        status_text.empty()

else:
    st.info("Please upload a PDF file to begin analysis.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
