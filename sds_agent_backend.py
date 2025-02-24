
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
import os
import logging
import pdfplumber
import requests
from io import BytesIO

# IMPORTANT: If you need Anthropic, import as well; 
# otherwise, you can rely on ChatOpenAI as you had originally.
from langchain.chat_models import ChatOpenAI
#from langchain.chat_models import ChatAnthropic  # If you use Anthropic
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults

###############################################################################
# GLOBALS
###############################################################################
# For demonstration, let's stick with ChatOpenAI. 
# If you have a functioning ChatAnthropic in your environment, 
# just swap out the commented code. 
# And remember to set the correct environment variables for Anthropic if you use it.

llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.2,
            max_tokens=1024,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )

tavily_search_call_count = 0
serp_search_call_count = 0
tool_call_cache = {}

###############################################################################
# PDF Parsing
###############################################################################
def parse_sds_pdf(file_obj) -> str:
    """
    Parse text from an in-memory PDF file object using pdfplumber.
    file_obj can be a BytesIO from Streamlit's file_uploader.
    """
    text_content = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text_content.append(text)
    return "\n".join(text_content)

###############################################################################
# Tavily Tools
###############################################################################
# Initialize Tavily tool for searching:
tavily_tool_base = TavilySearchResults(
    include_answer=True, 
    include_raw_content=True,
    search_depth="advanced", 
    max_results=5
)

def tavily_search(query: str) -> str:
    global tavily_search_call_count
    st.info(f"Searching the web with Tavily for: {query}")
    tavily_search_call_count += 1
    logging.info(f"[Tavily Tool] Call #{tavily_search_call_count} with query: {query}")
    return tavily_tool_base.run(query)

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

###############################################################################
# SERP Search Tool
###############################################################################
def serp_search(query: str) -> str:
    global serp_search_call_count
    st.info(f"Searching the web with SERP for: {query}")
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
    
    # Summarize the search results
    results_summary = ""
    for i, result in enumerate(organic_results):
        title = result.get("title", "No Title")
        snippet = result.get("snippet", "No Snippet")
        link = result.get("link", "No Link")
        results_summary += f"Result {i+1}:\nTitle: {title}\nSnippet: {snippet}\nLink: {link}\n\n"
    
    # Use the LLM to pick the best SDS link or "None"
    prompt = f"""
You are an expert in chemical safety data sheets (SDS/MSDS).
Below are search results from SERPAPI. Based solely on the titles, snippets, and links provided,
select the URL that most likely points to a valid SDS or MSDS document.
If none of the results appear to be relevant SDS documents, output only the word "None".

Search Results:
{results_summary}

Output only the chosen URL or "None".
"""
    llm_response = llm.call_as_llm(prompt)
    best_link = llm_response.strip()

    if best_link.lower() == "none" or best_link == "":
        return "No SDS Document link found in the search results."
    else:
        return f"Found SDS Document: {best_link}"

###############################################################################
# Tools for Agent
###############################################################################
tavily_search_tool = Tool(
    name="tavily_search",
    func=tavily_search,
    description="Use Tavily to retrieve advanced textual data regarding SDS updates and chemical safety information."
)

serp_search_tool = Tool(
    name="serp_search",
    func=serp_search,
    description="Use SERP API to retrieve general web search results for updated SDS versions."
)

tavily_extract_tool = Tool(
    name="tavily_extract",
    func=tavily_extract,
    description="Extract raw content from a URL using Tavily Extract."
)

###############################################################################
# ReAct Agent Setup
###############################################################################
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
2. Search for the latest updated SDS for the chemical using serp_search and tavily_search (the given version may be older).
   IMPORTANT: Use both "tavily_search" and "serp_search" at least once.
    Example  query: "latest Safety data sheet from [manufacturer name], for [insert chemical]", etc. 
3. If an updated SDS link is found, also consider using tavily_extract if needed to confirm details.
4. Compare the updated SDS with the old PDF version, find any differences in hazard/constituents.
5. Provide a final answer referencing the updated SDS date and link.
6. The final answer should begin with "Final Answer:" and mention the date of the latest SDS with a link.


**DRAFT FINAL ANSWER AFTER A MAXIMUM OF 5 TOOL CALLS** 
**PLEASE DO NOT CALL TOOLS FOR MORE THAN A COMBINED 5. AFTER 5, GIVE YOUR BEST FINAL ANSWER. WE DO NOT WANT TO KEEP THE USER WAITING.**
Example:
Question: What is the latest SDS for Bird Stop?
Thought: I need to find the updated SDS, so I will use tavily_search...
Final Answer: The updated SDS [last updated on ...] for Bird Stop is located at ...

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

def extract_and_assess_sds(pdf_text: str) -> str:
    # Prepare the agent as before...
    tools_for_agent = [tavily_search_tool, serp_search_tool]
    tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in tools_for_agent])
    tool_names = ", ".join([tool.name for tool in tools_for_agent])
    
    agent = create_react_agent(
        llm=llm,
        tools=tools_for_agent,
        prompt=prompt_template,
        stop_sequence=True
    )
    
    # Use "force" (not "generate") and enable returning intermediate steps
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_agent,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=7,
        early_stopping_method="force",  # changed from "generate"
        return_intermediate_steps=True  # add this flag to capture the scratchpad
    )
    
    user_msg = "Please extract key SDS info, check for updates, and summarize."
    inputs = {
        "pdf_text": pdf_text,
        "input": user_msg,
        "tools": tools_str,
        "tool_names": tool_names,
        "agent_scratchpad": ""
    }

    # Run the agent
    result = agent_executor.invoke(inputs)
    
    # Get the final output from the result
    if isinstance(result, dict):
        final_output = result.get("output", "")
    else:
        final_output = result

    # If the final output doesn't include a final answer, do one extra LLM call
    if "Final Answer:" not in final_output:
        # Extract the intermediate steps (if any) to build a final scratchpad
        intermediate_steps = result.get("intermediate_steps", [])
        final_scratchpad = ""
        for action, observation in intermediate_steps:
            final_scratchpad += f"Action: {action}\nObservation: {observation}\n"
        
        # Rebuild the prompt with the final scratchpad included
        final_inputs = {
            "pdf_text": pdf_text,
            "input": user_msg,
            "tools": tools_str,
            "tool_names": tool_names,
            "agent_scratchpad": final_scratchpad
        }
        final_prompt = prompt_template.format(**final_inputs)
        # Call the LLM directly with the final prompt to generate the final answer
        final_llm_response = llm.call_as_llm(final_prompt)
        final_llm_text = final_llm_response.strip()
        if "Final Answer:" in final_llm_text:
            final_output = final_llm_text.split("Final Answer:", 1)[-1].strip()
        else:
            final_output = final_llm_text

    # Optionally, if your final answer always begins with "Final Answer:", remove it:
    if "Final Answer:" in final_output:
        final_output = final_output.split("Final Answer:", 1)[-1].strip()

    return final_output

