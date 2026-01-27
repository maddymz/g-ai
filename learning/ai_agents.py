# Import necessary modules from LangChain
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langsmith import Client
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set LangChain project name (optional for tracing)
os.environ['LANGCHAIN_PROJECT'] = 'AI Agents Learning'

# Get the prompt to use - you can modify this!
client = Client()
prompt = client.pull_prompt("hwchase17/react")

# Define the tools the agent will use
tools = [TavilySearch(max_results=1)]

# Choose the LLM (Large Language Model) to use
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Construct the ReAct agent using the LLM, tools, and prompt
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Invoke the agent executor with a specific input
response = agent_executor.invoke({"input": "What is Educative?"})

# Print the response
print(response['output'])