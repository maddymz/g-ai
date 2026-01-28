import os
from dotenv import load_dotenv
import agentops
from crewai import Agent, Crew, TaskOutput
from crewai_tools import TavilyExtractorTool
from crewai.task import Task
from crewai.tasks.conditional_task import ConditionalTask
from pydantic import BaseModel
from typing import List

# Load environment variables
load_dotenv()

# Initialize AgentOps for monitoring
agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"))

tool = TavilyExtractorTool()

data_collector = Agent(
    role="Data Collector",
    goal="Retrieve event data using Tavily tool",
    backstory="You have a knack for finding the most exciting events happening around.",
    verbose=True,
    tools=[tool],
)

data_analyzer = Agent(
    role="Data Analyzer",
    goal="Analyze the collected data",
    backstory="You're known for your analytical skills, making sense of complex datasets.",
    verbose=True,
    tools=[tool],
)

summary_creator = Agent(
    role="Summary Creator",
    goal="Produce a concise summary from the event data",
    backstory="You're a skilled writer, able to summarize information clearly and effectively.",
    verbose=True,
)

class EventsData(BaseModel):
    events: List[str]

fetch_task = Task(
    description="Collect event data for New York City using Tavily tool",
    expected_output="A list of 8 exciting events happening in NYC this week",
    agent=data_collector,
    output_pydantic=EventsData,
)


def should_fetch_more_data(output: TaskOutput) -> bool:
    return len(output.pydantic.events) < 8

verify_data_task = ConditionalTask(
    description="""
        Ensure that sufficient event data has been collected. 
        If fewer than 8 events are found, gather more using the Tavily tool.
        """,
    expected_output="An updated list of at least 8 events happening in NYC this week",
    condition=should_fetch_more_data,
    agent=data_analyzer,
)

summary_task = Task(
    description="Summarize the collected events data for NYC",
    expected_output="summary_generated",
    agent=summary_creator,
)   

# Assemble the crew with the defined agents and tasks
crew = Crew(
    agents=[data_collector, data_analyzer, summary_creator],
    tasks=[fetch_task, verify_data_task, summary_task],
    verbose=True,
    planning=True  # Retain the planning feature
)

# Execute the tasks with the crew and track with AgentOps
try:
    result = crew.kickoff()
    agentops.end_session("Success")

    print("\n" + "="*50)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("="*50)
    print(result)

except Exception as e:
    agentops.end_session("Fail")
    print(f"\nExecution failed with error: {str(e)}")
    raise