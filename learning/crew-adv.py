from crewai import Agent, Task, Crew
import asyncio

# Create an agent with code execution enabled
coding_agent = Agent(
    role="Python Data Analyst",
    goal="Write and execute Python code to perform calculations",
    backstory="You are an experienced Python developer, skilled at writing efficient code to solve problems.",
    allow_code_execution=True
)

coding_agent_1 = Agent(
    role="Python Data Analyst",
    goal="Write and execute Python code to perform calculations",
    backstory="You are an experienced Python developer, skilled at writing efficient code to solve problems.",
    allow_code_execution=True
)

# Define the task with explicit instructions to generate and execute Python code
data_analysis_task = Task(
    description=(
        "Write Python code to calculate the average of the following list of ages: [23, 35, 31, 29, 40]. "
        "Output the result in the format: 'The average age of participants is: <calculated_average_age>'"
    ),
    agent=coding_agent,
    expected_output="The generated code based on the requirments and the average age of participants is: <calculated_average_age>."
)

data_analysis_task_1 = Task(
    description=(
        "Write Python code to calculate the sum of the following list of ages: [23, 35, 31, 29, 40]. "
        "Output the result in the format: 'The sum of the ages of participants is: <calculated_sum_of_ages>'"
    ),
    agent=coding_agent_1,
    expected_output="The generated code based on the requirments and the sum of ages of participants is: <calculated_sum_of_ages>."
)

# Create a crew and add the task
analysis_crew = Crew(
    agents=[coding_agent],
    tasks=[data_analysis_task], 
    verbose=True
)

analysis_crew_1 = Crew(
    agents=[ coding_agent_1],
    tasks=[ data_analysis_task_1], 
    verbose=True
)

# List of datasets to analyze
datasets = [
  { "ages": [25, 30, 35, 40, 45] },
  { "ages": [20, 25, 30, 35, 40] },
  { "ages": [30, 35, 40, 45, 50] }
]

async def kickoff_async():
    result_1 = await analysis_crew_1.kickoff_async(inputs={"ages": [1, 20, 30, 40, 50]})

asyncio.run(kickoff_async())
result = analysis_crew.kickoff_for_each(inputs=datasets)

