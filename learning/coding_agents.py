from crewai import Agent, Task, Crew

coding_agent = Agent(
    role="Python Data Analyst",
    goal="Write and execute Python code to perform calculations",
    backstory="You are an experienced Python developer, skilled at writing efficient code to solve problems.",
    allow_code_execution=True
)

# Create an agent with code execution enabled
coding_agent = Agent(
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

# Create a debugging agent with code execution enabled
debugging_agent = Agent(
    role="Python Debugger",
    goal="Identify and fix issues in existing Python code",
    backstory="You are an experienced Python developer with a knack for finding and fixing bugs.",
    allow_code_execution=True
)

debug_task = Task(
    description=("Review the python code"),
    agent=debugging_agent,
    expected_output="The corrected code should output the average age of the participants in the list. Provide the updated code and tell what was the bug and how you fixed it."
)

crew = Crew(
    agents=[coding_agent, debugging_agent],
    tasks=[data_analysis_task, debug_task],
    verbose=True
)

crew.kickoff()