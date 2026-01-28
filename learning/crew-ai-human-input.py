from crewai import Agent, Crew, Task
from crewai_tools import YoutubeVideoSearchTool

# Method # 1
# General search across Youtube content without specifying a video URL, so the agent can search within any Youtube video content it learns about irs url during its operation
search_tool = YoutubeVideoSearchTool()

# Method # 2
# Targeted search within a specific Youtube video's content
search_tool = YoutubeVideoSearchTool(youtube_video_url='https://www.youtube.com/watch?v=6N7NLJ9ox8w')

# Define the research agent
researcher = Agent(
    role='Video Content Researcher',
    goal='Extract key insights from YouTube videos',
    backstory=(
        "You are a skilled researcher who excels at extracting valuable insights from video content. "
        "You focus on gathering accurate and relevant information from YouTube to support your team."
    ),
    verbose=True,
    tools=[search_tool],
    memory = True
)

# Define the writing agent
writer = Agent(
    role='Tech Article Writer',
    goal='Craft an article based on the research insights',
    backstory=(
        "You are an experienced writer known for turning complex information into engaging and accessible articles. "
        "Your work helps make advanced technology topics understandable to a broad audience."
    ),
    verbose=True,
    tools=[search_tool],  # The writer may also use the YouTube tool for additional context 
    memory = True
)

# Create the research task
research_task = Task(
    description=(
        "Research and extract key insights from the given YouTube video. "
        "Compile your findings in a detailed summary."
    ),
    expected_output='A summary of the key insights from the YouTube video',
    agent=researcher
)

# Create the writing task
writing_task = Task(
    description=(
        "Using the summary provided by the researcher, write a compelling article on what the video conveys. "
        "Ensure the article is well-structured and engaging for a general audience."
    ),
    expected_output='A well-written article on the subject based on the YouTube video research.',
    agent=writer,
    human_input=True  # Allow for human feedback after the draft
)

from crewai import Agent, Crew, Task

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True,
    memory=True
)

# Get your crew to work!
result = crew.kickoff()

print("Execution Completed!")