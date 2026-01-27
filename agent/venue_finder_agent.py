from crewai import Agent, Task, Crew
from crewai_tools import TavilySearchTool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a search tool
search_tool = TavilySearchTool()

# Define agents
venue_finder = Agent(
  role='Conference Venue Finder',
  goal='Find the best venue for the upcoming conference',
  backstory=(
      "You are an experienced event planner with a knack for finding the perfect venues. "
      "Your expertise ensures that all conference requirements are met efficiently."
  ),
  verbose=True,
  tools=[search_tool]
)

venue_quality_assurance_agent = Agent(
    role="Venue Quality Assurance Specialist",
    goal="Ensure the selected venues meet all quality standards and client requirements",
    backstory=(
        "You are meticulous and detail-oriented, ensuring that the venue options provided "
        "are not only suitable but also exceed the client's expectations. "
        "Your job is to review the venue options and provide detailed feedback."
    ),
    tools=[search_tool],
    verbose=True
)

# Define tasks
find_venue_task = Task(
    description=(
        "Conduct a thorough search to find the best venue for the upcoming "
        "conference. Consider factors such as capacity, location, amenities, "
        "and pricing. Use online resources and databases to gather comprehensive "
        "information."
    ),
    expected_output=(
        "A list of 5 potential venues with detailed information on capacity, "
        "location, amenities, pricing, and availability."
    ),
    agent=venue_finder
)

quality_assurance_review_task = Task(
    description=(
        "Review the venue options provided by the Conference Venue Finder. "
        "Ensure that each venue meets all the specified requirements and standards. "
        "Provide a detailed report on the suitability of each venue."
    ),
    expected_output=(
        "A detailed review of the 5 potential venues, highlighting any issues, strengths, and overall suitability."
    ),
    tools=[search_tool],
    agent=venue_quality_assurance_agent,
)

#create crew
event_planning_crew = Crew(
  agents=[venue_finder, venue_quality_assurance_agent],
  tasks=[find_venue_task, quality_assurance_review_task],
  verbose=True,
  memory=True
)

def create_event_planning_crew():
    """Factory function to create and return the event planning crew."""
    return event_planning_crew

def run_venue_search(conference_name: str, requirements: str):
    """
    Execute venue search with given conference details.

    Args:
        conference_name: Name of the conference
        requirements: Detailed venue requirements

    Returns:
        str: Crew execution result with venue recommendations
    """
    crew = create_event_planning_crew()
    inputs = {
        "conference_name": conference_name,
        "requirements": requirements
    }
    result = crew.kickoff(inputs=inputs)
    return result

if __name__ == "__main__":
    # Example usage
    result = run_venue_search(
        conference_name="AI Innovations Summit",
        requirements="Capacity for 5000, central location, modern amenities, budget up to $50,000"
    )
    print(result)
