# Conference Venue Finder

An interactive Streamlit application that uses AI agents to find and evaluate conference venues based on your requirements.

## Overview

This application leverages CrewAI to orchestrate two specialized AI agents:

1. **Venue Finder Agent** - Searches for suitable conference venues using web search
2. **Quality Assurance Agent** - Reviews and validates the venue options

Both agents work together to provide comprehensive venue recommendations tailored to your conference needs.

## Prerequisites

- Python 3.9 or higher
- Virtual environment (recommended)
- API Keys:
  - OpenAI API key (for language models)
  - Tavily API key (for web search)

## Installation

1. **Navigate to the agent directory:**
   ```bash
   cd agent
   ```

2. **Activate your virtual environment:**
   ```bash
   # From project root
   source .venv/bin/activate
   # Or if venv is in current directory
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**

   Ensure you have a `.env` file in the project root with the following keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## Running the Application

### Start the Streamlit App

From the project root directory:

```bash
streamlit run agent/venue_finder_app.py
```

Or from the agent directory:

```bash
cd agent
streamlit run venue_finder_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Using the Command Line

You can also run the venue finder directly from the command line:

```bash
python agent/venue_finder_agent.py
```

This will execute a sample search with predefined parameters.

## How to Use

1. **Enter Conference Name**: Provide a name for your conference or event

2. **Specify Requirements**: Describe your venue needs in detail. Include:
   - Expected attendance/capacity
   - Location preferences (city, region, accessibility)
   - Required amenities (A/V equipment, Wi-Fi, catering, breakout rooms)
   - Budget range or maximum budget
   - Preferred dates or time of year
   - Any special requirements (parking, accessibility features, outdoor space)

3. **Submit**: Click the "Search for Venues" button

4. **Review Results**:
   - The AI agents will search and analyze venues (typically takes 30-60 seconds)
   - View the recommended venues with detailed information
   - Expand the "Agent Logs" section to see the reasoning process

## Example Input

**Conference Name:**
```
Tech Innovation Summit 2026
```

**Requirements:**
```
Capacity for 500 attendees, located in downtown San Francisco or nearby,
modern A/V equipment, high-speed Wi-Fi, catering services, 2-3 breakout rooms,
parking available, budget up to $30,000, dates in September 2026
```

## Features

- **Interactive Web Interface**: User-friendly form-based input
- **Real-time Search**: Agents search the web for current venue information
- **Quality Assurance**: Dual-agent system ensures thorough evaluation
- **Detailed Logs**: View agent reasoning and decision-making process
- **Helpful Tips**: Sidebar guidance for optimal results

## Technical Details

- **Framework**: Streamlit for the web interface
- **AI Orchestration**: CrewAI for multi-agent coordination
- **Language Model**: OpenAI GPT models
- **Web Search**: Tavily Search API
- **Port**: Default 8501 (configurable in `.streamlit/config.toml`)

## Troubleshooting

### Missing API Keys

If you see an error about missing API keys:
1. Verify `.env` file exists in project root
2. Check that API keys are properly formatted (no spaces around `=`)
3. Ensure `python-dotenv` is installed

### Import Errors

If you encounter `ModuleNotFoundError`:
```bash
pip install -r agent/requirements.txt
```

### Slow Response Times

- Initial search typically takes 30-60 seconds
- Web searches and API calls can be affected by network speed
- Complex requirements may take longer to process

### Port Already in Use

If port 8501 is already in use:
```bash
streamlit run agent/venue_finder_app.py --server.port 8502
```

## Project Structure

```
agent/
├── venue_finder_agent.py    # Core CrewAI agent logic
├── venue_finder_app.py       # Streamlit web interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Dependencies

- `crewai` - Multi-agent orchestration framework
- `crewai-tools` - Pre-built tools for CrewAI agents
- `streamlit` - Web application framework
- `python-dotenv` - Environment variable management

## Related Files

- Configuration: `../.streamlit/config.toml`
- Environment: `../.env`
- Project docs: `../CLAUDE.md`

## Support

For issues or questions:
- Check the troubleshooting section above
- Review agent logs in the expandable section
- Verify API keys are valid and have sufficient credits

## License

Part of the g-ai project.
