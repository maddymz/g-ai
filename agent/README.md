# CrewAI Multi-Agent Applications

AI agent applications built with CrewAI framework for collaborative task execution.

## Available Agents

### Conference Venue Finder

Finds and evaluates conference venues using two specialized agents:
- **Venue Finder Agent**: Searches for venues using web search
- **Quality Assurance Agent**: Reviews and validates options

```bash
streamlit run agent/venue_finder_app.py
# Open http://localhost:8501
```

## Setup

### Prerequisites

- Python 3.9+
- API keys for OpenAI and Tavily

### Installation

```bash
# From project root
pip install -r agent/requirements.txt
```

### Environment Configuration

Create `.env` file in project root:

```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

## Usage

### Venue Finder

**Web Interface:**
```bash
streamlit run agent/venue_finder_app.py
```

**Command Line:**
```bash
python agent/venue_finder_agent.py
```

**Input Parameters:**
- Conference name
- Venue requirements (capacity, location, amenities, budget, dates)

**Output:**
- Recommended venues with detailed analysis
- Agent reasoning logs

## Project Structure

```
agent/
├── venue_finder_agent.py    # CrewAI agent definitions
├── venue_finder_app.py       # Streamlit web interface
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Dependencies

- `crewai>=0.28.0` - Multi-agent orchestration
- `crewai-tools>=0.2.0` - Agent tools (web search, etc.)
- `streamlit>=1.30.0` - Web interface
- `python-dotenv>=1.0.0` - Environment variables

## Troubleshooting

**Missing API Keys:**
- Verify `.env` file exists in project root
- Check keys are properly formatted (no spaces)

**Import Errors:**
```bash
pip install -r agent/requirements.txt
```

**Port Conflicts:**
```bash
streamlit run agent/venue_finder_app.py --server.port 8502
```

## Related Examples

See `learning/` directory for CrewAI examples:
- `crew-ai-human-input.py` - Human-in-the-loop agents
- `crewai-conditional-agents.py` - Conditional tasks with AgentOps monitoring

## License

Part of the g-ai project.
