# g-ai

AI/ML learning and application project featuring BERT semantic search, multi-modal search, and CrewAI agents.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/maddymz/g-ai.git
cd g-ai
python3 -m venv venv
source venv/bin/activate

# For search app
pip install -r search-app/requirements.txt
cd search-app && uvicorn api:app --reload

# For CrewAI agents
pip install -r agent/requirements.txt
streamlit run agent/venue_finder_app.py
```

## Project Structure

```
g-ai/
├── search-app/     # BERT/CLIP semantic search (jobs, images, audio, video)
├── agent/          # CrewAI multi-agent applications
├── learning/       # NLP/ML educational scripts (BERT, GPT, LangChain, CrewAI)
└── scripts/        # Test data generation utilities
```

## Main Applications

### 1. Semantic Search Engine
**Location:** `search-app/`

BERT-powered search for jobs, images, audio, and video using CLIP and ResNet-18.

```bash
cd search-app && uvicorn api:app --reload
# Open http://localhost:8000
```

### 2. CrewAI Agents
**Location:** `agent/`

Multi-agent applications using CrewAI framework.

**Venue Finder:** Find conference venues with AI agents
```bash
streamlit run agent/venue_finder_app.py
```

### 3. Learning Examples
**Location:** `learning/`

NLP/ML examples including:
- BERT, GPT-based chatbots
- LangChain chains and tools
- CrewAI conditional agents with AgentOps monitoring
- Text processing (tokenization, embeddings, fine-tuning)

```bash
python learning/crewai-conditional-agents.py
python learning/chatbot_basic.py
```

## Key Technologies

- **Search**: BERT, CLIP, ResNet-18, FastAPI
- **Agents**: CrewAI, LangChain, AgentOps
- **ML/NLP**: PyTorch, Transformers, NLTK

## Environment Variables

Create `.env` file in project root:

```env
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
AGENTOPS_API_KEY=your_key
```

## License

MIT
