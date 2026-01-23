---
name: run-python
description: Run Python scripts using the virtual environment
---

You are tasked with running Python scripts using the project's virtual environment.

## Virtual Environment Path
The virtual environment is located at: `/Users/madhukarraj/personal/g-ai/venv`

## Instructions

1. When the user asks to run a Python script, use the full path to the Python interpreter in the virtual environment
2. The command format should be: `/Users/madhukarraj/personal/g-ai/venv/bin/python <script_path>`
3. If the script path is relative, make sure to use the correct path from the project root: `/Users/madhukarraj/personal/g-ai/`
4. Common script locations:
   - Learning scripts: `learning/`
   - Search application: `search-app/`
   - Utility scripts: `scripts/`

## Example Usage

To run a script in the learning directory:
```bash
/Users/madhukarraj/personal/g-ai/venv/bin/python learning/lang-chain.py
```

To run the main search application:
```bash
/Users/madhukarraj/personal/g-ai/venv/bin/python search-app/job_search_bert.py
```

## Notes
- Always use the venv Python interpreter, not the system Python
- If a script requires environment variables (like API keys), ensure they are loaded from the `.env` file
- Check the script's requirements before running - some scripts may need additional packages installed
