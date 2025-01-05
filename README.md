# Personal Assistant

A multi-agent system powered by LLMs to automate daily tasks such as email management and calendar scheduling.

## Features

- Intent detection using LLMs
- Email management (coming soon)
- Calendar management (coming soon)
- Extensible multi-agent architecture

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.template` to `.env` and fill in your API keys:
   ```bash
   cp .env.template .env
   ```
5. Configure your environment variables in `.env`:
   - Get OpenAI API key from: https://platform.openai.com/account/api-keys
   - Set up Google Cloud Project and get credentials from: https://console.cloud.google.com/
     - Enable Gmail API and Calendar API
     - Create OAuth 2.0 credentials

## Usage

Run the assistant:
```bash
python src/main.py chat
```

## Project Structure

- `src/`
  - `agents/` - Individual agent implementations
    - `base.py` - Base agent class
    - `manager.py` - Manager agent for orchestration
    - `intent_detector.py` - Intent detection agent
  - `config.py` - Configuration management
  - `main.py` - CLI interface

## Development

This project is under active development. Next steps:
1. Implement Email Agent
2. Implement Calendar Agent
3. Add authentication flow for Google APIs
4. Enhance intent detection capabilities
