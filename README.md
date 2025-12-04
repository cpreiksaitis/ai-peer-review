# Multi-Agent Peer Review System

A multi-agent system for peer reviewing biomedical manuscripts. Uses multiple specialized AI agents that debate and synthesize reviews, producing feedback similar to what a human peer reviewer would provide.

## Features

- **5 Specialized Agents**: Orchestrator + 4 reviewer agents with distinct expertise
- **Multi-Model Diversity**: Uses OpenAI, Anthropic, and Google models for cognitive diversity
- **Literature Grounding**: Searches PubMed for related work to contextualize reviews
- **Structured Debate**: Agents discuss and debate their assessments
- **Biomedical Focus**: Prompts tailored for clinical research, emergency medicine, and medical education
- **Cost Tracking**: Real-time cost tracking and estimates for all API calls
- **Web Interface**: Modern web UI for uploading manuscripts and viewing reviews
- **Review History**: SQLite database for persisting and tracking all reviews

## Agent Roles

| Agent | Model | Focus |
|-------|-------|-------|
| **Orchestrator** | Claude | Workflow management, consensus synthesis |
| **Methodologist** | GPT-4o | Study design, statistics, reproducibility |
| **Domain Expert** | Gemini 1.5 Pro | Clinical relevance, novelty, field context |
| **Communication** | Claude | Writing clarity, figures, organization |
| **Ethics** | GPT-4o | Research ethics, patient safety, limitations |

## Installation

```bash
# Clone the repository
cd reviewer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Configuration

### API Keys

Set the following environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."  # or GOOGLE_API_KEY
export PUBMED_EMAIL="your.email@example.com"  # Required for PubMed API
```

### Config File

Edit `config.yaml` to customize model assignments and settings:

```yaml
models:
  orchestrator: "claude-sonnet-4-20250514"
  methodologist: "gpt-4o"
  domain_expert: "gemini/gemini-1.5-pro"
  communication: "claude-sonnet-4-20250514"
  ethics: "gpt-4o"

debate:
  rounds: 2
  temperature: 0.7

literature:
  max_results: 10
```

## Usage

### Web Interface (Recommended)

```bash
# Start the web server
reviewer web

# With custom host/port
reviewer web --host 0.0.0.0 --port 8080

# With auto-reload for development
reviewer web --reload
```

Then open http://127.0.0.1:8000 in your browser.

### Command Line Review

```bash
reviewer review manuscript.pdf
```

### CLI Options

```bash
# Skip literature search
reviewer review manuscript.pdf --no-literature

# Custom output directory
reviewer review manuscript.pdf --output ./my_reviews

# Specify PubMed email
reviewer review manuscript.pdf --email your.email@example.com

# More debate rounds
reviewer review manuscript.pdf --rounds 3

# Don't save debate transcript
reviewer review manuscript.pdf --no-save-debate
```

### Cost Estimation

```bash
# Estimate cost before running
reviewer estimate --length 50000 --rounds 2
```

### Check Configuration

```bash
reviewer check
```

### List Agents

```bash
reviewer agents
```

## Output

The system generates:

1. **Final Review** (`manuscript_review_TIMESTAMP.md`):
   - Summary of the manuscript
   - Major comments (critical issues)
   - Minor comments (suggestions)
   - Questions for authors
   - Recommendation with justification

2. **Debate Log** (`manuscript_debate_TIMESTAMP.md`):
   - Initial reviews from all agents
   - Debate round responses
   - Final positions

## How It Works

1. **Document Ingestion**: Extracts text from PDF using PyMuPDF
2. **Literature Search**: Queries PubMed for related papers via PyPaperRetriever
3. **Initial Reviews**: Each agent independently reviews the manuscript
4. **Debate Rounds**: Agents see each other's reviews and respond
5. **Consensus Synthesis**: Orchestrator produces final balanced review

## Biomedical Focus

The agents are aware of:
- Reporting guidelines: CONSORT, STROBE, PRISMA, CARE, SQUIRE, STARD
- Evidence-based medicine principles
- Emergency medicine clinical workflow considerations
- Medical education frameworks (Kirkpatrick model, competency-based education)

## Dependencies

- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM API
- [PyPaperRetriever](https://github.com/josephisaacturner/pypaperretriever) - PubMed integration
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF extraction
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting

## Project Structure

```
reviewer/
├── pyproject.toml          # Dependencies and project config
├── config.yaml             # Model assignments and settings
├── .env                    # API keys (create from .env.example)
├── src/
│   ├── main.py             # CLI entry point
│   ├── orchestrator.py     # Multi-agent coordination
│   ├── document.py         # PDF parsing
│   ├── literature.py       # PubMed search
│   ├── costs.py            # Cost tracking
│   ├── database.py         # SQLite persistence
│   ├── prompts.py          # Agent prompts
│   ├── agents/             # Reviewer agents
│   │   ├── base.py
│   │   ├── methodologist.py
│   │   ├── domain_expert.py
│   │   ├── communication.py
│   │   └── ethics.py
│   └── web/                # Web interface
│       ├── app.py          # FastAPI application
│       └── templates/      # Jinja2 templates
├── data/                   # SQLite database
└── output/                 # Generated reviews
```

## Deployment

The web interface can be deployed using Docker or any ASGI server:

```bash
# Using uvicorn directly
uvicorn src.web.app:app --host 0.0.0.0 --port 8000

# Using gunicorn with uvicorn workers
gunicorn src.web.app:app -w 4 -k uvicorn.workers.UvicornWorker
```

