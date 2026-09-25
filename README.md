# langsmith-rag-evals

A production observability and evaluation suite for Retrieval-Augmented Generation (RAG) applications using LangChain and LangSmith.

---

## Project Overview

`langsmith-rag-evals` demonstrates end-to-end LLM observability, distributed tracing, and automated evaluation workflows. The repository is organized into two primary components:

1. **`monitoring/`**: An interactive RAG conversational assistant with integrated LangSmith tracing, logging, and Gradio chat interface.
2. **`evals/`**: An automated evaluation framework that assesses chatbot response quality using LangSmith datasets and LLM-as-a-judge evaluators.

---

## Features

- **Automated LLM Tracing**: Captures prompt inputs, token counts, model latency, retrieval contexts, and execution graphs in LangSmith.
- **Automated Quality Evaluation**:
  - **Correctness Evaluator**: LLM-as-a-judge assessing semantic factual correctness against ground-truth references.
  - **Conciseness Evaluator**: Evaluator measuring succinctness and output discipline.
- **RAG Knowledge Base & UI**: Conversational interface built with Gradio and LangChain, supporting live document uploads (PDF, TXT) and semantic retrieval.
- **Reproducible Dataset Management**: Creates and updates Q&A datasets programmatically in the LangSmith platform.

---

## Architecture & Directory Structure

```
.
├── evals/
│   ├── app.py             # Evaluation runner and dataset generator
│   ├── requirements.txt   # Evaluation dependencies (langsmith, openevals, openai)
│   └── .env.example       # Evaluation environment configuration template
├── monitoring/
│   ├── app.py             # Main entry point for RAG application with tracing
│   ├── Dockerfile         # Container definition for the web assistant
│   ├── deploy_to_railway.sh # Deployment automation script
│   ├── requirements.txt   # Application dependencies
│   ├── .env.example       # Application environment configuration template
│   └── src/               # Application source (agent, interface, rag)
└── SCRIPT.md              # Demonstration presentation notes
```

---

## Prerequisites

- **Python**: 3.10 or higher
- **OpenAI API Key**: Required for completion and embedding models
- **LangSmith API Key**: Required for trace ingestion and experiment management
- **Git**

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AntonioHellin/langsmith-rag-evals.git
   cd langsmith-rag-evals
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On macOS/Linux:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\Activate.ps1
   ```

---

## Environment Configuration

Configure environment variables for either component by creating `.env` files from their respective `.env.example` templates:

### Monitoring Environment (`monitoring/.env`)
```ini
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=rag-ai-assistant-monitoring
LANGSMITH_TRACING=true
```

### Evaluation Environment (`evals/.env`)
```ini
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=qa-evaluation
LANGSMITH_TRACING=true
```

> [!CAUTION]
> Ensure `.env` files are never committed to version control. Both root and subfolder configurations are excluded via `.gitignore`.

---

## Usage

### 1. Running the Monitored RAG Application

Navigate to the `monitoring` directory, install dependencies, and launch the assistant:

```bash
cd monitoring
pip install -r requirements.txt
python app.py
```

- Access the Gradio web UI at `http://localhost:7860`.
- All LLM invocations and retrieval spans will be streamed to your LangSmith project dashboard.

### 2. Running the Evaluation Suite

Navigate to the `evals` directory, install dependencies, and execute the experiment:

```bash
cd ../evals
pip install -r requirements.txt
python app.py
```

The script will:
1. Initialize the Q&A dataset in LangSmith.
2. Query the chatbot across evaluation inputs.
3. Compute correctness and conciseness scores via LLM evaluators.
4. Output a direct link to the LangSmith experiment dashboard for analysis.

---
