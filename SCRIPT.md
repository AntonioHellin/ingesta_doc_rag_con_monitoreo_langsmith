# Monitoring

1. Show requirements.txt with the new import
2. Update the .env file with the new LangSmith API key and project name
3. See the environment variable set to true in `app.py` in line 33
4. Explain that since we are using Langchain openai wrapper, it will automatically pick up the LangSmith configuration from the environment variables.
5. Showcase LangSmith's monitoring capabilities in action.

docs: https://docs.langchain.com/langsmith/trace-with-langchain

# Evaluations

This section demonstrates how to evaluate chatbots using LangSmith, following the official tutorial approach.

## 1. Setup
- Show `requirements.txt` with minimal dependencies
- Set environment variables for LangSmith and OpenAI

## 2. Simple Demo (`app.py`)
- Create a basic Q&A dataset (5 examples)
- Define a simple chatbot function
- Create 2 evaluators:
  - **Correctness**: LLM-as-a-judge for accuracy
  - **built-in Conciseness**: Length-based evaluation
- Run evaluation and show results in LangSmith

docs: https://docs.langchain.com/langsmith/evaluate-chatbot-tutorial

# Prompts in langsmith

1. Create a prompt
2. Show changelog
3. Show integration with langchain
