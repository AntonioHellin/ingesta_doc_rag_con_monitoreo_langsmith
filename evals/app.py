"""
Simple evaluation demo following the LangChain tutorial.
This creates a dataset, defines evaluators, and runs evaluation.
"""
import os
import openai
from langsmith import Client, wrappers
from dotenv import load_dotenv
from pathlib import Path
from openevals.prompts import CONCISENESS_PROMPT
from openevals.llm import create_llm_as_judge



# Load environment variables if .env file exists
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

_openai_client = None

def get_openai_client():
    """Get or initialize the wrapped OpenAI client for LangSmith tracing."""
    global _openai_client
    if _openai_client is None:
        _openai_client = wrappers.wrap_openai(openai.OpenAI())
    return _openai_client

def check_environment() -> bool:
    """Check if all required environment variables are set."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    required_vars = ["LANGSMITH_API_KEY", "OPENAI_API_KEY", "LANGSMITH_TRACING"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease set:")
        print("export LANGSMITH_TRACING='true'")
        print("export LANGSMITH_API_KEY='your_api_key'")
        print("export OPENAI_API_KEY='your_api_key'")
        print("export LANGSMITH_PROJECT='qa-evaluation'")
        return False
    return True

def create_dataset():
    """Create a simple Q&A dataset."""
    client = Client()
    
    dataset_name = "QA Example Dataset"
    print(f"📊 Creating dataset: {dataset_name}")
    
    # Create dataset
    dataset = client.create_dataset(dataset_name)
    
    # Add examples - simple Q&A pairs
    examples = [
        {
            "inputs": {"question": "What is LangChain?"},
            "outputs": {"answer": "A framework for building LLM applications"},
        },
        {
            "inputs": {"question": "What is LangSmith?"},
            "outputs": {"answer": "A platform for observing and evaluating LLM applications"},
        },
        {
            "inputs": {"question": "What is OpenAI?"},
            "outputs": {"answer": "A company that creates Large Language Models"},
        },
        {
            "inputs": {"question": "What is artificial intelligence?"},
            "outputs": {"answer": "Technology that enables machines to perform tasks requiring human intelligence"},
        },
        {
            "inputs": {"question": "What is machine learning?"},
            "outputs": {"answer": "A subset of AI that learns patterns from data"},
        }
    ]
    
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"✅ Created dataset with {len(examples)} examples")
    
    return dataset_name

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """
    Custom evaluator: Check if the answer is correct using LLM-as-a-judge.
    This follows the pattern from the LangChain tutorial.
    """
    eval_instructions = "You are an expert professor specialized in grading students' answers to questions."
    
    user_content = f"""You are grading the following question:
{inputs['question']}

Here is the real answer:
{reference_outputs['answer']}

You are grading the following predicted answer:
{outputs['response']}

Respond with CORRECT or INCORRECT:
Grade:"""
    
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": eval_instructions},
                {"role": "user", "content": user_content},
            ],
        )
        
        evaluation = response.choices[0].message.content or ""
        return evaluation.upper().startswith("CORRECT")
    except Exception as e:
        print(f"Evaluation error: {e}")
        return False

def wrapped_conciseness_evaluator(
    inputs: dict,
    outputs: dict,
):
    conciseness_evaluator = create_llm_as_judge(
        prompt=CONCISENESS_PROMPT,
        feedback_key="conciseness",
        model="openai:o3-mini",
    )
    return conciseness_evaluator(
        inputs=inputs,
        outputs=outputs,
    )

def my_chatbot(question: str) -> str:
    """
    Simple chatbot function - this is what we're evaluating.
    Follows the pattern from the LangChain tutorial.
    """
    instructions = "Respond to the user's question in a short, concise manner (one short sentence)."
    
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
    )
    
    return response.choices[0].message.content or "No response"

def chatbot_wrapper(inputs: dict) -> dict:
    """
    Wrapper function that maps dataset inputs to chatbot outputs.
    This is required for LangSmith evaluation.
    """
    question = inputs["question"]
    response = my_chatbot(question)
    return {"response": response}

def run_evaluation(dataset_name: str):
    """Run the evaluation using LangSmith."""
    client = Client()
    
    print(f"🚀 Running evaluation...")
    print(f"📊 Dataset: {dataset_name}")
    
    # Define evaluators
    evaluators = [correctness_evaluator, wrapped_conciseness_evaluator]

    # Run evaluation
    experiment_results = client.evaluate(
        chatbot_wrapper,  # Our AI system
        data=dataset_name,  # The data to predict and grade over
        evaluators=evaluators,  # The evaluators to score the results
        experiment_prefix="simple-qa-eval",  # Prefix for experiment names
        description="Simple Q&A chatbot evaluation demo"
    )
    
    print(f"✅ Evaluation completed!")
    print(f"🔗 View results: https://smith.langchain.com")
    print(f"📈 Experiment: {experiment_results.experiment_name}")
    
    return experiment_results

def main():
    """Main demo function."""
    print("🎯 Simple Chatbot Evaluation Demo")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        return
    
    try:
        # Step 1: Create dataset
        print("\n📋 Step 1: Creating Dataset")
        dataset_name = create_dataset()
        
        # Step 2: Run evaluation
        print("\n📋 Step 2: Running Evaluation")
        results = run_evaluation(dataset_name)
        
        # Step 3: Summary
        print("\n🎉 Demo Complete!")
        print("=" * 50)
        print("What happened:")
        print("• Created a dataset with 5 Q&A examples")
        print("• Defined 2 evaluators (correctness & conciseness)")
        print("• Evaluated a simple chatbot")
        print("• Results are now visible in LangSmith!")
        print("\n💡 Next steps:")
        print("• Check LangSmith dashboard for detailed results")
        print("• Try running with different models")
        print("• Add more test cases to the dataset")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
