import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensure the root directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import run_pipeline

if __name__ == "__main__":
    print("Starting Agentic Facebook Analyst...")
    default_question = (
        "Diagnose why ROAS has changed over the last 30 days and "
        "recommend new creative ideas for low-CTR campaigns."
    )
    
    # Allow user to pass a custom question via CLI arguments
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = default_question
        
    print(f"User Query: {question}")
    
    try:
        result = run_pipeline(question)
        print("Pipeline completed successfully.")
        print("Check the 'outputs' directory for insights.json and creatives.json.")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
