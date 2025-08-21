# Usually this app is ran through the entrypoint.sh in a dockerfile, 
# however you can also run locally on windows using "uv run run_local.py"
# run_fastapi.py
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Define the command
command = ["uvicorn", "food_access_model.main:app", "--reload", "--port", "8000"]

# Run the command in the terminal
subprocess.run(command)
