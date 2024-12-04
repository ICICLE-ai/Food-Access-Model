# Usually this app is ran through the entrypoint.sh in a dockerfile, 
# however you can also run locally on windows using "uv run run_local.py"
# run_fastapi.py
import subprocess

# Define the command
command = ["uvicorn", "api_server:app", "--reload", "--port", "8000"]

# Run the command in the terminal
subprocess.run(command)