# run_fastapi.py
import subprocess

# Define the command
command = ["uvicorn", "api_server:app", "--reload", "--port", "8000"]

# Run the command in the terminal
subprocess.run(command)
