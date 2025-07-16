import os

MAX_CPUS = 4

try:
    MAX_CPUS = int(os.getenv("MAX_CORES"))
except ValueError:
    MAX_CPUS = 4