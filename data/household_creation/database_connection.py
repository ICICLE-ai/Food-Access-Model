from sqlalchemy import create_engine
from config import USER, PASS

# Replace with your actual PostgreSQL connection details
username = USER
password = PASS
host = 'localhost'
port = '5432'
database = 'FASS_DB'

# Create the connection string
connection_string = f'postgresql://{username}:{password}@{host}:{port}/{database}'

# Create the SQLAlchemy engine
engine = create_engine(connection_string)