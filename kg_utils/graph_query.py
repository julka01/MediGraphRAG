import logging
from neo4j import GraphDatabase, basic_auth
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_graphDB_driver(uri, username, password, database="neo4j"):
    """
    Creates and returns a Neo4j database driver instance configured with the provided credentials.

    Returns:
    Neo4j.Driver: A driver object for interacting with the Neo4j database.
    """
    try:
        logging.info(f"Attempting to connect to the Neo4j database at {uri}")
        if all(v is None for v in [username, password]):
            username = os.getenv('NEO4J_USERNAME')
            database = os.getenv('NEO4J_DATABASE')
            password = os.getenv('NEO4J_PASSWORD')

        # Handle case where auth is disabled (empty credentials)
        if not username and not password:
            auth = None
        else:
            auth = basic_auth(username, password)

        enable_user_agent = os.environ.get("ENABLE_USER_AGENT", "False").lower() in ("true", "1", "yes")
        if enable_user_agent:
            driver = GraphDatabase.driver(uri, auth=auth, database=database, user_agent=os.environ.get('NEO4J_USER_AGENT'))
        else:
            driver = GraphDatabase.driver(uri, auth=auth, database=database)
        logging.info("Connection successful")
        return driver
    except Exception as e:
        error_message = f"graph_query module: Failed to connect to the database at {uri}."
        logging.error(error_message, exc_info=True)
        return None
