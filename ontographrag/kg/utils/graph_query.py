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
    Neo4j.Driver: A driver object for interacting with the database.
    """
    try:
        logging.info(f"Attempting to connect to the Neo4j database at {uri}")

        # Handle parameter defaults and environment fallbacks
        if username is None or username == '':
            username = os.getenv('NEO4J_USERNAME') or os.getenv('NEO4J_USER', '').strip()
        if password is None or password == '':
            password = os.getenv('NEO4J_PASSWORD', '').strip()
        if database is None or database == '':
            database = os.getenv('NEO4J_DATABASE', 'neo4j').strip()

        # Ensure URI is not empty
        if not uri or uri.strip() == '':
            uri = os.getenv('NEO4J_URI', '').strip()
            if not uri:
                raise ValueError("Neo4j URI is required but not provided")

        logging.info(f"Using credentials: uri={uri}, user='{username}' (password set: {bool(password)}), database='{database}'")

        # Handle case where auth is disabled (both username and password are empty)
        if not username and not password:
            logging.info("Using no-authentication mode")
            auth = None
        else:
            logging.info("Using basic authentication")
            from neo4j import Auth
            auth = Auth("basic", username, password)

        enable_user_agent = os.environ.get("ENABLE_USER_AGENT", "False").lower() in ("true", "1", "yes")
        if enable_user_agent:
            driver = GraphDatabase.driver(uri, auth=auth, database=database, user_agent=os.environ.get('NEO4J_USER_AGENT'))
        else:
            driver = GraphDatabase.driver(uri, auth=auth, database=database)

        # Test the connection
        with driver.session(database=database) as session:
            session.run("RETURN 1").single()

        logging.info("Connection successful")
        return driver
    except Exception as e:
        error_message = f"Failed to connect to Neo4j at {uri}: {str(e)}"
        logging.error(error_message, exc_info=True)
        return None
