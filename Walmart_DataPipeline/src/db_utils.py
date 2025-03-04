import pg8000
from pymongo import MongoClient
from config import DB_CONFIG, ADS_DB_CONFIG, MONGO_URI, DATABASE_NAME
from logger import logger

def get_postgres_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = pg8000.connect(**DB_CONFIG)
        logger.info("Successfully connected to PostgreSQL database.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL: {e}")
        raise

def get_postgres_connection_ads_query():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = pg8000.connect(**ADS_DB_CONFIG)
        logger.info("Successfully connected to PostgreSQL database.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL: {e}")
        raise

def get_mongo_client():
    """Returns a MongoDB client connection."""
    try:
        client = MongoClient(MONGO_URI)
        logger.info("Successfully connected to MongoDB.")
        return client[DATABASE_NAME]
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        raise

def close_postgres_connection(conn):
    """Closes the given PostgreSQL connection."""
    try:
        if conn:
            conn.close()
            logger.info("PostgreSQL connection closed.")
    except Exception as e:
        logger.error(f"Error closing PostgreSQL connection: {e}")

def close_mongo_client(client):
    """Closes the given MongoDB client connection."""
    try:
        if client:
            client.close()
            logger.info("MongoDB connection closed.")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}")
