from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger("walmart_pipeline")

# Load environment variables from .env file
load_dotenv()

# PostgreSQL Database Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "").strip('"'),
    "port": os.getenv("DB_PORT", "").strip('"'),
    "database": os.getenv("DB_NAME", "").strip('"'),
    "user": os.getenv("DB_USER", "").strip('"'),
    "password": os.getenv("DB_PASSWORD", "").strip('"')
}

ADS_DB_CONFIG = {
    "host": os.getenv("DB_HOST", "").strip('"'),
    "port": os.getenv("DB_PORT", "").strip('"'),
    "database": os.getenv("ADS_DB_NAME", "").strip('"'),
    "user": os.getenv("DB_USER", "").strip('"'),
    "password": os.getenv("DB_PASSWORD", "").strip('"')
}

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "").strip('"')
DATABASE_NAME = os.getenv("DATABASE_NAME", "").strip('"')

# AWS S3 Configuration
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "").strip('"')
S3_FOLDER = os.getenv("S3_FOLDER_NAME", "").strip('"')


# API Keys & Other Secrets
SMARTPROXY_API_KEY = os.getenv("SMARTPROXY_API_KEY", "").strip('"')

# Debugging Mode
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Ensure critical environment variables are set
required_env_vars = ["MONGO_URI", "DATABASE_NAME","DB_HOST","DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD", "SMARTPROXY_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")
    
logger.info("Configuration variables loaded successfully")
