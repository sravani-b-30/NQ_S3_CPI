import logging
import os
from config import DEBUG

# Define the log file name
LOG_FILE = "walmart_data_pipeline.log"

# Create a custom logger
logger = logging.getLogger("walmart_pipeline")
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

# Create a file handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)

# Define log format
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s"
)

# Set formatter for handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Test logging
if __name__ == "__main__":
    logger.debug("Debug mode is enabled.")
    logger.info("Logger is set up successfully.")
