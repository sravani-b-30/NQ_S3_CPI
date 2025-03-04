# Walmart Data Pipeline
A data processing pipeline to fetch, clean, and store Walmart product data using PostgreSQL, MongoDB, S3, and Smartproxy scraping.

## Overview
This project is designed to fetch Walmart product and keyword data, clean and transform it, and store it efficiently. 
It includes database operations, web scraping, S3 integration, and final data processing.

### 🔹 Features:
- Fetches data from **PostgreSQL & MongoDB**
- Scrapes **missing product details from Walmart**
- Cleans & merges the **fetched & scraped data**
- Saves the final dataset to **AWS S3**
- Structured in a modular way for **easy debugging and scalability**

## Project Structure
The repository follows a well-organized **modular structure**:

walmart-data-pipeline/
│── src/                        # Source code directory
│   │── config.py               # Configuration settings (DB, API keys)
│   │── logger.py               # Logger setup
│   │── db_utils.py             # Database interaction functions
│   │── data_fetching.py        # Fetching data from MongoDB, PostgreSQL
│   │── data_processing.py      # Data cleaning, transformation
│   │── walmart_scraper.py      # Web scraping functions
│   │── s3_utils.py             # Functions for saving data to AWS S3
│   │── pipeline.py             # Main pipeline execution script
│── requirements.txt            # Dependencies for the project
│── README.md                   # Documentation about the project
│── .gitignore                   # Ignore unnecessary files (e.g., `__pycache__`)
│── setup.py                     # Make it installable as a package


## Installation

### **1️⃣ Clone the repository**
```bash
git clone https://github.com/sravani-b-30/NQ_S3_CPI.git
cd walmart-data-pipeline

## **🔹2,3. Create Virtual Environment : 
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

## **🔹4. Install dependencies : 
pip install -r requirements.txt

---

## **🔹 5. Usage**
### **Run the main pipeline script**
To execute the pipeline, simply run:
```bash
python src/pipeline.py


---

## **🔹 6. Configuration**
## Configuration
Modify the `config.py` file to set up database credentials, API keys, and AWS S3 bucket details


## License & Usage
This project is proprietary and confidential. All rights are reserved by anarix.ai .  
Unauthorized copying, modification, distribution, or disclosure of this software and associated documentation  
without explicit written permission from anarix.ai is strictly prohibited.


