import pandas as pd
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from logger import logger
from config import DEBUG, S3_BUCKET_NAME, S3_FOLDER, SMARTPROXY_API_KEY
from s3_utils import fetch_product_details_from_s3, upload_file_to_s3

PRODUCT_DETAILS_FILE = "MATTRESS_PRODUCT_DETAILS.csv"
CDD_PRODUCT_DETAILS_FILE = "CALIFORNIA_DESIGN_DEN/CDD_PRODUCT_DETAILS.csv"

def get_missing_products(final_df, brand='NapQueen'):
    """Identify missing product IDs that need to be scraped.""" 
    if brand == 'NapQueen':
        existing_products_df = fetch_product_details_from_s3(S3_BUCKET_NAME, S3_FOLDER, PRODUCT_DETAILS_FILE)
        logger.info(f"Loaded existing product details from S3: {PRODUCT_DETAILS_FILE}")
        logger.info(f"Existing Product Details DataFrame Shape: {existing_products_df.shape}")
    elif brand == 'California Design Den Inc.':
        try:
            existing_products_df = fetch_product_details_from_s3(S3_BUCKET_NAME, S3_FOLDER, CDD_PRODUCT_DETAILS_FILE)
            logger.info(f"Loaded existing product details from S3: {CDD_PRODUCT_DETAILS_FILE}")
            logger.info(f"Existing Product Details DataFrame Shape: {existing_products_df.shape}")
        except FileNotFoundError:
            logger.warning(f"File {CDD_PRODUCT_DETAILS_FILE} not found in S3. Creating a new file.")
            existing_products_df = pd.DataFrame(columns=["ID", "URL", "SKU", "GTIN", "Title", "Rating", "Rating Count", "Seller ID", "Seller Name", "Currency", "Description", "Out of Stock", "Specifications","Price"])
    else:
        logger.error(f"Unsupported brand: {brand}")
        raise ValueError(f"Unsupported brand: {brand}")
    
    # if existing_products_df.empty:
    #     logger.warning("Existing product details file is empty or missing. All products need to be scraped.")
    #     return final_df["id"].astype(str).tolist()#, pd.DataFrame()  
    
    existing_ids = set(existing_products_df['ID'].astype(str))
    new_ids = set(final_df['id'].astype(str))

    missing_ids = list(new_ids - existing_ids)
    logger.info(f"Found {len(missing_ids)} missing product IDs to scrape.")
    
    return missing_ids, existing_products_df


def scrape_walmart_product(product_url, product_id):
    """Scrape product details from Walmart using the Smartproxy API."""
    url = "https://scraper-api.smartproxy.com/v2/scrape?walmart"
    payload = {
        "target": "universal",
        "url": product_url,
        "locale": "en-us",
        "parse": True,
        "geo": "United States",
        "device_type": "desktop",
        "headless": "html",
        "http_method": "POST",
        "successful_status_codes": [200, 201]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": SMARTPROXY_API_KEY
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 500:
            logger.error(f"Smartproxy API error for {product_url}: {response.text}")
            return None
        
        if response.status_code != 200:
            if DEBUG:
                logger.warning(f"Debug: Failed URL {product_url}")
                logger.debug(f"Status Code: {response.status_code}")
                logger.debug(f"Response Text: {response.text}")
            response.raise_for_status() 
            logger.warning(f"Failed to scrape {product_url}. Status Code: {response.status_code}")
            return None

        response_data = response.json()
        if 'results' not in response_data or not response_data['results']:
            if DEBUG:
                logger.warning(f"No results found for {product_url}")
                return None

        product_info = response_data['results'][0]['content']['results']

        return {
            "ID": product_id,
            "URL": product_info.get('general', {}).get('url', None),
            "SKU": product_info.get('general', {}).get('meta', {}).get('sku', {}),
            "GTIN": product_info.get('general', {}).get('meta', {}).get('gtin', {}),
            "Price": product_info.get('price', {}).get('price', None),
            "Title": product_info.get('general', {}).get('title', None),
            "Rating": product_info.get('rating', {}).get('rating', None),
            "Rating Count": product_info.get('rating', {}).get('count', None),
            "Seller ID": product_info.get('seller', {}).get('id', {}),
            "Seller Name": product_info.get('seller', {}).get('name', {}),
            "Currency": product_info.get('price', {}).get('currency', None),
            "Description": product_info.get('general', {}).get('description', None),
            "Out of Stock": product_info.get('fulfillment', {}).get('out_of_stock', None),
            "Specifications": product_info.get('specifications')
        }
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while scraping {product_url}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON for URL {product_url}: {e}")
    except Exception as e:
        logger.error(f"Error scraping {product_url} in walmart_scraper.py : {e}")
        return None

def process_product_id(product_id):
    """Process a single product ID with retries and exponential backoff."""
    global total_retries

    product_url = f"https://www.walmart.com/ip/{product_id}"
    retries = 5
    backoff = 2
    total_retries = 0

    for attempt in range(retries):
        product_details = scrape_walmart_product(product_url, product_id)
        if product_details:
            return product_details
        else:
            logger.info(f"Retry {attempt + 1}/{retries} for Product ID: {product_id}")
            total_retries += 1
            time.sleep(backoff ** attempt)

    logger.error(f"Failed to scrape product ID {product_id} after {retries} attempts in walmart_scraper.py.")
    logger.info(f"Total retries across scraping : {total_retries}")
    return None

def scrape_missing_products(missing_product_ids, max_workers=20):
    """Scrape missing products using a thread pool for efficiency."""
    scraped_data = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_product_id, product_id): product_id for product_id in missing_product_ids}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping Missing Products"):
            try:
                result = future.result()
                if result:
                    scraped_data.append(result)
            except Exception as e:
                logger.error(f"Error scraping product {futures[future]} walmart_scraper.py : {e}")
                
    return pd.DataFrame(scraped_data)

def clean_scraped_data(scraped_df):
    """Cleans the scraped product details file by replacing error messages with default values."""
    cleaned_scraped_df = scraped_df.copy()
    logger.info(f"Scraped Data into cleaning function : {scraped_df.shape}")

    # Define default values for cleaning
    default_values = {
        "SKU": {},
        "product_name": None,
        "Description": None,
        "Price": None,
        "Title": None,
        "Rating": None,
        "Rating Count": None,
        "Seller ID": {},
        "Seller Name": None,
        "Currency": None,
        "Out of Stock": None,
        "Specifications": {}
    }
    
    # Define a function to clean error messages
    def clean_field(value, default):
        if isinstance(value, str) and "Error while parsing" in value:
            return default  # Replace with the default value
        return value
    
    # Apply cleaning function to relevant columns
    for field, default in default_values.items():
        if field in cleaned_scraped_df.columns:  # Only clean columns that exist in the data
            cleaned_scraped_df[field] = cleaned_scraped_df[field].apply(lambda x: clean_field(x, default))
    

    logger.info(f"Cleaned data saved to {cleaned_scraped_df.shape}")
    logger.info(f"Cleaned Data Sample : {cleaned_scraped_df.head()}")
    return cleaned_scraped_df


def process_and_update_product_details(final_df , brand='NapQueen'):
    """Fetch existing product details from S3, scrape missing products, update and upload back to S3."""
    missing_product_ids, existing_products_df = get_missing_products(final_df, brand=brand)
    if not missing_product_ids:
        logger.info("No new products to scrape. Exiting process.")
        return
    
    scraped_df = scrape_missing_products(missing_product_ids)
    
    cleaned_scraped_df = clean_scraped_data(scraped_df)
    # Append new data to existing DataFrame
    product_details_df = pd.concat([existing_products_df, cleaned_scraped_df], ignore_index=True)
    
    # Upload the updated DataFrame back to S3
    try:
        if brand == 'NapQueen':
            upload_file_to_s3(product_details_df, S3_BUCKET_NAME, S3_FOLDER, PRODUCT_DETAILS_FILE)
            logger.info(f"Successfully uploaded updated product details to S3: {PRODUCT_DETAILS_FILE}")
        elif brand == 'California Design Den Inc.':
            upload_file_to_s3(product_details_df, S3_BUCKET_NAME, S3_FOLDER, CDD_PRODUCT_DETAILS_FILE)
            logger.info(f"Successfully uploaded updated product details to S3: {CDD_PRODUCT_DETAILS_FILE}")
    except Exception as e:
        logger.error(f"Error uploading product details file to S3 in walmart_scraper.py : {e}")
        
    return product_details_df
