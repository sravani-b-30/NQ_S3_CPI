# import pandas as pd
# import json
# import requests
# from multiprocessing import Pool, cpu_count
# from tqdm import tqdm
# import os


# def scrape_walmart_product(product_url):
#     url = "https://scraper-api.smartproxy.com/v2/scrape?walmart"

#     payload = {
#         "target": "universal",
#         "url": product_url,
#         "locale": "en-us",
#         "parse": "true",
#         "geo": "United States",
#         "device_type": "desktop",
#         "headless": "html",
#         "http_method": "post",
#         "successful_status_codes": [200, 201]
#     }
#     headers = {
#         "accept": "application/json",
#         "content-type": "application/json",
#         "authorization": "Basic VTAwMDAxNDUyNjI6RTZvanNmMmV1OGh3VlI1RGpq"
#     }

#     response = requests.post(url, json=payload, headers=headers)
    

#     if response.status_code == 200:
#         response_data = json.loads(response.text)

#         if 'results' in response_data and response_data['results']:
#             product_info = response_data['results'][0]['content']['results']
#             print(product_info)

#             product_details = {
#                 "URL": product_info.get('url'),
#                 "SKU": product_info['meta'].get('sku'),
#                 "GTIN": product_info['meta'].get('gtin'),
#                 "Price": product_info.get('price'),
#                 "Title": product_info.get('title'),
#                 "Images": product_info.get('images'),
#                 "Rating": product_info['rating'].get('rating'),
#                 "Rating Count": product_info['rating'].get('count'),
#                 "Seller ID": product_info['seller'].get('id'),
#                 "Seller Name": product_info['seller'].get('name'),
#                 "Seller Official Name": product_info['seller'].get('official_name'),
#                 "Currency": product_info.get('currency'),
#                 "Warranty": product_info.get('warranty'),
#                 "Warnings": product_info.get('_warnings'),
#                 "Variations": product_info.get('variations'),
#                 "Breadcrumbs": product_info.get('breadcrumbs'),
#                 "Description": product_info.get('description'),
#                 "Out of Stock": product_info.get('out_of_stock'),
#                 "Specifications": product_info.get('specifications')
#             }

#             return product_details
#         else:
#             return None
#     else:
#         print(response.raise_for_status())


# def process_product_id(product_id):
#     product_url = f"https://www.walmart.com/ip/{product_id}"
#     i = 0
#     for _ in range(3):
#         try:
#             product_details = scrape_walmart_product(product_url)

#             if product_details:
#                 # Save the product details incrementally
#                 df = pd.DataFrame([product_details])
#                 if not os.path.isfile('walmart_product_details_mattress.csv'):
#                     df.to_csv('walmart_product_details_mattress.csv', index=False, mode='w')
#                 else:
#                     df.to_csv('walmart_product_details_mattress.csv', index=False, mode='a', header=False)
#             return product_details
#         except Exception as e:
#             print(f"Error: {e}")
#             pass
#     return None


# def main():
#     df_id = pd.read_csv('walmart_mattress_asins_03-12.csv')
#     unique_ids = df_id['id'].unique()
#     print(unique_ids)

#     # Use multiprocessing to scrape product details concurrently
#     with Pool(cpu_count()) as pool:
#         # Wrap the iterable with tqdm for progress bar
#         for _ in tqdm(pool.imap_unordered(process_product_id, unique_ids), total=len(unique_ids)):
#             pass

#     print("Scraping completed and data saved incrementally to 'walmart_product_details_mattress.csv'")


# if __name__ == "__main__":
#     main()

import pandas as pd
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import time

DEBUG = True

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
        "authorization": "Basic VTAwMDAxNDY0ODg6UFcxYjU5NjI1NmIwNzk0ZjlkNGEyZjRhYmFkZmRkZDUzZjQ="
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200 :
            if DEBUG:
                print(f"Debug: Failed URL {product_url}")
                print(f"Status Code: {response.status_code}")
                print(f"Response Text: {response.text}")
            response.raise_for_status()  # Raise an exception for non-200 responses

        response_data = response.json()
        if DEBUG:
            print(f"Debug: Successful response for {product_url}")
            #print(json.dumps(response_data, indent=2))  # Pretty-print the response

        if 'results' not in response_data or not response_data['results']:
            if DEBUG:
                print(f"Debug: No results found for URL: {product_url}")
            return None

        product_info = response_data['results'][0]['content']['results']
        
        data = {
                    "ID": product_id,
                    "URL": product_info.get('general', {}).get('url', None),
                    "SKU": product_info.get('general', {}).get('meta', {}).get('sku', {}),
                    "GTIN": product_info.get('general', {}).get('meta', {}).get('gtin', {}),
                    "Price": product_info.get('price', {}).get('price', None),
                    "Title": product_info.get('general', {}).get('title', None),
                    # "Images": product_info.get('general', {}).get('images', []),
                    "Rating": product_info.get('rating', {}).get('rating', None),
                    "Rating Count": product_info.get('rating', {}).get('count', None),
                    "Seller ID": product_info.get('seller', {}).get('id', {}),
                    "Seller Name": product_info.get('seller', {}).get('name', {}),
                    "Currency": product_info.get('price', {}).get('currency', None),
                    "Description": product_info.get('general', {}).get('description', None),
                    "Out of Stock": product_info.get('fulfillment', {}).get('out_of_stock', None),
                    "Specifications": product_info.get('specifications')
                }
        return data 
    
    except requests.exceptions.RequestException as e:
        print(f"Network error while scraping {product_url}: {e}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON for URL {product_url}: {e}")
    except Exception as e:
        print(f"Unexpected error for URL {product_url}: {e}")

    return None

def save_product_details(product_details, filename):
    """Save product details to a CSV file incrementally."""
    # if product_details:
    df = pd.DataFrame([product_details])
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        df.to_csv(f, index=False, header=f.tell() == 0)  # Write header only for the first record

total_retries = 0

def process_product_id(product_id, output_file):
    """Process a single product ID with retries and exponential backoff."""
    global total_retries

    product_url = f"https://www.walmart.com/ip/{product_id}"
    retries = 5
    backoff = 2
    

    for attempt in range(retries):
        product_details = scrape_walmart_product(product_url, product_id)
        if product_details:
            save_product_details(product_details, output_file)
            print(f"Printing product details :")
            print(product_details)
            return product_details
        else :
            # Log retry attempt
            print(f"Retry {attempt + 1}/{retries} for Product ID: {product_id}")
            total_retries += 1
            time.sleep(backoff ** attempt)  # Exponential backoff

    # After exhausting all retries, save failed scrape data
    failed_data = {
        "ID": product_id,
        "URL": None,
        "SKU": {},
        "GTIN": {},
        "Price": None,
        "Title": None,
        "Rating": None,
        "Rating Count": None,
        "Seller ID": {},
        "Seller Name": {},
        "Currency": None,
        "Description": None,
        "Out of Stock": None,
        "Specifications": {}
    }
    save_product_details(failed_data, output_file)  # Save failed data
    print(f"Failed to scrape product ID {product_id} after {retries} attempts.")
    return None
        # else:
        #     print(f"Retry {attempt + 1}/{retries} for Product ID: {product_id}")
        #     time.sleep(backoff ** attempt)  # Exponential backoff

        # print(f"Failed to scrape product ID: {product_id}")
        # return None


def scrape_all_products(product_ids, output_file, max_workers=20):
    """Scrape all product details using ThreadPoolExecutor."""
    
    global total_retries

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_product_id, product_id, output_file): product_id for product_id in product_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping Products"):
            product_id = futures[future]
            try:
                result = future.result()
                if result:
                    print(f"Successfully scraped product ID: {product_id}")
                else:
                    print(f"Failed to scrape product ID: {product_id}")
            except Exception as e:
                print(f"Error with product ID {product_id}: {e}")
    
    print(f"Total retries across scraping : {total_retries}")


def main():
    """Main function to load data and initiate scraping."""
    input_file = 'remaining_asins_24_jan.csv'
    output_file = 'Pipeline/walmart_product_details_24_jan.csv'

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    df_id = pd.read_csv(input_file)
    unique_ids = df_id['id'].unique()

    print(f"Found {len(unique_ids)} unique product IDs. Starting scraping...")
    scrape_all_products(unique_ids, output_file)
    print(f"Scraping completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()
