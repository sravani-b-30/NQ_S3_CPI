import pg8000
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
import certifi
import logging
from decimal import Decimal
import boto3
import io
from io import StringIO

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logging
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Outputs to console
        logging.FileHandler("cpi_data_log.log")  # Outputs to a log file
    ]
)
logger = logging.getLogger(__name__)  # Use __name__ to track logs per module

def product_title(soup):
    try:
        title = soup.find("span", attrs={"id": 'productTitle'})
        title_value = title.string
        title_string = title_value.strip().replace(',', '')

    except AttributeError:
        logger.error("Product title not found.")
        title_string = "NA"
    return title_string

def scrape_product_details_v2(soup):
    """Scrapes product details from the new product details section using provided BeautifulSoup object."""
    product_details = {}
    try:
        product_facts = soup.find('div', {'id': 'productFactsDesktopExpander'})
        if product_facts:
            for fact in product_facts.find_all('div', {'class': 'a-fixed-left-grid product-facts-detail'}):
                key_div = fact.find('div', {'class': 'a-fixed-left-grid-col a-col-left'})
                value_div = fact.find('div', {'class': 'a-fixed-left-grid-col a-col-right'})
                if key_div and value_div:
                    key = key_div.text.strip()
                    value = value_div.text.strip()
                    product_details[key] = value
        return product_details
    except Exception as e:
        logger.error(f"Error scraping product details (v2): {e}")
        return None

def scrape_product_details(soup):
    """Scrapes product details from the main product details table using provided BeautifulSoup object."""
    product_details = {}
    try:
        table = soup.find('table', {'class': 'a-normal a-spacing-micro'})
        if table:
            for row in table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) == 2:
                    label = cells[0].text.strip()
                    value = cells[1].text.strip()
                    product_details[label] = value
        if len(product_details) == 0:
            product_details = scrape_product_details_v2(soup)
        return product_details
    except Exception as e:
        logger.error(f"Error scraping main product details: {e}")
        return None

def scrape_glance_icons_details(soup):
    """Scrapes product details from the 'glance_icons_div' using provided BeautifulSoup object."""
    glance_details = {}
    try:
        glance_div = soup.find('div', id='glance_icons_div')
        if glance_div:
            for table in glance_div.find_all('table'):
                for row in table.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) == 2:
                        label = cells[1].find('span', class_='a-text-bold').get_text(strip=True).replace(':', '')
                        value = ' '.join([span.get_text(strip=True) for span in cells[1].find_all('span')[1:]])
                        glance_details[label] = value
        return glance_details
    except Exception as e:
        logger.error(f"Error scraping glance icon details: {e}")
        return None

def scrape_selected_variation(soup):
    """Scrapes the selected variation from the twisterContainer div using the provided BeautifulSoup object."""
    selected_variation = {}
    try:
        container = soup.find('div', {'id': 'twisterContainer'})
        if container:
            for variation in container.find_all('div', {'class': 'a-section a-spacing-small'}):
                label_element = variation.find('label', {'class': 'a-form-label'})
                value_element = variation.find('span', {'class': 'selection'})
                if label_element and value_element:
                    label = label_element.text.strip().replace(':', '')
                    value = value_element.text.strip()
                    selected_variation[label] = value
        return selected_variation
    except Exception as e:
        logger.error(f"Error scraping selected variation: {e}")
        return None

def get_description(soup):
    try:
        ul_element = soup.find("ul", class_="a-unordered-list a-vertical a-spacing-mini")
        if ul_element:
            li_elements = ul_element.find_all("li", class_="a-spacing-mini")
            return ' '.join(li.get_text(strip=True) for li in li_elements)
        else:
            li_elements = soup.find_all("ul", class_="a-unordered-list a-vertical a-spacing-small")
            if li_elements:
                return ' '.join(li.get_text(strip=True) for li in li_elements)
    except Exception as e:
        logger.error(f"Error scraping description: {e}")
    return None

def scrape_rating_count(soup):
    try:
        review_count = soup.find("span", attrs={'id': 'acrCustomerReviewText'}).string.strip().replace(',', '')
    except AttributeError:
        logger.warning("Rating count not found.")
        review_count = "NA"
    return review_count

def scrape_selected_dropdown_option(soup):
    try:
        dropdown_container = soup.find('div', {'id': 'variation_size_name'})
        label = dropdown_container.find('label').text.strip().replace(":", "")
        selected_option = dropdown_container.find('option', selected=True)
        selected_value = selected_option.get('data-a-html-content').strip()
        result = {label: selected_value}
    except AttributeError:
        result = "NA"
    return result

def rating_out_of_5(soup):
    try:
        rating = soup.find("i", attrs={'class': 'a-icon a-icon-star a-star-4-5'}).string.strip().replace(',', '')

    except AttributeError:
        try:
            rating = soup.find("span", attrs={'class': 'a-icon-alt'}).string.strip().replace(',', '')
        except:
            rating = "NA"
    return rating

def extract_dropdown_info(soup):
    try:
        label_tag = soup.find('label', class_='a-form-label')
        label = label_tag.text.strip().rstrip(':') if label_tag else 'Label not found'
        select_tag = soup.find('span', class_='a-dropdown-prompt')
        selected_option = select_tag.text.strip() if select_tag else 'Selected option not found'
        return {label: selected_option}
    except:
        return None

def get_page_soup(url):
    """Fetches the web page and returns a BeautifulSoup object."""
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10, verify=certifi.where())
        if response.status_code == 200:
            logger.info(f"Successfully fetched page for URL: {url}")
            return BeautifulSoup(response.content, 'html.parser')
        else:
            logger.error(f"Error: Received status code {response.status_code} for URL: {url}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f" Request Error: {e}")
        return None

def scrape_data(asin, max_retries=100):
    """Scrapes product data from Amazon given an ASIN."""
    url = f'https://www.amazon.com/dp/{asin}?th=1&psc=1'
    retry_count = 0
    valid_data = False
    data = {}

    while retry_count < max_retries and not valid_data:
        try:
            soup = get_page_soup(url)
            if soup is None:
                retry_count += 1
                logger.warning(f"Retrying... {retry_count} for ASIN: {asin}")
                continue

            # Scrape the necessary details
            rating = rating_out_of_5(soup)
            review_count = scrape_rating_count(soup)
            title = product_title(soup)
            details = scrape_product_details(soup)
            glance_details = scrape_glance_icons_details(soup)
            description = get_description(soup)
            options = scrape_selected_variation(soup)
            drop_down = extract_dropdown_info(soup)

            # Validate if key data has been scraped properly
            if title and rating and details:
                valid_data = True
                data = {
                    'ASIN': asin,
                    'Title': title,
                    'Drop Down': drop_down,
                    'Product Details': details,
                    'Glance Icon Details': glance_details,
                    'Description': description,
                    'Option': options,
                    'Rating': rating,
                    'Review Count': review_count
                }
            else:
                retry_count += 1
                logger.error(f"Invalid data on attempt {retry_count} for ASIN: {asin}. Retrying...")
        except Exception as e:
            retry_count += 1
            logger.error(f'Attempt {retry_count} failed for ASIN {asin}: {e}')

    # Return the final data, even if it has failed to retrieve valid data after max_retries
    if not valid_data:
        logger.warning(f"Max retries reached for ASIN: {asin}. Saving incomplete data.")
        data = {
            'ASIN': asin,
            'Title': 'NA',
            'Drop Down': {},
            'Product Details': {},
            'Glance Icon Details': {},
            'Description': "No Description",
            'Option': {},
            'Rating': '',
            'Review Count': ''
        }

    return data

def save_to_csv(data, file_path):
    """Appends scraped data to CSV."""
    df = pd.DataFrame([data])
    if not os.path.isfile(file_path):
        df.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)
    logger.info(f"Data saved to CSV at {file_path}")

def parallel_scrape(asins, num_processes, file_path):
    """Runs the scraping process in parallel."""
    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap(scrape_data, asins), total=len(asins), desc="Scraping"):
            save_to_csv(result, file_path)
            logger.info(f"Scraped and saved data for ASIN.")

def save_to_s3(df, brand, file_name):
    """
    Saves a DataFrame as a CSV file to S3 in the specified brand folder.

    :param df: The DataFrame to save.
    :param brand: The brand name for the S3 folder path.
    :param file_name: The CSV file name.
    :param aws_access_key_id: AWS access key for authentication.
    :param aws_secret_access_key: AWS secret access key for authentication.
    """
    

    # Initialize the S3 client
    s3_client = boto3.client(
        's3',
    )

    # Convert DataFrame to CSV in-memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_key = f"{brand}/{file_name}"

    # Upload to S3
    try:
        s3_client.put_object(Bucket="anarix-cpi", Key=s3_key, Body=csv_buffer.getvalue())
        logger.info(f"DataFrame saved to S3: {s3_key}")
    except Exception as e:
        logger.error(f"Failed to save {file_name} to S3: {e}")


def scrapper_handler(df, bucket_name, brand, file_name="CRAZY_CUPS_PRODUCT_DETAILS.csv", num_workers=15):
    df['asin'] = df['asin'].str.upper()
    df_asins = df['asin'].unique().tolist()
    df_asins = [asin for asin in df_asins if asin.startswith('B')]
    
    total_collected = []

    asins = [asin for asin in df_asins if asin not in total_collected]

    logger.info(f"Total ASINs: {len(df_asins)}")
    logger.info(f"Collected ASINs: {len(total_collected)}")
    logger.info(f"Remaining ASINs: {len(asins)}")

    # Run parallel scraping for remaining ASINs
    if asins:
        logger.info(f"Starting parallel scraping with {num_workers} workers.")
        try:
            parallel_scrape(asins, num_workers, file_name)
            logger.info("Scraping completed successfully.")
            logger.info("Scraping completed and data appended to the S3 file.")
            
        except Exception as e:
            logger.error(f"Error during parallel scraping: {e}")
    else:
        logger.info("No new ASINs to scrape.")

    # Read the updated CSV file into a DataFrame and return it.
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    else:
        logger.error("Scraped CSV file not found!")
        return pd.DataFrame()



if __name__ == '__main__':
    multiprocessing.freeze_support()

    # List of brands to process
    brand =  "AMAZON/CRAZY_CUPS"
    
    logger.info(f"Processing brand: {brand}")

    #Step 7
    file_path = "Pipeline/Crazy_Cups/Files/crazy_cups_asins.csv"

    df_scrapped_info = scrapper_handler(
    df=pd.read_csv(file_path, on_bad_lines='skip'),
    bucket_name="anarix-cpi",
    brand="AMAZON/CRAZY_CUPS",
    file_name="CRAZY_CUPS_PRODUCT_DETAILS.csv"
    )

    # Read the scraped data from the CSV and save it to S3.
    scraped_df = pd.read_csv("CRAZY_CUPS_PRODUCT_DETAILS.csv")
    save_to_s3(
        df=scraped_df,
        brand=brand,
        file_name='CRAZY_CUPS_PRODUCT_DETAILS.csv'
    )

    logger.info("Scraping process completed.")
    logger.info("Data saved to S3 successfully.")
