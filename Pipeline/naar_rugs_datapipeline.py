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

def active_keyword_ids(brand):
    db_config = {
        "host": "postgresql-88164-0.cloudclusters.net",
        "database": "amazon",
        "user": "Pgstest",
        "password": "testwayfair",
        "port": 10102
    }

    profile_id_to_brand = {
        2330799067638737: "EUROPEAN_HOME_DESIGNS",
        57621727790771: "LAVA_SELLER",
        4481174563638304: "SFX_SELLER",
        629997099631956: "NAPQUEEN",
        998654473666848: "SETTON_FARMS",
        1218284306330889: "NAAR_RUGS_SELLER"
    }


    try:
        connection = pg8000.connect(**db_config)
        cursor = connection.cursor()
        query = """
            SELECT * FROM advertising_entities.sp_keywords;
        """
        cursor.execute(query)
        logger.info("Successfully executed query for keyword IDs.")

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        data = pd.DataFrame(rows, columns=columns)
        logger.info("Data successfully retrieved from database.")

        data = data[data['state'] == "ENABLED"]
        data['profileId'] = data['profileId'].astype(np.int64)
        data['brand'] = data['profileId'].map(profile_id_to_brand)
        data = data[data['brand'] == brand]
        logger.info("Process 1: Finding Keyword IDs")
        logger.info(f"Filtered active keywords for brand '{brand}'. Total records found: {len(data)}")
        return data
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()
            logger.info("Database connection closed.")

def fetch_serp_data(keyword_ids_df):

    keyword_ids_df = keyword_ids_df[['keyword_id','keyword']]
    logger.info(f"Filtered keyword and keyword_id columns from keyword_df : {keyword_ids_df.columns}")

    keyword_ids_df['keyword_id'] = keyword_ids_df['keyword_id'].astype(int)
    logger.info(f"Converted keyword_id to integer type: {keyword_ids_df['keyword_id'].dtype}")

    # Log data overview
    logger.info("Initial 'Keyword and Keyword_ID' DataFrame info:")
    logger.info(keyword_ids_df.info())
    
    keyword_id_list = keyword_ids_df['keyword_id'].unique().tolist()
    logger.info(f"Number of unique keyword IDs: {len(keyword_id_list)}")
    
    # Convert list to tuple for SQL IN clause
    keyword_id_tuple = tuple(keyword_id_list)

    # Establish connection to the database
    conn = pg8000.connect(
        host="postgresql-88164-0.cloudclusters.net",
        database="generic",
        user="Pgstest",
        password="testwayfair",
        port=10102
    )

    # Create a cursor object
    cursor = conn.cursor()

    # Define the date range
    start_date = datetime.now().date() - timedelta(days=30)
    end_date = datetime.now().date()
    logger.info(f"Fetching SERP data from {start_date} to {end_date}")

    # Initialize an empty list to collect dataframes
    dataframes = []
    # Function to fetch data for a given date range (1 week at a time)
    def fetch_data_for_range(start, end):
        logger.info("start date: "+ str(start))
        logger.info("end date: " + str(end))
        query_serp = f"""
            SELECT product_id, sale_price, scrapped_at, keyword_id
            FROM serp.amazon_serp
            WHERE keyword_id IN {keyword_id_tuple}
              AND "scrapped_at" BETWEEN '{start}' AND '{end}'
            ORDER BY "scrapped_at";
        """
        cursor.execute(query_serp)
        return pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    # Fetch data in weekly chunks
    current_start_date = start_date
    chunk_size = 7  # 7 days (1 week)
    while current_start_date <= end_date:
        current_end_date = min(current_start_date + timedelta(days=chunk_size), end_date)

        # Fetch data for the week
        week_df = fetch_data_for_range(current_start_date, current_end_date)
        logger.info(f"Fetched SERP Data for the chunk {current_start_date} to {current_end_date}")
        logger.info(f"Data shape: {week_df.shape}")

        # Append the dataframe to the list
        dataframes.append(week_df)

        # Move to the next chunk
        current_start_date += timedelta(days=chunk_size)
        logger.info(f"Moving to next chunk number {current_start_date}")

    # Close the cursor and connection
    cursor.close()
    conn.close()

    # Concatenate all dataframes into a single dataframe
    all_data_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Total SERP data fetched: {all_data_df.shape}")

    # Merge the two DataFrames on keyword_id
    merged_df = pd.merge(keyword_ids_df, all_data_df, on='keyword_id')

    # Rename the scrapped_at column to date
    merged_df.rename(columns={'scrapped_at': 'date' , 'sale_price':'price'}, inplace=True)
    logger.info(f"Renamed scrapped_at and sale_price columns from SERP Data : {merged_df.columns}")

    logger.info(f"Merged SERP data with keyword information: {merged_df.shape}")
    logger.info(f"Sample data of merged_df: {merged_df.head()}")

    logger.info("Process 2: Finishied fetching SERP Data")
    return merged_df

def fetch_and_merge_product_data(serp_df):
    """
    This function reads SERP data from a CSV file, fetches additional product details from the database in chunks,
    merges the two DataFrames, and saves the merged result as a CSV file.

    :param df: DataFrame containing SERP data with product IDs
    :param chunk_size: The number of product IDs to fetch in each query chunk
    :return: Merged DataFrame
    """
    # Extract product IDs from the DataFrame and convert to list
    product_id_list = serp_df['product_id'].unique().tolist()
    total_products = len(product_id_list)
    logger.info(f"Total unique product IDs from serp: {total_products}")

    chunk_size = int(len(product_id_list)/10)
    logger.info(f"Using chunk size: {chunk_size}")

    # Establish connection to the database
    conn = pg8000.connect(
        host="postgresql-88164-0.cloudclusters.net",
        database="generic",
        user="Pgstest",
        password="testwayfair",
        port=10102
    )

    merged_df = pd.DataFrame()  # Empty DataFrame to store merged data

    # Fetch product details in chunks to avoid memory issues
    for i in range(0, len(product_id_list), chunk_size):
        # Get the current chunk of product IDs
        product_id_chunk = product_id_list[i:i + chunk_size]
        logger.info(f"Processing chunk {i // chunk_size + 1}: {len(product_id_chunk)} products")

        # SQL query to fetch product details based on product_id
        query = """
        SELECT product_id , title , asin , brand
        FROM serp.products
        WHERE product_id = ANY(%s)
        ORDER BY brand;
        """

        # Execute the query with product_id_chunk as a parameter
        cursor = conn.cursor()
        cursor.execute(query, (product_id_chunk,))
        logger.info("Executed query to fetch product details.")
        logger.info(f"Processed chunk {i // chunk_size + 1} of {len(product_id_list) // chunk_size}")

        # Fetch the results and convert to DataFrame
        df1 = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

        # Merge the chunked data with the original DataFrame
        merged_df = pd.concat([merged_df, pd.merge(serp_df, df1, on='product_id')])
        logger.info(f"Merged SERP data with product information : {merged_df.shape}")
        logger.info(f"Sample data of merged_df : {merged_df.head()}")

    # Close the cursor and connection
    cursor.close()
    conn.close()

    # Save grouped ASIN and keyword_id_list to S3
    asin_keyword_df = merged_df.groupby('asin')['keyword_id'].apply(lambda x: list(set(x))).reset_index()
    asin_keyword_df.columns = ['asin', 'keyword_id_list']
    save_to_s3(asin_keyword_df, brand, "asin_keyword_id_mapping.csv")

    # Save keyword and keyword_id pairs to S3
    keyword_pairs_df = merged_df[['keyword_id', 'keyword']].drop_duplicates().reset_index(drop=True)
    save_to_s3(keyword_pairs_df, brand, "keyword_x_keyword_id.csv")

    logger.info("Process 3: Fetched Product Info and Merged with SERP Data")
    logger.info(f"Pushed asin and keyword mapping csv files to s3 Bucket successfully")
    return merged_df

def pre_processing_serp_data(merged_df) :

    merged_df['scrapped_at'] = pd.to_datetime(merged_df['scrapped_at'], errors='coerce', format='mixed')

    # Drop rows where `scrapped_at` could not be parsed
    merged_df = merged_df.dropna(subset=['scrapped_at'])

    # Extract the date part from `scrapped_at` and create a new `date` column
    merged_df['date'] = merged_df['scrapped_at'].dt.date
    logger.info(f"Extracted date from scrapped_at column : {merged_df.shape}")
    logger.info(f"Datatype of date column : {merged_df['date'].dtype}")

    # Sort the DataFrame by `walmart_id`, `date`, and `scrapped_at`
    merged_df = merged_df.sort_values(by=['asin', 'date', 'scrapped_at'])

    # Group by `walmart_id` and `date`, and take the last occurrence
    merged_df = merged_df.groupby(['asin', 'date']).tail(1)
    logger.info(f"Grouped by asin and date : {merged_df.shape}")

    
    merged_df.drop(columns=['scrapped_at'], inplace=True)
    logger.info(f"Dropped scrapped_at column from the dataframe : {merged_df.columns}")

    # Reset index to clean up the DataFrame
    merged_df = merged_df.reset_index(drop=True)
    
    return merged_df

# def monthly_analysis(merged_df , days=60):

#     merged_df.rename(columns={"walmart_id" : "id"}, inplace=True)
#     logger.info(f"Renamed walmart_id to id in SERP Dataset : {merged_df.columns}")

#     merged_df['date'] = pd.to_datetime(merged_df['date'], format='mixed').dt.date

#     # Initialize a list to store the results for each day
#     dfs = []

#     # Iterate over the last 'days' days
#     for i in range(days):

#         logger.info(f"Processing ASIN data for day {i+1}/{days}")
#         analysis_date = merged_df['date'].max() - timedelta(days=i)
#         logger.info(f"Analysis Date: {analysis_date}")

#         # Define the date range: last 30 days ending at analysis_date
#         start_date = analysis_date - timedelta(days=30)
#         logger.info(f"Start Date: {start_date}")

#         # Filter the DataFrame for the last 30 days
#         last_30_days_df = merged_df[(merged_df['date'] <= analysis_date) & (merged_df['date'] > start_date)]
#         logger.info(f"Filtered DataFrame for the last 30 days: {last_30_days_df.shape}")
        
#         # Sort the DataFrame by ASIN and date (descending order)
#         last_30_days_df = last_30_days_df.sort_values(by=['id', 'date'], ascending=[True, False])
#         logger.info(f"Sample data of id and date after sorting : {last_30_days_df[['id', 'date']].head()}")

#         # Get unique ASINs and their corresponding latest prices and other details
#         unique_asins = last_30_days_df.groupby('id').agg({
#             'product_title': 'first',
#             'sale_price': 'first',
#             'brand': 'first',
#             'rank' : 'first',
#             'organic_search_rank' : 'first',
#             'sponsored_search_rank' : 'first',
#             'keyword' : 'first',
#             'keyword_id' : 'first'
#         }).reset_index()

#         # Add 'analysis_date' column to track the date of analysis
#         unique_asins['analysis_date'] = analysis_date

#         # Append the processed DataFrame for the day to the list
#         dfs.append(unique_asins)
#         print(f"Processed ASIN data for {analysis_date}")

#     # Concatenate all DataFrames into a single DataFrame
#     final_df = pd.concat(dfs)

#     # Reset index and rename 'latest_sale_price' to 'price'
#     final_df = final_df.reset_index(drop=True)
#     final_df = final_df.rename(columns={"sale_price": "price" , "analysis_date" : "date"})
#     logger.info(f"Final DataFrame after processing for the last {days} days: {final_df.shape}")
#     logger.info(f"Price Column and Date Column has been renamed successfully : {final_df.columns}")

#     final_df.to_csv("walmart_serp_data_final.csv" , index=False)

#     return final_df 


def save_to_s3(df, brand, file_name):
    """
    Saves a DataFrame as a CSV file to S3 in the specified brand folder.

    :param df: The DataFrame to save.
    :param brand: The brand name for the S3 folder path.
    :param file_name: The CSV file name.
    :param aws_access_key_id: AWS access key for authentication.
    :param aws_secret_access_key: AWS secret access key for authentication.
    """
    import boto3
    from io import StringIO

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

def fetch_price_tracker_data(marketplace, days=30):
    """
    This function fetches price tracking data for a given marketplace from the last `days` days,
    loads it into a DataFrame, and saves the result as a CSV file.

    :param marketplace: The marketplace for which to fetch the data (e.g., 'Amazon').
    :param days: The number of days to look back for the data (default is 60 days).
    :param output_file_path: Path to save the resulting CSV file (default is 'data/ehd/ehd_price_data.csv').
    :return: DataFrame containing the fetched data.
    """

    # Establish connection to the database
    conn = pg8000.connect(
        host="postgresql-88164-0.cloudclusters.net",
        database="generic",
        user="Pgstest",
        password="testwayfair",
        port=10102
    )

    # Create a cursor object
    cursor = conn.cursor()

    product_ids_raw = """
    B0D4B3TVWJ
    B0D4B4N96T
    B0D4B4DFPP
    B0B8ZJL5Q2
    B0D4B4V7K6
    B0D4B4Q97J
    B0D4B5KKWJ
    B0D49ZL8B6
    B0D4B5978G
    B0D4B5FS41
    B0D4B3VWFT
    B0D4B4PNVH
    B0D4B4ZV67
    B0D4B6LJY9
    B0D4B4PSMC
    B0D4B74CR1
    B0D4B68BZS
    B0D4B5HCC9
    B0D4B5NZ18
    B0D4B2NQFX
    B0D4B64ZCS
    B0D4B4LLNX
    B0D4B66RMJ
    B0D4B5KJMD
    B0D4B55HHC
    B0D4B5M187
    B0D4B6QS97
    B0CK4VR75M
    B0CFM3NMKB
    B0B8ZJ1SPF
    B0B8ZH2D5N
    B0B8ZGCVMW
    B0B8ZFM5L9
    B0D4B43XJM
    B0D4B5J9FK
    B0B8ZH7S8Y
    B0D4B4NTYV
    B0D4B5BW9P
    B0B8ZGB6YJ
    B0D4B4TF83
    B0B8ZGMT29
    B0D4B5M6KS
    B0D4B547MX
    B0D4B55DKH
    B0D4B43P6F
    B0D4B51FM5
    B0D4B57WRM
    B0B8ZHPVGS
    B0D4B3BYXJ
    B0D4B4TB3N
    B0D4B476H5
    B0D4B3WC56
    B0D4B5KD49
    B0D4B5KT1K
    B0D4B59BZ2
    B0D4B41BZT
    B0D4B5HJXK
    B0D4B67ZRZ
    B0B8ZH3YJZ
    B0D4B5962G
    B0B9CBNSB9
    B0D4B4R1X4
    B0D4B638PB
    B0D4B4BW4N
    B0D4B4FF84
    B0D4B61YK9
    B0D4B5TVNK
    B0D4B46CMH
    B0D4B5R5M7
    B0B8ZGWMH4
    B0D4B6QMWC
    B0D4B4QNR1
    B0D4B6GYM7
    B0B8ZHCKBG
    B0D4B4SR5Y
    B0D4B6GF8R
    B0B8ZFRJFP
    B0B8ZNB3F6
    B0D4B4LBHP
    B0D4B4LH2J
    B0D4B66T3J
    B0B8ZFXKSH
    B0B8ZHRL6Z
    B0B8ZJ6B3P
    B0B8ZMW5RF
    B0B8ZGFW55
    B0D4B5BZL9
    B0B8ZJWRVP
    B0B8ZJJ5FL
    B0B8ZFB5Y9
    B0CFQVJRDZ
    B0D4B5S9SX
    B0CFQXKPH7
    B0CFQVCCRP
    B0CFQSW884
    B0B8ZKHVYP
    B0CFM3T16N
    B0CFM5RH8Z
    B0D4B5FN5X
    B0CFM3DJLK
    B0B8ZJHDZ7
    B0D4B4TQGQ
    B0B8ZHBNWD
    B0B8ZV6V6H
    B0D4B57N44
    B0D4B463RK
    B0B8ZLR374
    B0D4B5TV8T
    B0D4B3NJFT
    B0D4B5J4KJ
    B0D4B34D99
    B0D4B54BJF
    B0D4B4741K
    B0B8ZFMM6Z
    B0B8ZHS5NP
    B0B8ZHTB1W
    B0B8ZTJKFW
    B0B8ZKD7GW
    B0B8ZY62PV
    B0D4B4HC54
    B0B8ZFSYT4
    B0B8ZF1QHM
    B0CFQXJQCF
    B0B8ZJY6HH
    B0B8ZKGYC3
    B0CFQT4F4M
    B0B8ZHY5MK
    B0B8ZM3JNS
    B0D4B65176
    B0B8ZDZJ81
    B0B8ZHH6Y5
    B0B8ZJBP9J
    B0B8ZDPNTL
    B0D4B59Z9J
    B0D4B52MZN
    B0D4B73PCJ
    B0D4B4VXCK
    B0D4B6BTJC
    B0D4B64HV8
    B0D4B58QD7
    B0CK4W5FVZ
    B0CFQXSX8S
    B0B8ZGP61S
    B0B9CBZJJM
    B0B9CDPNFM
    B0BB77PM3Z
    B0BB78BGQQ
    B0BB78172X
    B0B9CD1Y63
    B0BB78FXHC
    B0BB7HHZ37
    B0B9CDR1QK
    B0B9CBLW46
    B0B9CB9DMC
    B0BB78B9DL
    B0BB77GR5D
    B0BB7BQ8GH
    B0BB78BBJ3
    B0BB82QV7X
    B0BB78PX2Y
    B0BB76N16Z
    B0B9CDDD6D
    B0CFR7LR5D
    B0B9C8YYDN
    B0CK4ZC67N
    B0CK4YQV1M
    B0CK4Z579T
    B0B8ZHR1RT
    B0CFLR5YRQ
    B0BB78J2GP
    B0BB783D7W
    B0BB78NQPX
    B0BB76VF7N
    B0BB7661ZZ
    B0BB777Q4L
    B0B9CCHNWZ
    B0B9CFFY79
    B0CFQK2JJF
    B0CFQNK13G
    B0CFQLWP3P
    B0CFLS9RMR
    B0CFLTNCLN
    B0CFQJXLNY
    B0CFQM3Y4N
    B0CFQJ12ZJ
    B0D4B4NTF3
    B0CFR768DP
    B0B8ZGNZMZ
    B0B9CDW8VD
    B0B9CDJZ4C
    B0B9CCZ15J
    B0B9C9WC7N
    B0B9CCPLQN
    B0BB78Z3W8
    B0B8ZFR6NS
    B0B8ZNJLTK
    B0B9CGVKQB
    B0B9CFWXYB
    B0B9CC2CSG
    B0B9CBKL5D
    B0B9CG1M62
    B0B9CFKRRT
    B0B9CCCL3B
    B0B9CG282L
    B0B9CCMGL5
    B0B9CC81YV
    B0B9CGLT75
    B0BB79P6TW
    B0BB76T6ZH
    B0D4B4P9FT
    B0D4B4HGKM
    B0D4B55B6D
    B0CK2L23LZ
    B0B9CFR129
    B0CK4T6VBP
    B0D4B5KSD2
    B0D4B4GZMK
    B0D4B5J8WJ
    B0D4B6MNLB
    B0D4B4ZWKH
    B0BB7B5ZJD
    B0B9CD6WF1
    B0D4B4RRS1
    B0D4B4BD25
    B0B9CGBBR7
    B0B9CD73DB
    B0BB76PDZ7
    B0CK4WGLX2
    B0BB77YK5Z
    B0BB78BGQT
    B0BB792VJH
    B0BB76SFFL
    B0BB78JHRC
    B0BB7Q8W8M
    B0B9CBKK86
    B0B9CCVDQZ
    B0BB787QMC
    B0BB7FS6DQ
    B0BB82MKT1
    B0BB799VYZ
    B0BB78VX88
    B0BB77LHRX
    B0BB784XWH
    B0BB795KH6
    B0BB76YV7N
    B0BB7BSL5K
    B0BB781BRP
    B0BB76X7LM
    B0BB771H18
    B0BB7D387X
    B0BB79PPQH
    B0B9CWPBTC
    B0B9CCJLZH
    B0B9CBLZKT
    B0B9CF8JKL
    B0B9CFY517
    B0B9CGHJM3
    B0B9CCS8KJ
    B0B9CFJ7DC
    B0B9CCYM2S
    B0B9CDFG4Z
    B0BB79CDMQ
    B0BB76Y7RM
    B0BB7B842J
    B0BB7KQZ76
    B0BB791J4H
    B0BB76NQMM
    B0BB79JSYY
    B0CFLRMLSR
    B0CFR6JFC8
    B0BB76Y7YV
    B0CFQM4FTH
    B0CFQKG9VY
    B0CFQLLWZD
    B0CFLV687G
    B0CFLRYBKN
    B0CFQMN9BH
    B0B9CHDMBB
    B0BB7BBKX3
    B0BB74KYJJ
    B0BB76XX4X
    B0BB77X1C3
    B0BB7734TV
    B0B9CC34B6
    B0B9CJHK3F
    B0BB82X8H5
    B0BB791LDM
    B0BB786R4T
    B0CFLSQQHT
    B0B9CCH5B6
    B0B9CCZ154
    B0CFLQWG19
    """
    
    naar_rugs_product_ids = [pid.strip() for pid in product_ids_raw.splitlines() if pid.strip()]
    logger.info(f"Number of product IDs: {len(naar_rugs_product_ids)}")

    # SQL query to fetch the price tracker data for the specified marketplace and date range
    placeholders = ', '.join(['%s'] * len(naar_rugs_product_ids))
    query = f"""
    SELECT "Date", "MarketPlace", "product_ID", availability, "listingPrice", "listPrice", "landedPrice", "shippingPrice", 
           "BSR_CategoryId1", "BSR_CategoryId1_rank", "BSR_CategoryId2", "BSR_CategoryId2_rank", 
           "sellerFeedbackCount", "sellerPositiveFeedbackRating", "size", thickness, 
           "BSR_CategoryId3", "BSR_CategoryId3_rank"
    FROM "Records"."PriceTracker"
    WHERE "MarketPlace" = '{marketplace}'
    AND "product_ID" IN ({placeholders})
    AND "Date" >= CURRENT_DATE - INTERVAL '{days} days';
    """

    # Execute the query
    params = list(naar_rugs_product_ids) 
    cursor.execute(query, params)
    logger.info("Executed price tracker query to fetch Naar Rugs Product Prices successfully.")

    # Fetch results and load them into a DataFrame
    df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    
    df.rename(columns= {"product_ID" : "asin", "Date" : "date", "listingPrice" : "price"}, inplace=True)
    logger.info(f"Renamed product_ID to asin and Date to date and listingPrice to price : {df.shape}")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    logger.info(f"Sorted by date: {df.shape}")
    logger.info(f"Sample data for price tracker data after sorting by date : {df.head()}")

    # Close the cursor and connection
    cursor.close()
    conn.close()

    # Return the DataFrame
    logger.info("Process 4: Fetched Brand's asin's prices successfully.")

    # Define S3 file path
    file_name_ = "price_tracker_data.csv"
    if brand:
        s3_folder = f"{brand}/"
    else:
        s3_folder = ""

    # Save DataFrame to S3
    save_to_s3(
        df=df,
        bucket_name="anarix-cpi",
        s3_folder=s3_folder,
        file_name=file_name_,
    )
    
    price_tracker_df = df[['asin', 'price', 'date']]
    logger.info(f"Columns after filtering asin, price and date : {price_tracker_df.columns}")

    return price_tracker_df


def fetch_product_information(price_tracker_df):
    """Fetch product information and merge with the main DataFrame."""
  
    logger.info(f"Loaded price_tracker_df into fetching product info function : {price_tracker_df.columns}")

    naar_rugs_asins = price_tracker_df["asin"].dropna().unique()
    logger.info(f"Number of unique asins to fetch product info : {len(naar_rugs_asins)}")

    try :
        conn = pg8000.connect(
        host="postgresql-88164-0.cloudclusters.net",
        database="generic",
        user="Pgstest",
        password="testwayfair",
        port=10102
        )

        cursor = conn.cursor()

        # Dynamically create the placeholders for the IN clause
        placeholders = ', '.join(['%s'] * len(naar_rugs_asins))
        query5 = f"""
        SELECT product_id , title , asin , brand
        FROM serp.products
        WHERE asin IN ({placeholders})
        ORDER BY brand;
        """

        cursor.execute(query5, tuple(naar_rugs_asins))
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        naar_rugs_df = pd.DataFrame(results, columns=columns)
        merged_naar_rugs_df = pd.merge(price_tracker_df, naar_rugs_df, on='asin', how='left')
        logger.info(f"Successfully fetched product information for Naar Rugs ASINs.")
        logger.info(f"Columns after merging price tracker data with product info : {merged_naar_rugs_df.columns}")
        logger.info(f"Sample data after merging price tracker data with product info : {merged_naar_rugs_df.head()}")
        
        
        return merged_naar_rugs_df
    except Exception as e :
        logger.error("An error occurred:", e)
        raise e
    finally :
        cursor.close()
        conn.close()

def cleaning_naar_rugs_data(merged_naar_rugs_df, merged_df):

    
    logger.info(f"Loaded merged_naar_rugs_df into cleaning function : {merged_naar_rugs_df.columns}")
    logger.info(f"Loaded merged_df into cleaning function : {merged_df.columns}")   

    # Step 3: Define full column order
    full_column_order = [
        'product_id', 'price', 'date', 'keyword_id', 'keyword', 'title', 'asin', 'brand'
    ]
        
    # Step 4: Add missing columns with None values
    for col in full_column_order:
        if col not in merged_naar_rugs_df.columns:
            merged_naar_rugs_df[col] = None

    # Step 5: Reorder the columns
    merged_naar_rugs_df = merged_naar_rugs_df.reindex(columns=full_column_order)
    logger.info(f"Reordered columns of NapQueen Dataset to match with SERP Dataset : {merged_naar_rugs_df.columns}")
    
    merged_df = merged_df.reindex(columns=full_column_order)
    logger.info(f"Reordered columns of NapQueen Dataset to match with SERP Dataset : {merged_df.columns}")

    # Step 6: Concatenate the SERP & NapQueen DataFrames
    logger.info(f"No.of rows in SERP Dataset before excluding NapQueen Products : {merged_df.shape}")
    merged_df= merged_df[merged_df['brand'] != 'naar']
    logger.info(f"Filtered SERP Dataset after excluding Naar Rugs products : {merged_df.shape}")

    final_merged_df = pd.concat([merged_df, merged_naar_rugs_df], ignore_index=True)
    logger.info(f"Concatenated SERP and NapQueen DataFrames : {final_merged_df.shape}")
    logger.info(f"Sample data after concatenation : {final_merged_df.head()} , {final_merged_df.info()}")

    return final_merged_df


def product_details_merge_data(df, df_scrapped_info):
    """
    Merges two DataFrames on 'ASIN' after filtering and renaming columns, and saves the result.

    :param df: Main DataFrame containing product information with an 'asin' column.
    :param df_scrapped_info: DataFrame containing scrapped product details, including the 'Option' column.
    :return: Merged DataFrame.
    """

    # Filter out rows in df_scrapped_info where 'Option' is '{}'
    df_scrapped_info = df_scrapped_info[df_scrapped_info['Option'] != '{}']

    # Rename 'asin' to 'ASIN' in df to match df_scrapped_info column
    df.rename(columns={'asin': 'ASIN'}, inplace=True)

    # Merge the DataFrames on the 'ASIN' column
    merged_df = pd.merge(df, df_scrapped_info, on='ASIN', how='left')
    logger.info("Process 7: Product Details Merging")
    logger.info(f"Sample data after merging product details : {merged_df.head()}")
    logger.info(f"Columns after merging product details : {merged_df.columns}")

    return merged_df


def query_and_save_to_s3(brand):
    """
    Queries data from the database based on the brand and saves the result to S3.
    
    :param brand: The brand for which the query is executed (e.g., 'EUROPEAN_HOME_DESIGNS', 'NAPQUEEN').
    :param aws_access_key_id: AWS access key ID for authentication.
    :param aws_secret_access_key: AWS secret access key for authentication.
    """
    # Establish connection to the database
    conn = pg8000.connect(
        host="postgresql-88164-0.cloudclusters.net",
        database="amazon",
        user="Pgstest",
        password="testwayfair",
        port=10102
    )
    cursor = conn.cursor()

    if brand == "EUROPEAN_HOME_DESIGNS":
        query = """
        SELECT "Ad Type", "date", asin, ads_date_ref, tsales_date_ref, anarix_id, impressions, clicks, ad_spend, units_sold, ad_sales, shippedunits_m, shippedrevenueamount_m, orderedunits_m, orderedrevenueamount_m, shippedunits_s, shippedrevenueamount_s
        FROM advertising.amazon_sp_seller_and_vendor_ehd;
        """
        file_name_ = "ehd_price_data.csv"
    elif brand == 'NAPQUEEN':
        query = """
        WITH seller_ads AS (
                 SELECT sp_advertised_product_report.date,
                    sp_advertised_product_report."advertisedAsin" AS asin,
                    sp_advertised_product_report.anarix_id::text AS anarix_id,
                    sum(sp_advertised_product_report.impressions) AS impressions,
                    sum(sp_advertised_product_report.clicks) AS clicks,
                    sum(sp_advertised_product_report.spend) AS ad_spend,
                    sum(sp_advertised_product_report."unitsSoldClicks14d") AS units_sold,
                    sum(sp_advertised_product_report.sales14d) AS ad_sales
                   FROM advertising.sp_advertised_product_report
                  WHERE sp_advertised_product_report.anarix_id::text = 'NAPQUEEN'
                  GROUP BY sp_advertised_product_report.date, sp_advertised_product_report."advertisedAsin", sp_advertised_product_report.anarix_id
                ), manufacturing_sales AS (
                 SELECT vendor_real_time_sales_day_wise_with_shipped_data.shipped_date,
                    vendor_real_time_sales_day_wise_with_shipped_data.asin,
                    vendor_real_time_sales_day_wise_with_shipped_data.anarix_id,
                    sum(vendor_real_time_sales_day_wise_with_shipped_data.shipped_units) AS shippedunits,
                    sum(vendor_real_time_sales_day_wise_with_shipped_data."shipped_revenue.amount") AS shippedrevenueamount,
                    sum(vendor_real_time_sales_day_wise_with_shipped_data.ordered_units) AS orderedunits,
                    sum(vendor_real_time_sales_day_wise_with_shipped_data."ordered_revenue.amount"::numeric(20,2)) AS orderedrevenueamount,
                    vendor_real_time_sales_day_wise_with_shipped_data.distributor_view
                   FROM selling_partner_api.vendor_real_time_sales_day_wise_with_shipped_data
                  WHERE vendor_real_time_sales_day_wise_with_shipped_data.distributor_view = 'MANUFACTURING'::text
                  and vendor_real_time_sales_day_wise_with_shipped_data.anarix_id = 'NAPQUEEN'
                  GROUP BY vendor_real_time_sales_day_wise_with_shipped_data.shipped_date, vendor_real_time_sales_day_wise_with_shipped_data.asin, vendor_real_time_sales_day_wise_with_shipped_data.anarix_id, vendor_real_time_sales_day_wise_with_shipped_data.distributor_view
                ), seller_ads_sb AS (
                 SELECT sb_ads.date,
                    COALESCE(x1.asin, 'SB_ASIN_BLANK'::character varying) AS asin,
                    sb_ads.anarix_id::text AS anarix_id,
                    sum(sb_ads.impressions) AS impressions,
                    sum(sb_ads.clicks) AS clicks,
                    sum(sb_ads.cost) AS ad_spend,
                    sum(sb_ads."attributedUnitsOrderedNewToBrand14d") AS units_sold,
                    sum(sb_ads."attributedSales14d") AS ad_sales
                   FROM advertising.sb_ads
                     LEFT JOIN adid_asin_map x1 ON sb_ads."adId" = x1.ad_id
                  WHERE sb_ads.anarix_id::text = 'NAPQUEEN'
                  GROUP BY x1.asin, sb_ads.date, sb_ads.anarix_id
                ), seller_ads_sd AS (
                 SELECT sd_product_ads.date,
                    sd_product_ads.asin,
                    sd_product_ads.anarix_id,
                    sum(sd_product_ads.impressions) AS impressions,
                    sum(sd_product_ads.clicks) AS clicks,
                    sum(sd_product_ads.cost) AS ad_spend,
                    sum(sd_product_ads."viewAttributedUnitsOrdered14d") AS units_sold,
                    sum(sd_product_ads."viewAttributedSales14d") AS ad_sales
                   FROM advertising.sd_product_ads
                  WHERE sd_product_ads.anarix_id::text = 'NAPQUEEN'
                  GROUP BY sd_product_ads.date, sd_product_ads.asin, sd_product_ads.anarix_id
                ), seller_ads_sd_target AS (
                 SELECT sd_product_ads_retarget.date,
                    sd_product_ads_retarget.asin,
                    sd_product_ads_retarget.anarix_id,
                    sum(sd_product_ads_retarget.impressions) AS impressions,
                    sum(sd_product_ads_retarget.clicks) AS clicks,
                    sum(sd_product_ads_retarget.cost) AS ad_spend,
                    sum(sd_product_ads_retarget."viewAttributedUnitsOrdered14d") AS units_sold,
                    sum(sd_product_ads_retarget."viewAttributedSales14d") AS ad_sales
                   FROM advertising.sd_product_ads_retarget
                  WHERE sd_product_ads_retarget.anarix_id::text = 'NAPQUEEN'
                  GROUP BY sd_product_ads_retarget.asin, sd_product_ads_retarget.date, sd_product_ads_retarget.anarix_id
                )
         SELECT 'SP'::text AS "Ad Type",
            COALESCE(manufacturing_sales.shipped_date, seller_ads.date) AS date,
            COALESCE(manufacturing_sales.asin, seller_ads.asin) AS asin,
            seller_ads.date AS ads_date_ref,
            manufacturing_sales.shipped_date AS tsales_date_ref,
            COALESCE(manufacturing_sales.anarix_id, seller_ads.anarix_id) AS anarix_id,
            COALESCE(seller_ads.impressions, 0::numeric) AS impressions,
            COALESCE(seller_ads.clicks, 0::numeric) AS clicks,
            COALESCE(seller_ads.ad_spend, 0::numeric) AS ad_spend,
            COALESCE(seller_ads.units_sold, 0::numeric) AS units_sold,
            COALESCE(seller_ads.ad_sales, 0::numeric) AS ad_sales,
            COALESCE(manufacturing_sales.shippedunits, 0::numeric) AS shippedunits,
            COALESCE(manufacturing_sales.shippedrevenueamount, 0::numeric) AS shippedrevenueamount,
            COALESCE(manufacturing_sales.orderedunits, 0::numeric) AS orderedunits,
            COALESCE(manufacturing_sales.orderedrevenueamount, 0::numeric) AS orderedrevenueamount
           FROM seller_ads
             FULL JOIN manufacturing_sales ON manufacturing_sales.shipped_date = seller_ads.date AND manufacturing_sales.asin::text = seller_ads.asin::text AND manufacturing_sales.anarix_id = seller_ads.anarix_id
        UNION
         SELECT 'SB'::text AS "Ad Type",
            seller_ads_sb.date,
            seller_ads_sb.asin,
            seller_ads_sb.date AS ads_date_ref,
            NULL::date AS tsales_date_ref,
            seller_ads_sb.anarix_id,
            seller_ads_sb.impressions,
            seller_ads_sb.clicks,
            seller_ads_sb.ad_spend,
            seller_ads_sb.units_sold,
            seller_ads_sb.ad_sales,
            NULL::numeric AS shippedunits,
            NULL::numeric AS shippedrevenueamount,
            NULL::numeric AS orderedunits,
            NULL::numeric AS orderedrevenueamount
           FROM seller_ads_sb
        UNION
         SELECT 'SD'::text AS "Ad Type",
            seller_ads_sd.date,
            seller_ads_sd.asin,
            seller_ads_sd.date AS ads_date_ref,
            NULL::date AS tsales_date_ref,
            seller_ads_sd.anarix_id,
            seller_ads_sd.impressions,
            seller_ads_sd.clicks,
            seller_ads_sd.ad_spend,
            seller_ads_sd.units_sold,
            seller_ads_sd.ad_sales,
            NULL::numeric AS shippedunits,
            NULL::numeric AS shippedrevenueamount,
            NULL::numeric AS orderedunits,
            NULL::numeric AS orderedrevenueamount
           FROM seller_ads_sd
        UNION
         SELECT 'SD'::text AS "Ad Type",
            seller_ads_sd_target.date,
            seller_ads_sd_target.asin,
            seller_ads_sd_target.date AS ads_date_ref,
            NULL::date AS tsales_date_ref,
            seller_ads_sd_target.anarix_id,
            seller_ads_sd_target.impressions,
            seller_ads_sd_target.clicks,
            seller_ads_sd_target.ad_spend,
            seller_ads_sd_target.units_sold,
            seller_ads_sd_target.ad_sales,
            NULL::numeric AS shippedunits,
            NULL::numeric AS shippedrevenueamount,
            NULL::numeric AS orderedunits,
            NULL::numeric AS orderedrevenueamount
           FROM seller_ads_sd_target;
        """
        file_name_ = "napqueen_price_tracker.csv"
    else:
            logger.error("Unknown brand specified for fetching sales data")
            return
    # Execute the query and fetch the data
    cursor.execute(query)
    df1 = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    df1['date'] = pd.to_datetime(df1['date'])
    df1 = df1.sort_values(by='date')

    # Save the result to S3
    save_to_s3(
        df=df1,
        bucket_name='anarix-cpi',
        s3_folder=f'{brand}/',
        file_name=file_name_
    )
    
    logger.info(f"Data for {brand} successfully queried and saved to S3 as {file_name}")

    # Close the cursor and connection
    cursor.close()
    conn.close()


def fetch_keyword_ids(df_keyword):
    """
    Fetches keyword IDs for the given keywords from the database and saves the results as a CSV file.

    :param df_keyword: DataFrame containing a column 'keyword' with keywords to query.
    :param db_config: A dictionary with database configuration parameters.
    :param output_file: The file path where the output CSV will be saved.
    """
    # Convert the 'keyword' column to a list
    keywords = df_keyword['keywordText'].to_list()

    logger.info("Starting fetch_keyword_ids for given keywords.")

    db_config = {
        'host': 'postgresql-88164-0.cloudclusters.net',
        'database': 'generic',
        'user': 'Pgstest',
        'password': 'testwayfair',
        'port': 10102
    }

    # Establish connection to the database
    conn = pg8000.connect(
        host=db_config['host'],
        database=db_config['database'],
        user=db_config['user'],
        password=db_config['password'],
        port=db_config['port']
    )

    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()

    # Prepare the SQL query to fetch keyword IDs
    query = """
    SELECT keyword_id, keyword
    FROM serp.keywords
    WHERE keyword = ANY(%s);
    """

    # Execute the query using the list of keywords
    cursor.execute(query, (keywords,))

    # Fetch all results and convert them into a DataFrame
    results = cursor.fetchall()
    df_results = pd.DataFrame(results, columns=['keyword_id', 'keyword'])

    logger.info("Successfully fetched keyword IDs from the database.")

    # Close the cursor and connection
    cursor.close()
    conn.close()
    return df_results


def scrapper_handler(df,file_path, num_workers=15):
    df['asin'] = df['asin'].str.upper()
    df_asins = df['asin'].unique().tolist()
    df_asins = [asin for asin in df_asins if asin.startswith('B')]

    try:
        existing_df = pd.read_csv(file_path, on_bad_lines='skip')
        total_collected = existing_df['ASIN'].tolist()
        asins_to_remove = existing_df[existing_df['Option'] == '{}']['ASIN'].str.upper().unique().tolist()
        total_collected = [asin for asin in total_collected if asin not in asins_to_remove]
        logger.info("Loaded existing ASIN data from file.")
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}. Starting with an empty list.")
        total_collected = []

    asins = [asin for asin in df_asins if asin not in total_collected]

    logger.info(f"Total ASINs: {len(df_asins)}")
    logger.info(f"Collected ASINs: {len(total_collected)}")
    logger.info(f"Remaining ASINs: {len(asins)}")

    # Run parallel scraping for remaining ASINs
    if asins:
        logger.info(f"Starting parallel scraping with {num_workers} workers.")
        try:
            parallel_scrape(asins, num_workers, file_path)
            logger.info("Scraping completed successfully.")
        except Exception as e:
            logger.error(f"Error during parallel scraping: {e}")
    else:
        logger.info("No new ASINs to scrape.")
        

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # List of brands to process
    brand =  "NAAR_RUGS_SELLER"
    
    logger.info(f"Processing brand: {brand}")

    df_keyword = active_keyword_ids(brand)
    keyword_ids_df = fetch_keyword_ids(df_keyword)

    # #Step 2
    serp_df = fetch_serp_data(keyword_ids_df)

    # #Step 3
    merged_df = fetch_and_merge_product_data(serp_df)

    processed_serp_df = pre_processing_serp_data(merged_df)

    #Step 4
    price_tracker_df = fetch_price_tracker_data(marketplace="Amazon", days=30)

    #Step 5
    merged_naar_rugs_df = fetch_product_information(price_tracker_df)

    final_merged_df = cleaning_naar_rugs_data(merged_naar_rugs_df, processed_serp_df)
    multiprocessing.freeze_support()


    today_date = datetime.now().strftime('%Y-%m-%d')
    file_name = f"serp_data_{today_date}.csv"
    save_to_s3(
        df=final_merged_df,
        bucket_name='anarix-cpi',
        s3_folder=f'{brand}/',
        file_name=file_name
    )

    #Step 7
    file_path = "Pipeline/NAAR_RUGS_PRODUCT_DETAILS.csv"
    scrapper_handler(final_merged_df,file_path)

    # Save the updated NAPQUEEN.csv to S3
    save_to_s3(
        df=pd.read_csv(file_path, on_bad_lines='skip'),  # Load the updated file into a DataFrame
        bucket_name='anarix-cpi',
        s3_folder=f'{brand}/',
        file_name='NAAR_RUGS_PRODUCT_DETAILS.csv'
    )

    logger.info(f"Uploaded {file_path} to S3 bucket 'anarix-cpi' in folder '{brand}/'")

    df_scrapped_info = pd.read_csv(file_path ,on_bad_lines='skip')
    final_df = product_details_merge_data(final_merged_df, df_scrapped_info)

    # # Example usage:
    today_date = datetime.now().strftime('%Y-%m-%d')
    file_name = f'merged_data_{today_date}.csv'

    #df.to_csv(file_name,index=False)
    
    save_to_s3(
        df=final_df,
        bucket_name='anarix-cpi',
        s3_folder=f'{brand}/',
        file_name=file_name
    )
    
    # Run query and save to S3 for the brand
    # query_and_save_to_s3(brand=brand)
    
    logger.info(f"Completed processing for brand: {brand}\n")
    

    logger.info("Data processing completed successfully.")