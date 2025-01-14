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
import io

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

DB_CONFIG = {
    "host": "postgresql-88164-0.cloudclusters.net",
    "database": "generic",
    "user": "Pgstest",
    "password": "testwayfair",
    "port": 10102,
}

def active_keyword_ids(brand):
    """
    Fetches active keyword IDs for a given brand from the database.
    """
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
        998654473666848: "SETTON_FARMS"
    }

    try:
        connection = pg8000.connect(**db_config)
        cursor = connection.cursor()
        query = "SELECT * FROM advertising_entities.sp_keywords;"
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        data = pd.DataFrame(rows, columns=columns)
        data = data[data['state'] == "ENABLED"]
        data['profileId'] = data['profileId'].astype(np.int64)
        data['brand'] = data['profileId'].map(profile_id_to_brand)
        data = data[data['brand'] == brand]
        logger.info("Process 1: Found active keyword IDs.")
        return data
    except Exception as e:
        logger.error(f"Error fetching active keyword IDs: {e}")
        raise
    finally:
        if connection:
            cursor.close()
            connection.close()

def fetch_keyword_ids(df_keyword):
    """
    Fetches keyword IDs for given keywords from the database.
    """
    keywords = df_keyword['keywordText'].to_list()

    db_config = {
        "host": "postgresql-88164-0.cloudclusters.net",
        "database": "generic",
        "user": "Pgstest",
        "password": "testwayfair",
        "port": 10102
    }

    try:
        conn = pg8000.connect(**db_config)
        cursor = conn.cursor()

        query = """
        SELECT keyword_id, keyword
        FROM serp.keywords
        WHERE keyword = ANY(%s);
        """
        cursor.execute(query, (keywords,))
        results = cursor.fetchall()
        df_results = pd.DataFrame(results, columns=['keyword_id', 'keyword'])
        logger.info("Fetched keyword IDs from the database.")
        return df_results
    except Exception as e:
        logger.error(f"Error fetching keyword IDs: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def filter_last_occurrence_by_day(df):
    """
    Filters the last occurrence of each ASIN for each day.
    """
    df['scrapped_at'] = pd.to_datetime(df['scrapped_at'])
    df['date'] = df['scrapped_at'].dt.date
    df = df.sort_values(by=['product_id', 'date', 'scrapped_at'], ascending=[True, True, True])
    filtered_df = df.groupby(['product_id', 'date'], as_index=False).last()
    filtered_df.drop(columns=['scrapped_at'], inplace=True)
    logger.info(f"Data type of date column in serp data after droppin duplicates occurrences : {filtered_df['date'].dtype}")
    logger.info("Filtered last occurrences of ASINs by day.")
    return filtered_df

def fetch_serp_data(updated_df):
    """
    Fetches SERP data for given keywords and processes it.
    """
    updated_df = updated_df[['keyword_id', 'keyword']]
    keyword_id_list = updated_df['keyword_id'].unique().tolist()
    logger.info("Initial DataFrame(keyword, keyword_id) info:")
    logger.info(updated_df.info())
    logger.info(f"Number of unique keyword IDs: {len(keyword_id_list)}")

    keyword_id_tuple = tuple(keyword_id_list)

    db_config = {
        "host": "postgresql-88164-0.cloudclusters.net",
        "database": "generic",
        "user": "Pgstest",
        "password": "testwayfair",
        "port": 10102
    }

    conn = pg8000.connect(**db_config)
    cursor = conn.cursor()

    end_date = datetime.now().date() - timedelta(days=21)
    start_date = end_date - timedelta(days=1)

    logger.info(f"Fetching SERP data from {start_date} to {end_date}")
    
    dataframes = []
    current_start_date = start_date
    chunk_size = 7

    while current_start_date <= end_date:
        current_end_date = min(current_start_date + timedelta(days=chunk_size), end_date)
        query = f"""
        SELECT product_id, sale_price AS price, scrapped_at, keyword_id
        FROM serp.amazon_serp
        WHERE keyword_id IN {keyword_id_tuple}
          AND scrapped_at BETWEEN '{current_start_date}' AND '{current_end_date}'
        ORDER BY scrapped_at;
        """
        cursor.execute(query)
        week_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        dataframes.append(week_df)
        current_start_date += timedelta(days=chunk_size)

    cursor.close()
    conn.close()

    all_data_df = pd.concat(dataframes, ignore_index=True)
    merged_df = pd.merge(updated_df, all_data_df, on='keyword_id')
    logger.info(f"Length of serp data before removing all the occurrences : {len(merged_df['product_id'])}")
    filtered_df = filter_last_occurrence_by_day(merged_df)
    logger.info(f"Length of serp data after removing all the occurrences : {len(filtered_df['product_id'])}")
    logger.info("Step-2 : Processed SERP data.")
    return filtered_df

def fetch_and_merge_product_data(df):
    """
    Fetches product details in chunks based on product IDs and merges with the input DataFrame.
    """
    product_id_list = df['product_id'].unique().tolist()
    logger.info(f"Total unique product IDs: {product_id_list}")
    chunk_size = max(1, len(product_id_list) // 10)
    logger.info(f"Using chunk size: {chunk_size}")

    try:
        conn = pg8000.connect(**DB_CONFIG)
        cursor = conn.cursor()
        merged_df = pd.DataFrame()

        for i in range(0, len(product_id_list), chunk_size):
            product_id_chunk = product_id_list[i:i + chunk_size]
            logger.info(f"Processing chunk {i // chunk_size + 1}: {len(product_id_chunk)} products")
            query = """
            SELECT *
            FROM serp.products
            WHERE product_id = ANY(%s)
            ORDER BY brand;
            """
            cursor.execute(query, (product_id_chunk,))
            chunk_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
            merged_df = pd.concat([merged_df, pd.merge(df, chunk_data, on='product_id', how='left')])

        cursor.close()
        conn.close()

        merged_df.rename(columns={'title': 'product_title'}, inplace=True)
        logger.info(f"After renaming price and title and processing serp data : {merged_df.columns}")
        logger.info(" Step-3 : Fetched and merged product data for SERP Data.")
        return merged_df
    except Exception as e:
        logger.error(f"Error fetching and merging product data: {e}")
        raise

def fetch_and_enrich_price_data_by_date_range():
    """
    Fetches and enriches data from sp_api_price_collector within a specific date range.
    """
    try:
        # conn = pg8000.connect(**DB_CONFIG)
        # cursor = conn.cursor()
        # query = """
        # SELECT date, product_id, asin, product_title, brand, price, availability, keyword_id, keyword
        # FROM serp.sp_api_price_collector
        # WHERE date BETWEEN %s AND %s;
        # """
        # cursor.execute(query, (start_date, end_date))
        # price_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        price_data = pd.read_csv("Pipeline\sp_api_24_23_dec.csv")

        price_data['date'] = pd.to_datetime(price_data['date']).dt.date
        unique_dates = price_data['date'].unique()
        logger.info(f"Printing unique dates from sp_api dataframe")
        logger.info(sorted(unique_dates))

        date_counts = price_data['date'].value_counts()

        # Display the counts
        logger.info("Frequency of dates in the dataframe:")
        logger.info(date_counts)

        filtered_data = price_data[price_data['date'] == pd.to_datetime('2024-12-23').date()]

        # Display the filtered dataframe
        logger.info(f"Filtered data for 23rd December 2024:")
        logger.info(filtered_data.head())

        # Optionally, check the number of rows in the filtered dataframe
        logger.info(f"Number of rows for 23rd December 2024: {len(filtered_data)}")

        # cursor.close()
        # conn.close()

        # enriched_data = pd.concat(enriched_data_list, ignore_index=True)
        logger.info("Fetched SP API price data and converted date column to datetime format.")
        logger.info("Step-4 : Processed SP-API Data")
        # return price_data
        return filtered_data
    except Exception as e:
        logger.error(f"Error fetching SP-API data: {e}")
        raise

def align_and_combine_serp_and_sp_api_data(serp_data, sp_api_data):
    """
    Combines SERP and SP API data, aligns columns, and deduplicates by ASIN and date.
    """
    required_columns = ['asin', 'product_id', 'keyword_id', 'product_title', 'brand', 'price', 'keyword', 'date']
    for col in required_columns:
        if col not in serp_data.columns:
            serp_data[col] = None
        if col not in sp_api_data.columns:
            sp_api_data[col] = None

    serp_data = serp_data[required_columns]
    sp_api_data = sp_api_data[required_columns]

    combined_data = pd.concat([serp_data, sp_api_data], ignore_index=True)
    logger.info(f"Length of ASINs before removing duplicates at day level after combining data : {len(combined_data['asin'])}")
    combined_data['date'] = pd.to_datetime(combined_data['date']).dt.date
    combined_data = combined_data.sort_values(by=['asin', 'date'], ascending=[True, True])
    deduplicated_data = combined_data.groupby(['asin', 'date'], as_index=False).last()
    logger.info(f"Length of ASINs after removing duplicates at day level after combining data : {len(deduplicated_data['asin'])}")
    #logger.info(f"Length of ASINs after removing duplicates: {len(deduplicated_data['asin'])}")
    logger.info("Step-5 : Combined and deduplicated SERP and SP API data in the final step.")

    asin_keyword_df = deduplicated_data.groupby('asin')['keyword_id'].apply(lambda x: list(set(x))).reset_index()
    asin_keyword_df.columns = ['asin', 'keyword_id_list']
    save_to_s3(asin_keyword_df, brand, "asin_keyword_id_mapping.csv")

    # Save keyword and keyword_id pairs to S3
    keyword_pairs_df = deduplicated_data[['keyword_id', 'keyword']].drop_duplicates().reset_index(drop=True)
    save_to_s3(keyword_pairs_df, brand, "keyword_x_keyword_id.csv")

    return deduplicated_data    


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

    return merged_df


import boto3
import pandas as pd
from io import StringIO


def save_df_to_s3(df, bucket_name, s3_folder, file_name):
    """
    Saves a DataFrame to an S3 bucket as a CSV file.

    :param df: The DataFrame to save.
    :param bucket_name: The name of the S3 bucket.
    :param s3_folder: The folder path in the S3 bucket (e.g., 'amazon_reviews/').
    :param file_name: The name of the CSV file to be saved (e.g., 'my_data.csv').
    :param aws_access_key_id: AWS access key ID for authentication.
    :param aws_secret_access_key: AWS secret access key for authentication.
    """
    # Create a session with AWS
    session = boto3.Session()

    # Create an S3 client
    s3_client = session.client('s3')

    # Convert DataFrame to CSV format in-memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    # Create the full path for the S3 object
    s3_object_path = f"{s3_folder}{file_name}"

    # Upload the CSV to S3
    s3_client.put_object(
        Bucket=bucket_name,
        Key=s3_object_path,
        Body=csv_buffer.getvalue()
    )

    logger.info(f"DataFrame saved to S3: {bucket_name}/{s3_object_path}")

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
                 SELECT vendor_real_time_sales_day_wise_with_shipped_data_ehd.shipped_date,
                    vendor_real_time_sales_day_wise_with_shipped_data_ehd.asin,
                    vendor_real_time_sales_day_wise_with_shipped_data_ehd.anarix_id,
                    sum(vendor_real_time_sales_day_wise_with_shipped_data_ehd.shipped_units) AS shippedunits,
                    sum(vendor_real_time_sales_day_wise_with_shipped_data_ehd."shipped_revenue.amount") AS shippedrevenueamount,
                    sum(vendor_real_time_sales_day_wise_with_shipped_data_ehd.ordered_units) AS orderedunits,
                    sum(vendor_real_time_sales_day_wise_with_shipped_data_ehd."ordered_revenue.amount"::numeric(20,2)) AS orderedrevenueamount,
                    vendor_real_time_sales_day_wise_with_shipped_data_ehd.distributor_view
                   FROM selling_partner_api.vendor_real_time_sales_day_wise_with_shipped_data_ehd
                  WHERE vendor_real_time_sales_day_wise_with_shipped_data_ehd.distributor_view = 'MANUFACTURING'::text
                  and vendor_real_time_sales_day_wise_with_shipped_data_ehd.anarix_id = 'NAPQUEEN'
                  GROUP BY vendor_real_time_sales_day_wise_with_shipped_data_ehd.shipped_date, vendor_real_time_sales_day_wise_with_shipped_data_ehd.asin, 
                  vendor_real_time_sales_day_wise_with_shipped_data_ehd.anarix_id, vendor_real_time_sales_day_wise_with_shipped_data_ehd.distributor_view
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
    save_df_to_s3(
        df=df1,
        bucket_name='anarix-cpi',
        s3_folder=f'{brand}/',
        file_name=file_name_
    )
    
    logger.info(f"Data for {brand} successfully queried and saved to S3 as {file_name_}")

    # Close the cursor and connection
    cursor.close()
    conn.close()

import boto3
import pandas as pd
import io

def fetch_latest_napqueen_file(bucket_name, brand, prefix="NAPQUEEN", file_extension=".csv"):
    """
    Fetches the latest file matching the prefix and extension in the brand folder from S3 based on LastModified property.

    :param bucket_name: S3 bucket name.
    :param brand: Brand folder in S3 bucket.
    :param prefix: Prefix of the file name to look for (default is "NAPQUEEN").
    :param file_extension: File extension to match (default is ".csv").
    :return: DataFrame loaded from the latest file.
    """
    s3_client = boto3.client('s3')
    folder_path = f"{brand}/"

    try:
        # List objects in the S3 bucket with the specified prefix
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
        if 'Contents' not in response:
            raise FileNotFoundError(f"No files found in bucket '{bucket_name}' with prefix '{folder_path}'.")

        # Filter files matching the prefix and extension
        files = [
            obj for obj in response['Contents']
            if obj['Key'].startswith(folder_path + prefix) and obj['Key'].endswith(file_extension)
        ]
        if not files:
            raise FileNotFoundError(f"No matching files found in bucket '{bucket_name}' for prefix '{prefix}'.")

        # Find the latest file based on LastModified timestamp
        latest_file = max(files, key=lambda x: x['LastModified'])
        latest_file_key = latest_file['Key']
        logger.info(f"Latest file found: {latest_file_key} (LastModified: {latest_file['LastModified']})")

        # Fetch the latest file
        obj = s3_client.get_object(Bucket=bucket_name, Key=latest_file_key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()), on_bad_lines='skip')
        logger.info(f"Fetched file '{latest_file_key}' from S3 bucket '{bucket_name}'.")
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch latest file from S3. Starting fresh. Error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if no file is found


def scrapper_handler(df, bucket_name, brand, file_name="NAPQUEEN.csv", num_workers=15):
    df['asin'] = df['asin'].str.upper()
    df_asins = df['asin'].unique().tolist()
    df_asins = [asin for asin in df_asins if asin.startswith('B')]

    try:
        existing_df = fetch_latest_napqueen_file(bucket_name, brand, file_name)
        total_collected = existing_df['ASIN'].tolist()
        asins_to_remove = existing_df[existing_df['Option'] == '{}']['ASIN'].str.upper().unique().tolist()
        total_collected = [asin for asin in total_collected if asin not in asins_to_remove]
        logger.info("Loaded existing ASIN data from file.")
    except FileNotFoundError:
        logger.warning(f"File not found: {file_name}. Starting with an empty list.")
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

            # Load the updated local file after scraping
            scraped_data = pd.read_csv(file_name, on_bad_lines='skip')
            updated_df = pd.concat([existing_df, scraped_data], ignore_index=True)
            logger.info("Scraping completed and data appended to the S3 file.")
        except Exception as e:
            logger.error(f"Error during parallel scraping: {e}")
            updated_df = existing_df  # Use existing data if scraping fails
    else:
        logger.info("No new ASINs to scrape.")
        updated_df = existing_df

    # Save the updated DataFrame to S3
    save_to_s3(
        df=updated_df,
        brand=brand,
        file_name=file_name
    )

    return updated_df

def fetch_latest_file_from_s3(bucket_name, prefix="NAPQUEEN/merged_data_", file_extension=".csv"):
    """
    Fetches the latest file matching the prefix and extension from the S3 bucket.
    """
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if 'Contents' not in response:
        raise FileNotFoundError(f"No files with prefix '{prefix}' found in the S3 bucket '{bucket_name}'.")

    # Find the latest file based on LastModified timestamp
    files = [
        obj for obj in response['Contents']
        if obj['Key'].startswith(prefix) and obj['Key'].endswith(file_extension)
    ]
    latest_file = max(files, key=lambda x: x['LastModified'])
    return latest_file['Key'], latest_file['LastModified']

def process_and_upload_analysis(bucket_name, new_analysis_df, brand, prefix="merged_data_", file_extension=".csv"):
    """
    Processes daily analysis results, checks the file's date, and appends or creates a new file based on month difference.
    """
    import io
    today = datetime.now() - timedelta(days=6)
    s3_client = boto3.client('s3')
    
    folder_path = f"{brand}/"

    # Step 1: Fetch the latest file
    try:
        latest_file_key, last_modified = fetch_latest_file_from_s3(bucket_name, prefix=folder_path + prefix, file_extension=file_extension)
        logger.info(f"Latest file found: {latest_file_key}, LastModified: {last_modified}")
    except FileNotFoundError:
        # If no files exist, create a new one
        latest_file_key = None
        logger.warning(f"No existing files found. Starting fresh with today's data.")
    
    # Step 2: Check if file exists
    if latest_file_key:
        # Load the file into a DataFrame directly from S3
        obj = s3_client.get_object(Bucket=bucket_name, Key=latest_file_key)
        existing_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        updated_df = pd.concat([existing_df, new_analysis_df], ignore_index=True)
        logging.info(f"Appended serp data to the existig file : {updated_df.info()}")
    else:
        # No existing file, start fresh with the new analysis data
        updated_df = new_analysis_df
        logging.info(f"Creating new file for sepr data as no existing file found : {updated_df.info()}")
        
    # Step 4: Upload the updated DataFrame directly to S3
    new_file_name = f"{folder_path}{prefix}{today.strftime('%Y-%m-%d')}{file_extension}"
    csv_buffer = io.StringIO()
    updated_df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket_name, Key=new_file_name, Body=csv_buffer.getvalue())
    logger.info(f"Uploaded file: {new_file_name} to S3 bucket: {bucket_name}")

    return updated_df

if __name__ == '__main__':
    multiprocessing.freeze_support()

    brand = "NAPQUEEN"
    logger.info(f"Processing for brand: {brand}")
    df = active_keyword_ids(brand)

    df_keywords = fetch_keyword_ids(df)
    df_serp = fetch_serp_data(df_keywords)

    df_product_data = fetch_and_merge_product_data(df_serp)

    end_date = datetime.now().date() - timedelta(days=21)
    start_date = end_date - timedelta(days=1)
    sp_api_data = fetch_and_enrich_price_data_by_date_range()

    final_combined_data = align_and_combine_serp_and_sp_api_data(df_product_data, sp_api_data)
    
    today_date = datetime.now().strftime('%Y-%m-%d')

    save_df_to_s3(
        df=final_combined_data,  # Load the updated file into a DataFrame
        bucket_name='anarix-cpi',
        s3_folder=f'{brand}/',
        file_name=f'serp_data_{today_date}.csv'
    )
    
    updated_napqueen_df = scrapper_handler(
    df=final_combined_data,
    bucket_name="anarix-cpi",
    brand="NAPQUEEN",
    file_name="NAPQUEEN.csv"
    )

    # Step 5: Merge with scrapped info for final output
    final_merged_df = product_details_merge_data(final_combined_data, updated_napqueen_df)

    # #Step 7
    # file_path = "Pipeline/NAPQUEEN.csv"
    # scrapper_handler(final_combined_data,file_path)

    # # Save the updated NAPQUEEN.csv to S3
    # save_df_to_s3(
    #     df=pd.read_csv(file_path, on_bad_lines='skip'),  # Load the updated file into a DataFrame
    #     bucket_name='anarix-cpi',
    #     s3_folder=f'{brand}/',
    #     file_name='NAPQUEEN.csv'
    # )

    # logger.info(f"Uploaded {file_path} to S3 bucket 'anarix-cpi' in folder '{brand}/'")

    # df_scrapped_info = pd.read_csv(file_path ,on_bad_lines='skip')
    # df = product_details_merge_data(final_combined_data, df_scrapped_info)
    
    # today_date = datetime.now().strftime('%Y-%m-%d')
    # file_name = f'merged_data_{today_date}.csv'

    # #df.to_csv(file_name,index=False)
    
    # save_df_to_s3(
    #     df=df,
    #     bucket_name='anarix-cpi',
    #     s3_folder=f'{brand}/',
    #     file_name=file_name
    # )

    merged_df = process_and_upload_analysis(
        bucket_name='anarix-cpi',
        new_analysis_df=final_merged_df,
        brand=brand,
        prefix="merged_data_",
        file_extension=".csv"
    )
    
    query_and_save_to_s3(brand=brand)

    logger.info(f"Completed processing for brand: {brand}\n")
    

    