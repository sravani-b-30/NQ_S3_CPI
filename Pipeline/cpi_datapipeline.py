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
        998654473666848: "SETTON_FARMS"
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

def fetch_serp_data(updated_df):
    # Extract product IDs as a list

    updated_df = updated_df[['keyword_id','keyword']]
    updated_df['keyword_id'] = updated_df['keyword_id'].astype(int)
    # Log data overview
    logger.info("Initial DataFrame info:")
    logger.info(updated_df.info())
    
    keyword_id_list = updated_df['keyword_id'].unique().tolist()
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
    # end_date = datetime.now().date() + timedelta(days=1)
    # start_date = end_date - timedelta(days=90)
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
        print(week_df)

        # Append the dataframe to the list
        dataframes.append(week_df)

        # Move to the next chunk
        logger.info(current_start_date)
        current_start_date += timedelta(days=chunk_size)
    # Close the cursor and connection
    cursor.close()
    conn.close()
    # Concatenate all dataframes into a single dataframe
    all_data_df = pd.concat(dataframes, ignore_index=True)
    # Merge the two DataFrames on keyword_id
    merged_df = pd.merge(updated_df, all_data_df, on='keyword_id')
    # Rename the scrapped_at column to date
    merged_df.rename(columns={'scrapped_at': 'date'}, inplace=True)
    # Return the merged DataFrame
    logger.info("Process 2: Finding SERP Data")
    return merged_df

def fetch_and_merge_product_data(df):
    """
    This function reads SERP data from a CSV file, fetches additional product details from the database in chunks,
    merges the two DataFrames, and saves the merged result as a CSV file.

    :param df: DataFrame containing SERP data with product IDs
    :param chunk_size: The number of product IDs to fetch in each query chunk
    :return: Merged DataFrame
    """
    # Extract product IDs from the DataFrame and convert to list
    product_id_list = df['product_id'].unique().tolist()
    total_products = len(product_id_list)
    logger.info(f"Total unique product IDs: {total_products}")

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
        SELECT *
        FROM serp.products
        WHERE product_id = ANY(%s)
        ORDER BY brand;
        """

        # Execute the query with product_id_chunk as a parameter
        cursor = conn.cursor()
        cursor.execute(query, (product_id_chunk,))

        # Fetch the results and convert to DataFrame
        df1 = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

        # Merge the chunked data with the original DataFrame
        merged_df = pd.concat([merged_df, pd.merge(df, df1, on='product_id')])
        print(i)

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

    logger.info("Process 3: Product Price Data fetched and files saved to S3")
    return merged_df

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

def fetch_price_tracker_data(marketplace, days=4):
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

    # SQL query to fetch the price tracker data for the specified marketplace and date range
    query = f"""
    SELECT "Date", "MarketPlace", "product_ID", availability, "listingPrice", "listPrice", "landedPrice", "shippingPrice", 
           "BSR_CategoryId1", "BSR_CategoryId1_rank", "BSR_CategoryId2", "BSR_CategoryId2_rank", 
           "sellerFeedbackCount", "sellerPositiveFeedbackRating", "size", thickness, 
           "BSR_CategoryId3", "BSR_CategoryId3_rank"
    FROM "Records"."PriceTracker"
    WHERE "MarketPlace" = '{marketplace}'
    AND "Date" >= CURRENT_DATE - INTERVAL '{days} days';
    """

    # Execute the query
    cursor.execute(query)
    logger.info("Executed price tracker query successfully.")

    # Fetch results and load them into a DataFrame
    df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    # Convert the 'Date' column to datetime format and sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    logger.info("Data fetched and processed into DataFrame.")

    # Close the cursor and connection
    cursor.close()
    conn.close()
    # Return the DataFrame
    logger.info("Process 4: Finding Brand's Sales Data")

    # Define S3 file path
    file_name_ = "price_tracker_data.csv"
    if brand:
        s3_folder = f"{brand}/"
    else:
        s3_folder = ""

    # Save DataFrame to S3
    save_pricetracker_df_to_s3(
        df=df,
        bucket_name="anarix-cpi",
        s3_folder=s3_folder,
        file_name=file_name_,
    )
    
    df.rename(columns={'product_ID': 'asin'}, inplace=True)
    price_tracker_df = df[['asin', 'listingPrice', 'Date']]

    return price_tracker_df

def save_pricetracker_df_to_s3(df, bucket_name, s3_folder, file_name):
    """
    Saves a DataFrame to an S3 bucket as a CSV file.

    :param df: The DataFrame to save.
    :param bucket_name: The name of the S3 bucket.
    :param s3_folder: The folder path in the S3 bucket (e.g., 'brand/').
    :param file_name: The name of the CSV file to be saved (e.g., 'my_data.csv').
    :param aws_access_key_id: AWS access key ID for authentication.
    :param aws_secret_access_key: AWS secret access key for authentication.
    """
    import boto3
    from io import StringIO

    # Initialize S3 client
    s3_client = boto3.client(
        's3',
    )

    # Convert DataFrame to CSV in-memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    # Define the S3 object key
    s3_key = f"{s3_folder}{file_name}"

    try:
        # Upload to S3
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue())
        logger.info(f"DataFrame saved to S3: {bucket_name}/{s3_key}")
    except Exception as e:
        logger.error(f"Failed to save {file_name} to S3: {e}")

def fetch_price_tracker_data_for_asins(df):
    """
    Fetches updated prices for the given ASINs from the price tracker table.

    :param df: DataFrame containing ASINs for which prices need to be updated.
    :return: DataFrame with updated prices.
    """
    # Get unique ASINs to fetch prices for
    asin_list = df['asin'].unique().tolist()
    if not asin_list:
        logger.warning("No ASINs provided for price fetching.")
        return df

    logger.info(f"Fetching prices for {len(asin_list)} ASINs.")

    # Establish database connection
    conn = pg8000.connect(
        host="postgresql-88164-0.cloudclusters.net",
        database="generic",
        user="Pgstest",
        password="testwayfair",
        port=10102
    )
    cursor = conn.cursor()

    # Fetch prices from the price tracker table
    query = """
        SELECT asin,
               price as listingprice
        FROM serp.sp_api_price_collector
        WHERE asin = ANY(%s);
    """
    cursor.execute(query, (asin_list,))
    rows = cursor.fetchall()

    if not rows:
        logger.warning("No rows returned from the price tracker query.")
        cursor.close()
        conn.close()
        return df  # Return the original DataFrame if no updates are available

    # Convert query results to a DataFrame
    price_data = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    logger.info(f"Fetched price data. Sample:\n{price_data.head()}")

    cursor.close()
    conn.close()

    # Check if 'listingPrice' exists in the fetched data
    if 'listingprice' not in price_data.columns:
        logger.error("'listingprice' column is missing in the fetched price data.")
        return df

    # Merge updated prices into the original DataFrame
    df = pd.merge(df, price_data, on='asin', how='left')
    logger.info("Merged price data with the original DataFrame.")

    # Replace the 'price' column with 'listingPrice' where available
    df['sale_price'] = np.where(df['listingprice'].notna(), df['listingprice'], df['sale_price'])
    df.drop(columns=['listingprice'], inplace=True)

    logger.info("Updated prices for ASINs with listingprice.")
    return df

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def process_asin_price_data(df, days=2):
    """
    Processes the ASIN price data for the last 'days' days and returns a consolidated DataFrame.
    Fetches and updates prices for excluded ASINs and aggregates data for each day.

    :param df: Input DataFrame containing ASIN data with associated details.
    :param days: The number of days to analyze (default is 3).
    :return: Consolidated DataFrame for the processed data.
    """
    # Ensure 'date' column is in proper datetime format
    df['date'] = pd.to_datetime(df['date'], format='mixed').dt.date

    # Initialize a list to store the results for each day
    dfs = []

    # Iterate over the last 'days' days
    for i in range(days):
        analysis_date = df['date'].max() - timedelta(days=i)

        # Define the date range: last 30 days ending at analysis_date
        start_date = analysis_date - timedelta(days=30)

        # Filter the DataFrame for the last 30 days
        last_30_days_df = df[(df['date'] <= analysis_date) & (df['date'] > start_date)]

        # Sort the DataFrame by ASIN and date (descending order)
        last_30_days_df = last_30_days_df.sort_values(by=['asin', 'date'], ascending=[False, False])
        
        last_30_days_df = last_30_days_df.apply(
            lambda col: col.astype(float) if col.apply(lambda x: isinstance(x, Decimal)).any() else col
        )
        
        # Group by ASIN and aggregate other metrics
        unique_asins = last_30_days_df.groupby('asin').agg({
            'title': 'first',
            'sale_price': 'first',
            'brand': 'first',
            'latest_rating_count': 'first',
            'latest_stars': 'first',
            'image_url': 'first',
            'date': 'first'
        }).reset_index()

        logger.info(f"Day {i}: Aggregated ASIN data. Total unique ASINs: {len(unique_asins)}")

        # Filter out ASINs whose last available date is not the analysis date
        asins_to_exclude = unique_asins[unique_asins['date'] != analysis_date]['asin'].tolist()
        logger.info(f"Day {i}: ASINs to exclude due to mismatched date: {len(asins_to_exclude)}")

        # Handle price updates for excluded ASINs
        if asins_to_exclude:
            logger.info(f"Day {i}: Fetching updated prices for {len(asins_to_exclude)} ASINs.")
            excluded_asins_df = unique_asins[unique_asins['asin'].isin(asins_to_exclude)]

            # Log a sample of ASINs to be updated
            logger.info(f"Day {i}: Sample of ASINs to update: {asins_to_exclude[:5]}")

            # Fetch updated prices for these ASINs
            updated_prices_df = fetch_price_tracker_data_for_asins(excluded_asins_df)
            logger.info(f"Day {i}: Fetched updated prices for {len(updated_prices_df)} ASINs.")
            
            updated_prices_df = updated_prices_df.apply(
                lambda col: col.astype(float) if col.apply(lambda x: isinstance(x, Decimal)).any() else col
            )

            # Ensure index alignment for update
            # updated_prices_df.set_index('asin', inplace=True)
            # unique_asins.set_index('asin', inplace=True)

            # Update the prices in the unique_asins DataFrame
            unique_asins.update(updated_prices_df)
            logger.info(f"Day {i}: Updated prices in the main DataFrame.")
            logger.info(f"After updating prices in unique_asins : {len(unique_asins)}")

            # Reset index back to default
            unique_asins.reset_index(inplace=True)
        else:
            logger.info(f"Day {i}: No ASINs require price updates.")

        # Drop the date column and assign the analysis date to all ASINs
        logger.info(f"Day {i}: Dropping date column and assigning analysis date.")
        unique_asins.drop(columns=['date'], inplace=True)
        unique_asins['date'] = analysis_date

        # Append the processed DataFrame for the day to the list
        dfs.append(unique_asins)
        logger.info(f"Day {i}: Appended processed data.")

    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Concatenated data for {days} days. Total rows: {len(final_df)}")

    # Log a sample of the final DataFrame
    logger.info(f"Final processed data sample:\n{final_df.head(5)}")

    logger.info(f"Processing complete for the last {days} days.")
    return final_df

def replace_napqueen_prices(df, df_price_tracker):
    
    # Rename 'analysis_date' to 'date'
    df = df.rename(columns={'analysis_date': 'date', 'sale_price':'price'})
    logger.info(f"After aggregating and renaming columns of serp data :")
    logger.info(df.columns)
    logger.info(df.info())

    napqueen_df = df.copy()
    napqueen_df.info()
    logger.info("Starting replacement of NapQueen prices.")
    
    # Step 1: Filter napqueen products from merged_df
    napqueen_df = napqueen_df[napqueen_df['brand'] == 'napqueen']
    napqueen_df.info()
    logger.info(f"Top 5 NapQueen ASINs before processing:\n{napqueen_df.head(5)}")
    
    # Ensure both columns are converted to the same type
    napqueen_df['date'] = pd.to_datetime(napqueen_df['date']).dt.date  # Convert to date (no time component)
    df_price_tracker['Date'] = pd.to_datetime(df_price_tracker['Date'], format='mixed').dt.date  # Convert to date

    logger.info(f"Data Type of Date column in price_tracker_df: {df_price_tracker['Date'].dtype}")
    logger.info(f"Data Type of date column in serp data: {napqueen_df['date'].dtype}")

    df_price_tracker = df_price_tracker.sort_values(by='Date', ascending=False)
    #price_tracker_df = price_tracker_df.drop_duplicates(subset=['asin'], keep='first')

    napqueen_asins = napqueen_df['asin'].head(5).tolist()
    logger.info(f"Listing prices from price_tracker_df for top 5 NapQueen ASINs:\n"
                  f"{df_price_tracker[df_price_tracker['asin'].isin(napqueen_asins)]}")

 
    # Step 2: Merge napqueen_df with price_tracker_df on 'asin'
    napqueen_df = pd.merge(napqueen_df, df_price_tracker[['asin', 'listingPrice', 'Date']], left_on= ['asin', 'date'], right_on=['asin', 'Date'], how='left')
    napqueen_df.info()
    logger.info(f"After merging with price_tracker_df:\n{napqueen_df.head(5)}")

    # Step 3: Replace sale_price with listingPrice in napqueen_df where available
    napqueen_df['price'] = np.where(
        napqueen_df['listingPrice'].notna(),  # Condition: If listingPrice is not null
        napqueen_df['listingPrice'],         # Replace with listingPrice
        napqueen_df['price']            # Otherwise, retain sale_price
    )
    logger.info(f"Updated sale_price in NapQueen products:\n{napqueen_df.head(5)}")

    # Step 4: Drop the listingPrice column as it's no longer needed in napqueen_df
    napqueen_df.drop(columns=['listingPrice', 'Date'], inplace=True)
    napqueen_df.info()
    # Step 5: Remove napqueen products from the original merged_df
    merged_df = df[df['brand'] != 'napqueen']
    logger.info(f"After removing napqueen products : ")
    merged_df.info()

    # Step 6: Append the updated napqueen_df back to merged_df
    merged_df = pd.concat([merged_df, napqueen_df], ignore_index=True)
    logger.info(f"After adding back napqueen products :")
    merged_df.info()
    logger.info("NapQueen price replacement complete.")
    logger.info(f"Final merged_df sample after replacement:\n{merged_df[merged_df['asin'].isin(napqueen_asins)]}")

    logger.info("Process 6: Replacing napqueen prices is done")
    
    return merged_df


# def merge_and_clean_data(df, df_price):
#     """
#     Merges two dataframes on 'asin' and 'date', fills missing price values, and cleans the merged data.

#     :param df: DataFrame containing ASIN and price data with 'analysis_date' to be renamed as 'date'.
#     :param df_price: DataFrame containing additional price data to merge.
#     :return: Merged and cleaned DataFrame.
#     """

#     # Rename 'analysis_date' to 'date'
#     df = df.rename(columns={'analysis_date': 'date'})

#     # Ensure 'date' columns in both dataframes are in datetime format
#     df['date'] = pd.to_datetime(df['date'])
#     df_price['Date'] = pd.to_datetime(df_price['Date'])
#     df_price.rename(columns={'asin': 'product_ID'}, inplace=True)

#     # Fill missing price values by forward and backward filling within 'asin' and 'date' groups
#     df['price'] = df.groupby(['asin', 'date'])['price'].transform(lambda group: group.ffill().bfill())

#     # Merge the two DataFrames on 'asin' and 'date' (using a left join)
#     merged_df = pd.merge(
#         df,
#         df_price[['Date', 'product_ID', 'listingPrice']],
#         how='left',
#         left_on=['asin', 'date'],
#         right_on=['product_ID', 'Date']
#     )

#     # Fill missing 'price' values with 'listingPrice'
#     # merged_df['price'] = merged_df['price'].fillna(merged_df['listingPrice'])

#     # Drop unnecessary columns from the merge
#     merged_df = merged_df.drop(columns=['product_ID', 'Date', 'listingPrice'])

    
#     return merged_df

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
    save_df_to_s3(
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

def fetch_latest_file_from_s3(bucket_name, prefix="merged_data_", file_extension=".csv"):
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

def process_and_upload_analysis(bucket_name, new_analysis_df, prefix="merged_data_", file_extension=".csv"):
    """
    Processes daily analysis results, checks the file's date, and appends or creates a new file based on month difference.
    """
    import io
    today = datetime.now()
    s3_client = boto3.client('s3')

    # Step 1: Fetch the latest file
    try:
        latest_file_key, last_modified = fetch_latest_file_from_s3(bucket_name, prefix, file_extension)
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
        
        # Extract the date from the file name
        file_date_str = latest_file_key.split('_')[-1].split('.')[0]
        file_date = datetime.strptime(file_date_str, '%Y-%m-%d')

        # Step 3: Compare months
        if today.month == file_date.month and today.year == file_date.year:
            logger.info("Same month. Appending to the existing file.")
            updated_df = pd.concat([existing_df, new_analysis_df], ignore_index=True)
        else:
            logger.info("Different month. Creating a new file.")
            updated_df = new_analysis_df
    else:
        # No existing file, start fresh
        updated_df = new_analysis_df

    # Step 4: Upload the updated DataFrame directly to S3
    new_file_name = f"{prefix}{today.strftime('%Y-%m-%d')}{file_extension}"
    csv_buffer = io.StringIO()
    updated_df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket_name, Key=new_file_name, Body=csv_buffer.getvalue())
    logger.info(f"Uploaded file: {new_file_name} to S3 bucket: {bucket_name}")

    return updated_df

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # List of brands to process
    brand =  "NAPQUEEN"
    
    logger.info(f"Processing brand: {brand}")

    df = active_keyword_ids(brand)
    df = fetch_keyword_ids(df)

    # #Step 2
    df = fetch_serp_data(df)

    # #Step 3
    df = fetch_and_merge_product_data(df)

    #Step 4
    df_price_tracker = fetch_price_tracker_data(marketplace="Amazon", days=4)
    #Step 5
    df = process_asin_price_data(df, days=2)
    multiprocessing.freeze_support()

    # df = replace_napqueen_prices(df, df_price_tracker)

    intermediate_file = f"/tmp/{brand}_testing.csv"
    df.to_csv(intermediate_file, index=False)
    today_date = datetime.now().strftime('%Y-%m-%d')
    file_name = f"serp_data_{today_date}.csv"
    save_df_to_s3(
        df=df,
        bucket_name='anarix-cpi',
        s3_folder=f'{brand}/',
        file_name=file_name
    )

    #Step 7
    file_path = "Pipeline/NAPQUEEN.csv"
    scrapper_handler(df,file_path)

    # Save the updated NAPQUEEN.csv to S3
    save_df_to_s3(
        df=pd.read_csv(file_path, on_bad_lines='skip'),  # Load the updated file into a DataFrame
        bucket_name='anarix-cpi',
        s3_folder=f'{brand}/',
        file_name='NAPQUEEN.csv'
    )

    logger.info(f"Uploaded {file_path} to S3 bucket 'anarix-cpi' in folder '{brand}/'")

    df_scrapped_info = pd.read_csv(file_path ,on_bad_lines='skip')
    df = product_details_merge_data(df, df_scrapped_info)
    
    merged_df = process_and_upload_analysis(
        bucket_name='anarix-cpi',
        new_analysis_df=df,
        brand=brand
    )

    # # Example usage:
    # today_date = datetime.now().strftime('%Y-%m-%d')
    # file_name = f'merged_data_{today_date}.csv'

    # #df.to_csv(file_name,index=False)
    
    # save_df_to_s3(
    #     df=merged_df,
    #     bucket_name='anarix-cpi',
    #     s3_folder=f'{brand}/',
    #     file_name=file_name
    # )
    
    # Run query and save to S3 for the brand
    query_and_save_to_s3(brand=brand)
    
    logger.info(f"Completed processing for brand: {brand}\n")
    

    