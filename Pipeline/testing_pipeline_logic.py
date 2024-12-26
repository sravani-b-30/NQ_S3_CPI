import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pg8000
import multiprocessing
import logging
import boto3
from io import StringIO

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

import logging

# Configure logging
LOG_FILE_NAME = "pipeline.log"  # Set the log file name
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_NAME),  # Write logs to the specified file
        logging.StreamHandler()  # Also print logs to console
    ]
)

logger = logging.getLogger(__name__)


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

    start_date = datetime.now().date() - timedelta(days=6)
    end_date = datetime.now().date()
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
    logger.info(f"Length of serp data after removing all the occurrences : {len(merged_df['product_id'])}")
    logger.info("Step-2 : Processed SERP data.")
    return filtered_df

def fetch_and_merge_product_data(df):
    """
    Fetches product details in chunks based on product IDs and merges with the input DataFrame.
    """
    product_id_list = df['product_id'].unique().tolist()
    # logger.info(f"Total unique product IDs: {product_id_list}")
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

def fetch_and_enrich_price_data_by_date_range(start_date, end_date):
    """
    Fetches and enriches data from sp_api_price_collector within a specific date range.
    """
    try:
        conn = pg8000.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = """
        SELECT date, product_id, asin, product_title, brand, price, availability, keyword_id, keyword
        FROM serp.sp_api_price_collector
        WHERE date BETWEEN %s AND %s;
        """
        cursor.execute(query, (start_date, end_date))
        price_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

        price_data['date'] = pd.to_datetime(price_data['date']).dt.date
        logger.info(f"SP-API date column type : {price_data['date'].dtype}")
        logger.info(f"Fetched data from sp_api_price_collector for the range {start_date} to {end_date}.")

        cursor.close()
        conn.close()

        # enriched_data = pd.concat(enriched_data_list, ignore_index=True)
        logger.info("Fetched SP API price data and converted date column to datetime format.")
        logger.info("Step-4 : Processed SP-API Data")
        return price_data
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
    logger.info(f"Length of ASINs after removing duplicates at day level after combining data : {len(combined_data['asin'])}")
    logger.info(f"Length of ASINs after removing duplicates: {len(deduplicated_data['asin'])}")
    logger.info("Step-5 : Combined and deduplicated SERP and SP API data in the final step.")

    asin_keyword_df = deduplicated_data.groupby('asin')['keyword_id'].apply(lambda x: list(set(x))).reset_index()
    asin_keyword_df.columns = ['asin', 'keyword_id_list']
    asin_keyword_df.to_csv("asin_keyword_id_mapping.csv", index=False)

    # Save keyword and keyword_id pairs to S3
    keyword_pairs_df = deduplicated_data[['keyword_id', 'keyword']].drop_duplicates().reset_index(drop=True)
    keyword_pairs_df.to_csv("keyword_x_keyword_id.csv", index=False)

    return deduplicated_data    

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

if __name__ == '__main__':
    multiprocessing.freeze_support()

    brand = "NAPQUEEN"
    logger.info(f"Processing for brand: {brand}")
    df = active_keyword_ids(brand)

    df_keywords = fetch_keyword_ids(df)
    df_serp = fetch_serp_data(df_keywords)

    df_product_data = fetch_and_merge_product_data(df_serp)

    start_date = '2024-12-20'
    end_date = datetime.now().date()
    sp_api_data = fetch_and_enrich_price_data_by_date_range(start_date, end_date)

    final_combined_data = align_and_combine_serp_and_sp_api_data(df_product_data, sp_api_data)
    final_combined_data.to_csv("final_combined_data.csv", index=False)

    # save_df_to_s3(
    #     df=final_combined_data,  # Load the updated file into a DataFrame
    #     bucket_name='anarix-cpi',
    #     s3_folder=f'{brand}/',
    #     file_name='serp_data_test.csv'
    # )

    logger.info("Final combined and deduplicated data saved as final_combined_data.csv.")
