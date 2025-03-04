import pandas as pd
from db_utils import get_postgres_connection, get_mongo_client, close_postgres_connection, close_mongo_client
from logger import logger
from datetime import datetime, timedelta

def fetch_serpkeywords_mongodb(brand):
    """Fetch SERP keywords from MongoDB for a specific brand."""
    client = get_mongo_client()
    try:
        collection1 = client["walmartmarketplaceaccounts"]
        collection2 = client["serpkeywords"]

        account_map = {str(doc["accountId"]): doc["partnerDisplayName"] for doc in collection1.find({})}
        df = pd.DataFrame(collection2.find({}))
        df["accountId"] = df["accountId"].astype(str)
        df["brand"] = df["accountId"].map(account_map)
        
        logger.info("Step 1 - Fetching SERP keywords from MongoDB completed successfully.")
        logger.info(f"Number of rows fetched: {df.shape[0]}")
        logger.info(f"Number of columns fetched: {df.shape[1]}")
        logger.info(f"Columns fetched: {df.columns}")
        logger.info(f"Number of keywords fetched from MongoDB: {len(df['keyword'].unique())}")
        logger.info(f"Sample keywords: {df['keyword'][:5]}") 

        # **Check if filtering removes all data**
        filtered_df = df[df["brand"] == brand]
        logger.info(f"Number of rows after filtering for brand '{brand}': {filtered_df.shape[0]}")

        if filtered_df.empty:
            logger.warning(f"No data found for brand '{brand}'. Check MongoDB records.")

        return filtered_df
    
    except Exception as e:
        logger.error(f"Error fetching SERP keywords from MongoDB in data_fetching.py : {e}")
        raise e 
    finally:
        close_mongo_client(client)

def fetch_keyword_mappings(serp_keywords_df):
    """Fetch keyword mappings from PostgreSQL."""
    conn = get_postgres_connection()
    try:
        logger.info(f"Sample of SERP Keywords Dataframe : {serp_keywords_df.head()}")
        logger.info(f"{serp_keywords_df.info()}")
        keywords = serp_keywords_df['keyword'].tolist()
        logger.info(f"Datatype of keywords: {type(keywords)}")
        logger.info(f"Number of keywords: {len(keywords)}")

        query = """
        SELECT keyword_id, keyword
        FROM walmart_serp.search_keywords
        WHERE keyword = ANY(%s);
        """
        with conn.cursor() as cursor:
            cursor.execute(query, (keywords,))
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

        keyword_df = pd.DataFrame(results, columns=columns)
        logger.info("Fetched keyword_id's from the database :")
        keyword_df.info()
        logger.info("Step 2 : Merging the keyword_id's with the SERP keywords dataframe completed successfully.")
        
        return keyword_df
    except Exception as e:
        logger.error(f"Error fetching keyword id's from PostgreSQL in data_fetching.py: {e}")
        raise e 
    finally:
        close_postgres_connection(conn)

def fetch_search_results(keyword_df):
    """Fetch search results from PostgreSQL."""
    conn = get_postgres_connection()
    try:
        keyword_df = keyword_df[['keyword_id', 'keyword']]
        keyword_ids = keyword_df['keyword_id'].unique().tolist()
        start_date = (datetime.now() - timedelta(days=60)).date()
        end_date = datetime.now().date()
        
        placeholders = ', '.join(['%s'] * len(keyword_ids))
        query = f"""
        SELECT * FROM walmart_serp.walmart_search_results
        WHERE keyword_id IN ({placeholders})
          AND scrapped_at BETWEEN %s AND %s
        ORDER BY scrapped_at;
        """
        logger.info("Fetching serp data from the database for the following dates :")
        logger.info(f"Start Date: {start_date}")
        logger.info(f"End Date: {end_date}")

        params = keyword_ids + [start_date, end_date]
        
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        serp_df = pd.DataFrame(results, columns=columns)
        serp_df = pd.merge(serp_df, keyword_df, on="keyword_id")

        logger.info("Fetched search results from the database.")
        logger.info(f"SERP datset after merging with keyword_df : {serp_df.shape}")
        logger.info("Step 3 : Merging the SERP data with the keyword_id's completed successfully.")

        return serp_df
    except Exception as e:
        logger.error(f"Error fetching search results from PostgreSQL in data_fetching.py: {e}")
        raise e
    finally:
        close_postgres_connection(conn)

def fetch_product_information(serp_df):
    """Fetch product information and merge with SERP results."""
    conn = get_postgres_connection()
    try:
        serp_df.rename(columns={"product_walmart_id": "walmart_id"}, inplace=True)
        logger.info(f"Columns after renaming product_walmart_id to walmart_id : {serp_df.columns}")
        
        walmart_ids = serp_df["walmart_id"].dropna().unique().tolist()
        logger.info(f"Number of unique Walmart IDs: {len(walmart_ids)}")

        placeholders = ', '.join(['%s'] * len(walmart_ids))
        query = f"""
        SELECT walmart_id, product_title, brand FROM walmart_serp.product_information
        WHERE walmart_id IN ({placeholders});
        """
        with conn.cursor() as cursor:
            cursor.execute(query, tuple(walmart_ids))
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        product_info_df = pd.DataFrame(results, columns=columns)
        merged_df = pd.merge(serp_df, product_info_df, on="walmart_id", how="left")

        logger.info(f"Merged product information : {merged_df.shape}")
        logger.info(f"Sample data after merging : {merged_df.head()}")
        logger.info("Step 4 : Merging the product information with the SERP data completed successfully.")

        return merged_df
    except Exception as e:
        logger.error(f"Error fetching product information from PostgreSQL in data_fetching.py: {e}")
        raise e
    finally:
        close_postgres_connection(conn)

def fetch_napqueen_products(marketplace='Walmart'):
    """Fetch NapQueen product data from PostgreSQL."""
    conn = get_postgres_connection()
    try:
        with open('Walmart_DataPipeline/src/napqueen_product_ids.txt', 'r') as file:
            product_ids = [line.strip() for line in file if line.strip()]
        
        placeholders = ', '.join(['%s'] * len(product_ids))
        query = f"""
        SELECT "Date", "MarketPlace", "product_ID", availability, "listingPrice", "listPrice", "landedPrice", "shippingPrice",
          "size", "thickness", "channel_type"
        FROM "Records"."PriceTracker"
        WHERE "MarketPlace" = %s
        AND "product_ID" IN ({placeholders})
        AND "Date" BETWEEN %s AND %s;
        """

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)

        with conn.cursor() as cursor:
            # Execute the query with the date range
            params = [marketplace] + list(product_ids) + [start_date, end_date]
            cursor.execute(query, params)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        nq_df = pd.DataFrame(results, columns=columns)
        nq_df.rename(columns= {"product_ID" : "walmart_id", "Date" : "date"}, inplace=True)
        logger.info(f"Renamed product_ID to walmart_id and Date to date : {nq_df.shape}")
        # serp_df = serp_df.groupby(['id', 'date']).tail(1)
        
        nq_df = (
            nq_df.groupby(['walmart_id', 'date'])
            .apply(lambda group: group.drop_duplicates(subset=['listingPrice']))
            .reset_index(drop=True)
        )

        logger.info(f"Fetched NapQueen product data : {nq_df.shape}")
        logger.info(f"Sample data after fetching NapQueen product data : {nq_df.head()}")
        logger.info("Step 7 : Fetching NapQueen product data completed successfully")

        return nq_df
    except Exception as e:
        logger.error(f"Error fetching NapQueen product data in data_fetching.py: {e}")
        raise e
    finally:
        close_postgres_connection(conn)

def fetch_nq_product_information(nq_df):
    """Fetch NapQueen product information and merge with NapQueen data."""
    conn = get_postgres_connection()
    try:
        nq_df = nq_df[['date', 'walmart_id', 'listingPrice', 'channel_type']]   
        logger.info(f"Columns after filtering date, walmart_id and listingPrice : {nq_df.columns}")

        walmart_ids = nq_df["walmart_id"].dropna().unique().tolist()
        logger.info(f"Number of unique NapQueen Walmart IDs to fetch product info : {len(walmart_ids)}")

        placeholders = ', '.join(['%s'] * len(walmart_ids))
        query = f"""
        SELECT walmart_id, product_title, brand
        FROM walmart_serp.product_information
        WHERE walmart_id IN ({placeholders});
        """
        with conn.cursor() as cursor:
            cursor.execute(query, tuple(walmart_ids))
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        product_info_df = pd.DataFrame(results, columns=columns)
        nq_df["walmart_id"] = nq_df["walmart_id"].astype(str)
        product_info_df["walmart_id"] = product_info_df["walmart_id"].astype(str)
        nq_merged_df = pd.merge(nq_df, product_info_df, on="walmart_id", how="left")

        logger.info(f"Merged product information with NapQueen Price Data : {nq_merged_df.shape}")
        logger.info(f"Sample data after merging : {nq_merged_df.head()}")
        logger.info("Step 8 : Merging the product information with the NapQueen Price data completed successfully.")

        return nq_merged_df
    except Exception as e:
        logger.error(f"Error fetching NapQueen product information in data_fetching.py: {e}")
        raise e
    finally:
        close_postgres_connection(conn)
