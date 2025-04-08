import pandas as pd
from datetime import datetime
from logger import logger
from config import S3_BUCKET_NAME , S3_FOLDER, BRAND_SELECTION
from db_utils import get_postgres_connection_ads_query, close_postgres_connection
from data_fetching import fetch_serpkeywords_mongodb, fetch_keyword_mappings, fetch_search_results, fetch_product_information, fetch_napqueen_products, fetch_nq_product_information
from data_processing import preprocess_serp_data, perform_monthly_analysis, clean_napqueen_data
from walmart_scraper import process_and_update_product_details
from s3_utils import upload_file_to_s3


def product_details_merge(final_df, product_details_df, brand='NapQueen'):
    """Merge the product details with the main DataFrame and save to S3."""
    today = datetime.now().date()

    # Convert IDs to string type for merging
    final_df['id'] = final_df['id'].astype(str)
    product_details_df['ID'] = product_details_df['ID'].astype(str)

    # Merge product details into main dataset
    final_merged_df = pd.merge(final_df, product_details_df, left_on='id', right_on='ID', how='left')
    logger.info(f"Merged DataFrame after merging product details : {final_merged_df.shape}")
    logger.info(f"Sample data after merging : {final_merged_df.head()}")

    # Save merged dataset to S3
    try:
        if brand == 'NapQueen':
            upload_file_to_s3(
                df=final_merged_df,
                bucket_name=S3_BUCKET_NAME,
                s3_folder=S3_FOLDER,
                file_name=f"merged_data_{today}.csv"
            )
            logger.info(f"Final merged data uploaded to S3 bucket {S3_BUCKET_NAME} in folder {S3_FOLDER} with file name merged_data_{today}.csv")
        elif brand == 'California Design Den':
            upload_file_to_s3(
                df=final_merged_df,
                bucket_name=S3_BUCKET_NAME,
                s3_folder=S3_FOLDER,
                file_name=f"CALIFORNIA_DESIGN_DEN/merged_data_{today}.csv"
            )
            logger.info(f"Final merged data uploaded to S3 bucket {S3_BUCKET_NAME} in folder {S3_FOLDER} with file name CALIFORNIA_DESIGN_DEN/merged_data_{today}.csv")
    except Exception as e:
        logger.error(f"Error uploading final merged data to S3 in pipeline.py script : {e}")
    
    return final_merged_df

def ads_query(brand='NapQueen'):
    """Fetch Ads data from PostgreSQL and save it to S3."""
    try:
        conn = get_postgres_connection_ads_query()
        cursor = conn.cursor()
        if brand == 'NapQueen':
            query = """
            SELECT COALESCE(ad_sales_wmt."Date", total_sales_wmt."Date") AS "Date",
                COALESCE(ad_sales_wmt."Item Id", total_sales_wmt."Item ID"::character varying) AS item_id,
                COALESCE(ad_sales_wmt.anarix_id, total_sales_wmt.anarix_id::text) AS anarix_id,
                COALESCE(ad_sales_wmt.impressions, 0::numeric) AS impressions,
                COALESCE(ad_sales_wmt.clicks, 0::numeric) AS clicks,
                COALESCE(ad_sales_wmt.ad_spend, 0::numeric) AS ad_spend,
                COALESCE(ad_sales_wmt.ad_sales, 0::numeric) AS ad_sales,
                COALESCE(ad_sales_wmt."Advertised SKU Sales", 0::numeric) AS "Advertised SKU Sales",
                COALESCE(ad_sales_wmt."Other SKU Sales", 0::numeric) AS "Other SKU Sales",
                COALESCE(ad_sales_wmt.units_sold, 0::numeric) AS units_sold_ads,
                COALESCE(ad_sales_wmt."Advertised SKU Units", 0::numeric) AS "Advertised SKU Units",
                COALESCE(ad_sales_wmt."Other SKU Units", 0::numeric) AS "Other SKU Units",
                COALESCE(total_sales_wmt.units_sold, 0::numeric) AS units_sold_retail,
                COALESCE(total_sales_wmt.auth_sales, 0::numeric) AS auth_sales,
                COALESCE(ad_sales_wmt."Orders", 0::numeric) AS "Orders",
                COALESCE(ad_sales_wmt."Brand Attributed Sales", 0::numeric) AS "Brand Attributed Sales",
                COALESCE(ad_sales_wmt."Direct Attributed Sales", 0::numeric) AS "Direct Attributed Sales",
                COALESCE(ad_sales_wmt."Instore Attributed Sales", 0::numeric) AS "Instore Attributed Sales",
                COALESCE(ad_sales_wmt."Instore Orders", 0::numeric) AS "Instore Orders",
                COALESCE(ad_sales_wmt."Instore Other Sales", 0::numeric) AS "Instore Other Sales",
                COALESCE(ad_sales_wmt."Instore Units Sold", 0::numeric) AS "Instore Units Sold",
                COALESCE(ad_sales_wmt."NTB Orders", 0::numeric) AS "NTB Orders",
                COALESCE(ad_sales_wmt."NTB Units Sold", 0::numeric) AS "NTB Units Sold",
                COALESCE(ad_sales_wmt."NTB Sales", 0::numeric) AS "NTB Sales",
                COALESCE(ad_sales_wmt."Related Attributed Sales", 0::numeric) AS "Related Attributed Sales",
                COALESCE(total_sales_wmt.gmv, 0::numeric) AS gmv
            FROM powerbi.walmartadsalesview_new ad_sales_wmt
                FULL JOIN powerbi.walmart_orders1p_all_new total_sales_wmt 
                ON total_sales_wmt."Date" = ad_sales_wmt."Date" 
                AND total_sales_wmt."Item ID" = ad_sales_wmt."Item Id"::text 
                AND total_sales_wmt.anarix_id::text = ad_sales_wmt.anarix_id
                WHERE COALESCE(ad_sales_wmt.anarix_id, total_sales_wmt.anarix_id::text) = ANY (ARRAY['NAPQUEEN_1P'::text, 'NAPQUEEN_3P'::text]);
            """
        elif brand == 'California Design Den':
            query = """
            SELECT COALESCE(ad_sales_wmt."Date", total_sales_wmt."Date") AS "Date",
                COALESCE(ad_sales_wmt."Item Id", total_sales_wmt."Item ID"::character varying) AS item_id,
                COALESCE(ad_sales_wmt.anarix_id, total_sales_wmt.anarix_id::text) AS anarix_id,
                COALESCE(ad_sales_wmt.impressions, 0::numeric) AS impressions,
                COALESCE(ad_sales_wmt.clicks, 0::numeric) AS clicks,
                COALESCE(ad_sales_wmt.ad_spend, 0::numeric) AS ad_spend,
                COALESCE(ad_sales_wmt.ad_sales, 0::numeric) AS ad_sales,
                COALESCE(ad_sales_wmt."Advertised SKU Sales", 0::numeric) AS "Advertised SKU Sales",
                COALESCE(ad_sales_wmt."Other SKU Sales", 0::numeric) AS "Other SKU Sales",
                COALESCE(ad_sales_wmt.units_sold, 0::numeric) AS units_sold_ads,
                COALESCE(ad_sales_wmt."Advertised SKU Units", 0::numeric) AS "Advertised SKU Units",
                COALESCE(ad_sales_wmt."Other SKU Units", 0::numeric) AS "Other SKU Units",
                COALESCE(total_sales_wmt.units_sold, 0::numeric) AS units_sold_retail,
                COALESCE(total_sales_wmt.auth_sales, 0::numeric) AS auth_sales,
                COALESCE(ad_sales_wmt."Orders", 0::numeric) AS "Orders",
                COALESCE(ad_sales_wmt."Brand Attributed Sales", 0::numeric) AS "Brand Attributed Sales",
                COALESCE(ad_sales_wmt."Direct Attributed Sales", 0::numeric) AS "Direct Attributed Sales",
                COALESCE(ad_sales_wmt."Instore Attributed Sales", 0::numeric) AS "Instore Attributed Sales",
                COALESCE(ad_sales_wmt."Instore Orders", 0::numeric) AS "Instore Orders",
                COALESCE(ad_sales_wmt."Instore Other Sales", 0::numeric) AS "Instore Other Sales",
                COALESCE(ad_sales_wmt."Instore Units Sold", 0::numeric) AS "Instore Units Sold",
                COALESCE(ad_sales_wmt."NTB Orders", 0::numeric) AS "NTB Orders",
                COALESCE(ad_sales_wmt."NTB Units Sold", 0::numeric) AS "NTB Units Sold",
                COALESCE(ad_sales_wmt."NTB Sales", 0::numeric) AS "NTB Sales",
                COALESCE(ad_sales_wmt."Related Attributed Sales", 0::numeric) AS "Related Attributed Sales",
                COALESCE(total_sales_wmt.gmv, 0::numeric) AS gmv
            FROM powerbi.walmartadsalesview_new ad_sales_wmt
                FULL JOIN powerbi.walmart_orders1p_all_new total_sales_wmt 
                ON total_sales_wmt."Date" = ad_sales_wmt."Date" 
                AND total_sales_wmt."Item ID" = ad_sales_wmt."Item Id"::text 
                AND total_sales_wmt.anarix_id::text = ad_sales_wmt.anarix_id
                WHERE COALESCE(ad_sales_wmt.anarix_id, total_sales_wmt.anarix_id::text) = ANY (California Design Den 3P);
            """
        cursor.execute(query)
        ads_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        ads_df.rename(columns={"Date": "date"}, inplace=True)
        ads_df['date'] = pd.to_datetime(ads_df['date'])
        ads_df = ads_df.sort_values(by='date')

        # Save to S3
        try:
            if brand == 'NapQueen':
                upload_file_to_s3(
                    df=ads_df,
                    bucket_name=S3_BUCKET_NAME,
                    s3_folder=S3_FOLDER,
                    file_name="napqueen_ads_data.csv"
                )
                logger.info(f"Ads data uploaded to S3 bucket {S3_BUCKET_NAME} in folder {S3_FOLDER} with file name napqueen_ads_data.csv")
            elif brand == 'California Design Den':
                upload_file_to_s3(
                    df=ads_df,
                    bucket_name=S3_BUCKET_NAME,
                    s3_folder=S3_FOLDER,
                    file_name="CALIFORNIA_DESIGN_DEN/cdd_ads_data.csv"
                )
                logger.info(f"Ads data uploaded to S3 bucket {S3_BUCKET_NAME} in folder {S3_FOLDER} with file name cdd_ads_data.csv")
        except Exception as e:
            logger.error(f"Error uploading Ads data to S3 in pipeline.py script : {e}")

        return ads_df

    except Exception as e:
        logger.error(f"Error fetching Ads data in pipeline.py script : {e}")
        raise e
    finally:
        cursor.close()
        close_postgres_connection(conn)

def get_brand_selection():
    """Prompt user to select a brand."""
    logger.info("Select a brand to run the pipeline:")
    logger.info("1. NapQueen")
    logger.info("2. California Design Den Inc.")
    
    choice = BRAND_SELECTION
    
    if choice == "1":
        return "NapQueen"
    elif choice == "2":
        return "California Design Den Inc."
    elif choice is None:
        choice = input("Enter 1 for NapQueen or 2 for California Design Den: ").strip()
    else:
        raise ValueError("Invalid choice. Use '1' or '2'.")


def main():
    """Main execution pipeline for Walmart Data Processing."""
    
    brand = get_brand_selection()
    logger.info(f"Selected brand: {brand}")
    logger.info(f"Starting pipeline execution for brand: {brand}")

    # Fetch Data
    logger.info("Fetching SERP Keywords...")
    serp_keywords_df = fetch_serpkeywords_mongodb(brand=brand)

    logger.info("Fetching Keyword IDs ...")
    keyword_df = fetch_keyword_mappings(serp_keywords_df)

    logger.info("Fetching Walmart Search Results...")
    serp_df = fetch_search_results(keyword_df)

    logger.info("Fetching Product Information...")
    merged_df = fetch_product_information(serp_df)
    
    # Data Preprocessing
    logger.info("Preprocessing SERP Data...")
    merged_df = preprocess_serp_data(merged_df)

    logger.info("Performing Monthly Analysis...")
    merged_df = perform_monthly_analysis(merged_df, days=60)

    logger.info("Fetching NapQueen Product's Price Data...")
    nq_df = fetch_napqueen_products(marketplace="Walmart", brand=brand)

    logger.info("Fetching NapQueen Product Information...")
    nq_merged_df = fetch_nq_product_information(nq_df, brand=brand)

    logger.info("Cleaning and Merging NapQueen Data and Preprocessed SERP Data...")
    final_df = clean_napqueen_data(nq_merged_df, merged_df , brand=brand)
    
    # Process and Update Product Details
    logger.info("Processing and Updating Product Details...")
    product_details_df = process_and_update_product_details(final_df, brand=brand)

    # Merge Product Details with Final Dataset
    final_merged_df = product_details_merge(final_df, product_details_df, brand=brand)

    # Fetch and Upload Ads Data
    logger.info("Fetching Ads Data...")
    ads_query(brand=brand)
    logger.info("Ads Data fetched and uploaded successfully!")

    logger.info("Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
