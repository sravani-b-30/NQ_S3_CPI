import pg8000
import pandas as pd
from datetime import datetime, timedelta

DB_CONFIG = {
    "host": "postgresql-88164-0.cloudclusters.net",
    "port": "10102",
    "database": "walmart",
    "user": "Pgstest",
    "password": "testwayfair"
}

def get_postgres_connection_ads_query():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = pg8000.connect(**DB_CONFIG)
        print("Successfully connected to PostgreSQL database.")
        print(conn)
        return conn
    except Exception as e:
        print.error(f"Error connecting to PostgreSQL: {e}")
        raise

def close_postgres_connection(conn):
    """Closes the given PostgreSQL connection."""
    try:
        if conn:
            conn.close()
            print("PostgreSQL connection closed.")
    except Exception as e:
        print(f"Error closing PostgreSQL connection: {e}")


def ads_query():
    """Fetch Ads data from PostgreSQL and save it to S3."""
    try:
        conn = get_postgres_connection_ads_query()
        cursor = conn.cursor()

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

        cursor.execute(query)
        ads_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        ads_df.rename(columns={"Date": "date"}, inplace=True)
        ads_df['date'] = pd.to_datetime(ads_df['date'])
        ads_df = ads_df.sort_values(by='date')
        print(f"Ads data fetched successfully. Shape: {ads_df.shape}")

        # Save to S3
        return ads_df

    except Exception as e:
        print(f"Error fetching Ads data in pipeline.py script : {e}")
        raise e
    finally:
        cursor.close()
        close_postgres_connection(conn)

if __name__ == "__main__":
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=2)
    print(start_date, end_date)
    ads_df = ads_query()
    print(ads_df.head())
    print(ads_df.columns)
    print(ads_df.shape)
    ads_df.to_csv("napqueen_ads_data.csv", index=False)
    print("Ads data saved to CSV.")