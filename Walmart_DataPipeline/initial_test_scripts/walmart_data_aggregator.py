# import pandas as pd
# from datetime import datetime , timedelta
# import numpy as np
# import pg8000


# # Database credentials
# DB_CONFIG = {
#     "host": "postgresql-88164-0.cloudclusters.net",       
#     "port": 10102,         
#     "database": "generic",  
#     "user": "Pgstest",  
#     "password": "testwayfair"  
# }

# def fetch_data_from_table(query):
#     try:
#         # Establish the connection
#         conn = pg8000.connect(**DB_CONFIG)
#         cursor = conn.cursor()
        
#         # Execute the query
#         cursor.execute(query)
        
#         # Fetch results
#         results = cursor.fetchall()
#         columns = [desc[0] for desc in cursor.description]
        
#         # Close the connection
#         cursor.close()
#         conn.close()
        
#         # Return results as a DataFrame
#         return pd.DataFrame(results, columns=columns)

#     except Exception as e:
#         print("An error occurred:", e)
#         return None

# def main():

#     merged_df = pd.DataFrame()

#     # Query to fetch data from walmart_search_results
#     last_30_days_date = datetime.now() - timedelta(days=30)
#     query1 = """
#     SELECT *
#     FROM walmart_serp.walmart_search_results
#     WHERE scrapped_at >= %s
#     """
    
#     # Fetch data
#     search_results_df = fetch_data_from_table(query1, (last_30_days_date,))
    
#     if search_results_df is not None:
#         # Save the search results to a CSV file
#         search_results_df.to_csv("walmart_serp_data.csv", index=False)
#         print("Saved search results to 'walmart_serp_data.csv'")
        
#         # Query to fetch keyword_id and keyword from search_keywords
#         query2 = """
#         SELECT keyword_id, keyword
#         FROM walmart_serp.search_keywords
#         """
        
#         # Fetch keyword mappings
#         keyword_mapping_df = fetch_data_from_table(query2)
        
#         if keyword_mapping_df is not None:
#             # Merge the dataframes on keyword_id
#             merged_df = pd.merge(search_results_df, keyword_mapping_df, on="keyword_id", how="left")
            
#             # Save the merged data to a new CSV file
#             merged_df.to_csv("walmart_data_with_keywords.csv", index=False)
#             print("Saved merged results to 'walmart_data_with_keywords.csv'")
#         else:
#             print("Failed to fetch data from search_keywords table.")
#     else:
#         print("Failed to fetch data from walmart_search_results table.")
#         return


#     query3 = """
#     SELECT badge_id , badge_name
#     FROM walmart_serp.product_badges
#     """ 
#     badge_name_df = fetch_data_from_table(query3)   

#     if badge_name_df is not None:
#         merged_df = pd.merge(merged_df, badge_name_df, on="badge_id", how='left')

#         merged_df.to_csv("walmart_data_with_badges.csv", index=False)
#         print("Saved merged results with badge names")
#     else: 
#         print("Failed to fetch badge names")
    
#     merged_df.rename(columns= {'listing_type_id' : 'type_id'})
#     print(merged_df.columns)

#     query4 = """
#     SELECT type_id , type_name
#     FROM walmart_serp.listing_types
#     """
#     type_name_df = fetch_data_from_table(query4)

#     if type_name_df is not None:
#         merged_df = pd.merge(merged_df , type_name_df, on='type_id', how='left')

#         merged_df.to_csv("walmart_data_with_listtype_name.csv", index=False)
#         print("Saved merged_df with list type name")
#     else:
#         print("Failed to fetch list type names")

#     merged_df.rename(columns= {'product_walmart_id' : 'walmart_id'})
    
#     walmart_ids = merged_df['walmart_id'].dropna().unique()

#     query5 = """
#     SELECT product_title , brand
#     FROM walmart_serp.product_information
#     WHERE walmart_id IN %s
#     """
#     product_info_df = fetch_data_from_table(query5, tuple(walmart_ids),)

#     if product_info_df is not None:
#         merged_df = pd.merge(merged_df, product_info_df, on='walmart_id', how='left')

#         merged_df.to_csv("walmart_cpi_napqueen_pre-analysis.csv", index=False)
#         print("Successfully added product title and brand as well , and fetched the final df")


# def monthly_analysis(df , days=30):

#     df['date'] = pd.to_datetime(df['date'], format='mixed').dt.date

#     # Initialize a list to store the results for each day
#     dfs = []

#     # Iterate over the last 'days' days
#     for i in range(days):
#         analysis_date = df['date'].max() - timedelta(days=i)

#         # Define the date range: last 30 days ending at analysis_date
#         start_date = analysis_date - timedelta(days=30)

#         # Filter the DataFrame for the last 30 days
#         last_30_days_df = df[(df['date'] <= analysis_date) & (df['date'] > start_date)]
        
#         # Sort the DataFrame by ASIN and date (descending order)
#         last_30_days_df = last_30_days_df.sort_values(by=['asin', 'date'], ascending=[True, False])

#         # Get unique ASINs and their corresponding latest prices and other details
#         unique_asins = last_30_days_df.groupby('asin').agg({
#             'title': 'first',
#             'sale_price': 'first',
#             'brand': 'first',
#             'latest_rating_count': 'first',
#             'latest_stars': 'first',
#             'image_url': 'first'
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
#     final_df = final_df.rename(columns={"sale_price": "price"})

#     final_df.to_csv("walmart_cpi_napqueen.csv" , index=False)

#     return final_df 

# if __name__ == "__main__":
#     main()

#     df = pd.read_csv("walmart_cpi_napqueen_pre-analysis.csv" , on_bad_lines='skip')
#     df.rename(columns= {'scrapped_at' : 'date'})

#     monthly_analysis()


import pandas as pd
from datetime import datetime, timedelta
import pg8000

# Database credentials
DB_CONFIG = {
        "host": "postgresql-88164-0.cloudclusters.net",
        "port": 10102,
        "database": "generic",
        "user": "Pgstest",
        "password": "testwayfair"
    }


def fetch_search_results():
    """Fetch data from walmart_search_results."""
    try :
        conn = pg8000.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Define start_date (60 days ago) and end_date (today)
        start_date = (datetime.now() - timedelta(days=60)).date()
        end_date = (datetime.now() + timedelta(days=1)).date()

        # Define the SQL query using DATE conversion
        query = """
        SELECT *
        FROM walmart_serp.walmart_search_results
        WHERE scrapped_at BETWEEN %s AND %s
        """
        print("Start Date:", start_date)
        print("End Date:", end_date)

        # Execute the query with the date range
        cursor.execute(query, (start_date, end_date))
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        serp_df = pd.DataFrame(results, columns=columns)
        #serp_df.to_csv("walmart_serp_data.csv", index=False)
        return serp_df
    
    except Exception as e:
        print("An error occurred:", e)
        return pd.DataFrame()
    
    finally :
        cursor.close()
        conn.close()

def fetch_keyword_mappings(serp_df):
    """Fetch keyword mappings and merge with the main DataFrame."""
    try :
        conn = pg8000.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query2 = """
        SELECT keyword_id, keyword
        FROM walmart_serp.search_keywords
        """
        cursor.execute(query2)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        keyword_mapping_df = pd.DataFrame(results, columns=columns)
        print("No.of Keyword objects in keyword df before filtering out Mattress Keywords :")
        keyword_mapping_df.info()
        keyword_ids = pd.read_csv("Pipeline\\Walmart_Mattress_Keywords.csv", on_bad_lines='skip')

        keyword_mapping_df = pd.merge(keyword_mapping_df, keyword_ids, on="keyword")
        print("After filtering out Mattress Keywords :")
        keyword_mapping_df.info()

        if not keyword_mapping_df.empty:
            merged_df = pd.merge(serp_df, keyword_mapping_df, on="keyword_id")
            print("Merged keyword mappings.")
        else:
            print("Failed to fetch data from search_keywords table.")
        #merged_df.to_csv("walmart_data_with_keywords.csv", index=False)
        return merged_df
    except Exception as e :
        print("An error occurred:", e)
        return pd.DataFrame()
    
    finally :
        cursor.close()
        conn.close()

def fetch_badge_names(merged_df):
    """Fetch badge names and merge with the main DataFrame."""
    try :
        conn = pg8000.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query3 = """
        SELECT badge_id, badge_name
        FROM walmart_serp.product_badges
        """

        cursor.execute(query3)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        badge_name_df = pd.DataFrame(results, columns=columns)
        
        if not badge_name_df.empty:
            merged_df = pd.merge(merged_df, badge_name_df, on="badge_id", how="left")
            print("Merged badge names.")
        else:
            print("Failed to fetch badge names.")
        #merged_df.to_csv("walmart_data_with_badges.csv", index=False)    
        return merged_df
    except Exception as e :
        print("An error occurred:", e)
        return pd.DataFrame()
    finally :
        cursor.close()
        conn.close()

def fetch_listing_types(merged_df):
    """Fetch listing types and merge with the main DataFrame."""
    try :
        conn = pg8000.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query4 = """
        SELECT type_id, type_name
        FROM walmart_serp.listing_types
        """
        cursor.execute(query4)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        type_name_df = pd.DataFrame(results , columns=columns)
        if not type_name_df.empty:
            merged_df = pd.merge(merged_df, type_name_df, on="type_id", how="left")
            print("Merged listing types.")
        else:
            print("Failed to fetch listing types.")
        #merged_df.to_csv("walmart_data_with_listtype.csv", index=False)    
        return merged_df
    except Exception as e :
        print("An error occurred:", e)
        return pd.DataFrame()
    finally :
        cursor.close()
        conn.close()

def fetch_product_information(merged_df, walmart_ids):
    """Fetch product information and merge with the main DataFrame."""
    try :
        conn = pg8000.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Dynamically create the placeholders for the IN clause
        placeholders = ', '.join(['%s'] * len(walmart_ids))
        query5 = f"""
        SELECT walmart_id, product_title, brand
        FROM walmart_serp.product_information
        WHERE walmart_id IN ({placeholders})
        """

        cursor.execute(query5, tuple(walmart_ids))
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        product_info_df = pd.DataFrame(results, columns=columns)
        if not product_info_df.empty:
            merged_df = pd.merge(merged_df, product_info_df, on="walmart_id", how="left")
            print("Merged product information.")
        else:
            print("Failed to fetch product information.")
        #merged_df.to_csv("walmart_cpi_napqueen_pre-analysis.csv", index=False)    
        return merged_df
    except Exception as e :
        print("An error occurred:", e)
        return pd.DataFrame()
    finally :
        cursor.close()
        conn.close()

def pre_processing_data(merged_df) :

    merged_df['scrapped_at'] = pd.to_datetime(merged_df['scrapped_at'], errors='coerce', format='mixed')

    # Drop rows where `scrapped_at` could not be parsed
    merged_df = merged_df.dropna(subset=['scrapped_at'])

    # Extract the date part from `scrapped_at` and create a new `date` column
    merged_df['date'] = merged_df['scrapped_at'].dt.date

    # Sort the DataFrame by `walmart_id`, `date`, and `scrapped_at`
    merged_df = merged_df.sort_values(by=['walmart_id', 'date', 'scrapped_at'])

    # Group by `walmart_id` and `date`, and take the last occurrence
    merged_df = merged_df.groupby(['walmart_id', 'date']).tail(1)
    
    merged_df.drop(columns=['scrapped_at'], inplace=True)
    print("After dropping scrapped_at :")
    print(merged_df.columns)
    
    # Reset index to clean up the DataFrame
    merged_df = merged_df.reset_index(drop=True)
    # Save the resulting DataFrame to a CSV file
    merged_df.to_csv("walmart_napqueen.csv", index=False)
    return merged_df

def monthly_analysis(merged_df , days=60):

    merged_df['date'] = pd.to_datetime(merged_df['date'], format='mixed').dt.date

    # Initialize a list to store the results for each day
    dfs = []

    # Iterate over the last 'days' days
    for i in range(days):
        analysis_date = merged_df['date'].max() - timedelta(days=i)

        # Define the date range: last 30 days ending at analysis_date
        start_date = analysis_date - timedelta(days=30)

        # Filter the DataFrame for the last 30 days
        last_30_days_df = merged_df[(merged_df['date'] <= analysis_date) & (merged_df['date'] > start_date)]
        
        # Sort the DataFrame by ASIN and date (descending order)
        last_30_days_df = last_30_days_df.sort_values(by=['id', 'date'], ascending=[True, False])

        # Get unique ASINs and their corresponding latest prices and other details
        unique_asins = last_30_days_df.groupby('id').agg({
            'product_title': 'first',
            'sale_price': 'first',
            'brand': 'first',
            'rank' : 'first',
            'organic_search_rank' : 'first',
            'sponsored_search_rank' : 'first',
            'keyword' : 'first',
            'keyword_id' : 'first'
        }).reset_index()

        # Add 'analysis_date' column to track the date of analysis
        unique_asins['analysis_date'] = analysis_date

        # Append the processed DataFrame for the day to the list
        dfs.append(unique_asins)
        print(f"Processed ASIN data for {analysis_date}")

    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(dfs)

    # Reset index and rename 'latest_sale_price' to 'price'
    final_df = final_df.reset_index(drop=True)
    final_df = final_df.rename(columns={"sale_price": "price"})

    final_df.to_csv("walmart_serp_data_30_Jan.csv" , index=False)

    return final_df 

print("Filtered dataset saved to 'walmart_napqueen.csv'.")

def main():
    serp_df = fetch_search_results()

    # Step-by-step data fetching and merging
    merged_df = fetch_keyword_mappings(serp_df)
    merged_df.info()
    print(len(merged_df.columns))

    merged_df = fetch_badge_names(merged_df)

    merged_df.rename(columns={"listing_type_id": "type_id"}, inplace=True)
    merged_df = fetch_listing_types(merged_df)

    merged_df.rename(columns={"product_walmart_id": "walmart_id"}, inplace=True)
    walmart_ids = merged_df["walmart_id"].dropna().unique()

    merged_df = fetch_product_information(merged_df, walmart_ids)

    merged_df = pre_processing_data(merged_df)
    # df = pd.read_csv("walmart_napqueen.csv", on_bad_lines='skip')
    merged_df.rename(columns={"walmart_id" : "id"}, inplace=True)
    print(merged_df.columns)

    print(f"Data fetching for walmart napqueen is complete")
    
    merged_df = monthly_analysis(merged_df , days=60)
        

if __name__ == "__main__":
    main()
