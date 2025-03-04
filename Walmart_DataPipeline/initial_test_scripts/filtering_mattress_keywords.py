import pandas as pd
from datetime import datetime, timedelta
import pg8000



# import pandas as pd

# # Load the dataset
# df = pd.read_csv("walmart_cpi_napqueen_pre-analysis.csv")  # Replace with your actual dataset filename

# # Convert `scrapped_at` to datetime, handling mixed formats
# df['scrapped_at'] = pd.to_datetime(df['scrapped_at'], errors='coerce', format='mixed')

# # Drop rows where `scrapped_at` could not be parsed
# df = df.dropna(subset=['scrapped_at'])

# # Extract the date part from `scrapped_at` and create a new `date` column
# df['date'] = df['scrapped_at'].dt.date

# # Sort the DataFrame by `walmart_id`, `date`, and `scrapped_at`
# df = df.sort_values(by=['walmart_id', 'date', 'scrapped_at'])

# # Group by `walmart_id` and `date`, and take the last occurrence
# final_df = df.groupby(['walmart_id', 'date']).tail(1)

# # Reset index to clean up the DataFrame
# final_df = final_df.reset_index(drop=True)
# # Save the resulting DataFrame to a CSV file
# final_df.to_csv("walmart_napqueen.csv", index=False)

# print("Filtered dataset saved to 'walmart_napqueen.csv'.")


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
#         last_30_days_df = last_30_days_df.sort_values(by=['id', 'date'], ascending=[True, False])

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
#     final_df = final_df.rename(columns={"sale_price": "price"})

#     final_df.to_csv("walmart_cpi_napqueen.csv" , index=False)

#     return final_df 

# if __name__ == "__main__":

#  df = pd.read_csv("walmart_napqueen.csv", on_bad_lines='skip')
#  df.rename(columns={"walmart_id" : "id"}, inplace=True)
#  print(df.columns)

#  df.drop(columns=['scrapped_at'], inplace=True)
#  print("After dropping scrapped_at :")
#  print(df.columns)
 
#  final_df = monthly_analysis(df , days=30)

import ast
import re

def parse_dict_str(dict_str):
    try:
        return ast.literal_eval(dict_str)
    except ValueError:
        return {}

# Function to convert list of key-value pairs to a dictionary
def list_to_dict(details):
    if isinstance(details, list):
        # Convert list of {'key': key, 'value': value} pairs into a single dictionary
        return {item['key']: item['value'] for item in details if 'key' in item and 'value' in item}
    return details  # If it's already a dictionary or another type, return as-is

def merge_dicts(dict1, dict2):
    merged = dict1.copy()
    merged.update(dict2)
    return merged
def extract_brand_from_title(title):
    if pd.isna(title) or not title:
        return 'unknown'
    return title.split()[0].lower()


def extract_style(title):
    title = str(title)
    style_pattern = r"\b(\d+)\s*(inches?|in|inch|\"|''|'\s*'\s*)\b"
    style_match = re.search(style_pattern, title.lower())

    if style_match:
        number = style_match.group(1)
        return f"{number} Inch"

    style_pattern_with_quote = r"\b(\d+)\s*(''{1,2})"
    style_match = re.search(style_pattern_with_quote, title.lower())

    if style_match:
        number = style_match.group(1)
        return f"{number} Inch"
    return None


def extract_size(title):
    title = str(title)
    size_patterns = {
        'Twin XL': r'\btwin[-\s]xl\b',
        'Queen': r'\bqueen\b',
        'Full': r'\b(full|double)\b',
        'Twin': r'\btwin\b',
        'King': r'\bking\b'
    }

    title_lower = title.lower()

    for size, pattern in size_patterns.items():
        if re.search(pattern, title_lower):
            return size

    return None

details_key_rename_map = {
    'Bed Size': 'Size',
    'Mattress Thickness': 'Style',
    'Maximum Load Weight': 'Weight',
    'Assembled Product Dimensions (L x W x H)': 'Product Dimensions'
}

def load_and_preprocess_data():
    # Load data from the single CSV file
    merged_data_df = pd.read_csv("CPI_Data_Preprocessing\\walmart_cpi_napqueen.csv", on_bad_lines='skip')
    
    merged_data_df['id'] = merged_data_df['id'].astype(str)
    # Rename columns as specified
    merged_data_df = merged_data_df.rename(columns={"id": "asin", "Specifications": "Product Details", "brand_name":"brand" , "analysis_date":"date"})

    merged_data_df['asin'] = merged_data_df['asin'].str.upper()
    
    # Convert 'Product Details' from string to dictionary
    merged_data_df['Product Details'] = merged_data_df['Product Details'].apply(parse_dict_str).apply(list_to_dict)

    # # Display the type and sample values of 'Product Details' to understand its format
    # st.write("Inspecting 'Product Details' format for the first few rows in the raw data:")
    # for idx, details in enumerate(merged_data_df['Product Details'].head(10)):
    #     st.write(f"Row {idx} - Type: {type(details)}, Value: {details}")

    def fill_missing_brand(df):
        # Fill 'brand' from 'Product Details' dictionary's 'Brand' key, if it exists
        has_brand_in_details = df['Product Details'].apply(lambda details: details.get('Brand') if isinstance(details, dict) else None)
        df['brand'] = df['brand'].fillna(has_brand_in_details)
        
        # For remaining products with missing 'brand', extract from 'product_title'
        missing_brand_mask = df['brand'].isna() | (df['brand'] == "")
        df.loc[missing_brand_mask, 'brand'] = df.loc[missing_brand_mask, 'product_title'].apply(extract_brand_from_title)
        
        return df

    # Apply function across partitions
    merged_data_df = fill_missing_brand(merged_data_df)

    merged_data_df['price'] = pd.to_numeric(merged_data_df['price'], errors='coerce')

    # Define regular expressions to normalize the style
    style_pattern = r"\b(\d+(?:\.\d+)?)(?:-|\s*)?(?:inches?|in|inch|\"|''|'\s*'\s*)\b"

    # Updated normalize_style function to handle decimal and hyphenated numbers
    def normalize_style(value):
        if isinstance(value, str):
            # Match using the enhanced pattern
            match = re.search(style_pattern, value, re.IGNORECASE)
            if match:
                # Convert to float to handle decimals
                number = float(match.group(1))
                # Return as integer if it has no fractional part, otherwise return as float
                if number.is_integer():
                    return f"{int(number)} Inch"
                else:
                    return f"{number} Inch"
        return value  # Return the original value if it doesn't match any pattern

    # Function to rename and normalize 'Product Details' keys
    def rename_and_normalize_product_details(details):
        if isinstance(details, dict):
            # Rename keys based on mapping and normalize the 'Style' value if it exists
            renamed_details = {details_key_rename_map.get(key, key): value for key, value in details.items()}
            
            if 'Size' not in renamed_details:
                renamed_details['Size'] = None
            if 'Style' not in renamed_details:
                renamed_details['Style'] = None

            # Normalize 'Style' if present in the renamed details
            if 'Style' in renamed_details:
                renamed_details['Style'] = normalize_style(renamed_details['Style'])
            return renamed_details
        return details

    # Step 1: Apply renaming and normalization function to 'Product Details'
    merged_data_df['Product Details'] = merged_data_df['Product Details'].apply(rename_and_normalize_product_details)
    
    unique_products_df = merged_data_df.drop_duplicates(subset='asin').copy()
    # Prepare verification_df with necessary columns
    verification_df = unique_products_df[['asin', 'product_title', 'Product Details']].copy()
    verification_df['Size'] = unique_products_df['Product Details'].apply(lambda details: details.get('Size', None))
    verification_df['Style'] = unique_products_df['Product Details'].apply(lambda details: details.get('Style', None))

    # Fill missing values using `product_title`
    verification_df['Size'] = verification_df.apply(lambda row: extract_size(row['product_title']) if pd.isna(row['Size']) else row['Size'], axis=1)
    verification_df['Style'] = verification_df.apply(lambda row: extract_style(row['product_title']) if pd.isna(row['Style']) else row['Style'], axis=1)

    verification_df['Product Dimensions'] = verification_df['Product Details'].apply(lambda details: details.get('Product Dimensions', None))

    def extract_style_from_dimensions(dimensions):
        if isinstance(dimensions, str):
            dimension_match = re.search(r'(\d+(?:\.\d+)?)\s*Inches$', dimensions)
            if dimension_match:
                number = float(dimension_match.group(1))
                return f"{int(number)} Inch" if number.is_integer() else f"{number} Inch"
        return None

    verification_df['Style'] = verification_df.apply(
        lambda row: extract_style_from_dimensions(row['Product Dimensions']) if pd.isna(row['Style']) else row['Style'], axis=1)

    def standardize_style_format(style_value):
        if isinstance(style_value, str):
            match = re.search(r'(\d+(?:\.\d+)?)', style_value)
            if match:
                number = match.group(1)
                return f"{number} Inch"
        return style_value

    verification_df['Style'] = verification_df['Style'].apply(standardize_style_format)

    # Update merged_data_df with final Size and Style
    merged_data_df = merged_data_df.merge(
    verification_df[['asin', 'Size', 'Style']],
    on='asin',
    how='left'
    )
    
    print("Columns after merge:", merged_data_df.columns)
    
    def update_product_details(row):
        details = row['Product Details']
        if isinstance(details, dict):
            details['Size'] = row['Size']
            details['Style'] = row['Style']
        return details

    merged_data_df['Product Details'] = merged_data_df.apply(update_product_details, axis=1)
    merged_data_df.drop(columns=['Size', 'Style'], inplace=True)

    # merged_data_df.to_csv('product_details_verification_final_standardized.csv', index=False)
    # print("Final CSV with standardized Style values saved as product_details_verification_final_standardized.csv.")

    asin_keyword_df = merged_data_df[['asin', 'keyword']].copy()
    asin_keyword_df = asin_keyword_df.drop_duplicates(subset='asin')
    asin_keyword_df.to_csv("asin_keywords.csv", index=False)
    return asin_keyword_df, merged_data_df

if __name__ == "__main__":
    load_and_preprocess_data()