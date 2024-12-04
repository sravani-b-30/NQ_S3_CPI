import pandas as pd
import ast
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime, timedelta
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import streamlit as st 
import boto3
from datetime import datetime
import dask.dataframe as dd
from dask import delayed
import io
import csv


nltk.download('punkt', quiet=True)


def format_details(details):
    return "\n".join([f"{key}: {value}" for key, value in details.items()])

def tokenize(text):
    tokens = word_tokenize(text.lower())
    return set(tokens)


def tokenize_with_delimiters(text):
    text = text.lower()
    tokens = re.split(r'[,;.\s]', text)
    return set(token for token in tokens if token)


def extract_numeric_metric(text):
    return set(re.findall(r'\d+\s?[a-zA-Z"]+', text.lower()))


def extract_thickness(dimension_str):
    match = re.search(r'\d+"Th', dimension_str)
    if match:
        return match.group()
    return ''


def tokenized_similarity(value1, value2):
    if value1 is None or value2 is None:
        return 0
    tokens1 = tokenize_with_delimiters(str(value1))
    tokens2 = tokenize_with_delimiters(str(value2))
    numeric_metric1 = extract_numeric_metric(str(value1))
    numeric_metric2 = extract_numeric_metric(str(value2))
    intersection = tokens1.intersection(tokens2).union(numeric_metric1.intersection(numeric_metric2))
    return len(intersection) / len(tokens1.union(tokens2))


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def title_similarity(title1, title2):
    # Tokenize the titles
    title1_tokens = tokenize_with_delimiters(title1)
    title2_tokens = tokenize_with_delimiters(title2)

    # Calculate intersection and union of title tokens
    intersection = title1_tokens.intersection(title2_tokens)
    union = title1_tokens.union(title2_tokens)

    # Calculate token similarity score
    token_similarity_score = len(intersection) / len(union)

    # Extract numeric metrics
    numeric_metric1 = extract_numeric_metric(title1)
    numeric_metric2 = extract_numeric_metric(title2)

    # Calculate numeric metric match score
    numeric_match_count = len(numeric_metric1.intersection(numeric_metric2))

    # Final similarity score
    similarity_score = (token_similarity_score + numeric_match_count) * 100

    return similarity_score, title1_tokens, title2_tokens, intersection


def description_similarity(desc1, desc2):
    desc1_tokens = tokenize_with_delimiters(desc1)
    desc2_tokens = tokenize_with_delimiters(desc2)
    intersection = desc1_tokens.intersection(desc2_tokens)
    union = desc1_tokens.union(desc2_tokens)
    similarity_score = len(intersection) / len(union) * 100
    return similarity_score, desc1_tokens, desc2_tokens, intersection


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


def convert_weight_to_kg(weight_str):
    weight_str = weight_str.lower()
    match = re.search(r'(\d+\.?\d*)\s*(pounds?|lbs?|kg)', weight_str)
    if match:
        value, unit = match.groups()
        value = float(value)
        if 'pound' in unit or 'lb' in unit:
            value *= 0.453592
        return value
    return None


def parse_weight(weight_str):
    weight_kg = convert_weight_to_kg(weight_str)
    return weight_kg


def parse_dimensions(dimension_str):
    matches = re.findall(r'(\d+\.?\d*)\s*"?([a-zA-Z]+)"?', dimension_str)
    if matches:
        return {unit: float(value) for value, unit in matches}
    return {}


def compare_weights(weight1, weight2):
    weight_kg1 = parse_weight(weight1)
    weight_kg2 = parse_weight(weight2)
    if weight_kg1 is not None and weight_kg2 is not None:
        return 1 if abs(weight_kg1 - weight_kg2) < 1e-2 else 0
    return 0


def compare_dimensions(dim1, dim2):
    dim1_parsed = parse_dimensions(dim1)
    dim2_parsed = parse_dimensions(dim2)
    if not dim1_parsed or not dim2_parsed:
        return 0
    matching_keys = set(dim1_parsed.keys()).intersection(set(dim2_parsed.keys()))
    matching_score = sum(1 for key in matching_keys if abs(dim1_parsed[key] - dim2_parsed[key]) < 1e-2)
    total_keys = len(set(dim1_parsed.keys()).union(set(dim2_parsed.keys())))
    return matching_score / total_keys


def calculate_similarity(details1, details2, title1, title2, desc1, desc2):
    score = 0
    total_keys = len(details1.keys())
    details_comparison = []
    for key in details1.keys():
        if key in details2:
            value1 = str(details1[key])
            value2 = str(details2[key])
            if 'weight' in key.lower():
                match_score = compare_weights(value1, value2)
                details_comparison.append(f"{key}: {value1} vs {value2} -> Match: {match_score}")
                score += match_score
            elif 'dimension' in key.lower() or key.lower() == 'product dimensions':
                match_score = compare_dimensions(value1, value2)
                details_comparison.append(f"{key}: {value1} vs {value2} -> Match Score: {match_score}")
                score += match_score
            else:
                match_score = tokenized_similarity(value1, value2)
                details_comparison.append(f"{key}: {value1} vs {value2} -> Match Score: {match_score}")
                score += match_score
    if total_keys > 0:
        details_score = (score / total_keys) * 100
    else:
        details_score = 0
    title_score, title1_tokens, title2_tokens, title_intersection = title_similarity(title1, title2)
    title_comparison = f"Title Tokens (Target): {title1_tokens}\nTitle Tokens (Competitor): {title2_tokens}\nCommon Tokens: {title_intersection}\nScore: {title_score}"
    desc_score, desc1_tokens, desc2_tokens, desc_intersection = description_similarity(desc1, desc2)
    desc_comparison = f"Description Tokens (Target): {desc1_tokens}\nDescription Tokens (Competitor): {desc2_tokens}\nCommon Tokens: {desc_intersection}\nScore: {desc_score}"

    return details_score, title_score, desc_score, details_comparison, title_comparison, desc_comparison


def calculate_weighted_score(details_score, title_score, desc_score):
    weighted_score = 0.5 * details_score + 0.4 * title_score + 0.1 * desc_score
    return weighted_score


def calculate_cpi_score(price, competitor_prices):
    percentile = 100 * (competitor_prices < price).mean()
    cpi_score = 10 - (percentile / 10)
    #st.write(f"CPI Score : {cpi_score}")
    return cpi_score

def calculate_cpi_score_updated(target_price, competitor_prices):
    # Compute distances from the target price
    distances = np.abs(competitor_prices - target_price)

    # Define a weighting function: closer prices get higher weights
    max_distance = np.max(distances)
    if max_distance == 0:
        weights = np.ones_like(distances)
    else:
        weights = 1 - (distances / max_distance)

    # Calculate the weighted average of competitor prices
    weighted_average_price = np.average(competitor_prices, weights=weights)

    # Calculate CPI Score
    if weighted_average_price > 0:
        percentile = 100 * (competitor_prices < weighted_average_price).mean()
    else:
        percentile = 100

    cpi_score = 10 - (percentile / 10)
    #st.write(f'Dynamic CPI Score : {cpi_score}')
    return cpi_score


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

# Connect to S3 Bucket:
s3_client = boto3.client(
    's3',
)
bucket_name = 'anarix-cpi'
s3_folder = 'WALMART/'
#prefix = 'walmart_cpi_trial_1.csv'


def get_latest_file_from_s3(s3_folder, prefix):
    """Fetches the latest file based on LastModified timestamp for a given prefix."""
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)
    all_files = [
        obj['Key'] for obj in response.get('Contents', [])
        if obj['Key'].startswith(f"{s3_folder}{prefix}")
    ]
    
    if not all_files:
        raise FileNotFoundError(f"No files found with prefix {prefix}")

    latest_file = max(
        all_files,
        key=lambda k: s3_client.head_object(Bucket=bucket_name, Key=k)['LastModified']
    )
    return latest_file

@delayed
def load_latest_csv_from_s3(s3_folder, prefix):
    """Loads the latest CSV file for a given prefix."""
    latest_file_key = get_latest_file_from_s3(s3_folder, prefix)
    obj = s3_client.get_object(Bucket=bucket_name, Key=latest_file_key)
    return pd.read_csv(obj['Body'], low_memory=False)


# @delayed
# def load_static_file_from_s3(s3_folder, file_name):
#     """Loads a static CSV file from S3 without searching for latest version."""
#     s3_key = f"{s3_folder}{file_name}"
#     obj = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
#     return pd.read_csv(obj['Body'], low_memory=False, on_bad_lines='skip')

# Define the mapping for renaming keys in the 'Product Details' dictionary
details_key_rename_map = {
    'Bed Size': 'Size',
    'Mattress Thickness': 'Style',
    'Maximum Load Weight': 'Weight',
    'Assembled Product Dimensions (L x W x H)': 'Product Dimensions'
}

@st.cache_resource
def load_and_preprocess_data(s3_folder):
    # Load data from the single CSV file
    merged_data_df = load_latest_csv_from_s3(s3_folder, 'walmart_cpi_napqueen').compute()
    
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
    asin_keyword_df['unique_combination'] = asin_keyword_df['asin'] + '_' + asin_keyword_df['keyword']

    # Drop duplicates based on this unique marker
    asin_keyword_df = asin_keyword_df.drop_duplicates(subset='unique_combination')

    # Drop the helper column if no longer needed
    asin_keyword_df.drop(columns=['unique_combination'], inplace=True)

    # Reset index for clean output
    asin_keyword_df.reset_index(drop=True, inplace=True)

    # asin_keyword_df = asin_keyword_df.drop_duplicates()
    # asin_keyword_df = asin_keyword_df.reset_index(drop=True , inplace=True)
    # asin_keyword_df = (asin_keyword_df.groupby('asin', as_index=False).agg({'keyword': lambda x: list(set(x))}))

    return asin_keyword_df, merged_data_df

def clear_cache():
    """Clears the cache for load_and_preprocess_data."""
    load_and_preprocess_data.clear()  # Clears the cache for this function

def get_data(s3_folder, refresh=False):
    """Handles data fetching with daily refresh logic."""
    if refresh:
        # Clear cache and fetch fresh data
        clear_cache()
        st.session_state.last_refresh_time = datetime.now()
    
    # Fetch the data (from cache or fresh)
    asin_keyword_df, merged_data_df = load_and_preprocess_data(s3_folder)
    
    # Store dataframes in session state
    st.session_state['asin_keyword_df'] = asin_keyword_df
    st.session_state['merged_data_df'] = merged_data_df

    return asin_keyword_df, merged_data_df

# Ensure data is loaded into session state during app initialization
if 'asin_keyword_df' not in st.session_state:
    # Load data initially
    get_data(s3_folder)

# Initialize last_refresh_time in session state
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = datetime.min  # Default to a very old date

# Manual refresh button logic
if st.button("Refresh Data"):
    with st.spinner("Refreshing data..."):
        # Force a refresh
        asin_keyword_df, merged_data_df = get_data(
            s3_folder, refresh=True
        )
        # Update session state
        st.session_state['show_features_df'] = merged_data_df
    st.success("Data refreshed successfully!")

# Access dataframes directly from session state
asin_keyword_df = st.session_state['asin_keyword_df']
merged_data_df = st.session_state['merged_data_df']

# Display last refresh time
st.write("Data last refreshed on:", st.session_state.last_refresh_time)

# Load cached data for session state
if 'show_features_df' not in st.session_state:
    _, _, merged_data_df, _ = get_data(s3_folder)
    st.session_state['show_features_df'] = merged_data_df
else:
    merged_data_df = st.session_state['show_features_df']


def check_compulsory_features_match(target_details, compare_details, compulsory_features):

    for feature in compulsory_features:
        if feature not in target_details:
            return False
        if feature not in compare_details:
            return False
        target_value = str(target_details[feature]).lower()
        compare_value = str(compare_details[feature]).lower()
        if target_value != compare_value:
            return False

    return True

def safe_literal_eval(val):
    """Safely evaluate a string as a Python literal, or return it unchanged if not a string."""
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val  # Return original if evaluation fails
    return val

def find_similar_asins(input_asin, asin_keyword_df):

    # Retrieve the keywords associated with the input ASIN
    input_keywords = asin_keyword_df.loc[asin_keyword_df['asin'] == input_asin, 'keyword'].tolist()
    if not input_keywords:
        print(f"ASIN {input_asin} not found in the data.")
        return []

    # Convert the input keywords and selected keywords to sets for comparison
    input_keyword_set = set(input_keywords)
    selected_keywords_set = set(st.session_state.get('selected_keyword_ids', []))

    # Initialize a list to store similar ASINs
    similar_asins = []

    # Loop through other ASINs and check for any keyword_id overlap
    for idx, row in asin_keyword_df.iterrows():
        # Skip the input ASIN itself
        if row['asin'] == input_asin:
            continue

        # Check if the current ASIN's keyword has any intersection with selected keywords
        row_keyword_set = {row['keyword']}
        if selected_keywords_set.intersection(row_keyword_set):
            similar_asins.append(row['asin'])

    return similar_asins

def find_dissimilar_asins(input_asin, asin_keyword_df):

    # Retrieve the keywords associated with the input ASIN
    input_keywords = asin_keyword_df.loc[asin_keyword_df['asin'] == input_asin, 'keyword'].tolist()
    if not input_keywords:
        print(f"ASIN {input_asin} not found in the data.")
        return []

    # Convert the input keywords and selected keywords to sets for comparison
    input_keyword_set = set(input_keywords)
    selected_keywords_set = set(st.session_state.get('selected_keyword_ids', []))

    # Initialize a list to store dissimilar ASINs
    dissimilar_asins = []

    # Loop through other ASINs and check for no keyword overlap with selected keywords
    for idx, row in asin_keyword_df.iterrows():
        if row['asin'] == input_asin:
            continue  # Skip the input ASIN itself

        # Check if the current ASIN's keyword has no intersection with selected keywords
        row_keyword_set = {row['keyword']}
        if not selected_keywords_set.intersection(row_keyword_set):
            dissimilar_asins.append(row['asin'])

    return dissimilar_asins


def find_similar_products(asin, price_min, price_max, merged_data_df, compulsory_features, same_brand_option, compulsory_keywords, non_compulsory_keywords):
    # If the user selected "Include Keywords", find similar ASINs
    if keyword_option == 'Include Keywords':
        similar_asin_list = find_similar_asins(asin, asin_keyword_df)
    elif keyword_option == 'Negate Keywords':
        similar_asin_list = find_dissimilar_asins(asin, asin_keyword_df)
    else:
        similar_asin_list = []  # No filtering based on ASINs if "No Keywords" is selected

    #merged_data_df['identified_brand'] = merged_data_df['product_title'].apply(extract_brand_from_title)

    target_product = merged_data_df[merged_data_df['asin'] == asin].iloc[0]
    target_details = {**target_product['Product Details']}

    target_brand = target_product['brand']
    target_title = str(target_product['product_title']).lower()
    target_desc = str(target_product['Description']).lower()

    similarities = []
    unique_asins = set()
    seen_combinations = set()

    for index, row in merged_data_df.iterrows():
        if row['asin'] == asin:
            continue
        compare_brand = row['brand']
        if same_brand_option == 'Include Brand Products' and compare_brand != target_brand:
            continue
        if same_brand_option == 'Exclude Brand Products' and compare_brand == target_brand:
            continue
        if price_min <= row['price'] <= price_max:
            compare_details = {**row['Product Details']}

            compare_title = str(row['product_title']).lower()
            compare_desc = str(row['Description']).lower()

            compulsory_match = check_compulsory_features_match(target_details, compare_details, compulsory_features)

            title = row['product_title']

            # Ensure title is a valid string, otherwise return False
            #if isinstance(title, str):
                # Check if all keywords from compulsory_keywords are present in the title
                #all_keywords_present = all(keyword.lower() in title.lower() for keyword in compulsory_keywords)
            #else:
                #all_keywords_present = False  # Return False if the title is NaN or not a string
            
            all_keywords_present = False
            any_excluded_word_present = False  # Invalid title, fail both checks
            
            if isinstance(title, str):
                    all_keywords_present = all(keyword.lower() in title.lower() for keyword in compulsory_keywords)
                    any_excluded_word_present = any(keyword.lower() in title.lower() for keyword in non_compulsory_keywords)
            
            if any_excluded_word_present or not all_keywords_present:
                continue 
            # Append product to similarities based on keyword filtering option
            if keyword_option == 'Include Keywords':
                # Check if the product matches the ASIN list and has all keywords present in the title
                #all_keywords_present = all(keyword.lower() in compare_title for keyword in compulsory_keywords)
                if compulsory_match and (row['asin'] in similar_asin_list):
                    # Append the product to the similarities list
                    asin = row['asin']
                    combination = (compare_title, row['price'], str(compare_details))
                    if combination not in seen_combinations and asin not in unique_asins:
                        details_score, title_score, desc_score, details_comparison, title_comparison, desc_comparison = calculate_similarity(
                            target_details, compare_details, target_title, compare_title, target_desc, compare_desc
                        )
                        weighted_score = calculate_weighted_score(details_score, title_score, desc_score)
                        #st.write(f"Tuple length: {len((asin, row['product_title'], row['price'], weighted_score, details_score, title_score, desc_score, compare_details, details_comparison, title_comparison, desc_comparison, compare_brand))}")
                        if weighted_score > 0:
                            matching_features = {}
                            for feature in compulsory_features:
                                if feature in target_details and feature in compare_details:
                                    if target_details[feature] == compare_details[feature]:
                                        matching_features[feature] = compare_details[feature]
                            similarities.append(
                                (asin, row['product_title'], row['price'], weighted_score, details_score,
                                 title_score, desc_score, compare_details, details_comparison, title_comparison,
                                 desc_comparison, compare_brand, matching_features)
                            )
                        unique_asins.add(asin)
                        seen_combinations.add(combination)
            elif keyword_option == 'Negate Keywords':
                if compulsory_match and (row['asin'] in similar_asin_list):
                    # Append the product to the similarities list
                    asin = row['asin']
                    combination = (compare_title, row['price'], str(compare_details))
                    if combination not in seen_combinations and asin not in unique_asins:
                        details_score, title_score, desc_score, details_comparison, title_comparison, desc_comparison = calculate_similarity(
                            target_details, compare_details, target_title, compare_title, target_desc, compare_desc
                        )
                        weighted_score = calculate_weighted_score(details_score, title_score, desc_score)
                        if weighted_score > 0:
                            matching_features = {}
                            for feature in compulsory_features:
                                if feature in target_details and feature in compare_details:
                                    if target_details[feature] == compare_details[feature]:
                                        matching_features[feature] = compare_details[feature]
                            similarities.append(
                                (asin, row['product_title'], row['price'], weighted_score, details_score,
                                 title_score, desc_score, compare_details, details_comparison, title_comparison,
                                 desc_comparison, compare_brand, matching_features)
                            )
                        unique_asins.add(asin)
                        seen_combinations.add(combination)
            else:
                # No keywords filtering, just use compulsory_match and add product directly
                if compulsory_match:
                    asin = row['asin']
                    combination = (compare_title, row['price'], str(compare_details))
                    if combination not in seen_combinations and asin not in unique_asins:
                        details_score, title_score, desc_score, details_comparison, title_comparison, desc_comparison = calculate_similarity(
                            target_details, compare_details, target_title, compare_title, target_desc, compare_desc
                        )
                        weighted_score = calculate_weighted_score(details_score, title_score, desc_score)
                        if weighted_score > 0:
                            matching_features = {}
                            for feature in compulsory_features:
                                if feature in target_details and feature in compare_details:
                                    if target_details[feature] == compare_details[feature]:
                                        matching_features[feature] = compare_details[feature]
                            similarities.append(
                                (asin, row['product_title'], row['price'], weighted_score, details_score,
                                 title_score, desc_score, compare_details, details_comparison, title_comparison,
                                 desc_comparison, compare_brand, matching_features)
                            )
                        unique_asins.add(asin)
                        seen_combinations.add(combination)

    similarities = sorted(similarities, key=lambda x: x[3], reverse=True)
    similarities = similarities[:100]  # Limit to top 100 results
    print(len(similarities))

    return similarities


def run_analysis(asin, price_min, price_max, target_price, compulsory_features, same_brand_option, merged_data_df, compulsory_keywords, non_compulsory_keywords):
    similar_products = find_similar_products(asin, price_min, price_max, merged_data_df, compulsory_features, same_brand_option, compulsory_keywords, non_compulsory_keywords)
    prices = [p[2] for p in similar_products]
    competitor_prices = np.array(prices)
    cpi_score = calculate_cpi_score(target_price, competitor_prices)
    cpi_score_dynamic = calculate_cpi_score_updated(target_price, competitor_prices)
    target_product = merged_data_df[merged_data_df['asin'] == asin].iloc[0]
    num_competitors_found = len(similar_products)
    target_product = merged_data_df[merged_data_df['asin'] == asin].iloc[0]
    size = target_product['Product Details'].get('Size', 'N/A')
    product_dimension = target_product['Product Details'].get('Product Dimensions', 'N/A')

    # Create a DataFrame to store competitor details
    competitor_details_df = pd.DataFrame(similar_products, columns=[
        'ASIN', 'Title', 'Price', 'Weighted Score', 'Details Score',
        'Title Score', 'Description Score', 'Product Details',
        'Details Comparison', 'Title Comparison', 'Description Comparison', 'Brand', 'Matching Features'
    ])

    # Extract Product Dimension and Matching Features
    competitor_details_df['Product Dimension'] = competitor_details_df['Product Details'].apply(
        lambda details: details.get('Product Dimensions', 'N/A'))
    
    # Add matching compulsory features
    # competitor_details_df['Matching Features'] = competitor_details_df['Product Details'].apply(
    #     lambda details: {feature: details.get(feature, 'N/A') for feature in compulsory_features}
    # )

    # Filter the dataframe to include only the required columns
    competitor_details_df = competitor_details_df[['ASIN', 'Title', 'Price', 'Product Dimension', 'Brand', 'Matching Features']]
    date = merged_data_df['date'].max().strftime('%Y-%m-%d')
    competitor_details_df['date'] = date

    return asin, target_price, cpi_score, num_competitors_found, size, product_dimension, prices, competitor_details_df, cpi_score_dynamic


def show_features(asin):

    if 'show_features_df' not in st.session_state:
        st.error("DataFrame is not initialized.")
        return
    show_features_df = st.session_state['show_features_df']
    if asin not in show_features_df['asin'].values:
        st.error("ASIN not found.")
        return
    target_product = show_features_df[show_features_df['asin'] == asin].iloc[0]
    product_details = target_product['Product Details']  # **target_product['Glance Icon Details']}

    st.subheader(f"Product Details for ASIN: {asin}")

    # Display product details
    st.text("Product Details:")
    st.text(format_details(product_details))

    return product_details

s3_client = boto3.client('s3')
bucket_name = 'anarix-cpi'
csv_folder = 'WALMART/' 

# Function to generate the competitor data CSV and upload it to S3
def upload_competitor_data_to_s3(competitors_data, asin):
    # Generate CSV content
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["ASIN", "Title", "Price", "Product Dimension", "Brand", "Matching Features"])
    writer.writeheader()
    writer.writerows(competitors_data)
    csv_content = output.getvalue().encode('utf-8')

    # Define the S3 key (file name) for the CSV
    s3_key = f"{csv_folder}competitors_analysis_{asin}.csv"
    
    # Upload the CSV to S3
    s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_content, ContentType='text/csv')

    # Generate a presigned URL for downloading the file
    presigned_url = s3_client.generate_presigned_url('get_object',
        Params={'Bucket': bucket_name, 'Key': s3_key},
        ExpiresIn=3600  # URL expires in 1 hour
    )
    
    return presigned_url

def perform_scatter_plot(asin, target_price, price_min, price_max, compulsory_features, same_brand_option, merged_data_df, compulsory_keywords, non_compulsory_keywords, generate_csv=False):
    
    # Find similar products
    similar_products = find_similar_products(asin, price_min, price_max, merged_data_df, compulsory_features, same_brand_option, compulsory_keywords, non_compulsory_keywords)

    # Debug: Check the length and contents of similar_products before creating DataFrame
    #st.write(f"Number of similar products found: {len(similar_products)}")
    #st.write(f"Contents of similar_products: {similar_products}")

    # Retrieve target product information
    target_product = merged_data_df[merged_data_df['asin'] == asin].iloc[0]
    target_title = str(target_product['product_title']).lower()
    target_desc = str(target_product['Description']).lower()
    target_details = target_product['Product Details']

    # Calculate similarity scores for the target product
    details_score, title_score, desc_score, details_comparison, title_comparison, desc_comparison = calculate_similarity(
    target_details, target_details, target_title, target_title, target_desc, target_desc
    )
    weighted_score = calculate_weighted_score(details_score, title_score, desc_score)

    target_product_entry = (
    asin, target_product['product_title'], target_price, weighted_score, details_score,
    title_score, desc_score, target_details, details_comparison, title_comparison, desc_comparison, target_product['brand']
    )

    # Ensure the target product is not included in the similar products list
    similar_products = [prod for prod in similar_products if prod[0] != asin]
    similar_products.insert(0, target_product_entry)

    # Extract price and weighted scores from similar products
    prices = [p[2] for p in similar_products]
    weighted_scores = [p[3] for p in similar_products]
    product_titles = [p[1] for p in similar_products]
    asin_list = [p[0] for p in similar_products]

    # Debug: Ensure the tuple format before DataFrame creation
    #st.write(f"Final similar_products list: {similar_products}")
    #st.write(f"Tuple length in similar_products (should be 12): {len(similar_products[0]) if similar_products else 'No products found'}")

    # Prepare competitors data for CSV
    competitors_data = [
        {
            "ASIN": product[0],
            "Title": product[1],
            "Price": product[2],
            "Product Dimension": product[7].get('Product Dimensions', ''),
            "Brand": product[11],
            "Matching Features": str(product[12]) if len(product) > 12 else "No Matching Features"
        }
        for product in similar_products
    ]

    # Plot using Plotly
    fig = go.Figure()

    # Add scatter plot for similar products
    fig.add_trace(go.Scatter(
        x=list(range(len(similar_products))),
        y=prices,
        mode='markers',
        marker=dict(
            size=10,
            color=weighted_scores,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Weighted Score")
        ),
        hoverinfo='text',
        text=[f"ASIN: {a}<br>Title: {t}<br>Price: ${p:.2f}" 
              for a, t, p in zip(asin_list, product_titles, prices)],
        name='Similar Products'
    ))

    # Plot the target product separately
    fig.add_trace(go.Scatter(
        x=[0], 
        y=[target_price], 
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        hoverinfo='text',
        text=[f"ASIN: {asin}<br>Title: {target_product['product_title']}<br>Price: ${target_price:.2f}"],
        name='Target Product'
    ))

    fig.update_layout(
        title=f"Comparison of Similar Products to ASIN: {asin}",
        xaxis_title="Index",
        yaxis_title="Price ($)",
        hovermode="closest",
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",  # Anchor the legend at the bottom
            y=1.03,  # Position the legend just below the title
            xanchor="left",  # Left-align the legend
            x=0.01,  # Adjust the position horizontally
            font=dict(size=10),  # Reduce the font size for the legend
        )
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Side panel content - competitor count and null price count
    competitor_count = len(similar_products)
    price_null_count = merged_data_df[merged_data_df['asin'].isin(asin_list) & merged_data_df['price'].isnull()].shape[0]

    # Display the side panel content below the scatter plot
    st.subheader("Product Comparison Details")
    st.write(f"**Competitor Count**: {competitor_count}")
    st.write(f"**Number of Competitors with Null Price**: {price_null_count}")

    # Store competitors data in session state
    st.session_state['competitors_data'] = competitors_data

    # If user requested a CSV, upload it to S3 and provide the download link
    if generate_csv:
        download_link = upload_competitor_data_to_s3(competitors_data, asin)
        st.session_state['csv_download_link'] = download_link

    # Display CSV download link if available
    if 'csv_download_link' in st.session_state:
        st.markdown(f"[Download Competitor Analysis CSV]({st.session_state['csv_download_link']})")

    # CPI Score Polar Plot
    competitor_prices = np.array(prices)
    cpi_score = calculate_cpi_score(target_price, competitor_prices)
    dynamic_cpi_score = calculate_cpi_score_updated(target_price, competitor_prices)

    st.subheader("CPI Score Comparison")

    # Create CPI radar charts (one for static and one for dynamic CPI)
    fig_cpi, (ax_cpi, ax_dynamic_cpi) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'polar': True})

    categories = [''] * 10
    angles = np.linspace(0, np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    values = [0] * 10
    values += values[:1]

    # Plot original CPI score
    ax_cpi.fill(angles, values, color='grey', alpha=0.25)
    score_angle = (cpi_score / 10) * np.pi
    ax_cpi.plot([0, score_angle], [0, 10], color='blue', linewidth=2, linestyle='solid')
    ax_cpi.set_title("CPI Score")

    # Remove the radial tick labels (2, 4, 6, 8, 10)
    ax_cpi.set_yticklabels([])
    ax_cpi.set_xticklabels([])

    # Add CPI score number in the center of the chart
    ax_cpi.text(0, 0, f"{cpi_score:.2f}", ha='center', va='center', fontsize=20, color='blue')

    # Plot dynamic CPI score
    ax_dynamic_cpi.fill(angles, values, color='grey', alpha=0.25)
    dynamic_score_angle = (dynamic_cpi_score / 10) * np.pi
    ax_dynamic_cpi.plot([0, dynamic_score_angle], [0, 10], color='green', linewidth=2, linestyle='solid')
    ax_dynamic_cpi.set_title("Dynamic CPI Score")

    # Remove the radial tick labels (2, 4, 6, 8, 10)
    ax_dynamic_cpi.set_yticklabels([])
    ax_dynamic_cpi.set_xticklabels([])

    # Add Dynamic CPI score number in the center of the chart
    ax_dynamic_cpi.text(0, 0, f"{dynamic_cpi_score:.2f}", ha='center', va='center', fontsize=20, color='green')

    # Display CPI score plots
    st.pyplot(fig_cpi)

# Initialize session state variables with additional checks
if 'result_df' not in st.session_state or st.session_state.get('recompute', False):
    st.session_state['result_df'] = None  # Force reset on recompute flag
if 'competitor_files' not in st.session_state:
    st.session_state['competitor_files'] = {}
if 'recompute' not in st.session_state:
    st.session_state['recompute'] = False

def process_date(merged_data_df, asin, date_str, price_min, price_max, compulsory_features, same_brand_option, compulsory_keywords, non_compulsory_keywords):
    """
    This function processes data for a single date and returns the results.
    """
    df_combined = merged_data_df.copy()
    df_combined['date'] = pd.to_datetime(df_combined['date'], format='%Y-%m-%d')
    df_current_day = df_combined[df_combined['date'] == date_str]

    #st.write(f"Data for {date_str}:")
    #st.write(df_current_day.head())

    if df_current_day.empty:
        st.error(f"No data found for date: {date_str}")
        return None

    try:
        target_price = df_current_day[df_current_day['asin'] == asin]['price'].values[0]
    except IndexError:
        st.error(f"ASIN {asin} not found for date {date_str}")
        return None

    # Calling run_analysis (assuming it's available and properly defined)
    result = run_analysis(asin, price_min, price_max, target_price, compulsory_features, same_brand_option, df_current_day, compulsory_keywords, non_compulsory_keywords)
    #st.write("run_analysis result:", result)

    # Calculate the number of products with missing or invalid prices
    daily_null_count = df_current_day['price'].isna().sum() + (df_current_day['price'] == 0).sum() + (df_current_day['price'] == '').sum()
    #st.write(f"Daily null count for {date_str}: {daily_null_count}")

    return {
        'date': date_str,
        'result': result,
        'daily_null_count': daily_null_count,
        'num_competitors_found': result[3],
        'competitors': result[7]
    }

def calculate_and_plot_cpi(merged_data_df, asin_list, start_date, end_date, price_min, price_max, compulsory_features, same_brand_option):
    asin = asin_list[0]
    dates_to_process = []

    compulsory_keywords = st.session_state.get('compulsory_keywords', [])
    non_compulsory_keywords = st.session_state.get('non_compulsory_keywords', [])

    combined_competitor_df = pd.DataFrame()

    # Detect changes in inputs and set recompute flag
    if st.session_state.get('recompute', False) or st.button('Run Analysis Again'):
        # Clear previously stored result_df and competitor_files on recompute
        st.session_state['result_df'] = None
        st.session_state['competitor_files'] = {}
        st.session_state['recompute'] = False  # Reset recompute flag after clearing

        all_results = []
        competitor_count_per_day = []
        null_price_count_per_day = []

        # Process each day in the date range
        current_date = start_date
        while current_date <= end_date:
            dates_to_process.append(current_date)
            
            st.write(f"Processing date: {current_date}")
            result = process_date(merged_data_df, asin, pd.to_datetime(current_date), price_min, price_max, compulsory_features, same_brand_option, compulsory_keywords, non_compulsory_keywords)

            if result is not None:
                daily_results = result['result'][:-1]
                daily_null_count = result['daily_null_count']
                num_competitors_found = result['num_competitors_found']

                all_results.append((result['date'], *daily_results))
                competitor_count_per_day.append(num_competitors_found)
                null_price_count_per_day.append(daily_null_count)

                # Append each day's competitor details to the combined DataFrame
                competitor_details_df = result['competitors']
                if not competitor_details_df.empty:
                    competitor_details_df['Date'] = result['date']  # Add date to track individual days
                    combined_competitor_df = pd.concat([combined_competitor_df, competitor_details_df], ignore_index=True)

            
            current_date += timedelta(days=1)
            
            # Save the combined competitor details DataFrame as a single CSV if it has data
        if not combined_competitor_df.empty:
            # Generate CSV content from DataFrame
            csv_content = combined_competitor_df.to_csv(index=False).encode('utf-8')

            # Define the S3 key
            s3_key = f"NAPQUEEN/combined_competitors_{asin}_{start_date}_{end_date}.csv"

            # Upload CSV to S3 and get presigned URL
            presigned_url = upload_competitor_data_to_s3(csv_content, s3_key)

            st.session_state['csv_download_link'] = presigned_url
        else:
            st.error("No competitor data available for the selected date range.")
            # Create result DataFrame and store in session state
        result_df = pd.DataFrame(all_results,
                                columns=['Date', 'ASIN', 'Target Price', 'CPI Score', 'Number Of Competitors Found',
                                        'Size', 'Product Dimension', 'Competitor Prices', 'Dynamic CPI'])
        st.session_state['result_df'] = result_df
    else:
        # Use the cached result if it exists
        result_df = st.session_state['result_df']

    # Display the result dataframe in Streamlit
    st.subheader("Analysis Results")
    st.dataframe(result_df)

    # # Merge with other necessary data for analysis
    # price_data_df_filtered = price_data_df[price_data_df['Ad Type'] == 'SP']
    # napqueen_df = price_data_df_filtered
    # #napqueen_df['Date'] = pd.to_datetime(napqueen_df['Date'], format='%d-%m-%Y', errors='coerce')
    # napqueen_df = napqueen_df.rename(columns={'asin': 'ASIN' , 'date' : 'Date'})

    # # Clean and ensure consistent data in ASIN columns
    # result_df['ASIN'] = result_df['ASIN'].str.upper().str.strip()  # Convert to uppercase and remove spaces
    # napqueen_df['ASIN'] = napqueen_df['ASIN'].str.upper().str.strip()

    # # Ensure consistent Date format in both DataFrames
    # result_df['Date'] = pd.to_datetime(result_df['Date'], format='%Y-%m-%d')
    # napqueen_df['Date'] = pd.to_datetime(napqueen_df['Date'], format='%Y-%m-%d', errors='coerce')

    # # Investigate mixed data types and force consistent types if necessary
    # napqueen_df['ad_spend'] = pd.to_numeric(napqueen_df['ad_spend'], errors='coerce')
    # napqueen_df['orderedunits'] = pd.to_numeric(napqueen_df['orderedunits'], errors='coerce')
    # napqueen_df['orderedrevenueamount'] = pd.to_numeric(napqueen_df['orderedrevenueamount'], errors='coerce')


    # try:
    #     #result_df['Date'] = pd.to_datetime(result_df['Date'], format='%d-%m-%Y')
    #     result_df = pd.merge(result_df, napqueen_df[['Date', 'ASIN', 'ad_spend', 'orderedunits', 'orderedrevenueamount']], on=['Date', 'ASIN'], how='left')

    #     st.success("Merging successful! Displaying the merged dataframe:")
    #     st.dataframe(result_df)

    # except KeyError as e:
    #     st.error(f"KeyError: {e} - Likely missing columns during merging.")
    # except Exception as e:
    #     st.error(f"An error occurred: {e}")

    # Plot the results
    st.subheader("Time-Series Analysis Results")
    plot_results(result_df, asin_list, start_date, end_date, selected_ax4_column)

    if not combined_competitor_df.empty:
        st.subheader("Combined Competitor Data for Selected Date Range")
        st.dataframe(combined_competitor_df)

        # Display the download link for the combined CSV
        if 'csv_download_link' in st.session_state:
            st.markdown(f"[Download Combined Competitor Details CSV]({st.session_state['csv_download_link']})")
    else:
        st.write("No competitor data available for the selected date range.")

def plot_competitor_vs_null_analysis(competitor_count_per_day, null_price_count_per_day, start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Competitors Found', color='tab:blue')
    ax1.plot(dates, competitor_count_per_day, color='tab:blue', label='Competitors Found')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Null Price Values', color='tab:orange')
    ax2.plot(dates, null_price_count_per_day, color='tab:orange', label='Null Price Values')

    # Set date format and tick locator for x-axis
    ax1.xaxis.set_major_locator(mdates.DayLocator())  # Set tick locator to daily
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))  # Format date labels
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    plt.title('Competitors Found vs Null Price Values Over Time')
    fig.tight_layout()
    st.pyplot(fig)


def plot_results(result_df, asin_list, start_date, end_date, selected_ax4_column):

    for asin in asin_list:
        asin_results = result_df[result_df['ASIN'] == asin]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot CPI Score on ax1
        ax1.set_xlabel('Date')
        ax1.set_ylabel('CPI Score', color='tab:blue')
        ax1.plot(pd.to_datetime(asin_results['Date']), asin_results['CPI Score'], label='CPI Score', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        ax1.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_xlim(start_date, end_date)

        # Plot Price on ax2
        ax2 = ax1.twinx()
        ax2.set_ylabel('Price', color='tab:orange')
        ax2.plot(pd.to_datetime(asin_results['Date']), asin_results['Target Price'], label='Price', linestyle='--', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # # Plot Ad Spend on ax3
        # ax3 = ax1.twinx()
        # ax3.spines['right'].set_position(('outward', 60))  # Offset the axis to the right
        # ax3.set_ylabel('Ad Spend', color='tab:green')
        # ax3.plot(pd.to_datetime(asin_results['Date']), asin_results['ad_spend'], label='Ad Spend', linestyle='-.', color='tab:green')
        # ax3.tick_params(axis='y', labelcolor='tab:green')

        # # Plot Ordered Units on ax4
        # ax4 = ax1.twinx()
        # ax4.spines['right'].set_position(('outward', 120))  # Offset further to the right
        # ax4.set_ylabel('Ordered Units' if selected_ax4_column == 'orderedunits' else 'Ordered Revenue Amount', color='tab:purple')
        # ax4.plot(
        #     pd.to_datetime(asin_results['Date']),
        #     asin_results[selected_ax4_column],  # Select column based on user choice
        #     label='Ordered Units' if selected_ax4_column == 'orderedunits' else 'Ordered Revenue Amount',
        #     color='tab:purple'
        # )
        # ax4.tick_params(axis='y', labelcolor='tab:purple')
        
        # Add title and ensure everything fits
        plt.title(f'CPI Score and Price Over Time for ASIN {asin}')#Ad Spend, and {selected_ax4_column} Over Time for ASIN {asin}')
        fig.tight_layout()

        # Display the plot
        st.pyplot(fig)

def get_distribution_date(result_df, asin):
     # Streamlit's date input widget
    selected_date = st.date_input("Select Distribution Date", datetime.now())

    if st.button("Plot Distribution Graph"):
        plot_distribution_graph(result_df, asin, selected_date)

def plot_distribution_graph(result_df, asin, selected_date):
    asin_results = result_df[result_df['ASIN'] == asin]
    selected_data = asin_results[asin_results['Date'] == selected_date]

    if selected_data.empty:
        st.error("Error", "No data available for the selected date.")
        return

    target_price = selected_data['Target Price'].values[0]
    competitor_prices = selected_data['Competitor Prices'].values[0]

    competitor_prices = [float(price) for price in competitor_prices]

    fig = make_subplots(rows=1, cols=1)

    # Competitor Prices Histogram
    fig.add_trace(
        go.Histogram(x=competitor_prices, name='Competitors', marker_color='blue', opacity=0.7, hoverinfo='x+y',
                     xbins=dict(size=10)), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=[target_price], y=[0], mode='markers', marker=dict(color='purple', size=12), name='Target Price',
                   hoverinfo='x'), row=1, col=1)

    fig.update_layout(barmode='overlay', title_text=f'Price Distribution on {selected_date.date()}', showlegend=True)
    fig.update_traces(marker_line_width=1.2, marker_line_color='black')

    fig.update_xaxes(title_text='Price')
    fig.update_yaxes(title_text='Frequency')

    st.plotly_chart(fig)

def run_analysis_button(merged_data_df, asin, price_min, price_max, target_price, start_date, end_date, same_brand_option, compulsory_features):
    # Set recompute flag
    st.session_state['recompute'] = True
    

    st.write("Inside Analysis")

    st.session_state['selected_keyword_ids'] = get_selected_keyword_ids()
    compulsory_keywords = st.session_state.get('compulsory_keywords', [])
    non_compulsory_keywords = st.session_state.get('non_compulsory_keywords', [])

    
    merged_data_df['date'] = pd.to_datetime(merged_data_df['date'], errors='coerce')
    df_recent = merged_data_df[merged_data_df['date'] == merged_data_df['date'].max()]
    df_recent = df_recent.drop_duplicates(subset=['asin'])

    # Ensure that ASIN exists in the dataset
    if asin not in merged_data_df['asin'].values:
        st.error("Error: ASIN not found.")
        return
    
    # Check if price fields are entered
    if price_min is None or price_max is None or target_price is None:
        if price_min is None:
            st.error("Error: Minimum Price is not entered!")
        if price_max is None:
            st.error("Error: Maximum Price is not entered!")
        if target_price is None:
            st.error("Error: Target Price is not entered!")
        return
    
    # Extract the product information for the target ASIN
    target_product = merged_data_df[merged_data_df['asin'] == asin].iloc[0]
    target_brand = target_product['brand'].lower() if 'brand' in target_product else None

    if target_brand is None:
        try:
            target_brand = target_product['Product Details'].get('Brand', '').lower()
        except:
            pass

    st.write(f"Brand: {target_brand}")

    # Check if we should perform time-series analysis (only if brand == 'napqueen' and dates are provided)
    if target_brand.upper() == "NAPQUEEN" and start_date and end_date:
        perform_scatter_plot(asin, target_price, price_min, price_max, compulsory_features, same_brand_option, df_recent, compulsory_keywords, non_compulsory_keywords, generate_csv=generate_csv_option)
        calculate_and_plot_cpi(merged_data_df, [asin], start_date, end_date, price_min, price_max, compulsory_features, same_brand_option)
    else:
        # Perform scatter plot only
        perform_scatter_plot(asin, target_price, price_min, price_max, compulsory_features, same_brand_option, df_recent, compulsory_keywords, non_compulsory_keywords, generate_csv=generate_csv_option)


# Load data globally before starting the Streamlit app
df = load_and_preprocess_data(s3_folder)


def load_keywords_for_asin(input_asin, asin_keyword_df):
    """Retrieve a list of keywords associated with the input ASIN directly."""
    keywords = asin_keyword_df.loc[asin_keyword_df['asin'] == input_asin, 'keyword'].tolist()
    return keywords

# Define a function to clear session state when dates change
def clear_session_state_on_date_change():
    if 'prev_start_date' not in st.session_state:
        st.session_state['prev_start_date'] = start_date
        st.session_state['prev_end_date'] = end_date
    else:
        if st.session_state['prev_start_date'] != start_date or st.session_state['prev_end_date'] != end_date:
            # Clear the result_df and competitor_files when date range changes
            st.session_state['result_df'] = None  # Clear cached result
            st.session_state['competitor_files'] = {}  # Clear cached competitor data
            st.session_state['prev_start_date'] = start_date
            st.session_state['prev_end_date'] = end_date

# Streamlit UI for ASIN Competitor Analysis
st.title("Walmart Competitor Analysis")

# Create columns to use horizontal space
col1, col2, col3 = st.columns(3)

# Input fields for ASIN and price range
with col1:
    asin = st.text_input("Enter Item ID").upper()

with col2:
    price_min = st.number_input("Price Min", value=0.00)

with col3:
    price_max = st.number_input("Price Max", value=0.00)

# Target price input
target_price = st.number_input("Product Price", value=0.00)

generate_csv_option = st.checkbox("Generate CSV file for download", value=True)

# Checkbox for including time-series analysis, placed directly after Target Price
include_dates = st.checkbox("Include Dates for Time-Series Analysis", value=True)

# Display empty date inputs if the user opts to include dates
if include_dates:
    col4, col5 = st.columns(2)
    with col4:
        start_date = st.date_input("Start Date", value=None)  # Ensure default is empty
    with col5:
        end_date = st.date_input("End Date", value=None)  # Ensure default is empty
else:
    start_date, end_date = None, None

# Dropdown selection for ax4 column
selected_ax4_column = st.selectbox(
    "Select data for Ordered Units/Revenue Amount",
    options=["orderedunits", "orderedrevenueamount"],
    index=0  # Set "orderedunits" as default
)

# Add the session state clearing logic at the beginning of the app
clear_session_state_on_date_change()

# Radio buttons for same brand option
same_brand_option = st.radio("Same Brand Option", ('all', 'Include Brand Products', 'Exclude Brand Products'))

# Initialize session state for button click tracking
if 'show_features_clicked' not in st.session_state:
    st.session_state['show_features_clicked'] = False

# Button to toggle show/hide for product details
if st.button("Show Features"):
    if asin in merged_data_df['asin'].values:
        # Toggle the session state value: If it's True, set it to False; if it's False, set it to True
        st.session_state['show_features_clicked'] = not st.session_state['show_features_clicked']
    else:
        st.error("ASIN not found in dataset.")

# Conditionally display the product details based on the session state
if st.session_state['show_features_clicked'] and asin in merged_data_df['asin'].values:
    show_features(asin)

# Automatically display checkboxes for each product detail feature (if ASIN exists)
compulsory_features_vars = {}
if asin in merged_data_df['asin'].values:
    product_details = merged_data_df[merged_data_df['asin'] == asin].iloc[0]['Product Details']
    st.write("Select compulsory features:")
    for feature in product_details.keys():
        compulsory_features_vars[feature] = st.checkbox(f"Include {feature}", key=f"checkbox_{feature}")

# Collect selected compulsory features
compulsory_features = [feature for feature, selected in compulsory_features_vars.items() if selected]


# Add a radio button for keyword-based filtering
keyword_option = st.radio(
    "Would you like to include keywords in filtering?",
    ('No Keywords', 'Include Keywords', 'Negate Keywords')
)

def get_words_in_title(asin=None):
    """Retrieve words from the Text box and return them as a list."""
    # Use the asin to generate a unique key if necessary
    key_suffix = f"_{asin}" if asin else ""
    
    words_in_title = st.text_area("Words must be in Title", value="", height=100, key=f"words_in_title_text_area{key_suffix}")
    # Split the input text by whitespace and return it as a list of words
    words_in_list = [word.strip() for word in re.split(r'[,;\s]+', words_in_title) if word.strip()]
    # Store the list in session state to persist it across the session
    st.session_state['compulsory_keywords'] = words_in_list

    return words_in_list

compulsory_keywords = []
# Initialize the compulsory_keywords list that stores words entered by the user
compulsory_keywords = get_words_in_title(asin)

def get_exclude_words_in_title(asin=None):
    """Retrieve words from the Text box and return them as a list."""
    # Use the asin to generate a unique key if necessary
    key_suffix = f"_{asin}" if asin else ""
    # New box for "Exclude words in title"
    exclude_words_in_title = st.text_area("Exclude words in Title", value="", height=100, key=f"exclude_words_in_title_text_area{key_suffix}")
    non_compulsory_keywords = [word.strip() for word in re.split(r'[,;\s]+', exclude_words_in_title) if word.strip()]
    st.session_state['non_compulsory_keywords'] = non_compulsory_keywords  # Store in session state for persistence

    return  non_compulsory_keywords

non_compulsory_keywords = []

non_compulsory_keywords = get_exclude_words_in_title(asin)

# If the user selects "Include Keywords", allow them to select keywords from multi-select
if keyword_option == 'Include Keywords':
    def update_keyword_ids(asin):
        # Load keywords associated with the given ASIN
        keywords = load_keywords_for_asin(asin, asin_keyword_df)

        # If no keywords are found, inform the user
        if not keywords:
            st.write(f"No keywords found for ASIN: {asin}")
            return []

        # Display multi-select box for keyword options
        selected_keywords = st.multiselect("Select Keywords for the given ASIN", options=keywords)

        # Update `selected_keyword_ids` with the selected keywords directly
        st.session_state['selected_keyword_ids'] = selected_keywords

        return selected_keywords
    
    if asin:
        selected_keywords = update_keyword_ids(asin)
elif keyword_option == 'Negate Keywords':
    def update_keyword_ids(asin):

        # Load keywords associated with the given ASIN
        keywords = load_keywords_for_asin(asin, asin_keyword_df)

        # If no keywords are found, inform the user
        if not keywords:
            st.write(f"No keywords found for ASIN: {asin}")
            return []

        # Display multi-select box for keyword options
        selected_keywords = st.multiselect("Select Keywords for the given ASIN", options=keywords)

        # Update `selected_keyword_ids` with the selected keywords directly
        st.session_state['selected_keyword_ids'] = selected_keywords

        return selected_keywords 
    
    if asin:
        selected_keywords = update_keyword_ids(asin)
else:
    # If "No Keywords" is selected, the `compulsory_keywords` list remains as the user-entered words only
    selected_keywords = []


def get_selected_keyword_ids():
    # Simply return the keyword IDs stored in session state
    return st.session_state.get('selected_keyword_ids', [])

# Store input values in session state for use in re-runs
if 'asin_list' not in st.session_state:
    st.session_state['asin_list'] = [asin]
if 'price_min' not in st.session_state:
    st.session_state['price_min'] = price_min
if 'price_max' not in st.session_state:
    st.session_state['price_max'] = price_max
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = start_date
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = end_date
if 'compulsory_features' not in st.session_state:
    st.session_state['compulsory_features'] = compulsory_features
if 'same_brand_option' not in st.session_state:
    st.session_state['same_brand_option'] = same_brand_option

if st.button("Analyze"):
    run_analysis_button(merged_data_df, asin, price_min, price_max, target_price, start_date, end_date, same_brand_option, compulsory_features)
