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
from datetime import datetime , date
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

    title1_tokens = tokenize_with_delimiters(title1)
    title2_tokens = tokenize_with_delimiters(title2)

    intersection = title1_tokens.intersection(title2_tokens)
    union = title1_tokens.union(title2_tokens)

    token_similarity_score = len(intersection) / len(union)

    numeric_metric1 = extract_numeric_metric(title1)
    numeric_metric2 = extract_numeric_metric(title2)

    numeric_match_count = len(numeric_metric1.intersection(numeric_metric2))

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
    return cpi_score

def calculate_cpi_score_updated(target_price, competitor_prices):

    distances = np.abs(competitor_prices - target_price)

    max_distance = np.max(distances)
    if max_distance == 0:
        weights = np.ones_like(distances)
    else:
        weights = 1 - (distances / max_distance)

    weighted_average_price = np.average(competitor_prices, weights=weights)

    if weighted_average_price > 0:
        percentile = 100 * (competitor_prices < weighted_average_price).mean()
    else:
        percentile = 100

    cpi_score = 10 - (percentile / 10)
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


@st.cache_resource
def load_and_preprocess_data():
    asin_keyword_df = pd.read_csv("Blue_Tees_Golf/asin_keyword_id_mapping.csv", on_bad_lines='skip')
    keyword_id_df = pd.read_csv("Blue_Tees_Golf/keyword_x_keyword_id.csv", on_bad_lines='skip')

    merged_data_df = pd.read_csv("Blue_Tees_Golf/merged_data_2024-12-17.csv", on_bad_lines='skip')
    #st.write("Latest merged_data file name loaded:", merged_data_df.head())

    merged_data_df = merged_data_df.rename(columns={"Title": "product_title", "ASIN":"asin", "sale_price":"price"})
    merged_data_df['asin'] = merged_data_df['asin'].str.upper()

    def fill_missing_brand(row):
        if pd.isna(row['brand']) or row['brand'] == "":
            return extract_brand_from_title(row['product_title'])
        return row['brand']

    # Apply the fill_missing_brand function row-wise
    merged_data_df['brand'] = merged_data_df.apply(fill_missing_brand, axis=1)

    merged_data_df['price'] =pd.to_numeric(merged_data_df['price'], errors='coerce')

    merged_data_df['Product Details'] = merged_data_df['Product Details'].apply(parse_dict_str)
    merged_data_df['Glance Icon Details'] = merged_data_df['Glance Icon Details'].apply(parse_dict_str)
    merged_data_df['Option'] = merged_data_df['Option'].apply(parse_dict_str)
    merged_data_df['Drop Down'] = merged_data_df['Drop Down'].apply(parse_dict_str)

    return asin_keyword_df, keyword_id_df, merged_data_df

asin_keyword_df, keyword_id_df, merged_data_df = load_and_preprocess_data()


if 'show_features_df' not in st.session_state:
    _, _, merged_data_df = load_and_preprocess_data()
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
            return val  
    return val

def find_similar_asins(input_asin, asin_keyword_df):

    asin_keyword_df['keyword_id_list'] = asin_keyword_df['keyword_id_list'].apply(safe_literal_eval)

    input_keyword_id_list = asin_keyword_df.loc[asin_keyword_df['asin'] == input_asin, 'keyword_id_list'].values

    if len(input_keyword_id_list) == 0:
        print(f"ASIN {input_asin} not found in the data.")
        return []

    input_keyword_id_list = input_keyword_id_list[0]

    input_keyword_id_set = set(st.session_state.get('selected_keyword_ids', []))
    similar_asins = []

    for idx, row in asin_keyword_df.iterrows():
        if row['asin'] == input_asin:
            continue

        row_keyword_id_set = set(row['keyword_id_list'])
        if input_keyword_id_set.intersection(row_keyword_id_set):
            similar_asins.append(row['asin'])

    return similar_asins

def find_dissimilar_asins(input_asin, asin_keyword_df):

    asin_keyword_df['keyword_id_list'] = asin_keyword_df['keyword_id_list'].apply(safe_literal_eval)

    input_keyword_id_list = asin_keyword_df.loc[asin_keyword_df['asin'] == input_asin, 'keyword_id_list'].values

    if len(input_keyword_id_list) == 0:
        print(f"ASIN {input_asin} not found in the data.")
        return []

    input_keyword_id_list = input_keyword_id_list[0]

    input_keyword_id_set = set(st.session_state.get('selected_keyword_ids', []))

    dissimilar_asins = []

    for idx, row in asin_keyword_df.iterrows():
        if row['asin'] == input_asin:
            continue

        row_keyword_id_set = set(row['keyword_id_list'])
        if not input_keyword_id_set.intersection(row_keyword_id_set):
            dissimilar_asins.append(row['asin'])

    return dissimilar_asins


def find_similar_products(asin, price_min, price_max, merged_data_df, compulsory_features, same_brand_option, compulsory_keywords, non_compulsory_keywords):

    if keyword_option == 'Include Keywords':
        similar_asin_list = find_similar_asins(asin, asin_keyword_df)
    elif keyword_option == 'Negate Keywords':
        similar_asin_list = find_dissimilar_asins(asin, asin_keyword_df)
    else:
        similar_asin_list = [] 


    target_product = merged_data_df[merged_data_df['asin'] == asin].iloc[0]
    target_details = {**target_product['Product Details'], **target_product['Glance Icon Details']}

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
        if same_brand_option == 'only' and compare_brand != target_brand:
            continue
        if same_brand_option == 'omit' and compare_brand == target_brand:
            continue
        if price_min <= row['price'] <= price_max:
            compare_details = {**row['Product Details'], **row['Glance Icon Details']}

            compare_title = str(row['product_title']).lower()
            compare_desc = str(row['Description']).lower()

            compulsory_match = check_compulsory_features_match(target_details, compare_details, compulsory_features)

            title = row['product_title']
            
            all_keywords_present = False
            any_excluded_word_present = False 
            
            if isinstance(title, str):
                    all_keywords_present = all(keyword.lower() in title.lower() for keyword in compulsory_keywords)
                    any_excluded_word_present = any(keyword.lower() in title.lower() for keyword in non_compulsory_keywords)
            
            if any_excluded_word_present or not all_keywords_present:
                continue 
           
            if keyword_option == 'Include Keywords':
                if compulsory_match and (row['asin'] in similar_asin_list):
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

            elif keyword_option == 'Negate Keywords':
                if compulsory_match and (row['asin'] in similar_asin_list):
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
    similarities = similarities[:100] 

    return similarities


def run_analysis(asin, price_min, price_max, target_price, compulsory_features, same_brand_option, merged_data_df, compulsory_keywords, non_compulsory_keywords):
    similar_products = find_similar_products(asin, price_min, price_max, merged_data_df, compulsory_features, same_brand_option, compulsory_keywords, non_compulsory_keywords)
    prices = [p[2] for p in similar_products]
    competitor_prices = np.array(prices)
    cpi_score = calculate_cpi_score(target_price, competitor_prices)
    cpi_score_dynamic = calculate_cpi_score_updated(target_price, competitor_prices)
    num_competitors_found = len(similar_products)

    competitor_details_df = pd.DataFrame(similar_products, columns=[
        'ASIN', 'Title', 'Price', 'Weighted Score', 'Details Score',
        'Title Score', 'Description Score', 'Product Details',
        'Details Comparison', 'Title Comparison', 'Description Comparison', 'Brand', 'Matching Features'
    ])

    
    competitor_details_df = competitor_details_df[['ASIN', 'Title', 'Price', 'Brand', 'Matching Features']]
    date = merged_data_df['date'].max().strftime('%Y-%m-%d')
    competitor_details_df['date'] = date

    return asin, target_price, cpi_score, num_competitors_found, prices, competitor_details_df, cpi_score_dynamic


def show_features(asin):

    if 'show_features_df' not in st.session_state:
        st.error("DataFrame is not initialized.")
        return
    show_features_df = st.session_state['show_features_df']
    if asin not in show_features_df['asin'].values:
        st.error("ASIN not found.")
        return
    target_product = show_features_df[show_features_df['asin'] == asin].iloc[0]
    product_details = target_product['Product Details']  

    st.subheader(f"Product Details for ASIN: {asin}")
    st.text("Product Details:")
    st.text(format_details(product_details))

    return product_details


s3_client = boto3.client('s3')
bucket_name = 'anarix-cpi'
csv_folder = 'NAPQUEEN/' 


def upload_competitor_data_to_s3(csv_content, s3_key):
    """
    Uploads a CSV file to S3 and generates a presigned URL for it.

    Parameters:
        csv_content (bytes): The CSV file content as bytes.
        s3_key (str): The S3 key (file name) under which the file should be stored.

    Returns:
        str: The presigned URL for the uploaded file.
    """
    # Upload the CSV to S3
    s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_content, ContentType='text/csv')

    # Generate a presigned URL for downloading the file
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': s3_key},
        ExpiresIn=3600  # URL expires in 1 hour
    )
    
    return presigned_url


def perform_scatter_plot(asin, target_price, price_min, price_max, compulsory_features, same_brand_option, merged_data_df, compulsory_keywords, non_compulsory_keywords, generate_csv=False):
    
    similar_products = find_similar_products(asin, price_min, price_max, merged_data_df, compulsory_features, same_brand_option, compulsory_keywords, non_compulsory_keywords)

    target_product = merged_data_df[merged_data_df['asin'] == asin].iloc[0]
    target_title = str(target_product['product_title']).lower()
    target_desc = str(target_product['Description']).lower()
    target_details = target_product['Product Details']

    details_score, title_score, desc_score, details_comparison, title_comparison, desc_comparison = calculate_similarity(
    target_details, target_details, target_title, target_title, target_desc, target_desc
    )
    weighted_score = calculate_weighted_score(details_score, title_score, desc_score)

    target_product_entry = (
    asin, target_product['product_title'], target_price, weighted_score, details_score,
    title_score, desc_score, target_details, details_comparison, title_comparison, desc_comparison, target_product['brand']
    )

    similar_products = [prod for prod in similar_products if prod[0] != asin]
    similar_products.insert(0, target_product_entry)

    prices = [p[2] for p in similar_products]
    weighted_scores = [p[3] for p in similar_products]
    product_titles = [p[1] for p in similar_products]
    asin_list = [p[0] for p in similar_products]

    competitors_data = [
        {
            "ASIN": product[0],
            "Title": product[1],
            "Price": product[2],
            "Brand": product[11],
            "Matching Features": str(product[12]) if len(product) > 12 else "No Matching Features"
        }
        for product in similar_products
    ]

    
    fig = go.Figure()

    
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
            orientation="h", 
            yanchor="bottom", 
            y=1.03,  
            xanchor="left", 
            x=0.01,  
            font=dict(size=10), 
        )
    )

    
    st.plotly_chart(fig)

    
    competitor_count = len(similar_products)
    price_null_count = merged_data_df[merged_data_df['asin'].isin(asin_list) & merged_data_df['price'].isnull()].shape[0]

    st.subheader("Product Comparison Details")
    st.write(f"**Competitor Count**: {competitor_count}")
    st.write(f"**Number of Competitors with Null Price**: {price_null_count}")
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["ASIN", "Title", "Price", "Brand", "Matching Features"])
    writer.writeheader()
    writer.writerows(competitors_data)
    csv_content = output.getvalue().encode('utf-8')

    s3_key = f"{csv_folder}competitors_analysis_{asin}.csv"

    st.session_state['competitors_data'] = competitors_data

    if generate_csv:
        download_link = upload_competitor_data_to_s3(csv_content, s3_key)
        st.session_state['csv_download_link'] = download_link

    if 'csv_download_link' in st.session_state:
        st.markdown(f"[Download Competitor Analysis CSV]({st.session_state['csv_download_link']})")

    competitor_prices = np.array(prices)
    cpi_score = calculate_cpi_score(target_price, competitor_prices)
    dynamic_cpi_score = calculate_cpi_score_updated(target_price, competitor_prices)

    st.subheader("CPI Score Comparison")

    fig_cpi, (ax_cpi, ax_dynamic_cpi) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'polar': True})

    categories = [''] * 10
    angles = np.linspace(0, np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    values = [0] * 10
    values += values[:1]

    ax_cpi.fill(angles, values, color='grey', alpha=0.25)
    score_angle = (cpi_score / 10) * np.pi
    ax_cpi.plot([0, score_angle], [0, 10], color='blue', linewidth=2, linestyle='solid')
    ax_cpi.set_title("CPI Score")

    ax_cpi.set_yticklabels([])
    ax_cpi.set_xticklabels([])

    ax_cpi.text(0, 0, f"{cpi_score:.2f}", ha='center', va='center', fontsize=20, color='blue')

    ax_dynamic_cpi.fill(angles, values, color='grey', alpha=0.25)
    dynamic_score_angle = (dynamic_cpi_score / 10) * np.pi
    ax_dynamic_cpi.plot([0, dynamic_score_angle], [0, 10], color='green', linewidth=2, linestyle='solid')
    ax_dynamic_cpi.set_title("Dynamic CPI Score")

    ax_dynamic_cpi.set_yticklabels([])
    ax_dynamic_cpi.set_xticklabels([])

    ax_dynamic_cpi.text(0, 0, f"{dynamic_cpi_score:.2f}", ha='center', va='center', fontsize=20, color='green')

    st.pyplot(fig_cpi)


if 'result_df' not in st.session_state or st.session_state.get('recompute', False):
    st.session_state['result_df'] = None 
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


    if df_current_day.empty:
        st.error(f"No data found for date: {date_str}")
        return None

    try:
        target_price = df_current_day[df_current_day['asin'] == asin]['price'].values[0]
    except IndexError:
        st.error(f"ASIN {asin} not found for date {date_str}")
        return None

    result = run_analysis(asin, price_min, price_max, target_price, compulsory_features, same_brand_option, df_current_day, compulsory_keywords, non_compulsory_keywords)

    daily_null_count = df_current_day['price'].isna().sum() + (df_current_day['price'] == 0).sum() + (df_current_day['price'] == '').sum()

    return {
        'date': date_str,
        'result': result,
        'daily_null_count': daily_null_count,
        'num_competitors_found': result[3],
        'competitors': result[5]
    }

def calculate_and_plot_cpi(merged_data_df, asin_list, start_date, end_date, price_min, price_max, compulsory_features, same_brand_option):
    asin = asin_list[0]
    dates_to_process = []

    compulsory_keywords = st.session_state.get('compulsory_keywords', [])
    non_compulsory_keywords = st.session_state.get('non_compulsory_keywords', [])

    combined_competitor_df = pd.DataFrame()

    if st.session_state.get('recompute', False) or st.button('Run Analysis Again'):
        st.session_state['result_df'] = None
        st.session_state['competitor_files'] = {}
        st.session_state['recompute'] = False 

        all_results = []
        competitor_count_per_day = []
        null_price_count_per_day = []

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

                competitor_details_df = result['competitors']
                if not competitor_details_df.empty:
                    competitor_details_df['Date'] = result['date'] 
                    combined_competitor_df = pd.concat([combined_competitor_df, competitor_details_df], ignore_index=True)

            
            current_date += timedelta(days=1)
            
            if not combined_competitor_df.empty:
                csv_content = combined_competitor_df.to_csv(index=False).encode('utf-8')
                s3_key = f"NAPQUEEN/combined_competitors_{asin}_{start_date}_{end_date}.csv"
                presigned_url = upload_competitor_data_to_s3(csv_content, s3_key)

                st.session_state['csv_download_link'] = presigned_url
            else:
                st.error("No competitor data available for the selected date range.")

            result_df = pd.DataFrame(all_results,
                                 columns=['Date', 'ASIN', 'Target Price', 'CPI Score', 'Number Of Competitors Found',
                                          'Competitor Prices', 'Dynamic CPI'])
            st.session_state['result_df'] = result_df
        else:
            result_df = st.session_state['result_df']

    st.subheader("Analysis Results")
    st.dataframe(result_df)

    result_df['asin'] = result_df['asin'].str.upper().str.strip()

    result_df['Date'] = pd.to_datetime(result_df['date'], format='%Y-%m-%d')

    st.subheader("Time-Series Analysis Results")
    plot_results(result_df, asin_list, start_date, end_date)

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


def plot_results(result_df, asin_list, start_date, end_date):

    for asin in asin_list:
        asin_results = result_df[result_df['asin'] == asin]
        
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

        plt.title(f'CPI Score, Price, Over Time for ASIN {asin}')
        fig.tight_layout()

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
  
    st.session_state['recompute'] = True
    

    st.write("Inside Analysis")

    st.session_state['selected_keyword_ids'] = get_selected_keyword_ids()
    compulsory_keywords = st.session_state.get('compulsory_keywords', [])
    non_compulsory_keywords = st.session_state.get('non_compulsory_keywords', [])

    
    merged_data_df['date'] = pd.to_datetime(merged_data_df['date'], errors='coerce')
    df_recent = merged_data_df[merged_data_df['date'] == merged_data_df['date'].max()]
    df_recent = df_recent.drop_duplicates(subset=['asin'])

    if asin not in merged_data_df['asin'].values:
        st.error("Error: ASIN not found.")
        return
    
    if price_min is None or price_max is None or target_price is None:
        if price_min is None:
            st.error("Error: Minimum Price is not entered!")
        if price_max is None:
            st.error("Error: Maximum Price is not entered!")
        if target_price is None:
            st.error("Error: Target Price is not entered!")
        return
    
    target_product = merged_data_df[merged_data_df['asin'] == asin].iloc[0]
    target_brand = target_product['brand'].lower() if 'brand' in target_product else None

    if target_brand is None:
        try:
            target_brand = target_product['Product Details'].get('Brand', '').lower()
        except:
            pass

    st.write(f"Brand: {target_brand}")

    if start_date and end_date:
        perform_scatter_plot(asin, target_price, price_min, price_max, compulsory_features, same_brand_option, df_recent, compulsory_keywords, non_compulsory_keywords, generate_csv=generate_csv_option)
        calculate_and_plot_cpi(merged_data_df, [asin], start_date, end_date, price_min, price_max, compulsory_features, same_brand_option)
    else:
        perform_scatter_plot(asin, target_price, price_min, price_max, compulsory_features, same_brand_option, df_recent, compulsory_keywords, non_compulsory_keywords, generate_csv=generate_csv_option)


df = load_and_preprocess_data()


def load_keyword_ids(input_asin, asin_keyword_df):

    df_grouped = asin_keyword_df
    df_grouped['keyword_id_list'] = df_grouped['keyword_id_list'].apply(safe_literal_eval)

    input_keyword_id_list = df_grouped.loc[df_grouped['asin'] == input_asin, 'keyword_id_list'].values
    if len(input_keyword_id_list) > 0:
        return list(input_keyword_id_list[0])  
    return [] 


def load_keyword_mapping(keyword_id_df):
    df_keywords = keyword_id_df
    keyword_mapping = dict(zip(df_keywords['keyword'], df_keywords['keyword_id']))
    return keyword_mapping


def clear_session_state_on_date_change():
    if 'prev_start_date' not in st.session_state:
        st.session_state['prev_start_date'] = start_date
        st.session_state['prev_end_date'] = end_date
    else:
        if st.session_state['prev_start_date'] != start_date or st.session_state['prev_end_date'] != end_date:
            st.session_state['result_df'] = None 
            st.session_state['competitor_files'] = {} 
            st.session_state['prev_start_date'] = start_date
            st.session_state['prev_end_date'] = end_date



st.title("ASIN Competitor Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    asin = st.text_input("Enter ASIN").upper()

with col2:
    price_min = st.number_input("Price Min", value=0.00)

with col3:
    price_max = st.number_input("Price Max", value=0.00)


target_price = st.number_input("Target Price", value=0.00)

generate_csv_option = st.checkbox("Generate CSV file for download", value=True)

include_dates = st.checkbox("Include Dates for Time-Series Analysis", value=True)

if include_dates:
    col4, col5 = st.columns(2)
    with col4:
        start_date = st.date_input("Start Date", value=None)
    with col5:
        end_date = st.date_input("End Date", value=None) 
else:
    start_date, end_date = None, None


clear_session_state_on_date_change()


same_brand_option = st.radio("Same Brand Option", ('all', 'only', 'omit'))

if 'show_features_clicked' not in st.session_state:
    st.session_state['show_features_clicked'] = False


if st.button("Show Features"):
    if asin in merged_data_df['asin'].values:
        st.session_state['show_features_clicked'] = not st.session_state['show_features_clicked']
    else:
        st.error("ASIN not found in dataset.")

if st.session_state['show_features_clicked'] and asin in merged_data_df['asin'].values:
    show_features(asin)

compulsory_features_vars = {}
if asin in merged_data_df['asin'].values:
    product_details = merged_data_df[merged_data_df['asin'] == asin].iloc[0]['Product Details']
    st.write("Select compulsory features:")
    for feature in product_details.keys():
        compulsory_features_vars[feature] = st.checkbox(f"Include {feature}", key=f"checkbox_{feature}")

compulsory_features = [feature for feature, selected in compulsory_features_vars.items() if selected]

keyword_mapping = load_keyword_mapping(keyword_id_df)

keyword_option = st.radio(
    "Would you like to include keywords in filtering?",
    ('No Keywords', 'Include Keywords', 'Negate Keywords')
)

def get_words_in_title(asin=None):
    """Retrieve words from the Text box and return them as a list."""
    key_suffix = f"_{asin}" if asin else ""
    
    words_in_title = st.text_area("Words must be in Title", value="", height=100, key=f"words_in_title_text_area{key_suffix}")
    # Split the input text by whitespace and return it as a list of words
    words_in_list = [word.strip() for word in re.split(r'[,;\s]+', words_in_title) if word.strip()]
    # Store the list in session state to persist it across the session
    st.session_state['compulsory_keywords'] = words_in_list

    return words_in_list

compulsory_keywords = []
compulsory_keywords = get_words_in_title(asin)

def get_exclude_words_in_title(asin=None):
    """Retrieve words from the Text box and return them as a list."""
    key_suffix = f"_{asin}" if asin else ""
    exclude_words_in_title = st.text_area("Exclude words in Title", value="", height=100, key=f"exclude_words_in_title_text_area{key_suffix}")
    non_compulsory_keywords = [word.strip() for word in re.split(r'[,;\s]+', exclude_words_in_title) if word.strip()]
    st.session_state['non_compulsory_keywords'] = non_compulsory_keywords  # Store in session state for persistence

    return  non_compulsory_keywords

non_compulsory_keywords = []
non_compulsory_keywords = get_exclude_words_in_title(asin)

if keyword_option == 'Include Keywords':
    def update_keyword_ids(asin):
        keyword_ids = load_keyword_ids(asin, asin_keyword_df)

        keyword_options = [keyword for keyword, id in keyword_mapping.items() if id in keyword_ids]

        if not keyword_options:
            st.write(f"No keywords found for ASIN: {asin}")
            return []

        selected_keywords = st.multiselect("Select Keywords for the given ASIN", options=keyword_options)

        selected_keyword_ids = [keyword_mapping[keyword] for keyword in selected_keywords]

        st.session_state['selected_keyword_ids'] = selected_keyword_ids

        return selected_keywords 
    if asin:
        selected_keywords = update_keyword_ids(asin)
elif keyword_option == 'Negate Keywords':
    def update_keyword_ids(asin):
        keyword_ids = load_keyword_ids(asin, asin_keyword_df)

        keyword_options = [keyword for keyword, id in keyword_mapping.items() if id in keyword_ids]

        if not keyword_options:
            st.write(f"No keywords found for ASIN: {asin}")
            return []

        selected_keywords = st.multiselect("Select Keywords for the given ASIN", options=keyword_options)

        selected_keyword_ids = [keyword_mapping[keyword] for keyword in selected_keywords]

        st.session_state['selected_keyword_ids'] = selected_keyword_ids

        return selected_keywords 
    if asin:
        selected_keywords = update_keyword_ids(asin)
else:
    selected_keywords = []


def get_selected_keyword_ids():
    return st.session_state.get('selected_keyword_ids', [])


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
